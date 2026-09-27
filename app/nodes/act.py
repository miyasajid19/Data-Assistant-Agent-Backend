from __future__ import annotations

from langchain_core.messages import SystemMessage, ToolMessage

from ..llm import get_llm
from ..state import AgentState, ToolRecord
from ..tools import ALL_TOOLS

SYSTEM_PROMPT = """You are the Data Assistant Agent — an autonomous data analyst.

Available tools:
- list_datasets() — list loaded datasets and their columns.
- profile_dataset(dataset_id) — get shape, dtypes, nulls, describe, head.
- execute_pandas(dataset_id, code) — run pandas code in a sandbox; df is pre-loaded; print() to see output.
                       NOTE: changes made here do NOT persist to the session.
- add_row(dataset_id, row) — append a row to the dataset (PERSISTS to the session).
                       row is a dict like {"col1": val1, "col2": val2}.
- save_dataset(dataset_id, filename) — save the current dataset to a downloadable CSV at /artifacts/<session_id>/.
- plot_bar(dataset_id, x, y="", title, top) — bar chart of x.
    Pass y to sum y per group; OMIT y to count rows per group (count/frequency chart).
- plot_histogram(dataset_id, column, title, bins) — histogram of a numeric column.
- plot_scatter(dataset_id, x, y, title, hue) — scatter of x vs y, optional hue.
- plot_correlation(dataset_id, title) — correlation heatmap of numeric columns.
- plot_violin(dataset_id, columns, title, group_by) — distribution-shape (violin) plot; pass a comma-separated list of columns to compare them side-by-side.

Workflow:
1. If you haven't yet, call list_datasets then profile_dataset on the relevant dataset.
2. Use execute_pandas to compute answers. Use print() and reference last-expression values.
3. When the user asks to MODIFY the data (add a row, delete a row, change a value),
   use the dedicated add_row tool — NOT execute_pandas (subprocess changes are lost).
4. When the user asks to SAVE / DOWNLOAD the data, call save_dataset.
5. When the user asks for a chart, ALWAYS call the appropriate plot_* tool — don't just
   describe the data in prose. For "bar chart of X" / "count of X" / "how many per X"
   queries, OMIT the y argument to plot_bar (it counts rows per group internally —
   do NOT try the two-step "add a count column first" recipe; execute_pandas changes
   do not persist, so that recipe silently fails).
6. When you have enough information, write a clear, well-formatted final answer
   (markdown OK) with the chart references inline.

CRITICAL — TABLE FORMAT:
- Whenever you need to show tabular data to the user, ALWAYS use **GitHub-flavored
  markdown tables** with pipe characters and a header separator row. Example:
    | name  | age | sex    |
    | ----- | --- | ------ |
    | sajid | 22  | male   |
    | aisha | 30  | female |
- NEVER paste raw `print(df)` output (space-aligned text) into your answer — the
  frontend renders pipes as tables but renders plain space-aligned text as a paragraph.
- For pandas Series or DataFrames, write them as markdown tables yourself.
- For a single row or short list, prefer a markdown bullet list over a table.

Rules:
- Always ground numbers in actual tool output. Never fabricate values.
- Prefer 1-3 well-chosen tool calls over many speculative ones.
- If a tool returns an error, read the error and adjust your approach.
- When the answer is complete, do NOT call any more tools — just respond.
- You MUST use the dataset_id values from the "Known dataset ids" line below
  verbatim — never invent your own. An invented id will fail with
  "dataset not found" and waste a turn.
"""


def act_node(state: AgentState) -> AgentState:
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    known_ids = state.get("dataset_ids") or []
    sys_text = SYSTEM_PROMPT
    if known_ids:
        sys_text += f"\n\nKnown dataset ids (use these verbatim in every tool call): {known_ids}\n"
    full_msgs = [SystemMessage(content=sys_text)] + list(state.get("messages", []))
    resp = llm_with_tools.invoke(full_msgs)

    if not getattr(resp, "tool_calls", None):
        return {
            "messages": [resp],
            "final_answer": resp.content if isinstance(resp.content, str) else str(resp.content),
        }

    tool_messages: list[ToolMessage] = []
    new_records: list[ToolRecord] = []
    for tc in resp.tool_calls:
        name = tc["name"]
        args = tc.get("args", {}) or {}
        tool_fn = next((t for t in ALL_TOOLS if t.name == name), None)
        if tool_fn is None:
            tool_messages.append(
                ToolMessage(content=f"Error: unknown tool '{name}'.", tool_call_id=tc["id"])
            )
            new_records.append(
                ToolRecord(step=state.get("current_step", 0), name=name, args=args, output="", ok=False, error="unknown tool")
            )
            continue
        try:
            out = tool_fn.invoke(args)
            out_str = str(out) if out is not None else ""
            ok = True
            err = None
        except Exception as e:
            out_str = f"Error: {type(e).__name__}: {e}"
            ok = False
            err = str(e)
        tool_messages.append(ToolMessage(content=out_str[:8000], tool_call_id=tc["id"]))
        new_records.append(
            ToolRecord(
                step=state.get("current_step", 0),
                name=name,
                args=args,
                output=out_str[:8000],
                ok=ok,
                error=err,
            )
        )

    return {
        "messages": [resp, *tool_messages],
        "tool_results": state.get("tool_results", []) + new_records,
        "artifacts": state.get("artifacts", []) + _extract_artifacts(new_records),
        "iteration": state.get("iteration", 0) + 1,
    }


def _extract_artifacts(records: list[ToolRecord]) -> list[str]:
    out: list[str] = []
    for r in records:
        if r.get("name", "").startswith("plot_") and r.get("ok"):
            for line in (r.get("output") or "").splitlines():
                if "/artifacts/" in line:
                    out.append(line.strip())
    return out