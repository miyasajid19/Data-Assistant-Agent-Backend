from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from ..state import AgentState

RESPOND_SYSTEM = """You are the Data Assistant Agent writing the final answer.

Write a clear, concise (3-8 sentence) answer to the user's original question.
Use the plan + tool results to ground every claim in real numbers from the data.

CRITICAL — DO NOT echo internal state:
- NEVER include the plan as JSON (e.g. `{"plan": [...]}`). The user already sees the
  plan in a dedicated card above. Just write the final prose answer.
- NEVER paste raw `print(df)` output, function-call arguments, tool-call JSON, or
  any internal protocol artifacts. Those are for the plan/tool panels, not the answer.

CRITICAL — Include generated charts inline using markdown image syntax.
- The current session_id is given to you explicitly below — use it verbatim
  in every chart URL. Do NOT invent your own session id, or the images will
  404 and the user will think nothing was saved.
- Copy each chart URL verbatim from the "Artifacts:" list in the context
  (those are the verified, working URLs). If a chart URL there doesn't
  match the current session_id, that's still the right URL — just use it.

Example:
    Here's the chart of profit by sub-category:
    ![Profit by Sub-Category](/artifacts/<SESSION_ID>/<slug>.png)

Use markdown image syntax — NOT raw `<img>` or `<span>` tags. The frontend renders
anything matching /artifacts/*.png as a clickable chart with a download button.
DO NOT cite charts as [figure-1] alone — that won't render.

CRITICAL — TABLE FORMAT:
- When showing tabular data, ALWAYS use GitHub-flavored markdown tables with pipes:
    | name  | age | sex    |
    | ----- | --- | ------ |
    | sajid | 22  | male   |
- NEVER paste raw `print(df)` output (space-aligned text) into your answer.

Format the rest of the response with markdown (tables, bullets, **bold** for key numbers).
If a tool errored, mention what went wrong and what you tried.
"""


def respond_node(state: AgentState) -> AgentState:
    if state.get("final_answer"):
        return {}

    llm = get_llm()
    plan = state.get("plan", []) or []
    artifacts = state.get("artifacts", []) or []
    results = state.get("tool_results", []) or []

    user_q = next(
        (m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        "",
    )

    plan_lines = [f"  - {s}" for s in plan] if plan else ["  (no plan)"]
    real_session_id = state.get("session_id") or "<unknown>"
    summary_lines = [
        f"Original question: {user_q}",
        "",
        f"Current session_id (use this verbatim in every chart URL): {real_session_id}",
        "",
        "Plan executed:",
        *plan_lines,
        "",
        f"Tool results ({len(results)} call(s)):",
    ]
    for i, r in enumerate(results, 1):
        ok = "OK" if r.get("ok") else "FAIL"
        out = (r.get("output") or "")[:400]
        summary_lines.append(f"  {i}. [{ok}] {r.get('name')}({r.get('args')}) -> {out}")
    if artifacts:
        summary_lines += ["", "Artifacts:", *(f"  - {a}" for a in artifacts)]

    messages = [
        SystemMessage(content=RESPOND_SYSTEM),
        HumanMessage(content="\n".join(summary_lines)),
    ]
    try:
        resp = llm.invoke(messages)
        answer = resp.content if isinstance(resp.content, str) else str(resp.content)
    except Exception as e:
        answer = (
            "I ran the analysis but couldn't synthesize a final answer due to: "
            f"{type(e).__name__}: {e}\n\n"
            "Here is what I found:\n\n"
            + "\n".join(summary_lines)
        )

    return {"messages": [resp] if 'resp' in locals() else [], "final_answer": answer}