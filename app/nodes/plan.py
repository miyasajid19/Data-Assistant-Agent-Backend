from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..llm import get_llm
from ..state import AgentState

PLANNER_SYSTEM = """You are a senior data analyst breaking a user question into ordered steps.

Each step should be ONE tool call the agent will make. Reference the actual
tool names available to the agent:
  - list_datasets()
  - profile_dataset(dataset_id)
  - execute_pandas(dataset_id, code)
  - add_row(dataset_id, row)
  - save_dataset(dataset_id, filename)
  - plot_bar(dataset_id, x, y="", title, top) — y is optional; omit for count charts
  - plot_histogram(dataset_id, column, title, bins)
  - plot_scatter(dataset_id, x, y, title, hue)
  - plot_correlation(dataset_id, title)
  - plot_violin(dataset_id, columns, title, group_by)

If multiple datasets are loaded (see the metadata), call list_datasets() first
to see their ids, then reference the right dataset_id in each step. For
cross-dataset analysis, plan separate execute_pandas calls per dataset and a
final merging/joiner step.

Return ONLY a JSON object: {"plan": ["step 1", "step 2", ...]}
- 1-5 steps; empty list for trivial questions.
- Each step is a concrete tool call spelled out (with arguments) the agent will make next.
- The first step is often list_datasets + profile_dataset to confirm schema.
- You MUST use the dataset_id values from the "Known dataset ids" list below
  verbatim — never invent your own. Invented ids will fail with "dataset not found".
"""


def plan_node(state: AgentState) -> AgentState:
    last_user = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    if last_user is None:
        return {"plan": [], "current_step": 0, "iteration": state.get("iteration", 0)}

    llm = get_llm()
    known_ids = state.get("dataset_ids") or []
    ids_hint = (
        f"\nKnown dataset ids (use these verbatim in every tool call): "
        f"{known_ids}"
        if known_ids
        else "\nKnown dataset ids: (none — no CSV/Excel uploaded yet)"
    )
    prompt = (
        "Dataset metadata (may be partial): "
        + json.dumps(state.get("df_meta") or {}, default=str)
        + ids_hint
        + "\n\nUser question:\n"
        + last_user.content
    )
    try:
        resp = llm.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        plan: list[str] = []
        if match:
            parsed = json.loads(match.group(0))
            raw = parsed.get("plan", [])
            if isinstance(raw, list):
                plan = [str(s).strip() for s in raw if str(s).strip()]
    except Exception:
        plan = []

    visible = "**Plan:**\n" + (
        "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
        if plan
        else "_No plan needed — answering directly._"
    )
    return {
        "plan": plan,
        "current_step": 0,
        "iteration": state.get("iteration", 0),
        "messages": [AIMessage(content=visible, name="plan")],
    }