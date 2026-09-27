from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ToolRecord(TypedDict, total=False):
    step: int
    name: str
    args: dict[str, Any]
    output: str
    ok: bool
    error: str | None


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    plan: list[str]
    current_step: int
    df_meta: dict[str, Any] | None
    session_id: str
    dataset_ids: list[str]
    tool_results: list[ToolRecord]
    artifacts: list[str]
    final_answer: str | None
    iteration: int
    failure_count: int