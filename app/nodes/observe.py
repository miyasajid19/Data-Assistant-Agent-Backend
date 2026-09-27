from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..config import settings
from ..state import AgentState


def observe_node(state: AgentState) -> AgentState:
    iteration = state.get("iteration", 0)
    plan = state.get("plan", []) or []
    step = state.get("current_step", 0)
    failure_count = state.get("failure_count", 0)
    tool_results = state.get("tool_results", []) or []

    if iteration >= settings.agent_max_iterations:
        return {"failure_count": failure_count}

    last = tool_results[-1] if tool_results else None
    if last is None:
        return {"current_step": step, "failure_count": failure_count}

    if not last.get("ok"):
        return {"current_step": step, "failure_count": failure_count + 1}

    return {
        "current_step": min(step + 1, max(len(plan) - 1, 0)),
        "failure_count": 0,
    }


def should_continue(state: AgentState) -> str:
    if state.get("final_answer"):
        return "respond"

    if state.get("failure_count", 0) >= 3:
        return "respond"

    last_msg = state.get("messages", [])[-1] if state.get("messages") else None

    if isinstance(last_msg, ToolMessage):
        return "act"

    if isinstance(last_msg, AIMessage):
        if not getattr(last_msg, "tool_calls", None):
            return "respond"
        return "act"

    return "respond"