from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from .config import settings
from .nodes.act import act_node
from .nodes.observe import observe_node, should_continue
from .nodes.plan import plan_node
from .nodes.respond import respond_node
from .state import AgentState

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent
_CHECKPOINT_PATH = _CHECKPOINT_DIR / "checkpoints.db"


def _redact_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if "@" not in rest:
        return url
    userinfo, host = rest.split("@", 1)
    if ":" not in userinfo:
        return url
    user, _ = userinfo.split(":", 1)
    return f"{scheme}://{user}:***@{host}"


_saver: BaseCheckpointSaver | None = None
_saver_ctx: Any = None
_saver_lock = asyncio.Lock()


async def get_saver() -> BaseCheckpointSaver:
    global _saver, _saver_ctx
    if _saver is not None:
        return _saver
    async with _saver_lock:
        if _saver is not None:
            return _saver
        if settings.database_url:
            from langgraph.checkpoint.postgres import PostgresSaver

            ctx = PostgresSaver.from_conn_string(settings.database_url)
            saver = ctx.__enter__()
            try:
                saver.setup()
            except Exception:
                ctx.__exit__(None, None, None)
                raise
            logger.info(
                "LangGraph checkpointer: Postgres at %s",
                _redact_url(settings.database_url),
            )
            _saver = saver
            _saver_ctx = ctx
        else:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            ctx = AsyncSqliteSaver.from_conn_string(str(_CHECKPOINT_PATH))
            saver = await ctx.__aenter__()
            try:
                await saver.setup()
            except Exception:
                await ctx.__aexit__(None, None, None)
                raise
            logger.info(
                "LangGraph checkpointer: AsyncSqliteSaver at %s",
                _CHECKPOINT_PATH,
            )
            _saver = saver
            _saver_ctx = ctx
        return _saver


def get_saver_sync() -> BaseCheckpointSaver:
    if _saver is not None:
        return _saver
    return asyncio.run(get_saver())


def build_graph():
    if settings.database_url:
        from langgraph.checkpoint.postgres import PostgresSaver
        ctx = PostgresSaver.from_conn_string(settings.database_url)
        saver = ctx.__enter__()
        saver.setup()
        _saver_ctx_holder = ctx
    else:
        from langgraph.checkpoint.sqlite import SqliteSaver
        _CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_CHECKPOINT_PATH), check_same_thread=False)
        saver = SqliteSaver(conn)
        _saver_ctx_holder = None

    g = StateGraph(AgentState)

    g.add_node("plan", plan_node)
    g.add_node("act", act_node)
    g.add_node("observe", observe_node)
    g.add_node("respond", respond_node)

    g.add_edge(START, "plan")
    g.add_edge("plan", "act")
    g.add_edge("act", "observe")
    g.add_conditional_edges(
        "observe",
        should_continue,
        {"act": "act", "respond": "respond"},
    )
    g.add_edge("respond", END)

    return g.compile(checkpointer=saver)


_agent_async = None
_agent_sync = None


def get_agent():
    global _agent_async
    if _agent_async is None:
        _agent_async = build_graph()
    return _agent_async


async def get_agent_async():
    global _agent_async
    if _agent_async is None:
        saver = await get_saver()
        g = StateGraph(AgentState)
        g.add_node("plan", plan_node)
        g.add_node("act", act_node)
        g.add_node("observe", observe_node)
        g.add_node("respond", respond_node)
        g.add_edge(START, "plan")
        g.add_edge("plan", "act")
        g.add_edge("act", "observe")
        g.add_conditional_edges(
            "observe",
            should_continue,
            {"act": "act", "respond": "respond"},
        )
        g.add_edge("respond", END)
        _agent_async = g.compile(checkpointer=saver)
    return _agent_async