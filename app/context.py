from __future__ import annotations

from contextvars import ContextVar

current_session_id: ContextVar[str | None] = ContextVar(
    "current_session_id", default=None
)