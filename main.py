"""Entry point.

Two ways to run:
  1) CLI:    python main.py path/to/data.csv "your question"
  2) Server: python main.py serve  (alias for uvicorn app.main:app)

Both share the same LangGraph agent defined in app.graph.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.config import settings
from app.context import current_session_id
from app.graph import get_agent
from app.sessions import SessionManager
from app.tools.pandas_tools import set_session_manager as bind_sessions_to_pandas


def _build_session_manager_for_cli() -> SessionManager:
    base = Path(__file__).resolve().parent
    mgr = SessionManager(base)
    bind_sessions_to_pandas(mgr)
    return mgr


def cmd_serve() -> None:
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
    )


async def _run_cli(dataset_path: str, question: str, thread_id: str | None = None) -> None:
    p = Path(dataset_path)
    if not p.exists():
        print(f"Dataset not found: {p}", file=sys.stderr)
        sys.exit(2)

    mgr = _build_session_manager_for_cli()
    session = mgr.create()
    content = p.read_bytes()
    if p.suffix.lower() in {".xlsx", ".xls"}:
        ds_id, df = mgr.load_excel(session, p.name, content)
    else:
        ds_id, df = mgr.load_csv(session, p.name, content)
    session.df_meta = {  # type: ignore[attr-defined]
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "first_dataset_id": ds_id,
    }
    print(f"[cli] loaded {p.name} -> session {session.id}, dataset {ds_id}, shape {df.shape}")

    agent = get_agent()
    cfg = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())[:8]}}
    token = current_session_id.set(session.id)
    try:
        result = await agent.ainvoke(
            {
                "messages": [HumanMessage(content=question)],
                "plan": [],
                "current_step": 0,
                "df_meta": session.df_meta,  # type: ignore[attr-defined]
                "tool_results": [],
                "artifacts": [],
                "final_answer": None,
                "iteration": 0,
                "failure_count": 0,
            },
            config=cfg,
        )
    finally:
        current_session_id.reset(token)

    print("\n=== FINAL ANSWER ===\n")
    print(result.get("final_answer") or "(no answer)")
    print("\n=== ARTIFACTS ===")
    for a in result.get("artifacts", []) or []:
        print(" -", a)


def cmd_cli(dataset_path: str, question: str) -> None:
    asyncio.run(_run_cli(dataset_path, question))


def main() -> None:
    parser = argparse.ArgumentParser(description="Data Assistant Agent")
    sub = parser.add_subparsers(dest="cmd", required=False)

    sub.add_parser("serve", help="Run the FastAPI server (uvicorn)")

    cli = sub.add_parser("ask", help="Ask a one-shot question against a local CSV")
    cli.add_argument("dataset", help="Path to a .csv / .xlsx file")
    cli.add_argument("question", help="Natural-language question")

    args = parser.parse_args()
    if args.cmd == "serve" or args.cmd is None:
        # Default: serve the API
        cmd_serve()
    elif args.cmd == "ask":
        cmd_cli(args.dataset, args.question)


if __name__ == "__main__":
    main()
