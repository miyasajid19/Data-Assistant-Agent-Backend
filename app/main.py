from __future__ import annotations

import asyncio
import io
import json
import logging
import re
from pathlib import Path
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .context import current_session_id
from .graph import get_agent
from .llm import get_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from .overview import (
    compute_overview,
    detect_cross_dataset_joins,
    compute_combined_summary,
    overview_to_dict,
    generate_followups,
)
from .schemas import ChatRequest, DatasetInfo, SessionInfo, UploadResponse
from .sessions import Session, SessionManager
from .tools.pandas_tools import set_session_manager as bind_sessions_to_pandas

logger = logging.getLogger("data_assistant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
ARTIFACTS_DIR = BASE_DIR / settings.artifacts_dirname
UPLOADS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

session_manager = SessionManager(BASE_DIR)
bind_sessions_to_pandas(session_manager)

_chat_history: dict[str, list] = {}

CHAT_ONLY_SYSTEM_PROMPT = """You are Data Assistant, a friendly, helpful AI companion.

You can chat freely about anything — explain concepts, brainstorm, draft text, answer questions, write code, summarize, etc. Be concise but thorough; use markdown for structure.

If the user asks for analysis of a CSV, dataset, or anything that requires their data, tell them to upload a file using the paperclip (📎) icon at the bottom of the chat. Don't pretend to have data you don't have — just point them at the upload.

When the user uploads data, this thread switches to data-analysis mode automatically. Until then, you're a normal conversational assistant."""

app = FastAPI(title="Data Assistant Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "base_url": settings.ollama_base_url,
    }


@app.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...)) -> UploadResponse:
    if not files:
        raise HTTPException(400, "No files provided")
    session = session_manager.create()
    infos: list[DatasetInfo] = []
    datasets_meta: list[dict] = []
    try:
        for f in files:
            content = await f.read()
            if not content:
                continue
            filename = f.filename or "uploaded.csv"
            try:
                if filename.lower().endswith((".xlsx", ".xls")):
                    ds_id, df = session_manager.load_excel(session, filename, content)
                else:
                    ds_id, df = session_manager.load_csv(session, filename, content)
            except Exception as e:
                raise HTTPException(400, f"Failed to parse '{filename}': {type(e).__name__}: {e}")

            preview = df.head(5).fillna("").astype(str).to_dict(orient="records")
            infos.append(
                DatasetInfo(
                    id=ds_id,
                    filename=filename,
                    rows=len(df),
                    columns=list(df.columns),
                    dtypes={c: str(t) for c, t in df.dtypes.items()},
                    preview=preview,
                    memory_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                )
            )
            datasets_meta.append(
                {
                    "id": ds_id,
                    "filename": filename,
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                }
            )
    except HTTPException:
        session_manager.delete(session.id)
        raise

    if not infos:
        session_manager.delete(session.id)
        raise HTTPException(400, "All uploaded files were empty")

    session.df_meta = {"datasets": datasets_meta}  # type: ignore[attr-defined]

    overviews = []
    for info in infos:
        df = session.datasets[info.id]
        ov = compute_overview(
            dataset_id=info.id,
            df=df,
            filename=info.filename,
            artifacts_dir=session.artifacts_dir,
            session_id=session.id,
        )
        overviews.append(ov)
    joins = detect_cross_dataset_joins(overviews)
    for ov, j in zip(overviews, [ov.potential_joins for ov in overviews]):
        ov.potential_joins = joins

    followups = generate_followups(
        overviews[0],
        cross_dataset_filenames=[ov.filename for ov in overviews] if len(overviews) > 1 else None,
    )

    return UploadResponse(
        session_id=session.id,
        datasets=infos,
        overviews=[overview_to_dict(o) for o in overviews],
        combined_summary=compute_combined_summary(overviews),
        cross_dataset_joins=joins,
        suggested_questions=followups,
    )


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    s: Session | None = session_manager.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    return SessionInfo(
        session_id=s.id,
        datasets=[
            {"id": ds_id, "shape": list(df.shape), "columns": list(df.columns)}
            for ds_id, df in s.datasets.items()
        ],
    )


@app.post("/sessions", response_model=SessionInfo)
async def create_empty_session() -> SessionInfo:
    s = session_manager.create()
    return SessionInfo(session_id=s.id, datasets=[])


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    ok = session_manager.delete(session_id)
    return {"deleted": ok}


def _build_dataset_info(ds_id: str, df, filename: str) -> DatasetInfo:
    preview = df.head(5).fillna("").astype(str).to_dict(orient="records")
    return DatasetInfo(
        id=ds_id,
        filename=filename,
        rows=len(df),
        columns=list(df.columns),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        preview=preview,
        memory_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
    )


def _safe_filename(name: str) -> str:
    name = name.split("/")[-1].split("\\")[-1].strip() or "dataset.csv"
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "dataset.csv"


def _parse_upload(filename: str, content: bytes):
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content))


@app.post("/sessions/{session_id}/datasets", response_model=UploadResponse)
async def add_dataset(
    session_id: str,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    s = session_manager.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    if not files:
        raise HTTPException(400, "No files provided")

    new_meta: list[dict] = []
    for f in files:
        content = await f.read()
        if not content:
            continue
        filename = _safe_filename(f.filename or "uploaded.csv")
        try:
            df = _parse_upload(filename, content)
        except Exception as e:
            raise HTTPException(400, f"Failed to parse '{filename}': {type(e).__name__}: {e}")
        s.add_dataset(df, filename, content)
        new_meta.append({
            "id": s.datasets.__class__.__name__,
            "filename": filename,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        })

    if not new_meta:
        raise HTTPException(400, "All uploaded files were empty")

    new_meta = []
    for ds_id, df in list(s.datasets.items())[-len(new_meta):]:
        new_meta.append({
            "id": ds_id,
            "filename": _filename_for(s, ds_id),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        })

    existing_meta = (getattr(s, "df_meta", None) or {}).get("datasets", [])
    s.df_meta = {"datasets": existing_meta + new_meta}  # type: ignore[attr-defined]

    all_infos = [
        _build_dataset_info(ds_id, df, _filename_for(s, ds_id))
        for ds_id, df in s.datasets.items()
    ]
    return UploadResponse(
        session_id=s.id,
        datasets=all_infos,
        overviews=[],
        combined_summary="",
        cross_dataset_joins=[],
    )


@app.put("/sessions/{session_id}/datasets/{dataset_id}", response_model=DatasetInfo)
async def replace_dataset(
    session_id: str,
    dataset_id: str,
    file: UploadFile = File(...),
) -> DatasetInfo:
    s = session_manager.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    if dataset_id not in s.datasets:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found in session")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    filename = _safe_filename(file.filename or f"{dataset_id}.csv")
    try:
        df = _parse_upload(filename, content)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse '{filename}': {type(e).__name__}: {e}")

    s.datasets[dataset_id] = df
    (s.uploads_dir / filename).write_bytes(content)

    if getattr(s, "df_meta", None):
        for entry in s.df_meta.get("datasets", []):
            if entry.get("id") == dataset_id:
                entry.update({
                    "filename": filename,
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                })
                break

    return _build_dataset_info(dataset_id, df, filename)


@app.patch("/sessions/{session_id}/datasets/{dataset_id}", response_model=DatasetInfo)
async def rename_dataset(
    session_id: str,
    dataset_id: str,
    body: dict,
) -> DatasetInfo:
    s = session_manager.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    if dataset_id not in s.datasets:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found in session")

    new_name = _safe_filename(str(body.get("filename", "")).strip() or f"{dataset_id}.csv")
    df = s.datasets[dataset_id]

    if getattr(s, "df_meta", None):
        for entry in s.df_meta.get("datasets", []):
            if entry.get("id") == dataset_id:
                entry["filename"] = new_name
                break

    return _build_dataset_info(dataset_id, df, new_name)


@app.delete("/sessions/{session_id}/datasets/{dataset_id}")
async def delete_dataset(session_id: str, dataset_id: str) -> dict:
    s = session_manager.get(session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    if dataset_id not in s.datasets:
        raise HTTPException(404, f"Dataset '{dataset_id}' not found in session")

    del s.datasets[dataset_id]

    if getattr(s, "df_meta", None):
        s.df_meta["datasets"] = [
            entry for entry in s.df_meta.get("datasets", []) if entry.get("id") != dataset_id
        ]

    return {"deleted": dataset_id, "remaining": len(s.datasets)}


def _filename_for(session, dataset_id: str) -> str:
    if getattr(session, "df_meta", None):
        for entry in session.df_meta.get("datasets", []):
            if entry.get("id") == dataset_id:
                return entry.get("filename") or dataset_id
    return f"{dataset_id}.csv"


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    s = session_manager.get(req.session_id)
    if s is None:
        raise HTTPException(404, "Session not found")

    tid = req.thread_id or "default"

    if not s.datasets:
        return _chat_only_stream(req, tid)

    from .graph import get_agent_async

    agent = await get_agent_async()
    cfg = {"configurable": {"thread_id": tid}}

    async def event_stream() -> AsyncIterator[str]:
        token = current_session_id.set(req.session_id)
        try:
            from langchain_core.messages import HumanMessage

            initial_state = {
                "messages": [HumanMessage(content=req.message)],
                "plan": [],
                "current_step": 0,
                "df_meta": getattr(s, "df_meta", None),
                "session_id": s.id,
                "dataset_ids": list(s.datasets.keys()),
                "tool_results": [],
                "artifacts": [],
                "final_answer": None,
                "iteration": 0,
                "failure_count": 0,
            }

            try:
                async for event in agent.astream_events(initial_state, config=cfg, version="v2"):
                    kind = event.get("event", "")
                    name = event.get("name", "")
                    data = event.get("data", {}) or {}
                    payload = _map_event(kind, name, data)
                    if payload is not None:
                        yield f"event: {payload['type']}\ndata: {json.dumps(payload['data'])}\n\n"
                        await asyncio.sleep(0)
                yield "event: done\ndata: {}\n\n"
            except Exception as e:
                logger.exception("agent stream failed")
                yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
        finally:
            current_session_id.reset(token)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _chat_only_stream(req: ChatRequest, tid: str) -> StreamingResponse:
    history = _chat_history.setdefault(tid, [])
    history.append(HumanMessage(content=req.message))

    async def event_stream() -> AsyncIterator[str]:
        llm = get_llm()
        messages = [SystemMessage(content=CHAT_ONLY_SYSTEM_PROMPT), *history]
        full = ""
        try:
            async for chunk in llm.astream(messages):
                content = getattr(chunk, "content", None)
                if isinstance(content, str) and content:
                    full += content
                    yield f"event: token\ndata: {json.dumps({'content': content})}\n\n"
                    await asyncio.sleep(0)
            history.append(AIMessage(content=full))
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.exception("chat-only stream failed")
            if full:
                history.append(AIMessage(content=full))
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _map_event(kind: str, name: str, data: dict) -> dict | None:
    if kind == "on_chain_start" and name in {"plan", "act", "observe", "respond"}:
        return {"type": "phase", "data": {"node": name}}
    if kind == "on_chain_end" and name == "plan":
        output = data.get("output") or {}
        if isinstance(output, dict):
            plan = output.get("plan") or []
            if plan:
                return {"type": "plan", "data": {"steps": list(plan)}}
    if kind == "on_chat_model_stream":
        chunk = data.get("chunk")
        if chunk is None:
            return None
        content = getattr(chunk, "content", None)
        if isinstance(content, str) and content:
            return {"type": "token", "data": {"content": content}}
        return None
    if kind == "on_tool_start":
        return {
            "type": "tool_call",
            "data": {"name": name, "input": data.get("input", {}) or {}},
        }
    if kind == "on_tool_end":
        out = data.get("output")
        out_str = str(out) if out is not None else ""
        return {"type": "tool_result", "data": {"name": name, "output": out_str[:2000]}}
    return None


app.mount("/artifacts", StaticFiles(directory=str(ARTIFACTS_DIR)), name="artifacts")
