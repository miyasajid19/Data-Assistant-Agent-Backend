from __future__ import annotations

import io
import shutil
import threading
import uuid
from pathlib import Path

import pandas as pd


class Session:
    def __init__(self, session_id: str, base_dir: Path) -> None:
        self.id = session_id
        self.base_dir = base_dir
        self.datasets: dict[str, pd.DataFrame] = {}
        self.uploads_dir = base_dir / "uploads" / session_id
        self.artifacts_dir = base_dir / settings_artifacts_subdir(base_dir) / session_id
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def add_dataset(self, df: pd.DataFrame, filename: str, raw_bytes: bytes) -> str:
        ds_id = f"ds_{uuid.uuid4().hex[:10]}"
        with self._lock:
            self.datasets[ds_id] = df
        path = self.uploads_dir / filename
        path.write_bytes(raw_bytes)
        return ds_id


def settings_artifacts_subdir(base_dir: Path) -> str:
    from .config import settings as _settings

    return _settings.artifacts_dirname


class SessionManager:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        (base_dir / "uploads").mkdir(exist_ok=True)
        (base_dir / settings_artifacts_subdir(base_dir)).mkdir(exist_ok=True)
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        sid = uuid.uuid4().hex[:12]
        session = Session(sid, self.base_dir)
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        shutil.rmtree(session.uploads_dir, ignore_errors=True)
        shutil.rmtree(session.artifacts_dir, ignore_errors=True)
        return True

    def load_csv(self, session: Session, filename: str, content: bytes) -> tuple[str, pd.DataFrame]:
        df = pd.read_csv(io.BytesIO(content))
        ds_id = session.add_dataset(df, filename, content)
        return ds_id, df

    def load_excel(self, session: Session, filename: str, content: bytes) -> tuple[str, pd.DataFrame]:
        df = pd.read_excel(io.BytesIO(content))
        ds_id = session.add_dataset(df, filename, content)
        return ds_id, df