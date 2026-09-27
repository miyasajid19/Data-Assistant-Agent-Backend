from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    ollama_api_key: str

    host: str
    port: int
    cors_origins: tuple[str, ...]

    agent_max_iterations: int
    code_exec_timeout: int
    artifacts_dirname: str = "artifacts"

    database_url: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return list(self.cors_origins)


def _load_settings() -> Settings:
    origins = tuple(
        o.strip()
        for o in os.environ.get(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
        ).split(",")
        if o.strip()
    )
    return Settings(
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud"),
        ollama_api_key=os.environ.get("OLLAMA_API_KEY", ""),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        cors_origins=origins,
        agent_max_iterations=int(os.environ.get("AGENT_MAX_ITERATIONS", "10")),
        code_exec_timeout=int(os.environ.get("CODE_EXEC_TIMEOUT", "30")),
        database_url=os.environ.get("DATABASE_URL", "").strip(),
    )


settings = _load_settings()