# Data Assistant Agent — Backend

FastAPI + LangGraph backend for the Data Assistant Agent.

## Stack

- Python 3.11, FastAPI, LangChain, LangGraph, langchain-ollama
- pandas, matplotlib, seaborn, openpyxl
- SQLite (default) or Postgres for chat-history checkpoints

## Layout

```
backend/
├── app/
│   ├── main.py            # FastAPI app, routes, SSE event mapping
│   ├── config.py          # Env-driven settings
│   ├── llm.py             # ChatOllama client
│   ├── sessions.py        # In-memory session + DataFrame registry
│   ├── context.py         # ContextVar for the active session
│   ├── schemas.py         # Pydantic request/response models
│   ├── state.py           # LangGraph AgentState TypedDict
│   ├── graph.py           # PLAN → ACT → OBSERVE → RESPOND state machine
│   ├── overview.py        # Auto-EDA on upload (dtypes, nulls, charts)
│   ├── nodes/             # plan, act, observe, respond
│   └── tools/             # 9 pandas + plot tools
├── main.py                # CLI entry: `python main.py ask <csv> "<question>"`
└── pyproject.toml
```

## Run

```bash
# 1. Install
uv sync        # or: pip install -e .

# 2. Configure
cp .env.example .env
# defaults: OLLAMA_BASE_URL=http://localhost:11434, OLLAMA_MODEL=gemma4:31b-cloud

# 3. Make sure Ollama is running with the model pulled
ollama serve &
ollama pull gemma4:31b-cloud

# 4. Start the server
uvicorn app.main:app --reload --port 8000
```

CLI one-shot mode (no server):

```bash
python main.py ask path/to/data.csv "what's the average profit by region?"
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness + model info |
| POST | `/upload` | Upload CSV/Excel → session + auto overview |
| POST | `/sessions` | Create empty session (chat without data) |
| GET | `/sessions/{id}` | Inspect a session |
| DELETE | `/sessions/{id}` | Drop a session |
| POST | `/sessions/{id}/datasets` | Add more files to a session |
| PUT | `/sessions/{id}/datasets/{ds_id}` | Replace a dataset |
| PATCH | `/sessions/{id}/datasets/{ds_id}` | Rename a dataset |
| DELETE | `/sessions/{id}/datasets/{ds_id}` | Remove a dataset |
| POST | `/chat` | SSE-streamed agent conversation |
| GET | `/artifacts/...` | Static-served chart PNGs |

`/chat` streams Server-Sent Events: `phase`, `plan`, `token`, `tool_call`, `tool_result`, `done`, `error`.

## Agent flow

```
START → plan → act → observe → (act | respond) → END
```

- **plan** — LLM breaks the user's question into ordered tool-call steps.
- **act** — LLM picks a tool (bound to `ALL_TOOLS`), we execute it inline and record the result.
- **observe** — cheap router: success advances the plan pointer; failure increments `failure_count`; 3 consecutive failures bail out.
- **respond** — LLM writes the final answer grounded in plan + tool results + chart URLs.

## Tools

9 `@tool`-decorated functions in `app/tools/`:

- `list_datasets`, `profile_dataset`, `execute_pandas` (subprocess sandbox), `add_row` (persists), `save_dataset`
- `plot_bar`, `plot_histogram`, `plot_scatter`, `plot_correlation`, `plot_violin`

`plot_bar` has a count mode: omit `y` to get rows-per-group instead of sum-per-group.

## Env vars

| Var | Default | Notes |
|-----|---------|-------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma4:31b-cloud` | Model tag |
| `OLLAMA_API_KEY` | `""` | Only needed for cloud auth |
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `8000` | Bind port |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated |
| `AGENT_MAX_ITERATIONS` | `10` | Safety cap on act→observe loops |
| `CODE_EXEC_TIMEOUT` | `30` | Seconds per `execute_pandas` call |
| `DATABASE_URL` | `""` | Set to a Postgres URL for cross-process checkpoints |

## Development

```bash
python -m py_compile app/*.py app/*/*.py    # syntax check
python -c "from app.main import app"        # import check
uvicorn app.main:app --reload --port 8000   # hot-reload server
```

Generated artifacts (charts, saved CSVs) go to `backend/artifacts/<session_id>/`. Uploaded files are kept at `backend/uploads/<session_id>/`. Chat history checkpoints live in `backend/checkpoints.db` (SQLite) or your configured Postgres.