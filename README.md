# Data Assistant Agent — Backend

FastAPI + LangGraph backend that turns a CSV/Excel file and a natural-language
question into a streamed, evidence-backed answer — complete with pandas-computed
tables and matplotlib/seaborn charts rendered in real time.

Built for the **Techvruk AI Agentic Systems Contest**.

---

## 1. Problem / Task Chosen

The contest asks for an AI agentic system that demonstrates **reasoning,
planning, tool use, and execution** instead of a one-shot LLM reply.

> **Chosen task** — *Data Assistant Agent.*
> Given a **CSV / Excel dataset** and a **natural-language question**,
> autonomously **plan**, **act** with tools (pandas, plotting, file I/O),
> **observe** results, and **respond** with a coherent answer that cites
> the chart(s) it generated.

Concretely the agent must:

1. Accept a tabular data file plus a free-form question.
2. Decompose the question into ordered steps (**planning**).
3. Choose and call real tools that actually execute code (**tool use**).
4. Inspect the result and decide whether to keep iterating or finalize
   (**reflection**).
5. Produce a final answer that **cites the chart(s) it produced**, with the
   reasoning trace streamed back to the UI.

This implementation runs **plan → act → observe → respond** as four explicit
LangGraph nodes rather than as a black-box `AgentExecutor`, so the agentic
loop is auditable end-to-end.

---

## 2. System Architecture & Workflow

### 2.1 High-level diagram

```
                  ┌─────────────────┐    POST /upload       ┌────────────────────┐
                  │   Browser / UI  │ ────────────────────► │   FastAPI backend  │
                  │  (React + Vite) │ ◄─────── SSE ──────── │   (this folder)    │
                  └────────┬────────┘   /chat events        └─────────┬──────────┘
                           │                                         │
                           │  upload CSV / Excel                     │
                           ▼                                         ▼
                  ┌─────────────────┐                       ┌────────────────────┐
                  │  on-disk store  │  parquet IPC          │   SessionManager   │
                  │  uploads/<sid>/ │ ◄──────────────────► │  (in-memory DFs)   │
                  │  artifacts/<sid>│                       └─────────┬──────────┘
                  └─────────────────┘                                 │
                                                                        │ tool calls
                                                                        ▼
                                                            ┌────────────────────┐
                                                            │     LangGraph      │
                                                            │   state machine    │
                                                            │                    │
                                                            │  START → plan      │
                                                            │   ↓                │
                                                            │   act  ──► observe │
                                                            │   ▲          │     │
                                                            │   └──────────┘     │
                                                            │          ↓         │
                                                            │       respond → END│
                                                            └─────────┬──────────┘
                                                                      │
                                                       bind_tools(ALL_TOOLS)
                                                                      │
                                                                      ▼
                                                          ┌────────────────────┐
                                                          │  Tools (10 total)  │
                                                          │  pandas_tools.py   │
                                                          │  plot_tools.py     │
                                                          │   ↓                │
                                                          │  subprocess sandbox │
                                                          │  (matplotlib Agg)  │
                                                          └─────────┬──────────┘
                                                                    │
                                                                    ▼
                                                          ┌────────────────────┐
                                                          │   Ollama LLM       │
                                                          │  (gemma4:31b-cloud │
                                                          │   via ChatOllama)  │
                                                          └────────────────────┘
```

### 2.2 Graph nodes

The agent is a LangGraph `StateGraph` with four explicit nodes — every transition
is visible in the streamed SSE event log so the UI can show each phase as it runs.

```
   ┌─────────┐    plan     ┌──────────┐
   │  PLAN   │ ──────────► │   ACT    │   LLM picks one of ALL_TOOLS,
   │  (LLM)  │             │  (tool)  │   we invoke it inline.
   └─────────┘             └────┬─────┘
                                │ tool result
                                ▼
                           ┌─────────────┐
                           │  OBSERVE    │   advance step / bump failure_count
                           │  (router)   │   - plan done    → RESPOND
                           └──────┬──────┘   - failure ≥ 3  → RESPOND
                                  │           - more steps    → ACT
                                  ▼
                           ┌─────────────┐
                           │  RESPOND    │   LLM synthesises a 3–8-sentence
                           │  (LLM)      │   markdown answer with chart links.
                           └─────────────┘
```

Wired with `START → plan → act → observe → (act | respond) → END`.

| Node | What it does |
|---|---|
| **plan** | Asks the LLM for `{"plan": [step, …]}` against the user's question + dataset metadata. Emits a `plan` SSE event with the steps. |
| **act**   | Binds `ALL_TOOLS` to the LLM, invokes the model, then runs each returned `tool_call` inline (subprocess for code/plots, in-memory for row appends). Emits `tool_call` + `tool_result`. |
| **observe** | Cheap router. Advances `current_step`, increments `failure_count`, respects `AGENT_MAX_ITERATIONS`. |
| **respond** | Builds a summary of (question, plan, per-tool outputs) and asks the LLM for a final markdown answer with `![alt](/artifacts/<sid>/<slug>.png)` references. |

### 2.3 Subprocess sandbox

`execute_pandas` and every `plot_*` tool follow the same pattern:

1. `df.to_parquet(tempfile.NamedTemporaryFile(suffix=".parquet"))`.
2. Build a Python wrapper that loads the parquet, sets
   `matplotlib.use("Agg")`, runs the user-supplied (or generated) code in a
   `try/except`, and either saves a PNG or prints stdout.
3. `subprocess.run([sys.executable, script_path], capture_output=True,
   text=True, timeout=settings.code_exec_timeout | 30)`.
4. Unlink both temp files in `finally`.

`add_row` is the only tool that mutates the session DataFrame in place; it
runs in the main process. Subprocess changes do **not** persist between
`execute_pandas` calls — they are isolated scratchpads.

### 2.4 Session persistence

| Layer | Where | Lifetime |
|---|---|---|
| Uploaded file bytes | `backend/uploads/<session_id>/` | Until `DELETE /sessions/{id}` |
| Generated charts / saved CSVs | `backend/artifacts/<session_id>/` | Until `DELETE /sessions/{id}` |
| Parsed DataFrames | In-memory `SessionManager` | Until process restart |
| Chat-history checkpoints | `backend/checkpoints.db` (SQLite) **or** your `DATABASE_URL` Postgres | Cross-restart / cross-worker |

`backend/checkpoints.db` is the LangGraph checkpointer; each `thread_id`
becomes a row of `agent_state` snapshots.

---

## 3. Setup / Run Instructions

### 3.1 Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.11+** | Pinned by `.python-version`. |
| **Ollama** | Local LLM runtime. Install from <https://ollama.com/download>. |
| **A pulled model** | Defaults to `gemma4:31b-cloud`. Override via `OLLAMA_MODEL`. |
| **uv** *(recommended)* | Fast Python package manager. Or use `pip` directly. |
| **(optional) Postgres** | Set `DATABASE_URL` if you want cross-process checkpoints. |

### 3.2 Install

```bash
cd backend
uv sync                                 # or: pip install -e .
cp .env.example .env                    # tweak OLLAMA_BASE_URL / OLLAMA_MODEL as needed
```

### 3.3 Configure `.env`

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP base URL |
| `OLLAMA_MODEL` | `gemma4:31b-cloud` | Model tag passed to `ChatOllama` |
| `OLLAMA_API_KEY` | `""` | Only needed for Ollama cloud auth |
| `HOST` | `0.0.0.0` | uvicorn bind host |
| `PORT` | `8000` | uvicorn bind port |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated allow-list |
| `AGENT_MAX_ITERATIONS` | `10` | Safety cap on `act → observe` loops |
| `CODE_EXEC_TIMEOUT` | `30` | Seconds per `execute_pandas` call |
| `DATABASE_URL` | `""` | Postgres URL → swaps SQLite checkpointer for Postgres |

### 3.4 Pull the model and start the server

```bash
# In one terminal
ollama serve &
ollama pull gemma4:31b-cloud

# In another terminal
python main.py serve                    # alias for: uvicorn app.main:app --host 0.0.0.0 --port 8000
# health check:
curl http://localhost:8000/health       # → {"status":"ok","model":"gemma4:31b-cloud","base_url":"…"}
```

### 3.5 CLI mode (no server)

For quick one-shot questions without spinning up FastAPI:

```bash
python main.py ask ../data/sample_superstore.csv \
    "Total sales and profit by region, sorted by profit descending."
```

Output: the final markdown answer plus any artifact URLs printed to stdout.

### 3.6 Development

```bash
python -m py_compile app/*.py app/*/*.py     # syntax check
python -c "from app.main import app"         # import check
uvicorn app.main:app --reload --port 8000    # hot-reload dev server
```

---

## 4. API Surface

All routes are defined in `backend/app/main.py`.

| Method | Path | Body / Params | Response |
|---|---|---|---|
| `GET`  | `/health` | — | `{status, model, base_url}` |
| `POST` | `/upload` | multipart `files: list[UploadFile]` | `UploadResponse` (session + auto-EDA + suggestions) |
| `POST` | `/sessions` | — | `SessionInfo` (empty chat session) |
| `GET`  | `/sessions/{session_id}` | — | `SessionInfo` |
| `DELETE` | `/sessions/{session_id}` | — | `{deleted: bool}` (also `rmtree`s uploads + artifacts) |
| `POST` | `/sessions/{session_id}/datasets` | multipart `files` | `UploadResponse` (no auto-EDA regen) |
| `PUT`  | `/sessions/{session_id}/datasets/{dataset_id}` | multipart `file` | `DatasetInfo` (replaces DataFrame + raw file) |
| `PATCH` | `/sessions/{session_id}/datasets/{dataset_id}` | JSON `{filename}` | `DatasetInfo` (renames in `df_meta`) |
| `DELETE` | `/sessions/{session_id}/datasets/{dataset_id}` | — | `{deleted, remaining}` |
| `POST` | `/chat` | JSON `ChatRequest`: `{session_id, thread_id?, message}` | `text/event-stream` SSE |
| `GET`  | `/artifacts/{session_id}/{file}.png` | — | PNG / CSV bytes from `backend/artifacts/<sid>/...` |

### 4.1 SSE event types emitted by `/chat`

| Event | Payload | When |
|---|---|---|
| `phase` | `{node: "plan" \| "act" \| "observe" \| "respond"}` | Each node enters |
| `plan` | `{steps: [...]}` | Once, after the planner finishes |
| `token` | `{content: "..."}` | Incremental assistant text |
| `tool_call` | `{name, input}` | When `act` invokes a tool |
| `tool_result` | `{name, output}` (truncated to 2000 chars) | When a tool returns |
| `done` | `{}` | Stream complete |
| `error` | `{message}` | Exception thrown anywhere in the graph |

---

## 5. Tools

10 `@tool`-decorated functions registered in `ALL_TOOLS` (see
`backend/app/tools/__init__.py`):

### 5.1 pandas_tools.py

| Tool | Purpose | Persists? |
|---|---|---|
| `list_datasets()` | List `{ds_id, shape, columns}` for the active session. | — |
| `profile_dataset(dataset_id)` | Shape, dtypes, null counts, unique counts, `df.describe()`, first 5 rows. | — |
| `execute_pandas(dataset_id, code)` | Run pandas code in a subprocess sandbox. `df` is pre-loaded; `print()` is captured; last-expression value (if a DataFrame/Series) is also printed. | No |
| `add_row(dataset_id, row)` | Append a coerced row to the in-memory DataFrame. | **Yes** |
| `save_dataset(dataset_id, filename="")` | Write the current DataFrame to `artifacts/<sid>/<safe>.csv` and return the public URL. | Side-effect only |

### 5.2 plot_tools.py

| Tool | Purpose |
|---|---|
| `plot_bar(dataset_id, x, y="", title, top=10)` | Seaborn bar — count-per-group if `y` is empty, else `sum(y)` per group. |
| `plot_histogram(dataset_id, column, title, bins=30)` | Matplotlib histogram. |
| `plot_scatter(dataset_id, x, y, title, hue=None)` | Seaborn scatter with optional category hue. |
| `plot_correlation(dataset_id, title)` | Heatmap of all numeric columns. |
| `plot_violin(dataset_id, columns, title, group_by=None)` | Side-by-side violins for one or more numeric columns. |

All plot tools return a string `"Chart saved at /artifacts/<sid>/<slug>.png\nchart_title: <title>"`. The respond node rewrites those URLs into the final markdown as `![title](/artifacts/<sid>/<slug>.png)`.

---

## 6. Project Layout

```
backend/
├── main.py                   # CLI entry: `python main.py [serve|ask …]`
├── pyproject.toml            # Python deps + hatchling build
├── .python-version           # pins 3.11
├── .env / .env.example       # Ollama + port + CORS + agent limits
├── checkpoints.db*           # SQLite LangGraph checkpointer (auto-created)
├── uploads/<session_id>/     # raw uploaded bytes
├── artifacts/<session_id>/   # rendered PNGs + saved CSVs
└── app/
    ├── __init__.py           # __version__ marker
    ├── main.py               # FastAPI app, all HTTP routes, SSE mapping
    ├── config.py             # frozen Settings dataclass from env
    ├── llm.py                # @lru_cache ChatOllama factory
    ├── state.py              # AgentState + ToolRecord TypedDicts
    ├── graph.py              # StateGraph assembly + saver (SQLite/Postgres)
    ├── sessions.py           # Session + SessionManager
    ├── context.py            # current_session_id ContextVar
    ├── schemas.py            # Pydantic API models
    ├── overview.py           # Auto-EDA, chart rendering, follow-up generator
    ├── nodes/
    │   ├── plan.py           # plan_node (LLM → JSON plan)
    │   ├── act.py            # act_node (LLM.bind_tools → execute)
    │   ├── observe.py        # observe_node + should_continue router
    │   └── respond.py        # respond_node (synthesise final answer)
    └── tools/
        ├── __init__.py       # ALL_TOOLS list
        ├── pandas_tools.py   # list_datasets, profile_dataset, execute_pandas, add_row, save_dataset
        └── plot_tools.py     # plot_bar, plot_histogram, plot_scatter, plot_correlation, plot_violin
```

---

## 7. Sample Input / Output

The full question bank lives at `../data/seed_questions.md` (10 questions).
Three illustrative transcripts follow.

### 7.1 Question — "Total sales and profit by region, sorted by profit descending."

```
Plan:
1. profile_dataset to confirm Region, Sales, Profit columns
2. execute_pandas: df.groupby('Region')[['Sales','Profit']].sum().sort_values('Profit', ascending=False)
3. format the answer

SSE stream:
  event: phase      data: {"node":"plan"}
  event: plan       data: {"steps":["profile_dataset","execute_pandas"]}
  event: phase      data: {"node":"act"}
  event: tool_call  data: {"name":"profile_dataset","input":{"dataset_id":"ds_a1b2c3"}}
  event: tool_result data: {"name":"profile_dataset","output":"Shape: 600 × 14 … Region (object) — 0 nulls, 4 unique …"}
  event: tool_call  data: {"name":"execute_pandas","input":{"dataset_id":"ds_a1b2c3","code":"df.groupby('Region')[['Sales','Profit']].sum().sort_values('Profit', ascending=False)"}}
  event: tool_result data: {"name":"execute_pandas","output":"Region\nWest     78,412.55   9,884.10\nEast     71,022.18   8,755.42\nCentral  67,580.40   7,991.66\nSouth    65,210.02   7,210.55"}
  event: phase      data: {"node":"respond"}
  event: token      data: {"content":"**Sales & profit by region (sorted by profit)**\n\n| Region  |    Sales |   Profit |\n|---------|---------:|---------:|\n| West    | 78,412.55 | 9,884.10 |\n…"}
  event: done       data: {}
```

### 7.2 Question — "Are higher discounts associated with lower profit? Show me a scatter."

```
Plan:
1. profile_dataset
2. execute_pandas: correlation between Discount and Profit
3. plot_scatter: Discount vs Profit

Final answer:
Yes — discount and profit are negatively correlated (Pearson r ≈ -0.42 in this
sample). Orders at 0% discount average +$58 profit; orders at ≥ 50% discount
average -$76.

![Discount vs Profit scatter](/artifacts/<sid>/discount_vs_profit.png)
```

### 7.3 Question — "Plot monthly total sales over time so I can see seasonality."

```
Plan:
1. execute_pandas: df.set_index('Order Date').resample('M')['Sales'].sum()
2. plot_bar: x=Month, y=Sales (or matplotlib line via execute_pandas)
3. respond

Artifacts emitted:
  /artifacts/<sid>/monthly_sales.png

Final answer:
Monthly sales oscillate between roughly $4k and $11k with a clear year-end
spike in November–December. The Feb–Apr trough is consistent across both years
in the sample.

![Monthly total sales](/artifacts/<sid>/monthly_sales.png)
```

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Health: model unavailable` on `/health` | Ollama not running or model not pulled | `ollama serve &` then `ollama pull <model>` |
| `execute_pandas` times out | User code is slow / infinite loop | `CODE_EXEC_TIMEOUT` already caps at 30 s by default; tighten if needed |
| 400 from `/chat` for `session_id not found` | Backend restarted, session is in-memory only | Re-upload the dataset, or set `DATABASE_URL` for cross-restart persistence |
| Chart URL returns 404 in the UI | LLM dropped the `/artifacts/<sid>/` prefix in its reply | Frontend `prepareAssistantContent()` rewrites it; ensure you're on the latest frontend |
| `add_row` "column not found" | Row dict keys don't match the DataFrame | The tool returns the actual column list in its error — fix the keys and retry |

---

## 9. License

MIT.
