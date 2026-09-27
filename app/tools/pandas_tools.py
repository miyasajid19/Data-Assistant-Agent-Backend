from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from ..config import settings
from ..context import current_session_id
from ..sessions import SessionManager

_session_manager: SessionManager | None = None


def set_session_manager(mgr: SessionManager) -> None:
    global _session_manager
    _session_manager = mgr


def _get_session():
    sid = current_session_id.get()
    if sid is None or _session_manager is None:
        return None
    return _session_manager.get(sid)


@tool
def list_datasets() -> str:
    """List all datasets currently loaded in this session."""
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if not session.datasets:
        return "No datasets loaded in this session. The user must upload a CSV or Excel file first."
    lines = []
    for ds_id, df in session.datasets.items():
        cols = ", ".join(map(str, df.columns))
        lines.append(
            f"- {ds_id}: {df.shape[0]} rows × {df.shape[1]} cols | columns: {cols}"
        )
    return "\n".join(lines)


@tool
def profile_dataset(dataset_id: str) -> str:
    """Get shape, dtypes, nulls, describe() and the first 5 rows of a dataset."""
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if dataset_id not in session.datasets:
        return f"Error: dataset '{dataset_id}' not found. Call list_datasets to see available."
    df = session.datasets[dataset_id]

    buf = io.StringIO()
    buf.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols\n\n")
    buf.write("Columns and dtypes:\n")
    for col, dtype in df.dtypes.items():
        nulls = int(df[col].isnull().sum())
        uniq = int(df[col].nunique())
        buf.write(f"  - {col} ({dtype}) — {nulls} nulls, {uniq} unique\n")
    buf.write("\nNumeric describe():\n")
    try:
        buf.write(df.describe().to_string())
    except ValueError:
        buf.write("(no numeric columns)")
    buf.write("\n\nFirst 5 rows:\n")
    buf.write(df.head(5).to_string(index=False))
    return buf.getvalue()


@tool
def execute_pandas(dataset_id: str, code: str) -> str:
    """Run pandas code in an isolated subprocess against a loaded dataset.

    `df` is pre-loaded. Use `print(...)` to produce output. Multi-line code is
    allowed. The last expression's value is captured and printed (truncated).
    """
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if dataset_id not in session.datasets:
        return f"Error: dataset '{dataset_id}' not found."
    df = session.datasets[dataset_id]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as pf:
        df.to_parquet(pf.name)
        parquet_path = pf.name

    try:
        code_json = json.dumps(code)
        wrapper = (
            "import sys, io, json\n"
            "import pandas as pd\n"
            f"df = pd.read_parquet({parquet_path!r})\n"
            f"_code = json.loads({code_json!r})\n"
            "_buf = io.StringIO()\n"
            "_old_out, _old_err = sys.stdout, sys.stderr\n"
            "sys.stdout = _buf\n"
            "_err = None\n"
            "try:\n"
            "    exec(_code, {'df': df, 'pd': pd})\n"
            "except Exception as e:\n"
            "    _err = f'{type(e).__name__}: {e}'\n"
            "sys.stdout = _old_out\n"
            "sys.stderr = _old_err\n"
            "_out = _buf.getvalue()\n"
            "try:\n"
            "    _lines = _code.rstrip().split('\\n')\n"
            "    _last = _lines[-1].strip() if _lines else ''\n"
            "    if _last and not _last.startswith('#') and '=' not in _last.split('#')[0].split(' if ')[0].split(' for ')[0]:\n"
            "        try:\n"
            "            _val = eval(_last, {'df': df, 'pd': pd})\n"
            "            if _val is not None:\n"
            "                if isinstance(_val, pd.DataFrame):\n"
            "                    print('\\n--- Last expression (DataFrame) ---')\n"
            "                    print(_val.head(20).to_string())\n"
            "                elif isinstance(_val, pd.Series):\n"
            "                    print('\\n--- Last expression (Series) ---')\n"
            "                    print(_val.head(20).to_string())\n"
            "                else:\n"
            "                    print('\\n--- Last expression ---')\n"
            "                    print(repr(_val))\n"
            "        except Exception:\n"
            "            pass\n"
            "except Exception:\n"
            "    pass\n"
            "if _err:\n"
            "    print(_err, file=_old_err)\n"
            "print(_out, end='')\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
            sf.write(wrapper)
            script_path = sf.name

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=settings.code_exec_timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            if not output.strip():
                output = "(code ran successfully with no printed output)"
            return output[:8000]
        except subprocess.TimeoutExpired:
            return f"Error: code timed out after {settings.code_exec_timeout}s"
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
        finally:
            Path(script_path).unlink(missing_ok=True)
    finally:
        Path(parquet_path).unlink(missing_ok=True)


@tool
def add_row(dataset_id: str, row: dict) -> str:
    """Append a row to a dataset. Persists to the in-memory DataFrame (unlike execute_pandas)."""
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if dataset_id not in session.datasets:
        return f"Error: dataset '{dataset_id}' not found."
    df = session.datasets[dataset_id]

    if not isinstance(row, dict) or not row:
        return "Error: 'row' must be a non-empty dict like {col: value, ...}."

    coerced = {}
    for col, val in row.items():
        if col not in df.columns:
            return f"Error: column '{col}' does not exist in dataset. Known columns: {list(df.columns)}"
        target_dtype = df[col].dtype
        try:
            if pd.api.types.is_integer_dtype(target_dtype) and not pd.isna(val):
                coerced[col] = int(val)
            elif pd.api.types.is_float_dtype(target_dtype) and not pd.isna(val):
                coerced[col] = float(val)
            else:
                coerced[col] = val
        except (ValueError, TypeError):
            coerced[col] = val

    new_df = pd.concat([df, pd.DataFrame([coerced])], ignore_index=True)
    session.datasets[dataset_id] = new_df
    last = new_df.tail(1).to_string(index=False)
    return f"Row appended. New shape: {new_df.shape[0]} rows × {new_df.shape[1]} cols.\nLast row:\n{last}"


@tool
def save_dataset(dataset_id: str, filename: str = "") -> str:
    """Save the dataset to a CSV file in the session's artifacts directory.

    The returned URL is served by FastAPI at /artifacts/<session_id>/<filename>.csv
    so the browser can download it directly.
    """
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if dataset_id not in session.datasets:
        return f"Error: dataset '{dataset_id}' not found."
    df = session.datasets[dataset_id]

    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", filename) if filename else f"{dataset_id}.csv"
    if not safe.lower().endswith(".csv"):
        safe += ".csv"
    save_path = session.artifacts_dir / safe
    df.to_csv(save_path, index=False)
    size_kb = round(save_path.stat().st_size / 1024, 1)
    return (
        f"Dataset saved: /artifacts/{session.id}/{safe} "
        f"({len(df)} rows, {size_kb} KB)"
    )