from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

from .pandas_tools import _get_session


def _slugify(text: str, default: str = "chart") -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return (text[:50] or default).lower()


def _chart_result(session_id: str, slug: str, title: str) -> str:
    url = f"/artifacts/{session_id}/{slug}.png"
    return f"Chart saved at {url}\nchart_title: {title}"


def _run_plot_script(dataset_id: str, code: str, save_path: Path) -> tuple[bool, str]:
    session = _get_session()
    if session is None or dataset_id not in session.datasets:
        return False, "session or dataset not found"
    df = session.datasets[dataset_id]

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as pf:
        df.to_parquet(pf.name)
        parquet_path = pf.name

    try:
        wrapper = (
            "import sys\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import pandas as pd\n"
            "sns.set_theme(style='darkgrid', palette='muted')\n"
            f"df = pd.read_parquet({parquet_path!r})\n"
            f"_save_path = {str(save_path)!r}\n"
            "try:\n"
            "    "
            + code.replace("\n", "\n    ")
            + "\n    plt.tight_layout()\n"
            "    plt.savefig(_save_path, bbox_inches='tight', dpi=110)\n"
            "    print(f'OK: saved {_save_path}')\n"
            "except Exception as e:\n"
            "    print(f'ERR: {type(e).__name__}: {e}', file=sys.stderr)\n"
            "    sys.exit(1)\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as sf:
            sf.write(wrapper)
            script_path = sf.name
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return False, (result.stderr or "plot failed").strip()
            return True, str(save_path)
        finally:
            Path(script_path).unlink(missing_ok=True)
    finally:
        Path(parquet_path).unlink(missing_ok=True)


@tool
def plot_bar(
    dataset_id: str,
    x: str,
    y: str = "",
    title: str = "",
    top: int = 10,
) -> str:
    """Render a vertical bar chart.

    Pass `y` to sum that numeric column per group. Omit `y` (leave empty) to
    count rows per group — the "how many per X" / frequency chart mode.
    """
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if dataset_id not in session.datasets:
        return f"Error: dataset '{dataset_id}' not found."
    if not x:
        return "Error: 'x' (category column) is required."

    is_count = not (y and y.strip())
    slug = _slugify(
        title
        or (f"count_of_{x}" if is_count else f"bar_{y}_by_{x}")
    )
    save_path = session.artifacts_dir / f"{slug}.png"

    if is_count:
        safe_title = title or f"Count of {x}"
        y_label = "count"
        code = (
            f"_data = df.groupby({x!r}).size().sort_values(ascending=False).head({top})\n"
            "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
            "sns.barplot(x=_data.index.astype(str), y=_data.values, ax=ax, palette='viridis')\n"
            f"ax.set_xlabel({x!r})\n"
            f"ax.set_ylabel({y_label!r})\n"
            f"ax.set_title({safe_title!r})\n"
            "plt.xticks(rotation=30, ha='right')\n"
        )
    else:
        safe_title = title or f"Top {top}: sum({y}) by {x}"
        y_label = f"sum({y})"
        code = (
            f"_data = df.groupby({x!r})[{y!r}].sum().sort_values(ascending=False).head({top})\n"
            "fig, ax = plt.subplots(figsize=(10, 5.5))\n"
            "sns.barplot(x=_data.index.astype(str), y=_data.values, ax=ax, palette='viridis')\n"
            f"ax.set_xlabel({x!r})\n"
            f"ax.set_ylabel({y_label!r})\n"
            f"ax.set_title({safe_title!r})\n"
            "plt.xticks(rotation=30, ha='right')\n"
        )
    ok, msg = _run_plot_script(dataset_id, code, save_path)
    if not ok:
        return f"Error rendering chart: {msg}"
    return _chart_result(session.id, save_path.name.removesuffix(".png"), safe_title)


@tool
def plot_histogram(
    dataset_id: str,
    column: str,
    title: str = "",
    bins: int = 30,
) -> str:
    """Render a histogram of a numeric column."""
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if dataset_id not in session.datasets:
        return f"Error: dataset '{dataset_id}' not found."
    if not column:
        return "Error: 'column' is required."

    slug = _slugify(title or f"hist_{column}")
    save_path = session.artifacts_dir / f"{slug}.png"
    safe_title = title or f"Distribution of {column}"

    code = (
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        f"ax.hist(df[{column!r}].dropna(), bins={bins}, color='#4C78A8', edgecolor='white')\n"
        f"ax.set_xlabel({column!r})\n"
        "ax.set_ylabel('count')\n"
        f"ax.set_title({safe_title!r})\n"
    )
    ok, msg = _run_plot_script(dataset_id, code, save_path)
    if not ok:
        return f"Error rendering chart: {msg}"
    return _chart_result(session.id, save_path.name.removesuffix(".png"), safe_title)


@tool
def plot_scatter(
    dataset_id: str,
    x: str,
    y: str,
    title: str = "",
    hue: Optional[str] = None,
) -> str:
    """Render a scatter plot of two numeric columns, optionally colored by a category."""
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if dataset_id not in session.datasets:
        return f"Error: dataset '{dataset_id}' not found."
    if not x or not y:
        return "Error: both 'x' and 'y' are required."

    slug = _slugify(title or f"scatter_{x}_{y}")
    save_path = session.artifacts_dir / f"{slug}.png"
    safe_title = title or f"{x} vs {y}" + (f" by {hue}" if hue else "")

    hue_kw = f", hue={hue!r}" if hue else ""
    code = (
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        f"sns.scatterplot(data=df, x={x!r}, y={y!r}{hue_kw}, ax=ax, alpha=0.75)\n"
        f"ax.set_xlabel({x!r})\n"
        f"ax.set_ylabel({y!r})\n"
        f"ax.set_title({safe_title!r})\n"
    )
    ok, msg = _run_plot_script(dataset_id, code, save_path)
    if not ok:
        return f"Error rendering chart: {msg}"
    return _chart_result(session.id, save_path.name.removesuffix(".png"), safe_title)


@tool
def plot_correlation(dataset_id: str, title: str = "") -> str:
    """Render a correlation heatmap of all numeric columns."""
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if dataset_id not in session.datasets:
        return f"Error: dataset '{dataset_id}' not found."

    slug = _slugify(title or "correlation_heatmap")
    save_path = session.artifacts_dir / f"{slug}.png"
    safe_title = title or "Correlation heatmap"

    code = (
        "_num = df.select_dtypes(include='number')\n"
        "if _num.shape[1] < 2:\n"
        "    raise ValueError('need at least 2 numeric columns for correlation')\n"
        "fig, ax = plt.subplots(figsize=(8, 6))\n"
        "sns.heatmap(\n"
        "    _num.corr(), annot=True, fmt='.2f', cmap='RdBu_r',\n"
        "    vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,\n"
        "    cbar_kws={'label': 'correlation'}, ax=ax,\n"
        ")\n"
        f"ax.set_title({safe_title!r})\n"
    )
    ok, msg = _run_plot_script(dataset_id, code, save_path)
    if not ok:
        return f"Error rendering chart: {msg}"
    return _chart_result(session.id, save_path.name.removesuffix(".png"), safe_title)


@tool
def plot_violin(
    dataset_id: str,
    columns: str,
    title: str = "",
    group_by: Optional[str] = None,
) -> str:
    """Render a violin plot of one or more numeric columns.

    Pass a comma-separated column list to compare distributions side-by-side.
    """
    session = _get_session()
    if session is None:
        return "Error: no active session."
    if dataset_id not in session.datasets:
        return f"Error: dataset '{dataset_id}' not found."

    df = session.datasets[dataset_id]
    cols = [c.strip() for c in columns.split(",") if c.strip()]
    if not cols:
        return "Error: 'columns' must list at least one numeric column."
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return f"Error: column(s) not in dataset: {missing}. Known: {list(df.columns)}"
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        return f"Error: columns must be numeric; got non-numeric: {non_numeric}"

    if group_by and group_by not in df.columns:
        return f"Error: group_by column '{group_by}' not in dataset."

    safe_title = title or f"Distribution of {', '.join(cols)}"
    slug = _slugify(safe_title)
    save_path = session.artifacts_dir / f"{slug}.png"

    cols_py = "[" + ", ".join(repr(c) for c in cols) + "]"
    melt_id_vars = f", id_vars={group_by!r}" if group_by else ""
    include_group = f" + [{group_by!r}]" if group_by else ""
    group_kw = f", hue={group_by!r}" if group_by else ""
    code = (
        f"_cols = {cols_py}\n"
        f"_df = df[_cols{include_group}].melt("
        f"value_vars=_cols{melt_id_vars}, var_name='column', value_name='value')\n"
        "fig, ax = plt.subplots(figsize=(9, 5))\n"
        f"sns.violinplot(data=_df, x='column', y='value'{group_kw}, "
        "palette='viridis', inner='quartile', cut=0, ax=ax)\n"
        "ax.set_xlabel('')\n"
        f"ax.set_title({safe_title!r})\n"
        "plt.tight_layout()\n"
    )
    ok, msg = _run_plot_script(dataset_id, code, save_path)
    if not ok:
        return f"Error rendering chart: {msg}"
    return _chart_result(session.id, save_path.name.removesuffix(".png"), safe_title)