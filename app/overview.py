from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class GeneratedChart:
    title: str
    url: str
    kind: str
    description: str


@dataclass
class DatasetOverview:
    dataset_id: str
    filename: str
    shape: list[int]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    null_counts: dict[str, int]
    duplicate_rows: int
    top_correlations: list[dict[str, Any]]
    charts: list[GeneratedChart] = field(default_factory=list)
    summary: str = ""
    potential_joins: list[dict[str, Any]] = field(default_factory=list)


def _slugify(text: str, default: str = "chart") -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return (text[:50] or default).lower()


def _render_chart(code: str, df: pd.DataFrame, save_path: Path) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as pf:
        df.to_parquet(pf.name)
        parquet_path = pf.name

    try:
        indented_body = "\n".join(
            ("    " + line) if line.strip() else line
            for line in code.splitlines()
        )

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
            f"{indented_body}\n"
            "    plt.tight_layout()\n"
            "    plt.savefig(_save_path, bbox_inches='tight', dpi=110)\n"
            "except Exception as e:\n"
            "    print(f'CHART_ERR: {type(e).__name__}: {e}', file=sys.stderr)\n"
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
                timeout=20,
            )
            if result.returncode != 0 and result.stderr:
                print(f"[chart error] {result.stderr.strip()[:300]}")
            return result.returncode == 0 and save_path.exists()
        finally:
            Path(script_path).unlink(missing_ok=True)
    finally:
        Path(parquet_path).unlink(missing_ok=True)


def _save_chart(df: pd.DataFrame, artifacts_dir: Path, session_id: str,
                kind: str, title: str, code: str) -> GeneratedChart | None:
    slug = _slugify(f"{kind}_{title}")
    save_path = artifacts_dir / f"{slug}.png"
    if _render_chart(code, df, save_path):
        return GeneratedChart(
            title=title,
            url=f"/artifacts/{session_id}/{slug}.png",
            kind=kind,
            description=title,
        )
    return None


def compute_overview(
    dataset_id: str,
    df: pd.DataFrame,
    filename: str,
    artifacts_dir: Path,
    session_id: str,
    max_charts_per_kind: int = 6,
) -> DatasetOverview:
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [
        c for c in df.columns
        if c not in numeric_cols
        and pd.api.types.is_string_dtype(df[c])
        and df[c].nunique(dropna=True) <= 50
        and df[c].nunique(dropna=True) >= 1
    ]
    dt_cols = [
        c for c in df.columns
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or (pd.api.types.is_string_dtype(df[c]) and _looks_like_date(df[c]))
    ]
    null_counts = {c: int(df[c].isnull().sum()) for c in df.columns if df[c].isnull().any()}
    duplicate_rows = int(df.duplicated().sum())

    top_corrs: list[dict[str, Any]] = []
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        pairs: list[tuple[float, str, str]] = []
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                c = corr.loc[a, b]
                if pd.notna(c):
                    pairs.append((float(c), a, b))
        pairs.sort(key=lambda x: abs(x[0]), reverse=True)
        for c, a, b in pairs[:5]:
            top_corrs.append({"a": a, "b": b, "corr": round(c, 3)})

    charts: list[GeneratedChart] = []

    if len(numeric_cols) >= 2:
        ch = _save_chart(
            df, artifacts_dir, session_id, "correlation_heatmap",
            "numeric_correlations",
            textwrap.dedent("""
                _num = df.select_dtypes(include='number')
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(_num.corr(), annot=True, fmt='.2f', cmap='RdBu_r',
                            vmin=-1, vmax=1, center=0, square=True,
                            linewidths=0.5, cbar_kws={'label': 'correlation'}, ax=ax)
                ax.set_title('Correlation heatmap (numeric columns)')
            """).strip(),
        )
        if ch: charts.append(ch)

    if numeric_cols:
        variances = (
            df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
        )
        chosen_num = [c for c in variances.index if variances[c] > 0][:max_charts_per_kind]
        for col in chosen_num:
            ch = _save_chart(
                df, artifacts_dir, session_id, "histogram",
                f"hist_{col}",
                textwrap.dedent(f"""
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.hist(df[{col!r}].dropna(), bins=30, color='#4C78A8', edgecolor='white')
                    ax.set_xlabel({col!r})
                    ax.set_ylabel('count')
                    ax.set_title(f'Distribution of {col}')
                """).strip(),
            )
            if ch: charts.append(ch)

    for col in cat_cols[:max_charts_per_kind]:
        try:
            top_vals = df[col].value_counts().head(8)
            if len(top_vals) == 0:
                continue
            ch = _save_chart(
                df, artifacts_dir, session_id, "bar",
                f"bar_{col}",
                textwrap.dedent(f"""
                    _counts = df[{col!r}].value_counts().head(8)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    sns.barplot(x=_counts.index.astype(str), y=_counts.values,
                                ax=ax, palette='viridis')
                    ax.set_xlabel({col!r})
                    ax.set_ylabel('count')
                    ax.set_title('Top values: ' + {col!r})
                    plt.xticks(rotation=30, ha='right')
                """).strip(),
            )
            if ch: charts.append(ch)
        except Exception:
            continue

    if null_counts:
        ch = _save_chart(
            df, artifacts_dir, session_id, "missing_values",
            "missing_values",
            textwrap.dedent("""
                _nulls = df.isnull().sum()
                _nulls = _nulls[_nulls > 0].sort_values(ascending=True)
                fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(_nulls))))
                ax.barh(_nulls.index.astype(str), _nulls.values, color='#d29922')
                ax.set_xlabel('null count')
                ax.set_title('Missing values per column')
            """).strip(),
        )
        if ch: charts.append(ch)

    lines = [
        f"**Loaded `{filename}`** — {len(df):,} rows × {df.shape[1]} columns.",
        f"• {len(numeric_cols)} numeric · {len(cat_cols)} categorical · {len(dt_cols)} date-like",
    ]
    if null_counts:
        worst = max(null_counts.items(), key=lambda kv: kv[1])
        lines.append(f"• Missing values in {len(null_counts)} column(s) — worst: `{worst[0]}` ({worst[1]})")
    else:
        lines.append("• No missing values.")
    if duplicate_rows:
        lines.append(f"• {duplicate_rows:,} duplicate row(s).")
    if top_corrs:
        top = top_corrs[0]
        sign = "positive" if top["corr"] > 0 else "negative"
        lines.append(
            f"• Strongest relationship: `{top['a']}` ↔ `{top['b']}` "
            f"(r = {top['corr']:.2f}, {sign})"
        )
    summary = "\n".join(lines)

    return DatasetOverview(
        dataset_id=dataset_id,
        filename=filename,
        shape=list(df.shape),
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        datetime_columns=dt_cols,
        null_counts=null_counts,
        duplicate_rows=duplicate_rows,
        top_correlations=top_corrs,
        charts=charts,
        summary=summary,
    )


def detect_cross_dataset_joins(overviews: list[DatasetOverview]) -> list[dict[str, Any]]:
    if len(overviews) < 2:
        return []
    joins: list[dict[str, Any]] = []
    for i, a in enumerate(overviews):
        for b in overviews[i + 1:]:
            common = set(a.numeric_columns) & set(b.numeric_columns)
            for col in common:
                joins.append({
                    "left_dataset": a.filename,
                    "left_column": col,
                    "right_dataset": b.filename,
                    "right_column": col,
                    "kind": "numeric",
                })
            common = set(a.categorical_columns) & set(b.categorical_columns)
            for col in common:
                joins.append({
                    "left_dataset": a.filename,
                    "left_column": col,
                    "right_dataset": b.filename,
                    "right_column": col,
                    "kind": "categorical",
                })
            common = set(a.datetime_columns) & set(b.datetime_columns)
            for col in common:
                joins.append({
                    "left_dataset": a.filename,
                    "left_column": col,
                    "right_dataset": b.filename,
                    "right_column": col,
                    "kind": "date",
                })
    return joins


def compute_combined_summary(overviews: list[DatasetOverview]) -> str:
    if len(overviews) == 1:
        return overviews[0].summary
    lines = [f"**Loaded {len(overviews)} datasets:**"]
    for ov in overviews:
        lines.append(f"• `{ov.filename}` — {ov.shape[0]:,} rows × {ov.shape[1]} cols")
    return "\n".join(lines)


def _looks_like_date(s: pd.Series, sample: int = 50) -> bool:
    sample = s.dropna().astype(str).head(sample)
    if len(sample) == 0:
        return False
    date_like = sum(1 for v in sample if re.match(r"^\d{4}-\d{2}-\d{2}", v))
    return date_like / len(sample) > 0.7


def overview_to_dict(ov: DatasetOverview) -> dict[str, Any]:
    return {
        "dataset_id": ov.dataset_id,
        "filename": ov.filename,
        "shape": ov.shape,
        "numeric_columns": ov.numeric_columns,
        "categorical_columns": ov.categorical_columns,
        "datetime_columns": ov.datetime_columns,
        "null_counts": ov.null_counts,
        "duplicate_rows": ov.duplicate_rows,
        "top_correlations": ov.top_correlations,
        "charts": [
            {"title": c.title, "url": c.url, "kind": c.kind, "description": c.description}
            for c in ov.charts
        ],
        "summary": ov.summary,
        "potential_joins": ov.potential_joins,
    }


_FOLLOWUP_FALLBACK = [
    "Summarize the key statistics of this dataset.",
    "What are the top 5 rows by some numeric column?",
    "Plot a chart of one numeric column grouped by a category.",
    "Are there any outliers or unusual patterns?",
    "Which categorical column has the most variation?",
]


def generate_followups(
    overview: DatasetOverview,
    max_questions: int = 5,
    cross_dataset_filenames: list[str] | None = None,
) -> list[str]:
    if not overview.numeric_columns and not overview.categorical_columns:
        return _FOLLOWUP_FALLBACK[:max_questions]

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from .llm import get_llm

        llm = get_llm()
        corrs = ", ".join(
            f"{c['a']} ↔ {c['b']} ({c['corr']:.2f})"
            for c in overview.top_correlations[:3]
        ) or "none"
        ctx_extra = ""
        if cross_dataset_filenames and len(cross_dataset_filenames) >= 2:
            ctx_extra = (
                f"\n\nThis is one of {len(cross_dataset_filenames)} uploaded files "
                f"({', '.join(cross_dataset_filenames)}). At least one suggested question "
                "should reference cross-dataset analysis or a join if applicable."
            )

        prompt = (
            f"Dataset: {overview.filename}\n"
            f"Shape: {overview.shape[0]} rows × {overview.shape[1]} columns\n"
            f"Numeric columns: {', '.join(overview.numeric_columns) or 'none'}\n"
            f"Categorical columns: {', '.join(overview.categorical_columns) or 'none'}\n"
            f"Top correlations: {corrs}"
            f"{ctx_extra}\n\n"
            f"Suggest exactly {max_questions} specific natural-language questions the user "
            "could ask next. Requirements:\n"
            "- Each question must reference actual column names from the dataset above.\n"
            "- Mix question types: 1 descriptive, 1 aggregative, 1 investigative (correlation/causation),\n"
            "  1 chart/visualization, 1 outlier or data-quality check.\n"
            "- Keep each question under 100 characters.\n"
            "- Return ONLY a JSON array of strings, no other text. Example:\n"
            '  ["What is the average Sales by Region?", "Plot profit by category"]'
        )
        resp = llm.invoke([
            SystemMessage(content="You are a senior data analyst."),
            HumanMessage(content=prompt),
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                cleaned = [str(q).strip().strip('"').strip("'") for q in arr if str(q).strip()]
                cleaned = [q for q in cleaned if 5 < len(q) < 200][:max_questions]
                if cleaned:
                    return cleaned
    except Exception as e:
        print(f"[followups error] {e}")

    out: list[str] = []
    if overview.numeric_columns and overview.categorical_columns:
        nc = overview.numeric_columns[0]
        cc = overview.categorical_columns[0]
        out.append(f"What is the average {nc} by {cc}?")
    if overview.categorical_columns:
        out.append(f"Plot a bar chart of {overview.categorical_columns[0]} counts.")
    if overview.top_correlations:
        c = overview.top_correlations[0]
        out.append(f"Plot a scatter of {c['a']} vs {c['b']}.")
    if overview.numeric_columns:
        out.append(f"Show the distribution of {overview.numeric_columns[0]}.")
    out.append("Are there any missing values or duplicates I should worry about?")
    return out[:max_questions]