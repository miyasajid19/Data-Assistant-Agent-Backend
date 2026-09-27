from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    id: str
    filename: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    preview: list[dict[str, Any]]
    memory_mb: float


class GeneratedChartInfo(BaseModel):
    title: str
    url: str
    kind: str
    description: str


class DatasetOverviewInfo(BaseModel):
    dataset_id: str
    filename: str
    shape: list[int]
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    null_counts: dict[str, int]
    duplicate_rows: int
    top_correlations: list[dict]
    charts: list[GeneratedChartInfo]
    summary: str
    potential_joins: list[dict] = []


class CrossDatasetJoin(BaseModel):
    left_dataset: str
    left_column: str
    right_dataset: str
    right_column: str
    kind: str


class UploadResponse(BaseModel):
    session_id: str
    datasets: list[DatasetInfo]
    overviews: list[DatasetOverviewInfo] = []
    combined_summary: str = ""
    cross_dataset_joins: list[CrossDatasetJoin] = []
    suggested_questions: list[str] = []


class ChatRequest(BaseModel):
    session_id: str
    thread_id: str = Field(default="default")
    message: str


class SessionInfo(BaseModel):
    session_id: str
    datasets: list[dict[str, Any]]