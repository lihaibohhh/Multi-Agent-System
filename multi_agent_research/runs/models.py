"""Pydantic contracts for research runs and persisted run events."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from ..sections.models import SectionRecord


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"


TERMINAL_RUN_STATUSES = frozenset({
    RunStatus.COMPLETED,
    RunStatus.CANCELLED,
})


class SessionCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    session_id: str | None = Field(default=None, max_length=128)


class SessionRecord(BaseModel):
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ParentContextSnapshot(BaseModel):
    schema_version: int = 1
    source_run_id: str
    source_question: str
    report_excerpt: str
    reference_excerpt: str = ""
    report_truncated: bool
    captured_at: datetime


class RunCreateRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    session_id: str | None = Field(default=None, max_length=128)
    parent_run_id: str | None = Field(default=None, max_length=128)
    run_id: str | None = Field(default=None, max_length=128)


class RunRecord(BaseModel):
    run_id: str
    session_id: str | None = None
    parent_run_id: str | None = None
    thread_id: str
    question: str
    status: RunStatus
    parent_context: ParentContextSnapshot | None = None
    sections: list[SectionRecord] = Field(default_factory=list)
    final_report: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class RunEventRecord(BaseModel):
    sequence: int
    run_id: str
    event_type: str
    payload: dict
    created_at: datetime


class SessionTimeline(BaseModel):
    session: SessionRecord
    runs: list[RunRecord]
