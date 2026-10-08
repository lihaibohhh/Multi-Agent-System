"""Pydantic contracts for research runs and persisted run events."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from ..sections.models import ReportReview, SectionRecord


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    BUDGET_LIMITED = "budget_limited"


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
    handoff: list[dict] = Field(default_factory=list)
    revision_sections: list[SectionRecord] = Field(default_factory=list)
    revision_target: str | None = None
    report_review: ReportReview | None = None
    section_operation: dict | None = None


class SectionRevisionRequest(BaseModel):
    instruction: str = Field(min_length=5, max_length=2000)
    run_id: str | None = Field(default=None, max_length=128)


class SectionOperationRequest(BaseModel):
    mode: Literal["continue", "supplement", "refresh"]
    instruction: str = Field(min_length=5, max_length=2000)
    run_id: str | None = Field(default=None, max_length=128)


class RunCreateRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    session_id: str | None = Field(default=None, max_length=128)
    parent_run_id: str | None = Field(default=None, max_length=128)
    run_id: str | None = Field(default=None, max_length=128)
    parent_section_ids: list[str] | None = Field(default=None, max_length=4)


class BudgetMigrationRequest(BaseModel):
    confirm: bool
    reason: str = Field(min_length=5, max_length=500)


class BudgetIncreaseRequest(BaseModel):
    confirm: bool
    request_id: str = Field(min_length=8, max_length=128, pattern=r"^[a-zA-Z0-9_-]+$")
    expected_tokens: int = Field(gt=0, strict=True)
    new_tokens: int = Field(gt=0, strict=True)
    reason: str = Field(min_length=5, max_length=500)


class RunRecord(BaseModel):
    run_id: str
    session_id: str | None = None
    parent_run_id: str | None = None
    thread_id: str
    question: str
    status: RunStatus
    execution_id: str | None = None
    model_usage: dict[str, int] = Field(default_factory=dict)
    budget: dict = Field(default_factory=dict)
    budget_id: str | None = None
    execution_deadline: float | None = None
    pause_requested: bool = False
    parent_context: ParentContextSnapshot | None = None
    sections: list[SectionRecord] = Field(default_factory=list)
    report_review: ReportReview | None = None
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


class RunSnapshot(BaseModel):
    run: RunRecord
    cursor: int = 0


class SessionTimeline(BaseModel):
    session: SessionRecord
    runs: list[RunRecord]
