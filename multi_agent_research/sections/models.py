"""Versioned chapter artifacts; source records are retrieved excerpts, not a KB copy."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SectionSpec(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=5, max_length=500)
    kind: Literal["research", "synthesis"] = "research"
    parent_section_ids: list[str] = Field(default_factory=list, max_length=4)


class SectionPlan(BaseModel):
    sections: list[SectionSpec] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_order(self):
        if self.sections[0].kind != "research":
            raise ValueError("the first section must perform research")
        if any(s.kind == "synthesis" for s in self.sections[:-1]):
            raise ValueError("synthesis is allowed only as the final section")
        if len({s.title.strip() for s in self.sections}) != len(self.sections):
            raise ValueError("section titles must be distinct")
        return self


class SectionReview(BaseModel):
    verdict: Literal["pass", "revise"]
    issues: list[str] = Field(default_factory=list, max_length=8)
    search_queries: list[str] = Field(default_factory=list, max_length=2)
    summary: str = Field(default="", max_length=1500)


class QuoteSpan(BaseModel):
    """Python character offsets [start, end) into the persisted original text."""
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    match: Literal["exact", "layout_whitespace"]


class EvidenceLink(BaseModel):
    source_number: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=1000)
    relation: Literal["supports", "contradicts", "context"] = "supports"
    evidence_id: str = ""  # Assigned by code, never trusted from model output.
    quote_span: QuoteSpan | None = None  # Assigned by code; None for old artifacts.


class Claim(BaseModel):
    claim_id: str = ""
    statement: str = Field(min_length=1, max_length=1000)
    draft_quote: str = Field(min_length=1, max_length=1000)
    draft_span: QuoteSpan | None = None
    assessment: Literal["supported", "uncertain", "unsupported"]
    evidence: list[EvidenceLink] = Field(default_factory=list, max_length=8)
    caveat: str = Field(default="", max_length=1000)


class ClaimExtraction(BaseModel):
    claims: list[Claim] = Field(min_length=1, max_length=12)


class ConsistencyIssue(BaseModel):
    kind: Literal["conflict", "scope", "duplication", "coverage", "dependency"]
    section_ids: list[str] = Field(default_factory=list, max_length=4)
    detail: str = Field(min_length=1, max_length=1500)


class ReportReview(BaseModel):
    verdict: Literal["pass", "revise"]
    issues: list[ConsistencyIssue] = Field(default_factory=list, max_length=12)
    summary: str = Field(default="", max_length=2000)


class SectionDraft(BaseModel):
    revision: int
    draft: str
    sources: list[dict] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    reviewed_at: datetime | None = None


class SectionRecord(SectionSpec):
    section_id: str
    status: Literal["pending", "stale", "researching", "drafted", "complete", "limited", "evidence_ready", "waiting_evidence", "claims_pending"] = "pending"
    artifact_version: int = 1
    depends_on: list[str] = Field(default_factory=list, max_length=4)
    dependency_revisions: dict[str, int] = Field(default_factory=dict)
    revision_instruction: str = ""
    revision_base: int = 0
    invalidated_by: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    reviewed_at: datetime | None = None
    search_rounds: int = 0
    revision: int = 0
    results: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    draft: str = ""
    previous_drafts: list[SectionDraft] = Field(default_factory=list)
    analyst: dict = Field(default_factory=dict)
    review: SectionReview | None = None
    gaps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence_update: dict = Field(default_factory=dict)
    claim_work: dict = Field(default_factory=dict)


class SectionPolicy(BaseModel):
    """Persisted limits keep resumed work bounded even if environment settings change."""
    max_search_rounds: int = Field(default=2, ge=1, le=4)
    max_revisions: int = Field(default=1, ge=0, le=3)
