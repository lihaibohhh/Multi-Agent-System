"""Typed artifacts exchanged through the run-level research workspace.

These models deliberately contain research outputs, not Agent transcripts.  Parent
Run conclusions and section conclusions use the same shape and are distinguished
by the enclosing coordination scope.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, model_validator


ScopeType = Literal["parent_run", "section"]
ScopeStatus = Literal["provisional", "coordinated", "accepted", "stale"]
WorkspaceStatus = Literal["collecting", "coordinating", "ready", "stale"]
ClaimStatus = Literal["pending", "accepted", "disputed", "rejected", "superseded"]
EvidenceStatus = Literal["active", "stale", "unavailable"]
SupportStatus = Literal["supports", "contradicts", "insufficient", "pending"]
MetricStatus = Literal["pending", "accepted", "disputed", "rejected", "superseded"]


class DocumentArtifact(BaseModel):
    """One canonical source document within a Run workspace."""

    document_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=1000)
    canonical_url: str | None = Field(default=None, max_length=4000)
    publisher: str | None = Field(default=None, max_length=500)
    author: str | None = Field(default=None, max_length=500)
    published_at: datetime | None = None
    source_type: str = Field(min_length=1, max_length=64)
    content_hash: str | None = Field(default=None, max_length=128)
    retrieved_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)


class EvidenceArtifact(BaseModel):
    """A located excerpt; many excerpts may point to the same document."""

    evidence_id: str = Field(min_length=1, max_length=256)
    document_id: str = Field(min_length=1, max_length=256)
    excerpt: str = Field(min_length=1)
    excerpt_hash: str = Field(min_length=1, max_length=128)
    locator: str | None = Field(default=None, max_length=1000)
    retrieval_query: str | None = None
    status: EvidenceStatus = "active"


class QuoteSpanArtifact(BaseModel):
    """Stable character offsets into the persisted source excerpt."""

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    match: Literal["exact", "layout_whitespace"]


class EvidenceQuoteArtifact(BaseModel):
    """One exact quote used to assess a Claim against a source excerpt."""

    source_number: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=1000)
    relation: Literal["supports", "contradicts", "context"]
    quote_span: QuoteSpanArtifact | None = None


class EvidenceBindingArtifact(BaseModel):
    """One semantic decision with all quote-level locators for an excerpt."""

    evidence_id: str = Field(min_length=1, max_length=256)
    support_status: SupportStatus
    reason: str | None = None
    required_supplement: str | None = None
    created_by_agent_run_id: str | None = Field(default=None, max_length=256)
    quote_refs: list[EvidenceQuoteArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_explanation_for_non_support(self):
        if self.support_status != "supports" and not (self.reason or "").strip():
            raise ValueError("non-support evidence bindings require a reason")
        return self


class ClaimArtifact(BaseModel):
    """A conclusion plus exact evidence bindings and provenance."""

    claim_id: str = Field(min_length=1, max_length=256)
    statement: str = Field(min_length=1)
    origin_run_id: str = Field(min_length=1, max_length=128)
    origin_section_id: str | None = Field(default=None, max_length=128)
    origin_agent_run_id: str | None = Field(default=None, max_length=256)
    claim_type: Literal["fact", "metric", "inference", "forecast", "recommendation"]
    status: ClaimStatus = "pending"
    confidence: float | None = Field(default=None, ge=0, le=1)
    revision: int = Field(default=1, ge=1)
    subject: str | None = None
    time_scope: str | None = None
    geography_scope: str | None = None
    industry_scope: str | None = None
    visibility: Literal["public", "internal"] = "public"
    caveat: str = ""
    evidence_bindings: list[EvidenceBindingArtifact] = Field(default_factory=list)


class MetricArtifact(BaseModel):
    """A number with enough dimensions to distinguish conflict from scope drift."""

    metric_id: str = Field(min_length=1, max_length=256)
    metric_name: str = Field(min_length=1, max_length=500)
    value_text: str = Field(min_length=1, max_length=500)
    value_numeric: Decimal | None = None
    unit: str | None = Field(default=None, max_length=128)
    period: str | None = Field(default=None, max_length=256)
    geography: str | None = Field(default=None, max_length=256)
    population: str | None = Field(default=None, max_length=500)
    sample_scope: str | None = None
    numerator_definition: str | None = None
    denominator_definition: str | None = None
    methodology: str | None = None
    status: MetricStatus = "pending"
    claim_id: str | None = Field(default=None, max_length=256)
    evidence_ids: list[str] = Field(default_factory=list)


class CoordinationUnitWrite(BaseModel):
    """Atomic replacement for one parent-Run or section coordination unit."""

    run_id: str = Field(min_length=1, max_length=128)
    scope_type: ScopeType
    scope_id: str = Field(min_length=1, max_length=128)
    source_run_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(default=1, ge=1)
    summary: str = ""
    summary_claim_ids: list[str] = Field(default_factory=list)
    summary_metric_ids: list[str] = Field(default_factory=list)
    status: ScopeStatus = "provisional"
    dependency_claim_ids: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    expected_workspace_version: int = Field(default=0, ge=0)
    documents: list[DocumentArtifact] = Field(default_factory=list)
    evidence: list[EvidenceArtifact] = Field(default_factory=list)
    referenced_evidence_ids: list[str] = Field(default_factory=list)
    claims: list[ClaimArtifact] = Field(default_factory=list)
    metrics: list[MetricArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_local_references(self):
        def unique(values: list[str], label: str) -> set[str]:
            result = set(values)
            if len(result) != len(values):
                raise ValueError(f"duplicate {label}")
            return result

        document_ids = unique([item.document_id for item in self.documents], "document_id")
        evidence_ids = unique([item.evidence_id for item in self.evidence], "evidence_id")
        referenced_evidence_ids = unique(self.referenced_evidence_ids, "referenced_evidence_id")
        if evidence_ids & referenced_evidence_ids:
            raise ValueError("evidence cannot be both owned and referenced")
        claim_ids = unique([item.claim_id for item in self.claims], "claim_id")
        unique([item.metric_id for item in self.metrics], "metric_id")
        for claim in self.claims:
            binding_ids = [item.evidence_id for item in claim.evidence_bindings]
            if len(binding_ids) != len(set(binding_ids)):
                raise ValueError(
                    f"duplicate evidence binding in claim {claim.claim_id}"
                )
        for metric in self.metrics:
            unique(metric.evidence_ids, f"metric evidence_id in {metric.metric_id}")

        missing_documents = {item.document_id for item in self.evidence} - document_ids
        if missing_documents:
            raise ValueError(f"evidence references unknown documents: {sorted(missing_documents)}")
        bound_evidence = {
            binding.evidence_id
            for claim in self.claims
            for binding in claim.evidence_bindings
        }
        metric_evidence = {
            evidence_id for metric in self.metrics for evidence_id in metric.evidence_ids
        }
        missing_evidence = (bound_evidence | metric_evidence) - (
            evidence_ids | referenced_evidence_ids
        )
        if missing_evidence:
            raise ValueError(f"bindings reference unknown evidence: {sorted(missing_evidence)}")
        missing_claims = {
            metric.claim_id for metric in self.metrics if metric.claim_id
        } - claim_ids
        if missing_claims:
            raise ValueError(f"metrics reference unknown claims: {sorted(missing_claims)}")
        unknown_summary_claims = set(self.summary_claim_ids) - claim_ids
        if unknown_summary_claims:
            raise ValueError(
                f"summary references unknown claims: {sorted(unknown_summary_claims)}"
            )
        metric_ids = {item.metric_id for item in self.metrics}
        unknown_summary_metrics = set(self.summary_metric_ids) - metric_ids
        if unknown_summary_metrics:
            raise ValueError(
                f"summary references unknown metrics: {sorted(unknown_summary_metrics)}"
            )
        return self


class CoordinationUnit(BaseModel):
    """Read model for the logical parent/section table consumed by coordination."""

    run_id: str
    scope_type: ScopeType
    scope_id: str
    source_run_id: str
    revision: int
    summary: str = ""
    summary_claim_ids: list[str] = Field(default_factory=list)
    summary_metric_ids: list[str] = Field(default_factory=list)
    status: ScopeStatus
    dependency_claim_ids: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    evidence: list[dict] = Field(default_factory=list)
    claims: list[dict] = Field(default_factory=list)
    metrics: list[dict] = Field(default_factory=list)
    updated_at: datetime


class CoordinationSnapshot(BaseModel):
    """Versioned Run-level view: one optional parent scope plus section scopes."""

    run_id: str
    schema_version: int = 1
    workspace_version: int = Field(ge=0)
    status: WorkspaceStatus
    summary: str = ""
    parent_run: CoordinationUnit | None = None
    sections: list[CoordinationUnit] = Field(default_factory=list)
    updated_at: datetime
