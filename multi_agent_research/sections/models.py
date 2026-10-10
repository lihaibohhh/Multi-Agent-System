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
    kind: Literal[
        "conflict",
        "scope",
        "duplication",
        "coverage",
        "dependency",
        "structure",
        "readability",
        "provenance",
    ]
    section_ids: list[str] = Field(default_factory=list, max_length=4)
    detail: str = Field(min_length=1, max_length=1500)


class ReportReview(BaseModel):
    verdict: Literal["pass", "revise"]
    issues: list[ConsistencyIssue] = Field(default_factory=list, max_length=12)
    summary: str = Field(default="", max_length=2000)


class EditorialIssueResolution(BaseModel):
    """How the editor handled one issue from the independent report review."""

    issue_index: int = Field(ge=0)
    action: Literal["resolved_by_edit", "preserved_as_limitation"]
    explanation: str = Field(min_length=1, max_length=1000)
    section_ids: list[str] = Field(default_factory=list, max_length=4)


class EditedReportSection(BaseModel):
    """One reader-facing section with stable evidence tokens and provenance."""

    title: str = Field(min_length=1, max_length=120)
    # This is a transport/candidate safety ceiling, not the publication target.
    # The coordinator applies the chapter-specific visible-length budget after
    # parsing so an overlong but otherwise valid candidate can be compressed.
    body: str = Field(min_length=1, max_length=12000)
    source_section_ids: list[str] = Field(min_length=1, max_length=4)
    claim_ids: list[str] = Field(default_factory=list, max_length=48)


class EditorialTerm(BaseModel):
    """One canonical term the whole report must use consistently."""

    term: str = Field(min_length=1, max_length=80)
    meaning: str = Field(min_length=1, max_length=300)


class EditorialSectionPlan(BaseModel):
    """Shared editorial contract for one source chapter."""

    source_section_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=120)
    purpose: str = Field(min_length=1, max_length=400)
    claim_ids: list[str] = Field(default_factory=list, max_length=24)
    evidence_ids: list[str] = Field(default_factory=list, max_length=24)
    transition_in: str = Field(default="", max_length=300)
    transition_out: str = Field(default="", max_length=300)
    target_chars: int = Field(default=2000, ge=400, le=3000)


class EditorialBlueprint(BaseModel):
    """Whole-report plan shared by every bounded ChiefEditor call."""

    verdict: Literal["ready", "limited"]
    report_title: str = Field(min_length=1, max_length=180)
    thesis: str = Field(min_length=1, max_length=1000)
    audience: str = Field(min_length=1, max_length=300)
    style_rules: list[str] = Field(min_length=1, max_length=8)
    terminology: list[EditorialTerm] = Field(default_factory=list, max_length=12)
    section_plans: list[EditorialSectionPlan] = Field(min_length=1, max_length=4)
    issue_resolutions: list[EditorialIssueResolution] = Field(
        default_factory=list,
        max_length=12,
    )
    unresolved_issues: list[str] = Field(default_factory=list, max_length=12)


class EditedSectionArtifact(BaseModel):
    """One evidence-safe chapter candidate; publication length is state-dependent."""

    section: EditedReportSection
    summary: str = Field(min_length=1, max_length=600)
    handoff: str = Field(default="", max_length=400)


class EditorialFraming(BaseModel):
    """Front/back matter generated only after all edited chapters exist."""

    executive_summary: str = Field(min_length=1, max_length=1300)
    conclusion: str = Field(min_length=1, max_length=1300)


class ChiefEditorStepResult(BaseModel):
    """Phase-tagged output contract for the single ChiefEditorAgent."""

    phase: Literal["plan", "section", "compress", "framing"]
    blueprint: EditorialBlueprint | None = None
    section_artifact: EditedSectionArtifact | None = None
    framing: EditorialFraming | None = None

    @model_validator(mode="after")
    def validate_phase_payload(self):
        expected = {
            "plan": self.blueprint,
            "section": self.section_artifact,
            "compress": self.section_artifact,
            "framing": self.framing,
        }
        populated = sum(value is not None for value in (
            self.blueprint,
            self.section_artifact,
            self.framing,
        ))
        if populated != 1 or expected[self.phase] is None:
            raise ValueError("phase 必须且只能携带对应的主编阶段产物")
        return self


class ChiefEditorResult(BaseModel):
    """Structured whole-report edit produced from already reviewed chapters."""

    verdict: Literal["ready", "limited"]
    report_title: str = Field(min_length=1, max_length=180)
    executive_summary: str = Field(min_length=1, max_length=3000)
    sections: list[EditedReportSection] = Field(min_length=1, max_length=8)
    conclusion: str = Field(min_length=1, max_length=3000)
    issue_resolutions: list[EditorialIssueResolution] = Field(
        default_factory=list,
        max_length=12,
    )
    used_claim_ids: list[str] = Field(default_factory=list, max_length=96)
    unresolved_issues: list[str] = Field(default_factory=list, max_length=12)


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
