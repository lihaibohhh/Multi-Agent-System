"""Versioned chapter artifacts; source records are retrieved excerpts, not a KB copy."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SectionSpec(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    question: str = Field(min_length=5, max_length=500)
    kind: Literal["research", "synthesis"] = "research"


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


class SectionDraft(BaseModel):
    revision: int
    draft: str
    sources: list[dict] = Field(default_factory=list)


class SectionRecord(SectionSpec):
    section_id: str
    status: Literal["pending", "researching", "drafted", "complete", "limited"] = "pending"
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


class SectionPolicy(BaseModel):
    """Persisted limits keep resumed work bounded even if environment settings change."""
    max_search_rounds: int = Field(default=2, ge=1, le=4)
    max_revisions: int = Field(default=1, ge=0, le=3)
