"""Strict model-facing contracts for Claim extraction and local repair.

These contracts intentionally exclude every program-owned identity and locator.
The persisted :mod:`sections.models` types are constructed only after catalog
membership and business validation have succeeded.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceSelection(BaseModel):
    """One relation backed by exactly one catalog-owned excerpt."""

    model_config = ConfigDict(extra="forbid")

    segment_id: str = Field(min_length=1, max_length=32)
    relation: Literal["supports", "contradicts", "context"] = "supports"


class ClaimCandidate(BaseModel):
    """Only the semantic choices a model is authorized to make."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=1000)
    draft_segment_id: str = Field(min_length=1, max_length=32)
    assessment: Literal["supported", "uncertain", "unsupported"]
    evidence: list[EvidenceSelection] = Field(default_factory=list, max_length=8)
    caveat: str = Field(default="", max_length=1000)


class ClaimCandidateBatch(BaseModel):
    """Initial model output; internal IDs and spans are absent by construction."""

    model_config = ConfigDict(extra="forbid")

    claims: list[ClaimCandidate] = Field(min_length=1, max_length=12)


class ClaimRepairCandidate(BaseModel):
    """A patch for one pending slot; the original statement is program-owned."""

    model_config = ConfigDict(extra="forbid")

    slot: int = Field(ge=1, le=12)
    draft_segment_id: str = Field(min_length=1, max_length=32)
    assessment: Literal["supported", "uncertain", "unsupported"]
    evidence: list[EvidenceSelection] = Field(default_factory=list, max_length=8)
    caveat: str = Field(default="", max_length=1000)


class ClaimRepairBatch(BaseModel):
    """Repair output containing only pending-slot selections."""

    model_config = ConfigDict(extra="forbid")

    repairs: list[ClaimRepairCandidate] = Field(min_length=1, max_length=12)


__all__ = [
    "ClaimCandidate",
    "ClaimCandidateBatch",
    "ClaimRepairBatch",
    "ClaimRepairCandidate",
    "EvidenceSelection",
]
