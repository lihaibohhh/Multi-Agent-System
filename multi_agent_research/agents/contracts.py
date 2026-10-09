"""Typed contracts shared by the independently managed research agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, NotRequired, Protocol, TypedDict

from ..sections.models import ReportReview, SectionRecord, SectionReview


class ModelCost(TypedDict):
    """Normalized usage returned by the checked model gateway."""

    tokens: int
    unknown: int
    attempts: NotRequired[int]


class ModelCall(Protocol):
    """Temporary gateway contract while model ownership moves out of the workflow."""

    def __call__(
        self,
        system: str,
        prompt: str,
        schema: Any = None,
        *,
        validator: Callable[[Any], Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> Awaitable[tuple[Any, ModelCost]]: ...


@dataclass(frozen=True, slots=True)
class SectionPlanningRequest:
    """Only the information the planner needs; never the full graph state."""

    research_question: str
    maximum_sections: int
    parent_view: str
    available_parent_section_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class EvidenceAnalysisRequest:
    """Bounded context for deciding whether one chapter has enough evidence."""

    section_id: str
    section_context: str
    evidence_text: str


@dataclass(frozen=True, slots=True)
class EvidenceAnalysisResult:
    """Typed result plus normalized model usage for orchestration accounting."""

    review: SectionReview
    cost: ModelCost


@dataclass(frozen=True, slots=True)
class SectionWritingRequest:
    """Bounded inputs for producing one chapter draft."""

    section_id: str
    next_revision: int
    section_context: str
    evidence_text: str
    sources: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]
    current_draft: str = ""
    previous_sources: tuple[dict[str, Any], ...] = ()
    review: SectionReview | None = None


@dataclass(frozen=True, slots=True)
class SectionWritingResult:
    """Generated chapter body plus normalized usage."""

    draft: str
    cost: ModelCost


@dataclass(frozen=True, slots=True)
class SectionReviewRequest:
    """Bounded inputs for reviewing one persisted chapter draft."""

    section_id: str
    section_context: str
    evidence_text: str
    draft: str


@dataclass(frozen=True, slots=True)
class SectionReviewResult:
    """Semantic chapter review plus normalized usage."""

    review: SectionReview
    cost: ModelCost


@dataclass(frozen=True, slots=True)
class ClaimExtractionRequest:
    """One bounded extraction or repair attempt for a persisted chapter."""

    section: SectionRecord
    section_context: str
    evidence_text: str
    work: dict[str, Any]
    attempt: int
    total_attempt: int


@dataclass(frozen=True, slots=True)
class ClaimExtractionResult:
    """Validated partial Claim work plus normalized usage for one attempt."""

    work: dict[str, Any]
    cost: ModelCost


@dataclass(frozen=True, slots=True)
class ReportReviewRequest:
    """Completed chapter artifacts needed for one whole-report review."""

    research_question: str
    sections: tuple[SectionRecord, ...]


@dataclass(frozen=True, slots=True)
class ReportReviewResult:
    """Validated whole-report review plus normalized usage."""

    review: ReportReview
    cost: ModelCost
