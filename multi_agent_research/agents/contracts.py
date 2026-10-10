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
    """Checked model gateway supplied by the Agent runtime composition root."""

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
class EvidenceResearchRequest:
    """Bounded inputs for an autonomous, read-only chapter research loop."""

    section_id: str
    question: str
    section_context: str
    initial_results: tuple[dict[str, Any], ...]
    initial_gaps: tuple[str, ...]
    parent_question: str
    starting_round: int
    max_search_rounds: int
    revision: int
    allow_retrieval: bool = True
    skip_retrieval_on_first_turn: bool = False
    stop_after_one_round: bool = False
    require_fresh_results: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceResearchResult:
    """Evidence, final sufficiency decision, and deterministic loop metadata."""

    review: SectionReview
    results: tuple[dict[str, Any], ...]
    search_rounds: int
    fresh_result_ids: tuple[str, ...] = ()


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
    """Validated chapter body and the number of bounded writing attempts."""

    draft: str
    attempts: int


@dataclass(frozen=True, slots=True)
class SectionReviewRequest:
    """Bounded inputs for reviewing one persisted chapter draft."""

    section_id: str
    section_context: str
    evidence_text: str
    draft: str


@dataclass(frozen=True, slots=True)
class ReportReviewRequest:
    """Completed chapter artifacts needed for one whole-report review."""

    research_question: str
    sections: tuple[SectionRecord, ...]
    candidate_report: str = ""


@dataclass(frozen=True, slots=True)
class ChiefEditorRequest:
    """Bounded, evidence-addressed inputs for whole-report semantic editing."""

    research_question: str
    sections: tuple[SectionRecord, ...]
    report_review: ReportReview
    coordination_context: str
    stable_sections: tuple[dict[str, Any], ...]
    evidence_ids: frozenset[str]
    previous_candidate: dict[str, Any] | None = None
