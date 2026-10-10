"""Typed projections from checkpointed graph state to Agent requests."""

from __future__ import annotations

from ..agents.contracts import (
    ChiefEditorRequest,
    EvidenceResearchRequest,
    ReportReviewRequest,
    SectionPlanningRequest,
    SectionReviewRequest,
    SectionWritingRequest,
)
from .editorial import evidence_tokens, stable_editorial_sections
from ..processors import ClaimBindingRequest
from .context_builder import evidence_text, parent_view, section_prompt
from .models import EditedSectionArtifact, EditorialBlueprint, ReportReview, SectionRecord


def planning_request(
    state: dict,
    *,
    maximum_sections: int,
    available_parent_section_ids: set[str],
) -> SectionPlanningRequest:
    return SectionPlanningRequest(
        research_question=state["research_question"],
        maximum_sections=maximum_sections,
        parent_view=parent_view(state),
        available_parent_section_ids=frozenset(available_parent_section_ids),
    )


def evidence_research_request(
    state: dict,
    section: SectionRecord,
    *,
    parent_question: str,
    max_search_rounds: int,
    synthesis_handoff: bool,
    analyze_existing: bool,
    supplement: bool,
    force_search: bool,
    coordination_brief: str = "",
) -> EvidenceResearchRequest:
    return EvidenceResearchRequest(
        section_id=section.section_id,
        question=section.question,
        section_context=section_prompt(state, section, coordination_brief),
        initial_results=tuple(section.results),
        initial_gaps=tuple(section.gaps[:2]),
        parent_question=parent_question,
        starting_round=section.search_rounds,
        max_search_rounds=max_search_rounds,
        revision=section.revision,
        allow_retrieval=True,
        skip_retrieval_on_first_turn=synthesis_handoff or analyze_existing,
        stop_after_one_round=supplement,
        require_fresh_results=force_search,
    )


def section_writing_request(
    state: dict,
    section: SectionRecord,
    selected: list[dict],
    coordination_brief: str = "",
) -> SectionWritingRequest:
    return SectionWritingRequest(
        section_id=section.section_id,
        next_revision=section.revision + 1,
        section_context=section_prompt(state, section, coordination_brief),
        evidence_text=evidence_text(selected),
        sources=tuple(selected),
        limitations=tuple(section.limitations),
        current_draft=section.draft,
        previous_sources=tuple(section.sources),
        review=section.review,
    )


def section_review_request(
    state: dict,
    section: SectionRecord,
    coordination_brief: str = "",
) -> SectionReviewRequest:
    return SectionReviewRequest(
        section_id=section.section_id,
        section_context=section_prompt(state, section, coordination_brief),
        evidence_text=evidence_text(section.sources),
        draft=section.draft,
    )


def claim_binding_request(
    state: dict,
    section: SectionRecord,
    coordination_brief: str = "",
) -> ClaimBindingRequest:
    return ClaimBindingRequest(
        section=section,
        section_context=section_prompt(state, section, coordination_brief),
        evidence_text=evidence_text(section.sources),
    )


def report_review_request(
    state: dict,
    sections: list[SectionRecord],
    *,
    candidate_report: str = "",
) -> ReportReviewRequest:
    return ReportReviewRequest(
        research_question=state["research_question"],
        sections=tuple(sections),
        candidate_report=candidate_report,
    )


def chief_editor_request(
    state: dict,
    sections: list[SectionRecord],
    coordination_context: str,
    *,
    phase: str = "plan",
    target_section_id: str | None = None,
    section_candidate: EditedSectionArtifact | None = None,
    target_min_chars: int | None = None,
    target_max_chars: int | None = None,
) -> tuple[ChiefEditorRequest, dict[str, dict]]:
    stable_sections, evidence_registry = stable_editorial_sections(sections)
    blueprint = (
        EditorialBlueprint.model_validate(state["editorial_blueprint"])
        if state.get("editorial_blueprint")
        else None
    )
    edited_sections = tuple(
        EditedSectionArtifact.model_validate(item)
        for item in state.get("editorial_sections", [])
    )
    allowed_evidence = set(evidence_registry)
    if phase == "framing":
        allowed_evidence = set().union(*(
            evidence_tokens(item.section.body) for item in edited_sections
        )) if edited_sections else set()
    return ChiefEditorRequest(
        phase=phase,
        research_question=state["research_question"],
        sections=tuple(sections),
        report_review=ReportReview.model_validate(state["report_review"]),
        coordination_context=coordination_context,
        stable_sections=stable_sections,
        evidence_ids=frozenset(allowed_evidence),
        blueprint=blueprint,
        target_section_id=target_section_id,
        edited_sections=edited_sections,
        section_candidate=section_candidate,
        target_min_chars=target_min_chars,
        target_max_chars=target_max_chars,
        previous_candidate=state.get("edited_report"),
    ), evidence_registry


__all__ = [
    "claim_binding_request",
    "chief_editor_request",
    "evidence_research_request",
    "planning_request",
    "report_review_request",
    "section_review_request",
    "section_writing_request",
]
