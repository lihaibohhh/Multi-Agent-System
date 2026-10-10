"""Deterministic chapter selection, retry, and graph routing policies."""

from __future__ import annotations

from ..core.config import settings
from . import claim_repair
from .context_builder import current_section
from .models import SectionPolicy, SectionRecord, SectionReview
from .operations import FINISHED
from .rendering import evidence_key, merge_results


def default_section_policy() -> SectionPolicy:
    """Create the limits persisted with a newly planned Run."""
    return SectionPolicy(
        max_search_rounds=settings.agent.section_max_search_rounds,
        max_revisions=settings.agent.section_max_revisions,
    )


def section_policy(state: dict) -> SectionPolicy:
    return SectionPolicy.model_validate(state["section_policy"])


def select_sources(section: SectionRecord) -> list[dict]:
    """Select a bounded chapter-local evidence view, preserving fresh supplements."""
    ranked = sorted(section.results, key=lambda item: item.get("score", 0), reverse=True)
    fresh_ids = set(section.evidence_update.get("source_ids", []))
    fresh = [result for result in ranked if evidence_key(result) in fresh_ids][:8]
    return merge_results(fresh, ranked)[:15]


def next_step_after_review(
    section: SectionRecord,
    review: SectionReview,
    policy: SectionPolicy,
) -> str | None:
    """Return a retry step, or None when semantic review is finished."""
    revisions_used = section.revision - section.revision_base
    if review.verdict != "revise" or revisions_used > policy.max_revisions:
        return None
    if review.search_queries and section.search_rounds < policy.max_search_rounds:
        return "research"
    return "write"


def claim_gate(state: dict) -> dict:
    section = current_section(state)
    work = section.claim_work
    if work.get("epoch") == claim_repair.epoch() and work.get("attempts", 0) >= 3:
        raise claim_repair.ClaimsPending(section)
    return {"section_step": "claims"}


def advance_section(state: dict) -> dict:
    operation = (state.get("parent_context") or {}).get("section_operation")
    if operation:
        for index in range(state["active_section"] + 1, len(state["sections"])):
            section = state["sections"][index]
            if section["section_id"] in operation["work_ids"] and section["status"] not in FINISHED:
                return {"active_section": index, "section_step": "research"}
        complete = all(section["status"] in FINISHED for section in state["sections"])
        return {"section_step": "report_review" if complete else "assemble"}
    index = state["active_section"] + 1
    while (
        index < len(state["sections"])
        and state["sections"][index]["status"] in FINISHED
    ):
        index += 1
    return {
        "active_section": index,
        "section_step": "research" if index < len(state["sections"]) else "report_review",
    }


def route_parent(state: dict) -> str:
    """Route only between parent-level planning, chapter, and report boundaries."""
    step = state["section_step"]
    if step in {"research", "write", "review", "advance", "claims", "claim_gate"}:
        return "section_cycle"
    if step == "report_review":
        return "report_review"
    if step == "chief_edit":
        return "chief_edit"
    if step == "edited_report_review":
        return "edited_report_review"
    if step == "assemble":
        return "assemble_report"
    raise ValueError(f"父图无法处理 section_step={step!r}")


_policy = section_policy
_select_sources = select_sources


__all__ = [
    "advance_section",
    "claim_gate",
    "default_section_policy",
    "next_step_after_review",
    "route_parent",
    "section_policy",
    "select_sources",
]
