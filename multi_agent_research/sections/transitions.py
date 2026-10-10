"""Deterministic transformations from domain results to graph state updates."""

from __future__ import annotations

from datetime import datetime, timezone

from ..agents.contracts import EvidenceResearchResult, ModelCost, SectionWritingResult
from ..processors import ClaimBindingResult
from .context_builder import dependency_revisions
from .models import SectionDraft, SectionPlan, SectionPolicy, SectionRecord, SectionReview
from .policies import next_step_after_review
from .rendering import citation_issues


def usage_delta(state: dict, cost: ModelCost | dict) -> dict:
    return {
        "token_budget_used": state.get("token_budget_used", 0) + cost["tokens"],
        "model_calls": state.get("model_calls", 0) + cost.get("attempts", 1),
        "usage_unknown_calls": state.get("usage_unknown_calls", 0) + cost["unknown"],
    }


def update_section(state: dict, section: SectionRecord, step: str, **extra) -> dict:
    sections = list(state["sections"])
    sections[state["active_section"]] = section.model_dump(mode="json")
    return {"sections": sections, "section_step": step, **extra}


def archive_draft(section: SectionRecord) -> None:
    if section.draft:
        section.previous_drafts.append(
            SectionDraft(
                revision=section.revision,
                draft=section.draft,
                sources=section.sources,
                claims=section.claims,
                reviewed_at=section.reviewed_at,
            )
        )


def planned_state(
    state: dict,
    plan: SectionPlan,
    policy: SectionPolicy,
    cost: ModelCost | dict,
) -> dict:
    sections = [
        SectionRecord(**spec.model_dump(), section_id=f"section_{index}").model_dump(
            mode="json"
        )
        for index, spec in enumerate(plan.sections, 1)
    ]
    for index, section in enumerate(sections):
        section["artifact_version"] = 2
        section["depends_on"] = (
            [prior["section_id"] for prior in sections[:index]]
            if section["kind"] == "synthesis"
            else []
        )
    return {
        "sections": sections,
        "active_section": 0,
        "section_policy": policy.model_dump(),
        "section_step": "research",
        **usage_delta(state, cost),
    }


def researched_state(
    state: dict,
    section: SectionRecord,
    research: EvidenceResearchResult,
    *,
    initial_search_rounds: int,
    operation: dict | None,
    force_search: bool,
    cost: ModelCost | dict,
) -> dict:
    section.results = list(research.results)
    section.search_rounds = research.search_rounds
    section.analyst = research.review.model_dump(mode="json")
    section.gaps = list(research.review.search_queries)
    if force_search and operation:
        section.evidence_update = {
            "mode": operation["mode"],
            "result_count": len(research.fresh_result_ids),
            "source_ids": list(research.fresh_result_ids),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        }
    if operation and operation["mode"] == "supplement":
        section.status = (
            "evidence_ready"
            if section.results and research.review.verdict == "pass"
            else "waiting_evidence"
        )
        section.limitations = list(research.review.issues)
        next_step = "assemble"
    else:
        if research.review.verdict == "revise":
            section.limitations = list(
                dict.fromkeys(section.limitations + research.review.issues)
            )
            if not research.review.issues:
                section.limitations.append("检索预算已用完，证据审查未通过")
        next_step = "write"
    return update_section(
        state,
        section,
        next_step,
        iteration_count=(
            state.get("iteration_count", 0)
            + max(0, section.search_rounds - initial_search_rounds)
        ),
        **usage_delta(state, cost),
    )


def missing_evidence_state(state: dict, section: SectionRecord) -> dict:
    """Finish a chapter deterministically when retrieval found no usable evidence."""
    archive_draft(section)
    section.draft = "未取得可用检索证据，无法对本章问题作出可靠结论。"
    section.sources = []
    section.claims = []
    section.claim_work = {}
    section.revision += 1
    section.reviewed_at = None
    section.dependency_revisions = dependency_revisions(state, section)
    section.limitations = list(dict.fromkeys(section.limitations + ["未检索到可用证据"]))
    section.status = "limited"
    return update_section(state, section, "advance")


def written_state(
    state: dict,
    section: SectionRecord,
    writing: SectionWritingResult,
    selected: list[dict],
    cost: ModelCost | dict,
) -> dict:
    archive_draft(section)
    section.draft = writing.draft
    section.sources = selected
    section.claims = []
    section.claim_work = {}
    section.reviewed_at = None
    section.revision += 1
    section.status = "drafted"
    return update_section(state, section, "review", **usage_delta(state, cost))


def reviewed_state(
    state: dict,
    section: SectionRecord,
    review: SectionReview,
    policy: SectionPolicy,
    cost: ModelCost | dict,
) -> dict:
    invalid = citation_issues(section.draft, section.sources)
    if invalid:
        review.verdict = "revise"
        review.issues = list(dict.fromkeys(review.issues + invalid))[:8]
    section.review = review
    retry_step = next_step_after_review(section, review, policy)
    if retry_step:
        section.gaps = review.search_queries
        return update_section(
            state,
            section,
            retry_step,
            **usage_delta(state, cost),
        )
    if invalid:
        raise ValueError("chapter citation validation failed: " + "; ".join(invalid))
    if review.verdict == "revise":
        section.limitations = list(
            dict.fromkeys(
                section.limitations
                + review.issues
                + ["章节审校仍有未解决问题，修订预算已用完"]
            )
        )
    section.dependency_revisions = dependency_revisions(state, section)
    # Semantic review has finished; Claim validation owns final complete/limited status.
    section.status = "drafted"
    return update_section(state, section, "claims", **usage_delta(state, cost))


def claim_binding_state(
    state: dict,
    section: SectionRecord,
    binding: ClaimBindingResult,
) -> dict:
    section.claim_work = binding.work
    section.claims = list(binding.claims)
    if not binding.completed:
        section.status = "claims_pending"
        return update_section(
            state,
            section,
            "claim_gate",
            **usage_delta(state, binding.cost),
        )
    section.reviewed_at = datetime.now(timezone.utc)
    section.limitations.extend(binding.limitations)
    section.limitations = list(dict.fromkeys(section.limitations))
    section.status = "limited" if section.limitations else "complete"
    return update_section(
        state,
        section,
        "advance",
        **usage_delta(state, binding.cost),
    )


_usage = usage_delta
_update = update_section
_archive_draft = archive_draft


__all__ = [
    "archive_draft",
    "claim_binding_state",
    "missing_evidence_state",
    "planned_state",
    "researched_state",
    "reviewed_state",
    "update_section",
    "usage_delta",
    "written_state",
]
