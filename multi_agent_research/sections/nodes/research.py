"""Evidence research node and its deterministic state preparation."""

from __future__ import annotations

from copy import deepcopy

from langchain_core.runnables import RunnableConfig

from ...agents.bootstrap import agent_runner, resumable_agent_context
from ...agents.registry import agent_registry
from ...coordination.briefing import current_research_brief
from ..context_builder import current_section
from ..models import SectionRecord
from ..policies import section_policy
from ..rendering import merge_results
from ..request_factory import evidence_research_request
from ..transitions import researched_state


def _inherit_parent_sources(state: dict, section) -> None:
    context = state.get("parent_context") or {}
    inherited = []
    for parent in context.get("handoff", []):
        if parent["section_id"] not in section.parent_section_ids:
            continue
        for raw in parent["sources"]:
            source = deepcopy(raw)
            source.setdefault("metadata", {})["inherited_from_run"] = context["source_run_id"]
            source["metadata"]["inherited_from_section"] = parent["section_id"]
            inherited.append(source)
    section.results = merge_results(section.results, inherited)


def _inherit_shared_section_sources(state: dict, section) -> None:
    """Offer evidence behind prior accepted Claims; the research Agent re-evaluates it."""

    shared = []
    for raw in state["sections"][: state["active_section"]]:
        prior = SectionRecord.model_validate(raw)
        if prior.status not in {"complete", "limited"}:
            continue
        source_numbers = {
            link.source_number
            for claim in prior.claims
            if claim.assessment == "supported"
            for link in claim.evidence
            if link.relation == "supports"
        }
        for source_number in sorted(source_numbers):
            if not 1 <= source_number <= len(prior.sources):
                continue
            source = deepcopy(prior.sources[source_number - 1])
            source.setdefault("metadata", {})["shared_from_section"] = prior.section_id
            shared.append(source)
    section.results = merge_results(section.results, shared[:8])


def _prepare_synthesis_handoff(state: dict, section, force_search: bool) -> bool:
    synthesis_handoff = (
        section.kind == "synthesis" and section.search_rounds == 0 and not force_search
    )
    if not synthesis_handoff:
        return False
    section.results = merge_results(
        section.results,
        [
            source
            for prior in state["sections"][: state["active_section"]]
            if prior["section_id"] in section.depends_on
            for source in prior["sources"]
        ],
    )
    section.search_rounds += 1
    return True


async def research_section(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    section = current_section(state)
    initial_search_rounds = section.search_rounds
    section.status = "researching"
    parent_context = state.get("parent_context") or {}
    if section.search_rounds == 0 and not parent_context.get("revision_target"):
        _inherit_parent_sources(state, section)
        _inherit_shared_section_sources(state, section)

    operation = parent_context.get("section_operation")
    force_search = bool(
        operation
        and operation["target"] == section.section_id
        and operation["mode"] in {"refresh", "supplement"}
    )
    analyze_existing = bool(
        operation and operation["mode"] == "continue" and section.results
    )
    synthesis_handoff = _prepare_synthesis_handoff(state, section, force_search)
    retrieval_context = (
        operation.get("retrieval_context")
        if operation and operation["mode"] == "continue"
        else parent_context
    )
    coordination_brief = await current_research_brief(section)
    result = await runner.run(
        agent_registry.evidence_research,
        evidence_research_request(
            state,
            section,
            parent_question=str((retrieval_context or {}).get("source_question", "")).strip(),
            max_search_rounds=section_policy(state).max_search_rounds,
            synthesis_handoff=synthesis_handoff,
            analyze_existing=analyze_existing,
            supplement=bool(operation and operation["mode"] == "supplement"),
            force_search=force_search,
            coordination_brief=coordination_brief,
        ),
        context=await resumable_agent_context(
            agent_registry.evidence_research.spec.name,
            config,
            section_id=section.section_id,
        ),
    )
    if result.output is None:
        raise RuntimeError("EvidenceResearchAgent 未返回研究结果")
    return researched_state(
        state,
        section,
        result.output,
        initial_search_rounds=initial_search_rounds,
        operation=operation,
        force_search=force_search,
        cost=result.usage,
    )


__all__ = ["research_section"]
