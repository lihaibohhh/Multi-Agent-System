"""Chapter semantic review and deterministic Claim binding nodes."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ...agents.bootstrap import agent_context, agent_runner, call_model as default_call_model
from ...agents.registry import agent_registry
from ...coordination.briefing import current_research_brief
from ...processors import claim_binding_processor
from ..context_builder import current_section
from ..policies import section_policy
from ..request_factory import claim_binding_request, section_review_request
from ..transitions import claim_binding_state, reviewed_state


async def review_section(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    section = current_section(state)
    coordination_brief = await current_research_brief(section)
    result = await runner.run(
        agent_registry.section_reviewer,
        section_review_request(state, section, coordination_brief),
        context=agent_context(config, section_id=section.section_id),
    )
    if result.output is None:
        raise RuntimeError("SectionReviewerAgent 未返回章节审校结果")
    return reviewed_state(
        state,
        section,
        result.output,
        section_policy(state),
        result.usage,
    )


async def extract_claims(state: dict, *, call_model=default_call_model) -> dict:
    section = current_section(state)
    coordination_brief = await current_research_brief(section)
    binding = await claim_binding_processor.process_attempt(
        claim_binding_request(state, section, coordination_brief),
        call_model=call_model,
    )
    return claim_binding_state(state, section, binding)


__all__ = ["extract_claims", "review_section"]
