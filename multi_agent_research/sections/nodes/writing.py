"""Chapter writing node."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ...agents.bootstrap import agent_runner, resumable_agent_context
from ...agents.registry import agent_registry
from ...coordination.briefing import current_research_brief
from ..context_builder import current_section
from ..policies import select_sources
from ..request_factory import section_writing_request
from ..transitions import missing_evidence_state, written_state


async def write_section(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    section = current_section(state)
    selected = select_sources(section)
    if not selected:
        return missing_evidence_state(state, section)

    coordination_brief = await current_research_brief(section)
    result = await runner.run(
        agent_registry.section_writer,
        section_writing_request(state, section, selected, coordination_brief),
        context=await resumable_agent_context(
            agent_registry.section_writer.spec.name,
            config,
            section_id=section.section_id,
        ),
    )
    if result.output is None:
        raise RuntimeError("SectionWriterAgent 未返回通过校验的章节正文")
    return written_state(state, section, result.output, selected, result.usage)


__all__ = ["write_section"]
