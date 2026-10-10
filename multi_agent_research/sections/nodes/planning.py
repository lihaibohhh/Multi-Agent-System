"""Planning node: prepare one typed request and persist its validated result."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from ...agents.bootstrap import agent_context, agent_runner
from ...agents.registry import agent_registry
from ...core.config import settings
from ..artifacts import validate_dependencies
from ..models import SectionPolicy, SectionRecord
from ..operations import continue_step
from ..policies import default_section_policy
from ..request_factory import planning_request
from ..transitions import planned_state


async def plan_sections(
    state: dict,
    config: RunnableConfig | None = None,
    *,
    runner=agent_runner,
) -> dict:
    context = state.get("parent_context") or {}
    policy = default_section_policy()
    if context.get("revision_target"):
        sections = [SectionRecord.model_validate(raw) for raw in context["revision_sections"]]
        validate_dependencies(sections)
        operation = context.get("section_operation")
        if operation and operation.get("section_policy"):
            policy = SectionPolicy.model_validate(operation["section_policy"])
        index = (
            next(i for i, section in enumerate(sections) if section.section_id == operation["target"])
            if operation
            else next(i for i, section in enumerate(sections) if section.status == "stale")
        )
        step = (
            continue_step(sections[index], policy)
            if operation and operation["mode"] == "continue"
            else "research"
        )
        return {
            "sections": [section.model_dump(mode="json") for section in sections],
            "active_section": index,
            "section_policy": policy.model_dump(),
            "section_step": step,
        }

    available = {item["section_id"] for item in context.get("handoff", [])}
    result = await runner.run(
        agent_registry.planner,
        planning_request(
            state,
            maximum_sections=settings.agent.section_max_count,
            available_parent_section_ids=available,
        ),
        context=agent_context(config),
    )
    if result.output is None:
        raise RuntimeError("PlannerAgent 未返回章节计划")
    return planned_state(state, result.output, policy, result.usage)


__all__ = ["plan_sections"]
