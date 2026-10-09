"""Runtime-managed section-planning agent."""

from __future__ import annotations

from .context import AgentContext
from .contracts import ModelCall, SectionPlanningRequest
from .runtime import AgentTurnResult
from .spec import AgentSpec
from ..sections.models import SectionPlan
from ..sections.validation import validate_plan


PLANNER_SYSTEM_PROMPT = (
    "你是专题研究编辑。返回 JSON，字段 sections 为章节数组；每章含 title、question、"
    "kind(research/synthesis)。每个问题须包含研究对象和必要口径，可独立检索。"
    "短问题只设一章。复杂问题按论证需要分章，避免重复；需要综合结论时仅放在最后，"
    "kind=synthesis。不要凭空添加用户未要求的时间、地区或统计口径。"
    "可选 parent_section_ids：只选择与本章有关的父任务章节 ID，无关则空数组。"
)


class PlannerAgent:
    """Turn a research request and bounded parent context into a validated plan."""

    spec = AgentSpec(
        name="planner",
        description="根据研究问题和受限父任务上下文生成章节计划",
        model_ref="section_model",
        input_type=SectionPlanningRequest,
        output_type=SectionPlan,
        version="2",
        max_turns=1,
        # The checked gateway may perform up to three bounded correction calls.
        # Each individual call retains the existing configured model timeout.
        timeout_seconds=3600,
    )

    async def run_turn(
        self,
        request: SectionPlanningRequest,
        *,
        context: AgentContext,
        call_model: ModelCall,
    ) -> AgentTurnResult[SectionPlan]:
        prompt = (
            f"研究问题：{request.research_question}\n"
            f"最多 {request.maximum_sections} 章。\n"
            f"{request.parent_view}"
        )
        plan, _ = await call_model(
            PLANNER_SYSTEM_PROMPT,
            prompt,
            SectionPlan,
            validator=lambda value: validate_plan(
                value,
                request.maximum_sections,
                set(request.available_parent_section_ids),
            ),
            context={
                "agent": self.spec.name,
                "agent_version": self.spec.version,
                "agent_run_id": context.agent_run_id,
            },
        )
        return AgentTurnResult(
            status="completed",
            output=plan,
            handoff={"plan": plan.model_dump(mode="json")},
        )
