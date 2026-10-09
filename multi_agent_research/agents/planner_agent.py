"""Section-planning agent with an explicit input/output boundary."""

from __future__ import annotations

from .contracts import ModelCall, ModelCost, SectionPlanningRequest
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

    name = "planner"

    async def run(
        self,
        request: SectionPlanningRequest,
        *,
        call_model: ModelCall,
    ) -> tuple[SectionPlan, ModelCost]:
        prompt = (
            f"研究问题：{request.research_question}\n"
            f"最多 {request.maximum_sections} 章。\n"
            f"{request.parent_view}"
        )
        plan, cost = await call_model(
            PLANNER_SYSTEM_PROMPT,
            prompt,
            SectionPlan,
            validator=lambda value: validate_plan(
                value,
                request.maximum_sections,
                set(request.available_parent_section_ids),
            ),
            context={"agent": self.name},
        )
        return plan, cost
