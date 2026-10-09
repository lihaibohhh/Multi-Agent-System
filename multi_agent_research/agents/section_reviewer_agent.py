"""Runtime-managed Agent for semantic review of one chapter draft."""

from __future__ import annotations

from .context import AgentContext
from .contracts import ModelCall, SectionReviewRequest
from .runtime import AgentTurnResult
from .spec import AgentSpec
from ..sections.models import SectionReview
from ..sections.validation import validate_section_review


SECTION_REVIEWER_SYSTEM_PROMPT = (
    "核查章节草稿与给定来源，返回 JSON：verdict(pass/revise)、issues、search_queries、"
    "summary(供后续章节使用的简短结论，必须保留不确定性)。"
    "检查结论有无原文支持、是否回答本章问题、是否忽略反证。需要新证据才填写查询词；"
    "纯文字/引用修订不填查询词。前章交接仅供一致性检查，不能当新证据。"
)


class SectionReviewerAgent:
    """Review chapter meaning without deciding graph transitions or retry limits."""

    spec = AgentSpec(
        name="section_reviewer",
        description="审查单章草稿的证据支持、问题覆盖和反证处理",
        model_ref="section_model",
        input_type=SectionReviewRequest,
        output_type=SectionReview,
        version="2",
        max_turns=1,
        timeout_seconds=3600,
    )

    async def run_turn(
        self,
        request: SectionReviewRequest,
        *,
        context: AgentContext,
        call_model: ModelCall,
    ) -> AgentTurnResult[SectionReview]:
        review, _ = await call_model(
            SECTION_REVIEWER_SYSTEM_PROMPT,
            request.section_context
            + "来源：\n"
            + request.evidence_text
            + f"\n草稿：\n{request.draft}",
            SectionReview,
            validator=validate_section_review,
            context={
                "agent": self.spec.name,
                "agent_version": self.spec.version,
                "agent_run_id": context.agent_run_id,
                "section_id": request.section_id,
            },
        )
        return AgentTurnResult(
            status="completed",
            output=review,
            handoff={"review": review.model_dump(mode="json")},
        )
