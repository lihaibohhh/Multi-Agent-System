"""Agent responsible only for semantic review of one chapter draft."""

from __future__ import annotations

from .contracts import ModelCall, SectionReviewRequest, SectionReviewResult
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

    name = "section_reviewer"

    async def run(
        self,
        request: SectionReviewRequest,
        *,
        call_model: ModelCall,
    ) -> SectionReviewResult:
        review, cost = await call_model(
            SECTION_REVIEWER_SYSTEM_PROMPT,
            request.section_context
            + "来源：\n"
            + request.evidence_text
            + f"\n草稿：\n{request.draft}",
            SectionReview,
            validator=validate_section_review,
            context={"agent": self.name, "section_id": request.section_id},
        )
        return SectionReviewResult(review=review, cost=cost)
