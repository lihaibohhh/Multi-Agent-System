"""Agent responsible only for chapter-level evidence sufficiency analysis."""

from __future__ import annotations

from .contracts import EvidenceAnalysisRequest, EvidenceAnalysisResult, ModelCall
from ..sections.models import SectionReview
from ..sections.validation import validate_section_review


EVIDENCE_ANALYST_SYSTEM_PROMPT = (
    "审查本章证据是否足以回答问题。返回 JSON：verdict(pass/revise)、issues(缺口列表)、"
    "search_queries(最多2条可直接检索的关键词)、summary(简短分析)。"
    "没有来源不得通过；相关性分数不等于事实核验。检查反例、日期、单位和口径。"
)


class EvidenceAnalystAgent:
    """Judge evidence sufficiency without mutating graph or chapter state."""

    name = "evidence_analyst"

    async def run(
        self,
        request: EvidenceAnalysisRequest,
        *,
        call_model: ModelCall,
    ) -> EvidenceAnalysisResult:
        review, cost = await call_model(
            EVIDENCE_ANALYST_SYSTEM_PROMPT,
            request.section_context + "来源：\n" + request.evidence_text,
            SectionReview,
            validator=validate_section_review,
            context={"agent": self.name, "section_id": request.section_id},
        )
        return EvidenceAnalysisResult(review=review, cost=cost)
