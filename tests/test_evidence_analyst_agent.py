from __future__ import annotations

import pytest

from multi_agent_research.agents.contracts import EvidenceAnalysisRequest
from multi_agent_research.agents.evidence_analyst_agent import EvidenceAnalystAgent
from multi_agent_research.sections.models import SectionReview


@pytest.mark.asyncio
async def test_evidence_analyst_owns_prompt_schema_and_context() -> None:
    captured = {}

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        captured.update(system=system, prompt=prompt, schema=schema, context=context)
        value = SectionReview(
            verdict="pass",
            summary="证据覆盖了成本和反例",
        )
        return validator(value), {"tokens": 9, "unknown": 0, "attempts": 1}

    result = await EvidenceAnalystAgent().run(
        EvidenceAnalysisRequest(
            section_id="section_2",
            section_context="本章：成本优势\n本章必须回答：成本优势能否持续\n",
            evidence_text="[来源1] 成本数据",
        ),
        call_model=call_model,
    )

    assert result.review.verdict == "pass"
    assert result.cost["tokens"] == 9
    assert captured["schema"] is SectionReview
    assert captured["context"] == {"agent": "evidence_analyst", "section_id": "section_2"}
    assert "检查反例" in captured["system"]
    assert "[来源1] 成本数据" in captured["prompt"]


@pytest.mark.asyncio
async def test_evidence_analyst_downgrades_contradictory_pass() -> None:
    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        value = SectionReview(
            verdict="pass",
            issues=["缺少反面证据"],
            search_queries=["公司 成本 反例"],
        )
        return validator(value), {"tokens": 1, "unknown": 0}

    result = await EvidenceAnalystAgent().run(
        EvidenceAnalysisRequest(
            section_id="section_1",
            section_context="本章：成本\n",
            evidence_text="[来源1] 正面材料",
        ),
        call_model=call_model,
    )

    assert result.review.verdict == "revise"
    assert result.review.issues == ["缺少反面证据"]
