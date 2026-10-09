from __future__ import annotations

import json

import pytest

from multi_agent_research.agents.claim_extractor_agent import ClaimExtractorAgent
from multi_agent_research.agents.contracts import ClaimExtractionRequest
from multi_agent_research.sections import claim_repair
from multi_agent_research.sections.model_output import ModelOutputError
from multi_agent_research.sections.models import ClaimExtraction, SectionRecord


def _example() -> tuple[SectionRecord, dict, dict]:
    section = SectionRecord(
        section_id="section_2",
        title="行业",
        question="行业趋势如何？",
        draft="头部企业的马太效应凸显。其他已经通过的结论。",
        revision=2,
        sources=[
            {
                "content": "细分行业进入头部企业强者恒强阶段。",
                "title": "来源",
                "source": "knowledge",
                "metadata": {},
                "score": 0.9,
                "query": "行业趋势",
            }
        ],
    )
    bad = {
        "statement": "头部企业优势加强",
        "draft_quote": "细分行业进入头部企业强者恒强阶段",
        "assessment": "supported",
        "evidence": [
            {
                "source_number": 1,
                "quote": "细分行业进入头部企业强者恒强阶段",
                "relation": "supports",
            }
        ],
    }
    good = dict(
        bad,
        statement="不能被重写的通过项",
        draft_quote="其他已经通过的结论",
    )
    return section, bad, good


def _request(section: SectionRecord, work: dict) -> ClaimExtractionRequest:
    return ClaimExtractionRequest(
        section=section,
        section_context="本章：行业\n本章必须回答：行业趋势如何？\n",
        evidence_text="[来源1] 细分行业进入头部企业强者恒强阶段。",
        work=work,
        attempt=1,
        total_attempt=4,
    )


@pytest.mark.asyncio
async def test_claim_extractor_owns_initial_prompt_schema_and_context() -> None:
    section, _, good = _example()
    initial = claim_repair.new_work(section)
    captured = {}

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        captured.update(system=system, prompt=prompt, schema=schema, context=context)
        extraction = ClaimExtraction(claims=[good])
        return validator(extraction), {"tokens": 17, "unknown": 0, "attempts": 1}

    result = await ClaimExtractorAgent().run(
        _request(section, initial),
        call_model=call_model,
    )

    assert set(result.work["accepted"]) == {"1"}
    assert result.work["pending"] == []
    assert initial["accepted"] == {}
    assert result.cost["tokens"] == 17
    assert captured["schema"] is ClaimExtraction
    assert captured["context"] == {
        "agent": "claim_extractor",
        "section_id": "section_2",
        "revision": 2,
        "single_attempt": True,
        "claim_attempt": 1,
        "claim_total_attempt": 4,
        "claim_mode": "extract",
    }
    assert "不声称穷尽" in captured["system"]
    assert "草稿：\n头部企业的马太效应凸显" in captured["prompt"]


@pytest.mark.asyncio
async def test_claim_extractor_repairs_only_pending_slots() -> None:
    section, bad, good = _example()
    work = claim_repair.split_extraction(section, [good, bad])
    accepted_before = dict(work["accepted"])
    captured = {}

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        captured.update(system=system, prompt=prompt, schema=schema, context=context)
        view = json.loads(prompt)
        draft_span = next(key for key in view["excerpts"] if key.startswith("D:"))
        source_span = next(key for key in view["excerpts"] if key.startswith("S"))
        patches = claim_repair.ClaimRepairs(
            repairs=[
                {
                    "slot": 2,
                    "statement": bad["statement"],
                    "draft_span_id": draft_span,
                    "assessment": "supported",
                    "evidence": [
                        {"source_span_id": source_span, "relation": "supports"}
                    ],
                }
            ]
        )
        return validator(patches), {"tokens": 9, "unknown": 0}

    result = await ClaimExtractorAgent().run(
        _request(section, work),
        call_model=call_model,
    )

    assert captured["schema"] is claim_repair.ClaimRepairs
    assert captured["context"]["claim_mode"] == "repair"
    assert "只修复给出的 pending Claim" in captured["system"]
    assert "不能被重写的通过项" not in captured["prompt"]
    assert result.work["accepted"]["1"] == accepted_before["1"]
    assert set(result.work["accepted"]) == {"1", "2"}
    assert result.work["pending"] == []


@pytest.mark.asyncio
async def test_claim_extractor_salvages_valid_siblings_from_retryable_output() -> None:
    section, bad, good = _example()
    malformed = dict(bad, assessment="invalid")

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        raise ModelOutputError(
            "schema failed",
            record={
                "retryable": True,
                "raw": json.dumps({"claims": [good, malformed]}, ensure_ascii=False),
                "errors": [{"field": "claims.1.assessment", "message": "invalid"}],
                "diagnostic_id": "diag-claim",
            },
            cost={"tokens": 11, "unknown": 0, "attempts": 1},
        )

    result = await ClaimExtractorAgent().run(
        _request(section, claim_repair.new_work(section)),
        call_model=call_model,
    )

    assert set(result.work["accepted"]) == {"1"}
    assert [pending["slot"] for pending in result.work["pending"]] == [2]
    assert result.work["diagnostic_id"] == "diag-claim"
    assert result.cost["tokens"] == 11
