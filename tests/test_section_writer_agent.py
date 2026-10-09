from __future__ import annotations

import pytest

from multi_agent_research.agents.contracts import SectionWritingRequest
from multi_agent_research.agents.section_writer_agent import SectionWriterAgent
from multi_agent_research.sections.models import SectionReview
from multi_agent_research.sections.validation import BusinessValidationError


def source(name: str = "成本") -> dict:
    return {
        "query": name,
        "source": "knowledge",
        "content": f"{name}的实际证据",
        "score": 0.9,
        "metadata": {"source": f"{name}.pdf"},
    }


@pytest.mark.asyncio
async def test_section_writer_owns_prompt_validation_and_usage() -> None:
    captured = {}

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        captured.update(system=system, prompt=prompt, schema=schema, context=context)
        return validator("成本优势得到证据支持[来源1]。"), {
            "tokens": 11,
            "unknown": 0,
            "attempts": 1,
        }

    result = await SectionWriterAgent().run(
        SectionWritingRequest(
            section_id="section_1",
            next_revision=2,
            section_context="本章：成本\n本章必须回答：成本优势能否持续\n",
            evidence_text="[来源1] 成本材料",
            sources=(source(),),
            limitations=("仍需披露时间口径",),
        ),
        call_model=call_model,
    )

    assert result.draft.endswith("[来源1]。")
    assert result.cost["tokens"] == 11
    assert captured["schema"] is None
    assert captured["context"] == {
        "agent": "section_writer",
        "section_id": "section_1",
        "revision": 2,
    }
    assert "每个事实性判断" in captured["system"]
    assert "仍需披露时间口径" in captured["prompt"]


@pytest.mark.asyncio
async def test_section_writer_includes_old_draft_and_rejects_bad_citation() -> None:
    captured = {}

    async def call_model(system, prompt, schema=None, *, validator=None, context=None):
        captured["prompt"] = prompt
        return validator("错误引用[来源2]。"), {"tokens": 1, "unknown": 0}

    with pytest.raises(BusinessValidationError):
        await SectionWriterAgent().run(
            SectionWritingRequest(
                section_id="section_1",
                next_revision=3,
                section_context="本章：成本\n",
                evidence_text="[来源1] 新材料",
                sources=(source("新成本"),),
                limitations=(),
                current_draft="旧结论[来源1]。",
                previous_sources=(source("旧成本"),),
                review=SectionReview(verdict="revise", issues=["更新成本口径"]),
            ),
            call_model=call_model,
        )

    assert "待修改草稿" in captured["prompt"]
    assert "旧成本的实际证据" in captured["prompt"]
    assert "更新成本口径" in captured["prompt"]
