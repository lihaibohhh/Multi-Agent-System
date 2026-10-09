from __future__ import annotations

import pytest

from multi_agent_research.agents import AgentContext, AgentRunner, AgentTurnLimitError
from multi_agent_research.agents.contracts import SectionWritingRequest
from multi_agent_research.agents.section_writer_agent import SectionWriterAgent
from multi_agent_research.sections.models import SectionReview


def source(name: str = "成本") -> dict:
    return {
        "query": name,
        "source": "knowledge",
        "content": f"{name}的实际证据",
        "score": 0.9,
        "metadata": {"source": f"{name}.pdf"},
    }


def request(**updates) -> SectionWritingRequest:
    values = {
        "section_id": "section_1",
        "next_revision": 2,
        "section_context": "本章：成本\n本章必须回答：成本优势能否持续\n",
        "evidence_text": "[来源1] 成本材料",
        "sources": (source(),),
        "limitations": ("仍需披露时间口径",),
    }
    values.update(updates)
    return SectionWritingRequest(**values)


@pytest.mark.asyncio
async def test_section_writer_owns_prompt_validation_and_usage() -> None:
    captured = {}

    def resolve(spec):
        async def call_model(system, prompt, schema=None, *, validator=None, context=None):
            captured.update(system=system, prompt=prompt, schema=schema,
                            validator=validator, context=context)
            return "成本优势得到证据支持[来源1]。", {
                "tokens": 11,
                "unknown": 0,
                "attempts": 1,
            }
        return call_model

    result = await AgentRunner(resolve).run(
        SectionWriterAgent(),
        request(),
        context=AgentContext.create("run-writer", section_id="section_1"),
    )

    assert result.output is not None
    assert result.output.draft.endswith("[来源1]。")
    assert result.output.attempts == 1
    assert result.usage["tokens"] == 11
    assert captured["schema"] is None
    assert captured["validator"] is None
    assert captured["context"] == {
        "agent": "section_writer",
        "section_id": "section_1",
        "revision": 2,
        "writer_attempt": 1,
        "single_attempt": True,
    }
    assert "每个事实性判断" in captured["system"]
    assert "仍需披露时间口径" in captured["prompt"]


@pytest.mark.asyncio
async def test_section_writer_repairs_invalid_citation_in_its_own_second_turn() -> None:
    prompts = []
    drafts = iter(["错误引用[来源2]。", "修正后的结论[来源1]。"])

    def resolve(_spec):
        async def call_model(system, prompt, schema=None, *, validator=None, context=None):
            prompts.append(prompt)
            return next(drafts), {"tokens": 2, "unknown": 0, "attempts": 1}
        return call_model

    result = await AgentRunner(resolve).run(
        SectionWriterAgent(),
        request(
            next_revision=3,
            current_draft="旧结论[来源1]。",
            previous_sources=(source("旧成本"),),
            review=SectionReview(verdict="revise", issues=["更新成本口径"]),
        ),
        context=AgentContext.create("run-repair", section_id="section_1"),
    )

    assert result.turns == 2
    assert result.output is not None
    assert result.output.draft == "修正后的结论[来源1]。"
    assert result.output.attempts == 2
    assert "待修改草稿" in prompts[0]
    assert "旧成本的实际证据" in prompts[0]
    assert "更新成本口径" in prompts[0]
    assert "上一次候选正文未通过" in prompts[1]
    assert "引用不存在的来源编号" in prompts[1]


@pytest.mark.asyncio
async def test_section_writer_completed_checkpoint_resumes_without_model_call() -> None:
    calls = 0

    def resolve(_spec):
        async def call_model(*args, **kwargs):
            nonlocal calls
            calls += 1
            return "有效结论[来源1]。", {"tokens": 3, "unknown": 0, "attempts": 1}
        return call_model

    runner = AgentRunner(resolve)
    writing_request = request()
    first = await runner.run(
        SectionWriterAgent(),
        writing_request,
        context=AgentContext.create("run-resume", section_id="section_1"),
    )
    resumed = await runner.run(
        SectionWriterAgent(),
        writing_request,
        context=AgentContext.create(
            "run-resume",
            parent_agent_run_id="prior-writer",
            section_id="section_1",
            local_state=first.local_state,
        ),
    )

    assert resumed.output == first.output
    assert resumed.usage == {"tokens": 0, "unknown": 0, "attempts": 0}
    assert calls == 1


@pytest.mark.asyncio
async def test_section_writer_stops_after_three_invalid_drafts() -> None:
    calls = 0

    def resolve(_spec):
        async def call_model(*args, **kwargs):
            nonlocal calls
            calls += 1
            return "始终错误[来源9]。", {"tokens": 1, "unknown": 0, "attempts": 1}
        return call_model

    lifecycle = []
    with pytest.raises(AgentTurnLimitError, match="3 个 turn"):
        await AgentRunner(resolve, event_sink=lifecycle.append).run(
            SectionWriterAgent(),
            request(),
            context=AgentContext.create("run-limit", section_id="section_1"),
        )

    assert calls == 3
    assert lifecycle[-1].event_type == "agent_failed"
    assert lifecycle[-1].details["checkpoint"]["local_state"]["attempts"] == 3
