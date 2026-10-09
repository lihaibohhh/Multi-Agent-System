from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver

from multi_agent_research.core.budget import ExecutionPaused
from multi_agent_research.core.run_context import checkpoint_config
from multi_agent_research.core.state import initial_state
from multi_agent_research.sections import workflow
from multi_agent_research.sections.models import SectionRecord
from multi_agent_research.sections.subgraph import (
    SectionSubgraphState,
    build_section_subgraph,
)


def _state() -> dict:
    first = SectionRecord(
        section_id="section_1",
        title="成本",
        question="公司成本优势能否持续？",
    )
    second = SectionRecord(
        section_id="section_2",
        title="渠道",
        question="公司渠道优势能否持续？",
    )
    state = initial_state("分析公司的竞争优势")
    state.update(
        sections=[first.model_dump(mode="json"), second.model_dump(mode="json")],
        section_policy={"max_search_rounds": 2, "max_revisions": 1},
        section_step="research",
    )
    return state


def test_subgraph_state_excludes_parent_report_ownership() -> None:
    fields = SectionSubgraphState.__required_keys__
    assert {
        "research_question",
        "workflow_version",
        "sections",
        "active_section",
        "section_policy",
        "section_step",
        "model_calls",
        "usage_unknown_calls",
        "iteration_count",
        "token_budget_used",
    } <= fields
    assert {"report_review", "report_quality", "writer_status", "final_report"}.isdisjoint(
        fields
    )


@pytest.mark.asyncio
async def test_subgraph_completes_only_current_chapter_then_returns_to_parent(
    monkeypatch,
) -> None:
    calls = []

    async def research(state, config=None):
        calls.append("research")
        return {"section_step": "write"}

    async def write(state, config=None):
        calls.append("write")
        return {"section_step": "review"}

    async def review(state, config=None):
        calls.append("review")
        return {"section_step": "claims"}

    async def claims(state):
        calls.append("claims")
        sections = list(state["sections"])
        sections[0] = {**sections[0], "status": "complete", "draft": "已完成章节"}
        return {"sections": sections, "section_step": "advance"}

    def advance(state):
        calls.append("advance")
        return {"active_section": 1, "section_step": "research"}

    monkeypatch.setattr(workflow, "research_section", research)
    monkeypatch.setattr(workflow, "write_section", write)
    monkeypatch.setattr(workflow, "review_section", review)
    monkeypatch.setattr(workflow, "extract_claims", claims)
    monkeypatch.setattr(workflow, "advance_section", advance)

    result = await build_section_subgraph().compile().ainvoke(_state())

    assert calls == ["research", "write", "review", "claims", "advance"]
    assert result["active_section"] == 1
    assert result["section_step"] == "research"
    assert result["sections"][0]["status"] == "complete"
    assert result["sections"][1]["status"] == "pending"


@pytest.mark.asyncio
async def test_subgraph_pause_resumes_at_pending_local_node(monkeypatch) -> None:
    write_calls = 0

    async def research(state, config=None):
        return {"section_step": "write"}

    async def write(state, config=None):
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            raise ExecutionPaused("controlled pause")
        return {"section_step": "advance"}

    def advance(state):
        sections = list(state["sections"])
        sections[0] = {**sections[0], "status": "complete"}
        return {
            "sections": sections,
            "active_section": 1,
            "section_step": "research",
        }

    monkeypatch.setattr(workflow, "research_section", research)
    monkeypatch.setattr(workflow, "write_section", write)
    monkeypatch.setattr(workflow, "advance_section", advance)

    app = build_section_subgraph().compile(checkpointer=MemorySaver())
    config = checkpoint_config("section-subgraph-pause")
    with pytest.raises(ExecutionPaused, match="controlled pause"):
        await app.ainvoke(_state(), config=config)

    snapshot = await app.aget_state(config)
    assert snapshot.next == ("section_write",)
    assert snapshot.values["section_step"] == "write"

    result = await app.ainvoke(None, config=config)
    assert write_calls == 2
    assert result["sections"][0]["status"] == "complete"
    assert result["active_section"] == 1
