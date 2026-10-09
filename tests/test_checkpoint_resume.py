from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from multi_agent_research.core import streaming
from multi_agent_research.core.run_context import checkpoint_config
from multi_agent_research.core.state import CURRENT_WORKFLOW_VERSION


class ResumeState(TypedDict, total=False):
    question: str
    workflow_version: int
    sections: list
    final_report: str
    writer_status: str
    iteration_count: int
    token_budget_used: int


@pytest.mark.asyncio
async def test_resume_retries_pending_node_from_same_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def flaky_assembly(state: ResumeState) -> ResumeState:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("simulated process failure")
        return {"final_report": "recovered", "writer_status": "complete"}

    graph = StateGraph(ResumeState)
    graph.add_node("assemble_report", flaky_assembly)
    graph.add_edge(START, "assemble_report")
    graph.add_edge("assemble_report", END)
    app = graph.compile(checkpointer=MemorySaver())
    config = checkpoint_config("run-real-resume")
    initial: ResumeState = {
        "question": "resume checkpoint",
        "workflow_version": CURRENT_WORKFLOW_VERSION,
        "sections": [],
        "final_report": "",
        "writer_status": "not_started",
        "iteration_count": 0,
        "token_budget_used": 0,
    }

    with pytest.raises(RuntimeError, match="simulated process failure"):
        async for _ in app.astream(initial, config=config):
            pass

    async def fake_get_app():
        return app

    monkeypatch.setattr(streaming, "_get_app", fake_get_app)
    events = [event async for event in streaming.aresume_research("run-real-resume")]

    assert attempts == 2
    assert events[0][1]["resumed"] is True
    assert events[-1][0] == "done"
    assert events[-1][1]["report"] == "recovered"


@pytest.mark.asyncio
async def test_resume_reconciles_completed_checkpoint_without_rerunning_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def assemble(state: ResumeState) -> ResumeState:
        nonlocal calls
        calls += 1
        return {"final_report": "already complete", "writer_status": "complete"}

    graph = StateGraph(ResumeState)
    graph.add_node("assemble_report", assemble)
    graph.add_edge(START, "assemble_report")
    graph.add_edge("assemble_report", END)
    app = graph.compile(checkpointer=MemorySaver())
    config = checkpoint_config("run-completed-checkpoint")
    initial: ResumeState = {
        "question": "completed checkpoint",
        "workflow_version": CURRENT_WORKFLOW_VERSION,
        "sections": [],
        "final_report": "",
        "writer_status": "not_started",
        "iteration_count": 0,
        "token_budget_used": 0,
    }
    async for _ in app.astream(initial, config=config):
        pass

    async def fake_get_app():
        return app

    monkeypatch.setattr(streaming, "_get_app", fake_get_app)
    events = [
        event
        async for event in streaming.aresume_research("run-completed-checkpoint")
    ]

    assert calls == 1
    assert [event_type for event_type, _ in events] == [
        "start",
        "section_snapshot",
        "done",
    ]
    assert events[-1][1]["report"] == "already complete"


@pytest.mark.asyncio
async def test_resume_rejects_removed_workflow_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def finish(state: ResumeState) -> ResumeState:
        return {"final_report": "old", "writer_status": "complete"}

    graph = StateGraph(ResumeState)
    graph.add_node("old_flow", finish)
    graph.add_edge(START, "old_flow")
    graph.add_edge("old_flow", END)
    app = graph.compile(checkpointer=MemorySaver())
    config = checkpoint_config("removed-workflow")
    initial: ResumeState = {
        "question": "old checkpoint",
        "workflow_version": CURRENT_WORKFLOW_VERSION - 1,
        "sections": [],
        "writer_status": "not_started",
    }
    async for _ in app.astream(initial, config=config):
        pass

    async def fake_get_app():
        return app

    monkeypatch.setattr(streaming, "_get_app", fake_get_app)
    with pytest.raises(RuntimeError, match="已移除的旧工作流"):
        async for _ in streaming.aresume_research("removed-workflow"):
            pass
