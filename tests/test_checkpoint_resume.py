from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from multi_agent_research.core import streaming
from multi_agent_research.core.run_context import checkpoint_config


class ResumeState(TypedDict, total=False):
    question: str
    final_report: str
    writer_status: str
    iteration_count: int
    search_results: list
    token_budget_used: int


@pytest.mark.asyncio
async def test_resume_retries_pending_node_from_same_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def flaky_writer(state: ResumeState) -> ResumeState:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("simulated process failure")
        return {"final_report": "recovered", "writer_status": "complete"}

    graph = StateGraph(ResumeState)
    graph.add_node("writer_agent", flaky_writer)
    graph.add_edge(START, "writer_agent")
    graph.add_edge("writer_agent", END)
    app = graph.compile(checkpointer=MemorySaver())
    config = checkpoint_config("run-real-resume")
    initial: ResumeState = {
        "question": "resume checkpoint",
        "final_report": "",
        "writer_status": "not_started",
        "iteration_count": 0,
        "search_results": [],
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

    async def writer(state: ResumeState) -> ResumeState:
        nonlocal calls
        calls += 1
        return {"final_report": "already complete", "writer_status": "complete"}

    graph = StateGraph(ResumeState)
    graph.add_node("writer_agent", writer)
    graph.add_edge(START, "writer_agent")
    graph.add_edge("writer_agent", END)
    app = graph.compile(checkpointer=MemorySaver())
    config = checkpoint_config("run-completed-checkpoint")
    initial: ResumeState = {
        "question": "completed checkpoint",
        "final_report": "",
        "writer_status": "not_started",
        "iteration_count": 0,
        "search_results": [],
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
    assert [event_type for event_type, _ in events] == ["start", "done"]
    assert events[-1][1]["report"] == "already complete"
