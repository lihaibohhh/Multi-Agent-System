from __future__ import annotations

from types import SimpleNamespace

import pytest

from multi_agent_research.core import streaming
from multi_agent_research.core.state import CURRENT_WORKFLOW_VERSION


@pytest.mark.asyncio
async def test_stream_events_include_the_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.state_reads = 0
            self.input_state: dict | None = None

        async def aget_state(self, config: dict, subgraphs: bool = False):
            self.state_reads += 1
            if self.state_reads == 1:
                return SimpleNamespace(values={})
            return SimpleNamespace(values={
                "workflow_version": CURRENT_WORKFLOW_VERSION,
                "final_report": "完成",
                "writer_status": "complete",
                "iteration_count": 1,
                "sections": [],
                "token_budget_used": 12,
            })

        async def astream(self, state: dict, *, config: dict, subgraphs: bool = False):
            assert subgraphs is True
            self.input_state = state
            yield {
                "plan_sections": {"sections": []}
            }
            yield {
                "assemble_report": {
                    "final_report": "完成",
                    "writer_status": "complete",
                }
            }

    fake_app = FakeApp()

    async def fake_get_app():
        return fake_app

    monkeypatch.setattr(streaming, "_get_app", fake_get_app)

    parent_context = {
        "source_run_id": "run-parent",
        "source_question": "父问题",
        "report_excerpt": "父报告",
    }
    events = [
        event
        async for event in streaming.astream_research(
            "测试研究问题",
            "run-test-1",
            parent_context=parent_context,
        )
    ]

    assert [name for name, _ in events] == [
        "start",
        "section_plan",
        "report_ready",
        "done",
    ]
    assert all(data["run_id"] == "run-test-1" for _, data in events)
    assert events[0][1]["parent_run_id"] == "run-parent"
    assert fake_app.input_state is not None
    assert fake_app.input_state["parent_context"] == parent_context
    assert events[-1][1]["report"] == "完成"


@pytest.mark.asyncio
async def test_resume_continues_with_none_input(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.graph_input = "not-called"
            self.state_reads = 0

        async def aget_state(self, config: dict, subgraphs: bool = False):
            self.state_reads += 1
            values = {
                "research_question": "原始问题",
                "workflow_version": CURRENT_WORKFLOW_VERSION,
                "final_report": "" if self.state_reads == 1 else "恢复完成",
                "writer_status": "not_started" if self.state_reads == 1 else "complete",
                "iteration_count": 2,
                "sections": [],
                "token_budget_used": 20,
            }
            return SimpleNamespace(values=values)

        async def astream(self, state: dict | None, *, config: dict, subgraphs: bool = False):
            assert subgraphs is True
            self.graph_input = state
            yield {
                "assemble_report": {
                    "final_report": "恢复完成",
                    "writer_status": "complete",
                }
            }

    fake_app = FakeApp()

    async def fake_get_app():
        return fake_app

    monkeypatch.setattr(streaming, "_get_app", fake_get_app)

    events = [event async for event in streaming.aresume_research("run-resume-1")]

    assert fake_app.graph_input is None
    assert events[0] == (
        "start",
        {
            "question": "原始问题",
            "run_id": "run-resume-1",
            "parent_run_id": None,
            "resumed": True,
        },
    )
    assert events[-1][0] == "done"
    assert events[-1][1]["report"] == "恢复完成"
