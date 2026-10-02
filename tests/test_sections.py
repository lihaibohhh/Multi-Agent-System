from collections import Counter

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from multi_agent_research.core import streaming
from multi_agent_research.core.graph import build_graph
from multi_agent_research.core.state import initial_state
from multi_agent_research.sections import workflow
from multi_agent_research.sections.models import SectionPlan, SectionRecord, SectionReview
from multi_agent_research.sections.rendering import assemble_report, merge_results


def evidence(name, score=0.9):
    return {
        "query": name, "source": "knowledge", "content": f"{name} 的实际证据摘录",
        "score": score, "iteration": 0,
        "metadata": {"source": f"{name}.pdf", "page": 2, "chunk_id": f"{name}::2"},
    }


class FakeModels:
    def __init__(self, *, fail_second=False, revise=False, no_results=False):
        self.calls = Counter()
        self.fail_second = fail_second
        self.revise = revise
        self.no_results = no_results
        self.questions = []

    async def model(self, system, prompt, schema=None):
        cost = {"tokens": 10, "unknown": 0}
        if schema is SectionPlan:
            self.calls["plan"] += 1
            return SectionPlan(sections=[
                {"title": "成本", "question": "公司X的成本优势如何"},
                {"title": "渠道", "question": "公司X的渠道优势如何"},
                {"title": "结论", "question": "公司X的整体优势是否持续", "kind": "synthesis"},
            ]), cost
        chapter = next(name for name in ("成本", "渠道", "结论") if f"本章：{name}\n" in prompt)
        if schema is SectionReview:
            stage = "review" if "核查章节草稿" in system else "analyze"
            self.calls[f"{stage}:{chapter}"] += 1
            revision = self.revise and chapter == "成本" and stage == "review"
            return SectionReview(
                verdict="revise" if revision else "pass",
                issues=["缺少反面证据"] if revision else [],
                search_queries=["公司X 成本反例"] if revision else [],
                summary=f"{chapter}已分析，仍应关注口径变化",
            ), cost
        self.calls[f"write:{chapter}"] += 1
        if chapter == "渠道" and self.fail_second:
            self.fail_second = False
            raise RuntimeError("simulated writer outage")
        return f"{chapter}的分析结论[来源1]。", cost

    async def search(self, state):
        self.questions.append(state["research_question"])
        self.calls["search"] += 1
        return {"search_results": [] if self.no_results else [evidence(state["research_question"])]}


def install(monkeypatch, fake):
    monkeypatch.setattr(workflow, "call_model", fake.model)
    monkeypatch.setattr(workflow, "search_agent_node", fake.search)
    monkeypatch.setattr(workflow.settings.agent, "section_max_count", 4)
    monkeypatch.setattr(workflow.settings.agent, "section_max_search_rounds", 2)
    monkeypatch.setattr(workflow.settings.agent, "section_max_revisions", 1)


@pytest.mark.asyncio
async def test_serial_chapters_scoped_search_and_dependent_synthesis(monkeypatch):
    fake = FakeModels()
    install(monkeypatch, fake)
    app = build_graph()
    state = initial_state("分析公司X竞争优势")
    result = await app.ainvoke(state, streaming.research_config("chapters", state))
    assert fake.questions == ["公司X的成本优势如何", "公司X的渠道优势如何"]
    assert result["writer_status"] == "complete"
    assert result["report_quality"] == "reviewed"
    assert [s["status"] for s in result["sections"]] == ["complete"] * 3
    assert all(s["revision"] == 1 for s in result["sections"])
    assert "成本的分析结论[来源1]" in result["final_report"]
    assert "渠道的分析结论[来源2]" in result["final_report"]
    assert result["model_calls"] == 10  # planner + analyze/write/review per chapter
    assert result["token_budget_used"] == 100


@pytest.mark.asyncio
async def test_resume_from_sqlite_preserves_completed_chapter_and_draft(monkeypatch, tmp_path):
    fake = FakeModels(fail_second=True)
    install(monkeypatch, fake)
    state = initial_state("分析公司X竞争优势")
    config = streaming.research_config("restart-chapters", state)
    database = str(tmp_path / "chapters.sqlite")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        app = build_graph(saver)
        with pytest.raises(RuntimeError, match="writer outage"):
            await app.ainvoke(state, config)
        snapshot = await app.aget_state(config)
        assert snapshot.values["sections"][0]["status"] == "complete"
        assert snapshot.values["sections"][0]["draft"]
        assert snapshot.next == ("section_write",)

    # A new saver and compiled graph model a process restart, not just another call.
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        app = build_graph(saver)

        async def get_app():
            return app

        monkeypatch.setattr(streaming, "_get_app", get_app)
        events = [event async for event in streaming.aresume_research("restart-chapters")]
    assert fake.calls["plan"] == 1
    assert fake.calls["write:成本"] == 1
    assert fake.calls["write:渠道"] == 2
    assert fake.calls["search"] == 2
    assert events[1][0] == "section_snapshot"
    assert events[1][1]["sections"][0]["status"] == "complete"
    assert events[-1][0] == "done"
    assert len(events[-1][1]["sections"]) == 3


@pytest.mark.asyncio
async def test_review_supplements_only_its_chapter_with_bounded_revisions(monkeypatch):
    fake = FakeModels(revise=True)
    install(monkeypatch, fake)
    state = initial_state("分析公司X竞争优势")
    result = await build_graph().ainvoke(state, streaming.research_config("revise-chapter", state))
    first, second, _ = result["sections"]
    assert first["search_rounds"] == 2
    assert first["revision"] == 2
    assert len(first["previous_drafts"]) == 1
    assert first["previous_drafts"][0]["sources"]
    assert first["previous_drafts"][0]["revision"] == 1
    assert first["status"] == "limited"
    assert second["status"] == "complete"
    assert second["search_rounds"] == 1
    assert "未解决" in result["final_report"]
    assert result["report_quality"] == "limited"


@pytest.mark.asyncio
async def test_no_results_produces_explicit_limitations_not_fake_findings(monkeypatch):
    fake = FakeModels(no_results=True)
    install(monkeypatch, fake)
    state = initial_state("分析公司X竞争优势")
    result = await build_graph().ainvoke(state, streaming.research_config("no-evidence", state))
    assert all(s["status"] == "limited" for s in result["sections"])
    assert fake.calls["write:成本"] == 0
    assert all(s["search_rounds"] == 2 for s in result["sections"])
    assert all(s["analyst"]["verdict"] == "revise" for s in result["sections"])
    assert "无法对本章问题作出可靠结论" in result["final_report"]


@pytest.mark.asyncio
async def test_invalid_citation_never_completes_run(monkeypatch):
    fake = FakeModels()
    install(monkeypatch, fake)

    async def wrong_citations(system, prompt, schema=None):
        if schema is None:
            return "无效结论[来源99]", {"tokens": 10, "unknown": 0}
        return await fake.model(system, prompt, schema)

    monkeypatch.setattr(workflow, "call_model", wrong_citations)
    state = initial_state("分析公司X竞争优势")
    app = build_graph()
    config = streaming.research_config("bad-citation", state)
    with pytest.raises(ValueError, match="citation validation failed"):
        await app.ainvoke(state, config)
    snapshot = await app.aget_state(config)
    assert snapshot.values["writer_status"] == "not_started"
    assert not snapshot.values["final_report"]
    assert snapshot.values["sections"][0]["revision"] == 2


def test_assembly_preserves_paragraphs_and_renumbers_shared_evidence():
    a, b = evidence("A"), evidence("B")
    first = SectionRecord(section_id="1", title="甲", question="研究第一个问题",
                          draft="甲[来源1]", sources=[a], status="complete")
    second = SectionRecord(section_id="2", title="乙", question="研究第二个问题",
                           draft="乙[来源1]；再看甲[来源2]", sources=[b, a], status="complete")
    report = assemble_report("研究问题", [first, second])
    assert "乙[来源2]；再看甲[来源1]" in report
    assert report.count("chunk_id: A::2") == 1
    assert "p.2" in report
    assert len(merge_results([a], [a, b])) == 2
    assert len(merge_results([a], [{**a, "content": "更新的正文"}])) == 2


def test_plan_rejects_synthesis_before_required_research():
    with pytest.raises(ValueError):
        SectionPlan(sections=[{"title": "结论", "question": "研究综合结论", "kind": "synthesis"}])


@pytest.mark.asyncio
async def test_new_stream_publishes_sections_and_final_snapshot(monkeypatch):
    fake = FakeModels()
    install(monkeypatch, fake)
    app = build_graph()

    async def get_app():
        return app

    monkeypatch.setattr(streaming, "_get_app", get_app)
    events = [event async for event in streaming.astream_research("研究公司X竞争优势", "stream-chapters")]
    assert events[0][0] == "start"
    assert events[1][0] == "section_plan"
    assert any(kind == "section_progress" and any(s["draft"] for s in data["sections"])
               for kind, data in events)
    assert [kind for kind, _ in events][-2:] == ["report_ready", "done"]
    assert events[-1][1]["report_quality"] == "reviewed"
    assert events[-1][1]["total_results"] == 4  # two local + two synthesis observations


@pytest.mark.asyncio
async def test_existing_checkpoint_uses_legacy_nodes(monkeypatch):
    from multi_agent_research.core import graph as graph_module
    from multi_agent_research.core.run_context import checkpoint_config

    async def old_supervisor(state):
        return {"next_agent": "writer_agent"}

    async def old_writer(state):
        return {"final_report": "旧流程报告", "writer_status": "complete"}

    monkeypatch.setattr(graph_module, "supervisor_node", old_supervisor)
    monkeypatch.setattr(graph_module, "writer_agent_node", old_writer)
    legacy = initial_state("旧任务研究问题")
    legacy.pop("workflow_version")
    result = await build_graph().ainvoke(legacy, checkpoint_config("legacy-run"))
    assert result["final_report"] == "旧流程报告"
