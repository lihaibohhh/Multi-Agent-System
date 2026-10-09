from __future__ import annotations

import pytest

from multi_agent_research.agents import AgentContext, AgentRunner
from multi_agent_research.agents.contracts import EvidenceResearchRequest
from multi_agent_research.agents.evidence_research_agent import EvidenceResearchAgent
from multi_agent_research.retrieval import RetrievalRequest
from multi_agent_research.sections.models import SectionReview


def _result(query: str, score: float = 0.8) -> dict:
    return {
        "query": query,
        "source": "knowledge",
        "content": f"evidence for {query}",
        "score": score,
        "metadata": {"chunk_id": query},
        "iteration": 0,
    }


@pytest.mark.asyncio
async def test_evidence_research_agent_owns_bounded_query_retrieval_analysis_loop() -> None:
    retrievals: list[RetrievalRequest] = []
    model_calls = 0

    async def retrieve(request: RetrievalRequest):
        retrievals.append(request)
        return [_result(request.gaps[0] if request.gaps else request.question)]

    def resolve(spec):
        async def call_model(system, prompt, schema=None, *, validator=None, context=None):
            nonlocal model_calls
            model_calls += 1
            review = (
                SectionReview(verdict="revise", issues=["缺少反例"], search_queries=["反例 查询"])
                if model_calls == 1
                else SectionReview(verdict="pass", summary="正反证据已覆盖")
            )
            return validator(review), {"tokens": 5, "unknown": 0, "attempts": 1}
        return call_model

    events = []
    runner = AgentRunner(
        resolve,
        tools={"retrieve_evidence": retrieve},
        event_sink=events.append,
    )
    result = await runner.run(
        EvidenceResearchAgent(),
        EvidenceResearchRequest(
            section_id="section_1",
            question="成本优势能否持续？",
            section_context="本章：成本优势\n",
            initial_results=(),
            initial_gaps=(),
            parent_question="",
            starting_round=0,
            max_search_rounds=2,
            revision=0,
        ),
        context=AgentContext.create("run-1", section_id="section_1"),
    )

    assert result.turns == 2
    assert result.usage == {"tokens": 10, "unknown": 0, "attempts": 2}
    assert result.output is not None
    assert result.output.review.verdict == "pass"
    assert result.output.search_rounds == 2
    assert len(result.output.results) == 2
    assert retrievals[1].gaps == ("反例 查询",)
    assert [event.event_type for event in events].count("agent_retrying") == 1
    assert [event.event_type for event in events].count("agent_tool_started") == 2
    assert [event.event_type for event in events].count("agent_tool_completed") == 2
    tool_events = [event for event in events if event.event_type.startswith("agent_tool_")]
    assert all(event.details["tool_name"] == "retrieve_evidence" for event in tool_events)
    assert all("query" not in event.details and "result" not in event.details for event in tool_events)


@pytest.mark.asyncio
async def test_evidence_research_agent_resumes_completed_checkpoint_without_calls() -> None:
    agent = EvidenceResearchAgent()
    request = EvidenceResearchRequest(
        section_id="section_2",
        question="市场规模如何变化？",
        section_context="本章：市场\n",
        initial_results=(),
        initial_gaps=(),
        parent_question="",
        starting_round=0,
        max_search_rounds=1,
        revision=0,
    )

    calls = {"model": 0, "retrieval": 0}
    async def retrieve(_request):
        calls["retrieval"] += 1
        return [_result("market")]
    def resolve(_spec):
        async def call_model(*args, validator=None, **kwargs):
            calls["model"] += 1
            value = SectionReview(verdict="pass", summary="充分")
            return validator(value), {"tokens": 3, "unknown": 0, "attempts": 1}
        return call_model

    runner = AgentRunner(resolve, tools={"retrieve_evidence": retrieve})
    first = await runner.run(agent, request, context=AgentContext.create("run-2", section_id="section_2"))
    resumed = await runner.run(
        agent,
        request,
        context=AgentContext.create(
            "run-2",
            parent_agent_run_id="prior",
            section_id="section_2",
            local_state=first.local_state,
        ),
    )

    assert resumed.output == first.output
    assert resumed.usage == {"tokens": 0, "unknown": 0, "attempts": 0}
    assert calls == {"model": 1, "retrieval": 1}


@pytest.mark.asyncio
async def test_evidence_research_agent_requires_fresh_supplement_results() -> None:
    async def retrieve(_request):
        return []
    def resolve(_spec):
        async def call_model(*args, validator=None, **kwargs):
            return validator(SectionReview(verdict="pass")), {"tokens": 1, "unknown": 0}
        return call_model

    runner = AgentRunner(resolve, tools={"retrieve_evidence": retrieve})
    result = await runner.run(
        EvidenceResearchAgent(),
        EvidenceResearchRequest(
            section_id="section_3",
            question="补充最新证据",
            section_context="本章：更新\n",
            initial_results=(_result("old"),),
            initial_gaps=("latest",),
            parent_question="",
            starting_round=0,
            max_search_rounds=2,
            revision=1,
            stop_after_one_round=True,
            require_fresh_results=True,
        ),
        context=AgentContext.create("run-3", section_id="section_3"),
    )

    assert result.turns == 1
    assert result.output is not None
    assert result.output.review.verdict == "revise"
    assert "没有返回可用新结果" in result.output.review.issues[0]
