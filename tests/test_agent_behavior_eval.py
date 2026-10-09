from __future__ import annotations

import pytest

from multi_agent_research.agents import AgentContext, AgentRunner
from multi_agent_research.agents.contracts import (
    EvidenceResearchRequest,
    ReportReviewRequest,
    SectionPlanningRequest,
    SectionReviewRequest,
    SectionWritingRequest,
)
from multi_agent_research.agents.evidence_research_agent import EvidenceResearchAgent
from multi_agent_research.agents.planner_agent import PlannerAgent
from multi_agent_research.agents.report_reviewer_agent import ReportReviewerAgent
from multi_agent_research.agents.section_reviewer_agent import SectionReviewerAgent
from multi_agent_research.agents.section_writer_agent import SectionWriterAgent
from multi_agent_research.eval import (
    BehaviorCheck,
    BehaviorExpectation,
    build_suite_report,
    evaluate_behavior,
    observe_agent_run,
)
from multi_agent_research.retrieval import RetrievalRequest
from multi_agent_research.sections.models import (
    ReportReview,
    SectionPlan,
    SectionRecord,
    SectionReview,
)
from multi_agent_research.sections.rendering import citation_issues


def _source(name: str, *, iteration: int = 0) -> dict:
    return {
        "query": name,
        "source": "knowledge",
        "content": f"{name}的证据正文",
        "score": 0.9,
        "metadata": {"source": f"{name}.pdf", "chunk_id": name},
        "iteration": iteration,
    }


def _section() -> SectionRecord:
    return SectionRecord(
        section_id="section_1",
        title="成本",
        question="公司的成本优势能否持续？",
        status="complete",
        revision=1,
        draft="单位成本下降，但持续性仍有不确定性[来源1]。",
        sources=[_source("成本")],
        limitations=["缺少更长时间序列"],
    )


async def _run(agent, request, resolver, *, tools=None, section_id=None):
    events = []
    observation = await observe_agent_run(
        AgentRunner(resolver, tools=tools, event_sink=events.append),
        agent,
        request,
        context=AgentContext.create(
            f"eval-{agent.spec.name}",
            section_id=section_id,
        ),
        events=events,
    )
    return observation


@pytest.mark.asyncio
async def test_five_agent_behavior_baseline_is_deterministic_and_passes() -> None:
    async def planner_model(*args, validator=None, **kwargs):
        value = SectionPlan(sections=[{
            "title": "成本",
            "question": "公司的成本优势能否持续？",
            "parent_section_ids": ["parent_1"],
        }])
        return validator(value), {"tokens": 3, "unknown": 0, "attempts": 1}

    planner = await _run(
        PlannerAgent(),
        SectionPlanningRequest(
            research_question="分析公司的竞争优势",
            maximum_sections=2,
            parent_view="{}",
            available_parent_section_ids=frozenset({"parent_1"}),
        ),
        lambda _spec: planner_model,
    )

    research_model_calls = 0

    async def research_model(*args, validator=None, **kwargs):
        nonlocal research_model_calls
        research_model_calls += 1
        value = (
            SectionReview(
                verdict="revise",
                issues=["缺少反例"],
                search_queries=["成本优势反例"],
            )
            if research_model_calls == 1
            else SectionReview(verdict="pass", summary="正反证据均已覆盖")
        )
        return validator(value), {"tokens": 4, "unknown": 0, "attempts": 1}

    retrievals: list[RetrievalRequest] = []

    async def retrieve(request: RetrievalRequest):
        retrievals.append(request)
        name = request.gaps[0] if request.gaps else "成本优势"
        return [_source(name, iteration=request.iteration)]

    research = await _run(
        EvidenceResearchAgent(),
        EvidenceResearchRequest(
            section_id="section_1",
            question="公司的成本优势能否持续？",
            section_context="本章必须回答成本优势能否持续。\n",
            initial_results=(),
            initial_gaps=(),
            parent_question="",
            starting_round=0,
            max_search_rounds=2,
            revision=0,
        ),
        lambda _spec: research_model,
        tools={"retrieve_evidence": retrieve},
        section_id="section_1",
    )

    writer_calls = 0

    async def writer_model(*args, **kwargs):
        nonlocal writer_calls
        writer_calls += 1
        draft = (
            "错误编号[来源2]。"
            if writer_calls == 1
            else "成本结论受证据支持[来源1]；长期口径仍有限。"
        )
        return draft, {"tokens": 5, "unknown": 0, "attempts": 1}

    sources = (_source("成本优势"),)
    writer = await _run(
        SectionWriterAgent(),
        SectionWritingRequest(
            section_id="section_1",
            next_revision=1,
            section_context="本章必须回答成本优势能否持续。\n",
            evidence_text="[来源1] 成本优势证据",
            sources=sources,
            limitations=("长期口径仍有限",),
        ),
        lambda _spec: writer_model,
        section_id="section_1",
    )

    async def section_review_model(*args, validator=None, **kwargs):
        value = SectionReview(
            verdict="pass",
            issues=["仍缺少长期反例"],
            search_queries=["长期成本反例"],
        )
        return validator(value), {"tokens": 2, "unknown": 0, "attempts": 1}

    section_review = await _run(
        SectionReviewerAgent(),
        SectionReviewRequest(
            section_id="section_1",
            section_context="审查成本章节。\n",
            evidence_text="[来源1] 成本优势证据",
            draft="成本结论受证据支持[来源1]。",
        ),
        lambda _spec: section_review_model,
        section_id="section_1",
    )

    async def report_review_model(*args, validator=None, **kwargs):
        value = ReportReview(
            verdict="pass",
            issues=[{
                "kind": "scope",
                "section_ids": ["section_1"],
                "detail": "长期与短期口径需要区分",
            }],
        )
        return validator(value), {"tokens": 2, "unknown": 0, "attempts": 1}

    report_review = await _run(
        ReportReviewerAgent(),
        ReportReviewRequest(
            research_question="公司的竞争优势能否持续？",
            sections=(_section(),),
        ),
        lambda _spec: report_review_model,
    )

    cases = [
        (
            planner,
            BehaviorExpectation(
                scenario_id="planner-valid-parent-scope",
                agent_name="planner",
                max_turns=1,
                expected_model_calls=1,
                required_events=frozenset({"agent_completed"}),
                checks=(BehaviorCheck(
                    "plan_scope",
                    "规划结果只引用提供的父章节",
                    lambda item: item.result.output.sections[0].parent_section_ids
                    == ["parent_1"],
                ),),
            ),
        ),
        (
            research,
            BehaviorExpectation(
                scenario_id="research-bounded-gap-loop",
                agent_name="evidence_research",
                max_turns=4,
                allowed_tools=frozenset({"retrieve_evidence"}),
                expected_model_calls=2,
                required_events=frozenset({
                    "agent_retrying",
                    "agent_tool_started",
                    "agent_tool_completed",
                }),
                checks=(BehaviorCheck(
                    "gap_driven_retrieval",
                    "第二轮检索使用第一轮保留的证据缺口",
                    lambda _item: len(retrievals) == 2
                    and retrievals[1].gaps == ("成本优势反例",),
                ),),
            ),
        ),
        (
            writer,
            BehaviorExpectation(
                scenario_id="writer-validates-and-repairs-citations",
                agent_name="section_writer",
                max_turns=3,
                expected_model_calls=2,
                required_events=frozenset({"agent_retrying", "agent_completed"}),
                checks=(
                    BehaviorCheck(
                        "source_bound_draft",
                        "最终正文只引用当前来源表中的编号",
                        lambda item: not citation_issues(
                            item.result.output.draft,
                            list(sources),
                        ),
                    ),
                    BehaviorCheck(
                        "no_unsupported_certainty",
                        "固定样本不得把证据限制改写成无依据的确定性结论",
                        lambda item: "长期口径仍有限" in item.result.output.draft
                        and "必然" not in item.result.output.draft,
                    ),
                ),
            ),
        ),
        (
            section_review,
            BehaviorExpectation(
                scenario_id="section-review-preserves-objections",
                agent_name="section_reviewer",
                max_turns=1,
                expected_model_calls=1,
                checks=(BehaviorCheck(
                    "review_objection_preserved",
                    "存在审查问题时不得维持 pass",
                    lambda item: item.result.output.verdict == "revise"
                    and item.result.output.issues == ["仍缺少长期反例"],
                ),),
            ),
        ),
        (
            report_review,
            BehaviorExpectation(
                scenario_id="report-review-known-section-conflict",
                agent_name="report_reviewer",
                max_turns=1,
                expected_model_calls=1,
                checks=(BehaviorCheck(
                    "report_issue_preserved",
                    "全篇问题保留并将结论降级为 revise",
                    lambda item: item.result.output.verdict == "revise"
                    and item.result.output.issues[0].section_ids == ["section_1"],
                ),),
            ),
        ),
    ]

    report = build_suite_report(
        "five-agent-behavior-baseline",
        [evaluate_behavior(expectation, observation) for observation, expectation in cases],
    )

    assert report.passed, report.as_dict()
    assert report.as_dict()["total"] == 5
    assert report.as_dict()["failed"] == 0


def test_behavior_evaluator_reports_precise_failed_codes() -> None:
    evaluation = evaluate_behavior(
        BehaviorExpectation(
            scenario_id="missing-lifecycle",
            agent_name="planner",
            max_turns=1,
        ),
        observation=type("Observation", (), {
            "result": None,
            "events": (),
            "error": RuntimeError("missing"),
        })(),
    )

    assert not evaluation.passed
    assert {"identity", "lifecycle", "outcome"} <= set(evaluation.failed_codes)
