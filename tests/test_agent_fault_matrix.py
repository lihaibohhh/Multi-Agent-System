from __future__ import annotations

from dataclasses import dataclass

import pytest

from multi_agent_research.agents import (
    AgentContext,
    AgentRunner,
    AgentTimeoutError,
    AgentToolExecutionError,
    AgentTurnResult,
)
from multi_agent_research.agents.contracts import (
    EvidenceResearchRequest,
    SectionPlanningRequest,
)
from multi_agent_research.agents.evidence_research_agent import EvidenceResearchAgent
from multi_agent_research.agents.planner_agent import PlannerAgent
from multi_agent_research.agents.spec import AgentSpec
from multi_agent_research.core.budget import BudgetExceeded, CallTimeout, ExecutionPaused
from multi_agent_research.eval import (
    BehaviorExpectation,
    BehaviorObservation,
    FaultInjector,
    FaultRule,
    InjectedProcessCrash,
    evaluate_behavior,
    observe_agent_run,
)
from multi_agent_research.runs.repository import StaleExecutionError
from multi_agent_research.sections.models import SectionReview


@dataclass(frozen=True)
class _Request:
    value: str


@dataclass(frozen=True)
class _Output:
    value: str


class _ModelAgent:
    spec = AgentSpec(
        name="fault_model",
        description="fault injection model boundary",
        model_ref="test",
        input_type=_Request,
        output_type=_Output,
        timeout_seconds=0.01,
    )

    async def run_turn(self, request, *, context, call_model):
        value, _ = await call_model("system", request.value)
        return AgentTurnResult(status="completed", output=_Output(value))


def _research_request() -> EvidenceResearchRequest:
    return EvidenceResearchRequest(
        section_id="section_1",
        question="成本优势能否持续？",
        section_context="本章研究成本优势。\n",
        initial_results=(),
        initial_gaps=(),
        parent_question="",
        starting_round=0,
        max_search_rounds=2,
        revision=0,
    )


def _evidence(request) -> list[dict]:
    name = request.gaps[0] if request.gaps else f"round-{request.iteration}"
    return [{
        "query": name,
        "source": "knowledge",
        "content": f"{name} evidence",
        "score": 0.8,
        "metadata": {"chunk_id": name},
        "iteration": request.iteration,
    }]


def _business_output(result) -> dict:
    """Ignore observation timestamps that are expected to change after a resume."""
    output = result.output
    return {
        "review": output.review.model_dump(mode="json"),
        "search_rounds": output.search_rounds,
        "fresh_result_ids": output.fresh_result_ids,
        "results": [
            {
                **item,
                "metadata": {
                    key: value
                    for key, value in item.get("metadata", {}).items()
                    if key != "retrieved_at"
                },
            }
            for item in output.results
        ],
    }


@pytest.mark.asyncio
async def test_fault_matrix_model_timeout_is_bounded_and_observable() -> None:
    async def model(*args, **kwargs):
        import asyncio

        await asyncio.sleep(1)
        return "late", {"tokens": 1, "unknown": 0, "attempts": 1}

    events = []
    observation = await observe_agent_run(
        AgentRunner(lambda _spec: model, event_sink=events.append),
        _ModelAgent(),
        _Request("slow"),
        context=AgentContext.create("fault-model-timeout"),
        events=events,
    )
    evaluation = evaluate_behavior(
        BehaviorExpectation(
            scenario_id="model-timeout",
            agent_name="fault_model",
            max_turns=1,
            expected_outcome="failed",
            expected_model_calls=1,
            required_events=frozenset({"agent_failed"}),
        ),
        observation,
    )

    assert isinstance(observation.error, AgentTimeoutError)
    assert evaluation.passed, evaluation.as_dict()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_id", "error"),
    [
        ("retrieval-timeout", CallTimeout("retrieval timed out")),
        ("pause-mid-tool", ExecutionPaused("pause requested")),
        ("budget-exhausted", BudgetExceeded("retrieval budget exhausted")),
    ],
)
async def test_fault_matrix_run_controls_pause_without_fabricating_results(
    case_id,
    error,
) -> None:
    async def retrieve(_request):
        raise error

    async def model(*args, **kwargs):
        raise AssertionError("模型不应在检索控制异常后被调用")

    events = []
    observation = await observe_agent_run(
        AgentRunner(
            lambda _spec: model,
            tools={"retrieve_evidence": retrieve},
            event_sink=events.append,
        ),
        EvidenceResearchAgent(),
        _research_request(),
        context=AgentContext.create("fault-tool-control", section_id="section_1"),
        events=events,
    )
    evaluation = evaluate_behavior(
        BehaviorExpectation(
            scenario_id=case_id,
            agent_name="evidence_research",
            max_turns=4,
            expected_outcome="paused",
            allowed_tools=frozenset({"retrieve_evidence"}),
            expected_model_calls=0,
            required_events=frozenset({"agent_tool_failed", "agent_paused"}),
        ),
        observation,
    )

    assert observation.error is error
    assert evaluation.passed, evaluation.as_dict()


@pytest.mark.asyncio
async def test_fault_matrix_dependency_error_fails_safely() -> None:
    async def retrieve(_request):
        raise RuntimeError("PRIVATE_PROVIDER_MESSAGE")

    async def model(*args, **kwargs):
        raise AssertionError("模型不应在依赖失败后被调用")

    events = []
    observation = await observe_agent_run(
        AgentRunner(
            lambda _spec: model,
            tools={"retrieve_evidence": retrieve},
            event_sink=events.append,
        ),
        EvidenceResearchAgent(),
        _research_request(),
        context=AgentContext.create("fault-dependency", section_id="section_1"),
        events=events,
    )
    evaluation = evaluate_behavior(
        BehaviorExpectation(
            scenario_id="dependency-error",
            agent_name="evidence_research",
            max_turns=4,
            expected_outcome="failed",
            allowed_tools=frozenset({"retrieve_evidence"}),
            expected_model_calls=0,
            required_events=frozenset({"agent_tool_failed", "agent_failed"}),
        ),
        observation,
    )

    assert isinstance(observation.error, AgentToolExecutionError)
    assert "PRIVATE_PROVIDER_MESSAGE" not in str(observation.error)
    assert "PRIVATE_PROVIDER_MESSAGE" not in repr([event.details for event in events])
    assert evaluation.passed, evaluation.as_dict()


@pytest.mark.asyncio
async def test_fault_matrix_process_crash_after_checkpoint_resumes_equivalently() -> None:
    request = _research_request()

    def resolver():
        calls = 0

        async def model(*args, validator=None, **kwargs):
            nonlocal calls
            calls += 1
            review = (
                SectionReview(
                    verdict="revise",
                    issues=["缺少反例"],
                    search_queries=["成本反例"],
                )
                if calls == 1
                else SectionReview(verdict="pass", summary="证据充分")
            )
            return validator(review), {"tokens": 2, "unknown": 0, "attempts": 1}

        return model

    async def retrieve(value):
        return _evidence(value)

    baseline = await AgentRunner(
        lambda _spec: resolver(),
        tools={"retrieve_evidence": retrieve},
    ).run(
        EvidenceResearchAgent(),
        request,
        context=AgentContext.create("fault-crash-baseline", section_id="section_1"),
    )

    crash_events = []
    injector = FaultInjector(FaultRule(
        boundary="agent_retrying",
        phase="after",
        exception_factory=lambda: InjectedProcessCrash("simulated crash"),
    ))
    crashing_runner = AgentRunner(
        lambda _spec: resolver(),
        tools={"retrieve_evidence": retrieve},
        event_sink=injector.wrap_event_sink(crash_events.append),
    )
    with pytest.raises(InjectedProcessCrash):
        await crashing_runner.run(
            EvidenceResearchAgent(),
            request,
            context=AgentContext.create("fault-crash-resume", section_id="section_1"),
        )

    retry_event = next(
        event for event in crash_events if event.event_type == "agent_retrying"
    )
    interrupted = BehaviorObservation(
        result=None,
        events=tuple(crash_events),
        error=InjectedProcessCrash("simulated crash"),
    )
    interrupted_evaluation = evaluate_behavior(
        BehaviorExpectation(
            scenario_id="crash-after-agent-checkpoint",
            agent_name="evidence_research",
            max_turns=4,
            expected_outcome="interrupted",
            allowed_tools=frozenset({"retrieve_evidence"}),
            expected_model_calls=1,
            required_events=frozenset({"agent_retrying"}),
        ),
        interrupted,
    )
    assert interrupted_evaluation.passed, interrupted_evaluation.as_dict()

    resume_calls = 0

    async def resume_model(*args, validator=None, **kwargs):
        nonlocal resume_calls
        resume_calls += 1
        return validator(SectionReview(verdict="pass", summary="证据充分")), {
            "tokens": 2,
            "unknown": 0,
            "attempts": 1,
        }

    resumed = await AgentRunner(
        lambda _spec: resume_model,
        tools={"retrieve_evidence": retrieve},
    ).run(
        EvidenceResearchAgent(),
        request,
        context=AgentContext.create(
            "fault-crash-resume",
            parent_agent_run_id=retry_event.agent_run_id,
            section_id="section_1",
            local_state=retry_event.details["checkpoint"]["local_state"],
        ),
    )

    assert _business_output(resumed) == _business_output(baseline)
    assert resume_calls == 1
    assert resumed.output.search_rounds == 2


@pytest.mark.asyncio
async def test_fault_matrix_stale_execution_stops_before_model_call() -> None:
    async def model(*args, **kwargs):
        raise AssertionError("过期执行不应调用模型")

    events = []
    injector = FaultInjector(FaultRule(
        boundary="agent_started",
        phase="after",
        exception_factory=lambda: StaleExecutionError("stale execution"),
    ))
    observation = await observe_agent_run(
        AgentRunner(
            lambda _spec: model,
            event_sink=injector.wrap_event_sink(events.append),
        ),
        PlannerAgent(),
        SectionPlanningRequest(
            research_question="分析竞争优势",
            maximum_sections=1,
            parent_view="{}",
            available_parent_section_ids=frozenset(),
        ),
        context=AgentContext.create("fault-stale"),
        events=events,
    )
    evaluation = evaluate_behavior(
        BehaviorExpectation(
            scenario_id="stale-execution-fence",
            agent_name="planner",
            max_turns=1,
            expected_outcome="interrupted",
            expected_model_calls=0,
        ),
        observation,
    )

    assert isinstance(observation.error, StaleExecutionError)
    assert injector.triggered_rules == 1
    assert evaluation.passed, evaluation.as_dict()


def test_fault_rules_reject_invalid_or_synchronous_boundaries() -> None:
    with pytest.raises(ValueError, match="延迟或异常"):
        FaultRule(boundary="model")
    with pytest.raises(TypeError, match="异步"):
        FaultInjector(FaultRule(
            boundary="tool",
            exception_factory=RuntimeError,
        )).wrap_async("tool", lambda: None)
