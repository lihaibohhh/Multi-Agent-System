"""Deterministic behavior checks for Agent runs and their lifecycle traces."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from ..agents.events import AgentEvent, AgentEventType
from ..agents.runtime import AgentResult, AgentRunner, RuntimeAgent
from ..agents.context import AgentContext


ExpectedOutcome = Literal["completed", "paused", "failed", "interrupted"]
BehaviorPredicate = Callable[["BehaviorObservation"], bool]


@dataclass(frozen=True, slots=True)
class BehaviorCheck:
    """One scenario-specific, deterministic acceptance condition."""

    code: str
    description: str
    predicate: BehaviorPredicate

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.description.strip():
            raise ValueError("BehaviorCheck code 和 description 不能为空")


@dataclass(frozen=True, slots=True)
class BehaviorExpectation:
    """Expected runtime envelope for one curated Agent scenario."""

    scenario_id: str
    agent_name: str
    max_turns: int
    expected_outcome: ExpectedOutcome = "completed"
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    required_events: frozenset[AgentEventType] = field(default_factory=frozenset)
    expected_model_calls: int | None = None
    allow_incomplete_tool_calls: bool = False
    checks: tuple[BehaviorCheck, ...] = ()

    def __post_init__(self) -> None:
        if not self.scenario_id.strip() or not self.agent_name.strip():
            raise ValueError("BehaviorExpectation 场景和 Agent 名称不能为空")
        if self.max_turns < 1:
            raise ValueError("BehaviorExpectation.max_turns 必须至少为 1")
        if self.expected_model_calls is not None and self.expected_model_calls < 0:
            raise ValueError("expected_model_calls 不能为负数")


@dataclass(frozen=True, slots=True)
class BehaviorObservation:
    """Captured result, failure, and events from a single Agent execution."""

    result: AgentResult[Any] | None
    events: tuple[AgentEvent, ...]
    error: BaseException | None = None


@dataclass(frozen=True, slots=True)
class EvaluationFinding:
    code: str
    passed: bool
    message: str


@dataclass(frozen=True, slots=True)
class BehaviorEvaluation:
    scenario_id: str
    agent_name: str
    findings: tuple[EvaluationFinding, ...]

    @property
    def passed(self) -> bool:
        return all(finding.passed for finding in self.findings)

    @property
    def failed_codes(self) -> tuple[str, ...]:
        return tuple(finding.code for finding in self.findings if not finding.passed)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "agent_name": self.agent_name,
            "passed": self.passed,
            "findings": [
                {
                    "code": finding.code,
                    "passed": finding.passed,
                    "message": finding.message,
                }
                for finding in self.findings
            ],
        }


@dataclass(frozen=True, slots=True)
class EvaluationSuiteReport:
    name: str
    evaluations: tuple[BehaviorEvaluation, ...]

    @property
    def passed(self) -> bool:
        return bool(self.evaluations) and all(item.passed for item in self.evaluations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "total": len(self.evaluations),
            "failed": sum(not item.passed for item in self.evaluations),
            "evaluations": [item.as_dict() for item in self.evaluations],
        }


async def observe_agent_run(
    runner: AgentRunner,
    agent: RuntimeAgent[Any, Any],
    request: Any,
    *,
    context: AgentContext,
    events: list[AgentEvent],
) -> BehaviorObservation:
    """Run an Agent and retain expected execution failures for evaluation.

    ``KeyboardInterrupt`` and ``SystemExit`` deliberately remain process controls. A
    simulated process crash is assembled into an observation by the fault test after
    its checkpoint event has been captured.
    """

    try:
        result = await runner.run(agent, request, context=context)
    except asyncio.CancelledError as exc:
        return BehaviorObservation(result=None, events=tuple(events), error=exc)
    except Exception as exc:
        return BehaviorObservation(result=None, events=tuple(events), error=exc)
    return BehaviorObservation(result=result, events=tuple(events))


def _finding(code: str, passed: bool, success: str, failure: str) -> EvaluationFinding:
    return EvaluationFinding(code=code, passed=passed, message=success if passed else failure)


def _outcome_matches(
    expectation: BehaviorExpectation,
    observation: BehaviorObservation,
) -> bool:
    terminal = [
        event.event_type
        for event in observation.events
        if event.event_type in {"agent_completed", "agent_paused", "agent_failed"}
    ]
    result = observation.result
    if expectation.expected_outcome == "completed":
        return (
            observation.error is None
            and result is not None
            and result.status == "completed"
            and terminal == ["agent_completed"]
        )
    if expectation.expected_outcome == "paused":
        return (
            (result is not None and result.status == "paused" or observation.error is not None)
            and terminal == ["agent_paused"]
        )
    if expectation.expected_outcome == "failed":
        return result is None and observation.error is not None and terminal == ["agent_failed"]
    return result is None and observation.error is not None and not terminal


def _tool_trace_findings(
    expectation: BehaviorExpectation,
    events: Sequence[AgentEvent],
) -> tuple[EvaluationFinding, EvaluationFinding, EvaluationFinding]:
    tool_events = [event for event in events if event.event_type.startswith("agent_tool_")]
    used_tools = {str(event.details.get("tool_name", "")) for event in tool_events}
    policy_ok = used_tools <= expectation.allowed_tools and "" not in used_tools

    calls: dict[str, list[str]] = {}
    for event in tool_events:
        call_id = str(event.details.get("tool_call_id", ""))
        calls.setdefault(call_id, []).append(event.event_type)
    balance_ok = "" not in calls
    for lifecycle in calls.values():
        starts = lifecycle.count("agent_tool_started")
        terminals = sum(
            lifecycle.count(kind)
            for kind in ("agent_tool_completed", "agent_tool_failed")
        )
        balance_ok = balance_ok and starts == 1 and (
            terminals in {0, 1}
            if expectation.allow_incomplete_tool_calls
            else terminals == 1
        )

    safe_fields = {
        "agent_tool_started": {"tool_name", "tool_call_id"},
        "agent_tool_completed": {"tool_name", "tool_call_id", "duration_ms"},
        "agent_tool_failed": {
            "tool_name",
            "tool_call_id",
            "duration_ms",
            "error_type",
        },
    }
    privacy_ok = all(set(event.details) <= safe_fields[event.event_type] for event in tool_events)
    return (
        _finding(
            "tool_policy",
            policy_ok,
            "所有工具调用均在 Agent 白名单内",
            f"检测到未授权工具：{sorted(used_tools - expectation.allowed_tools)}",
        ),
        _finding(
            "tool_trace_balance",
            balance_ok,
            "工具开始与终态事件配对正确",
            "工具 Trace 存在重复、缺失或无开始事件的终态",
        ),
        _finding(
            "tool_trace_privacy",
            privacy_ok,
            "工具 Trace 仅包含安全元数据",
            "工具 Trace 包含参数、结果或未授权字段",
        ),
    )


def evaluate_behavior(
    expectation: BehaviorExpectation,
    observation: BehaviorObservation,
) -> BehaviorEvaluation:
    """Apply common runtime invariants plus scenario-specific behavior checks."""

    events = observation.events
    starts = [event for event in events if event.event_type == "agent_started"]
    identity_ok = bool(events) and all(
        event.agent_name == expectation.agent_name for event in events
    ) and len({event.agent_run_id for event in events}) == 1
    lifecycle_ok = len(starts) == 1 and events[0].event_type == "agent_started"

    turn_events = [event.turn for event in events if event.event_type == "agent_turn_started"]
    turns_ok = (
        len(turn_events) <= expectation.max_turns
        and turn_events == list(range(1, len(turn_events) + 1))
        and (
            observation.result is None
            or observation.result.turns == len(turn_events)
        )
    )
    model_calls = sum(event.event_type == "agent_model_called" for event in events)
    model_calls_ok = (
        expectation.expected_model_calls is None
        or model_calls == expectation.expected_model_calls
    )
    present_events = {event.event_type for event in events}
    required_ok = expectation.required_events <= present_events
    unique_events_ok = len({event.event_id for event in events}) == len(events)

    findings = [
        _finding(
            "identity",
            identity_ok,
            "事件身份保持在同一 Agent 执行内",
            "事件的 Agent 名称或 agent_run_id 不一致",
        ),
        _finding(
            "lifecycle",
            lifecycle_ok,
            "生命周期从唯一 agent_started 开始",
            "生命周期缺少、重复或未从 agent_started 开始",
        ),
        _finding(
            "outcome",
            _outcome_matches(expectation, observation),
            f"执行结果符合预期：{expectation.expected_outcome}",
            f"执行结果不符合预期：{expectation.expected_outcome}",
        ),
        _finding(
            "turn_bound",
            turns_ok,
            f"执行轮次未超过 {expectation.max_turns}",
            "轮次不连续、结果计数不一致或超过上限",
        ),
        _finding(
            "model_call_count",
            model_calls_ok,
            f"模型调用次数符合预期：{model_calls}",
            f"模型调用次数为 {model_calls}，预期 {expectation.expected_model_calls}",
        ),
        _finding(
            "required_events",
            required_ok,
            "要求的生命周期事件均已出现",
            f"缺少事件：{sorted(expectation.required_events - present_events)}",
        ),
        _finding(
            "event_id_uniqueness",
            unique_events_ok,
            "生命周期事件 ID 唯一",
            "生命周期事件 ID 存在重复",
        ),
        *_tool_trace_findings(expectation, events),
    ]
    for check in expectation.checks:
        try:
            passed = bool(check.predicate(observation))
        except Exception as exc:
            findings.append(EvaluationFinding(
                code=check.code,
                passed=False,
                message=f"评测条件执行失败：{type(exc).__name__}",
            ))
        else:
            findings.append(EvaluationFinding(
                code=check.code,
                passed=passed,
                message=check.description if passed else f"未满足：{check.description}",
            ))
    return BehaviorEvaluation(
        scenario_id=expectation.scenario_id,
        agent_name=expectation.agent_name,
        findings=tuple(findings),
    )


def build_suite_report(
    name: str,
    evaluations: Sequence[BehaviorEvaluation],
) -> EvaluationSuiteReport:
    if not name.strip():
        raise ValueError("评测套件名称不能为空")
    return EvaluationSuiteReport(name=name, evaluations=tuple(evaluations))
