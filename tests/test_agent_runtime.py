from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from multi_agent_research.agents import (
    AgentConfigurationError,
    AgentContext,
    AgentContractError,
    AgentRunner,
    AgentSpec,
    AgentToolUnavailableError,
    AgentTimeoutError,
    AgentToolExecutionError,
    AgentTurnLimitError,
    AgentTurnResult,
)
from multi_agent_research.core.budget import ExecutionPaused


@dataclass(frozen=True)
class Request:
    value: str


@dataclass(frozen=True)
class Output:
    value: str


def _gateway(events: list[str]):
    def resolve(spec):
        assert spec.model_ref == "reasoning-model"

        async def call_model(system, prompt, schema=None, *, validator=None, context=None):
            events.append(f"model:{prompt}")
            return "observed", {"tokens": 3, "unknown": 0, "attempts": 1}

        return call_model

    return resolve


def _gateway_without_attempt_count():
    def resolve(spec):
        async def call_model(system, prompt, schema=None, *, validator=None, context=None):
            return "observed", {"tokens": 3, "unknown": 0}

        return call_model

    return resolve


class TwoTurnAgent:
    spec = AgentSpec(
        name="two_turn",
        description="exercise the bounded runtime",
        model_ref="reasoning-model",
        input_type=Request,
        output_type=Output,
        max_turns=2,
        allowed_tools=frozenset({"echo"}),
    )

    async def run_turn(self, request, *, context, call_model):
        assert set(context.tools) == {"echo"}
        observed, _ = await call_model("system", f"turn-{context.turn}")
        if context.turn == 1:
            return AgentTurnResult(
                status="continue",
                state_updates={"observation": observed},
                reason="需要第二轮",
            )
        echo = context.require_tool("echo")
        return AgentTurnResult(
            status="completed",
            output=Output(await echo(request.value)),
            state_updates={"finished": True},
            unresolved=("保留一项说明",),
            handoff={"artifact": "result-1"},
        )


@pytest.mark.asyncio
async def test_runner_owns_identity_context_tools_turns_usage_and_events() -> None:
    model_events: list[str] = []
    lifecycle = []
    async def echo(value):
        return value.upper()

    runner = AgentRunner(
        _gateway(model_events),
        tools={"echo": echo, "not_allowed": object()},
        event_sink=lifecycle.append,
    )
    context = AgentContext.create(
        "run-1",
        parent_agent_run_id="parent-agent",
        section_id="section_1",
        local_state={"seed": 1},
    )

    result = await runner.run(TwoTurnAgent(), Request("done"), context=context)

    assert result.status == "completed"
    assert result.output == Output("DONE")
    assert result.turns == 2
    assert result.usage == {"tokens": 6, "unknown": 0, "attempts": 2}
    assert result.local_state == {"seed": 1, "observation": "observed", "finished": True}
    assert result.unresolved == ("保留一项说明",)
    assert result.handoff == {"artifact": "result-1"}
    assert context.turn == 0
    assert context.local_state == {"seed": 1}
    assert model_events == ["model:turn-1", "model:turn-2"]
    assert [event.event_type for event in lifecycle] == [
        "agent_started",
        "agent_turn_started",
        "agent_model_called",
        "agent_retrying",
        "agent_turn_started",
        "agent_model_called",
        "agent_tool_started",
        "agent_tool_completed",
        "agent_completed",
    ]
    assert lifecycle[0].agent_name == "two_turn"
    assert lifecycle[0].agent_run_id == context.agent_run_id
    assert lifecycle[0].details == {"model_ref": "reasoning-model"}
    assert lifecycle[-1].turn == 2
    assert lifecycle[-1].details["checkpoint"] == {
        "usage": {"tokens": 6, "unknown": 0, "attempts": 2},
        "local_state": {"seed": 1, "observation": "observed", "finished": True},
        "handoff": {"artifact": "result-1"},
        "unresolved": ["保留一项说明"],
    }


@pytest.mark.asyncio
async def test_runner_counts_legacy_gateway_call_without_attempt_field() -> None:
    async def echo(value):
        return value

    runner = AgentRunner(
        _gateway_without_attempt_count(),
        tools={"echo": echo},
    )

    result = await runner.run(
        TwoTurnAgent(),
        Request("done"),
        context=AgentContext.create("run-legacy-cost"),
    )

    assert result.usage == {"tokens": 6, "unknown": 0, "attempts": 2}


def test_spec_and_context_reject_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match="max_turns"):
        AgentSpec(
            name="invalid",
            description="invalid",
            model_ref="model",
            input_type=Request,
            output_type=Output,
            max_turns=0,
        )
    context = AgentContext.create("run-1")
    with pytest.raises(AgentToolUnavailableError, match="未授权"):
        context.require_tool("search")


@pytest.mark.asyncio
async def test_runner_rejects_missing_declared_tool_before_execution() -> None:
    lifecycle = []
    runner = AgentRunner(_gateway([]), event_sink=lifecycle.append)

    with pytest.raises(AgentConfigurationError, match="echo"):
        await runner.run(TwoTurnAgent(), Request("x"), context=AgentContext.create("run-1"))

    assert lifecycle == []


@pytest.mark.asyncio
async def test_runner_rejects_synchronous_tools_before_execution() -> None:
    lifecycle = []
    runner = AgentRunner(
        _gateway([]),
        tools={"echo": lambda value: value},
        event_sink=lifecycle.append,
    )

    with pytest.raises(AgentConfigurationError, match="必须是异步"):
        await runner.run(TwoTurnAgent(), Request("x"), context=AgentContext.create("run-sync"))

    assert lifecycle == []


@pytest.mark.asyncio
async def test_tool_trace_records_safe_failure_metadata_without_arguments_or_message() -> None:
    class ToolFailureAgent:
        spec = AgentSpec(
            name="tool_failure",
            description="exercise safe tool failure telemetry",
            model_ref="reasoning-model",
            input_type=Request,
            output_type=Output,
            allowed_tools=frozenset({"lookup"}),
        )

        async def run_turn(self, request, *, context, call_model):
            lookup = context.require_tool("lookup")
            await lookup(request.value)
            return AgentTurnResult(status="completed", output=Output("unexpected"))

    async def lookup(query):
        raise ValueError(f"private tool failure for {query}")

    lifecycle = []
    runner = AgentRunner(
        _gateway([]),
        tools={"lookup": lookup},
        event_sink=lifecycle.append,
    )
    with pytest.raises(AgentToolExecutionError, match="lookup.*ValueError"):
        await runner.run(
            ToolFailureAgent(),
            Request("SECRET_QUERY"),
            context=AgentContext.create("run-tool-failure"),
        )

    assert [event.event_type for event in lifecycle] == [
        "agent_started",
        "agent_turn_started",
        "agent_tool_started",
        "agent_tool_failed",
        "agent_failed",
    ]
    started, failed = lifecycle[2:4]
    assert started.details["tool_name"] == "lookup"
    assert failed.details["tool_call_id"] == started.details["tool_call_id"]
    assert failed.details["error_type"] == "ValueError"
    serialized = json.dumps([event.as_record() for event in lifecycle])
    assert "SECRET_QUERY" not in serialized
    assert "private tool failure" not in serialized


@pytest.mark.asyncio
async def test_tool_pause_preserves_run_control_and_emits_safe_trace() -> None:
    class ToolPauseAgent:
        spec = AgentSpec(
            name="tool_pause",
            description="pause while executing a tool",
            model_ref="reasoning-model",
            input_type=Request,
            output_type=Output,
            allowed_tools=frozenset({"lookup"}),
        )

        async def run_turn(self, request, *, context, call_model):
            await context.require_tool("lookup")(request.value)
            return AgentTurnResult(status="completed", output=Output("unexpected"))

    async def lookup(_query):
        raise ExecutionPaused("pause requested")

    lifecycle = []
    with pytest.raises(ExecutionPaused, match="pause requested"):
        await AgentRunner(
            _gateway([]),
            tools={"lookup": lookup},
            event_sink=lifecycle.append,
        ).run(
            ToolPauseAgent(),
            Request("private"),
            context=AgentContext.create("run-tool-pause"),
        )

    assert [event.event_type for event in lifecycle][-2:] == [
        "agent_tool_failed",
        "agent_paused",
    ]
    assert lifecycle[-2].details["error_type"] == "ExecutionPaused"


@pytest.mark.asyncio
async def test_agent_timeout_cancels_tool_and_records_both_boundaries() -> None:
    class SlowToolAgent:
        spec = AgentSpec(
            name="slow_tool",
            description="time out during a tool call",
            model_ref="reasoning-model",
            input_type=Request,
            output_type=Output,
            timeout_seconds=0.01,
            allowed_tools=frozenset({"lookup"}),
        )

        async def run_turn(self, request, *, context, call_model):
            await context.require_tool("lookup")(request.value)
            return AgentTurnResult(status="completed", output=Output("late"))

    async def lookup(_query):
        import asyncio
        await asyncio.sleep(1)

    lifecycle = []
    with pytest.raises(AgentTimeoutError):
        await AgentRunner(
            _gateway([]),
            tools={"lookup": lookup},
            event_sink=lifecycle.append,
        ).run(
            SlowToolAgent(),
            Request("private"),
            context=AgentContext.create("run-tool-timeout"),
        )

    assert [event.event_type for event in lifecycle][-2:] == [
        "agent_tool_failed",
        "agent_failed",
    ]
    assert lifecycle[-2].details["error_type"] == "CancelledError"
    assert lifecycle[-1].details["error_type"] == "AgentTimeoutError"


@pytest.mark.asyncio
async def test_runner_rejects_wrong_input_before_starting_agent() -> None:
    lifecycle = []
    async def echo(value):
        return value

    runner = AgentRunner(
        _gateway([]),
        tools={"echo": echo},
        event_sink=lifecycle.append,
    )

    with pytest.raises(AgentContractError, match="输入应为 Request"):
        await runner.run(TwoTurnAgent(), "wrong", context=AgentContext.create("run-1"))

    assert lifecycle == []


@pytest.mark.asyncio
async def test_runner_emits_failure_for_output_contract_violation() -> None:
    class InvalidOutputAgent:
        spec = AgentSpec(
            name="invalid_output",
            description="returns a wrong type",
            model_ref="reasoning-model",
            input_type=Request,
            output_type=Output,
        )

        async def run_turn(self, request, *, context, call_model):
            return AgentTurnResult(status="completed", output="wrong")

    lifecycle = []
    runner = AgentRunner(_gateway([]), event_sink=lifecycle.append)

    with pytest.raises(AgentContractError, match="输出应为 Output"):
        await runner.run(
            InvalidOutputAgent(),
            Request("x"),
            context=AgentContext.create("run-1"),
        )

    assert [event.event_type for event in lifecycle] == [
        "agent_started",
        "agent_turn_started",
        "agent_failed",
    ]
    assert lifecycle[-1].details["error_type"] == "AgentContractError"


@pytest.mark.asyncio
async def test_runner_enforces_turn_limit_without_fake_retry_event() -> None:
    class NeverDoneAgent:
        spec = AgentSpec(
            name="never_done",
            description="always asks for another turn",
            model_ref="reasoning-model",
            input_type=Request,
            output_type=Output,
            max_turns=1,
        )

        async def run_turn(self, request, *, context, call_model):
            return AgentTurnResult(status="continue", reason="again")

    lifecycle = []
    runner = AgentRunner(_gateway([]), event_sink=lifecycle.append)

    with pytest.raises(AgentTurnLimitError, match="1 个 turn"):
        await runner.run(
            NeverDoneAgent(),
            Request("x"),
            context=AgentContext.create("run-1"),
        )

    assert [event.event_type for event in lifecycle] == [
        "agent_started",
        "agent_turn_started",
        "agent_failed",
    ]


@pytest.mark.asyncio
async def test_runner_returns_explicit_paused_result() -> None:
    class PausingAgent:
        spec = AgentSpec(
            name="pausing",
            description="pauses with local progress",
            model_ref="reasoning-model",
            input_type=Request,
            output_type=Output,
        )

        async def run_turn(self, request, *, context, call_model):
            return AgentTurnResult(
                status="paused",
                state_updates={"saved": True},
                unresolved=("等待外部条件",),
                reason="dependency unavailable",
            )

    lifecycle = []
    runner = AgentRunner(_gateway([]), event_sink=lifecycle.append)
    result = await runner.run(
        PausingAgent(),
        Request("x"),
        context=AgentContext.create("run-1"),
    )

    assert result.status == "paused"
    assert result.local_state == {"saved": True}
    assert result.unresolved == ("等待外部条件",)
    assert lifecycle[-1].event_type == "agent_paused"
    assert lifecycle[-1].details["reason"] == "dependency unavailable"
    assert lifecycle[-1].details["checkpoint"]["local_state"] == {"saved": True}


@pytest.mark.asyncio
async def test_runner_emits_pause_and_preserves_run_control_exception() -> None:
    class ControlledAgent:
        spec = AgentSpec(
            name="controlled",
            description="propagates outer budget or pause controls",
            model_ref="reasoning-model",
            input_type=Request,
            output_type=Output,
        )

        async def run_turn(self, request, *, context, call_model):
            raise ExecutionPaused("pause requested")

    lifecycle = []
    runner = AgentRunner(_gateway([]), event_sink=lifecycle.append)

    with pytest.raises(ExecutionPaused, match="pause requested"):
        await runner.run(
            ControlledAgent(),
            Request("x"),
            context=AgentContext.create("run-1"),
        )

    assert lifecycle[-1].event_type == "agent_paused"
    assert lifecycle[-1].details["reason"] == "ExecutionPaused"
    assert lifecycle[-1].details["error_message"] == "pause requested"


@pytest.mark.asyncio
async def test_runner_enforces_agent_timeout() -> None:
    class SlowAgent:
        spec = AgentSpec(
            name="slow",
            description="exceeds its local runtime limit",
            model_ref="reasoning-model",
            input_type=Request,
            output_type=Output,
            timeout_seconds=0.01,
        )

        async def run_turn(self, request, *, context, call_model):
            import asyncio

            await asyncio.sleep(1)
            return AgentTurnResult(status="completed", output=Output("late"))

    lifecycle = []
    runner = AgentRunner(_gateway([]), event_sink=lifecycle.append)

    with pytest.raises(AgentTimeoutError, match="0.01 秒"):
        await runner.run(
            SlowAgent(),
            Request("x"),
            context=AgentContext.create("run-1"),
        )

    assert lifecycle[-1].event_type == "agent_failed"
    assert lifecycle[-1].details["error_type"] == "AgentTimeoutError"
    assert lifecycle[-1].details["checkpoint"]["usage"]["attempts"] == 0


@pytest.mark.asyncio
async def test_runner_rejects_non_json_checkpoint_state() -> None:
    class UnsafeStateAgent:
        spec = AgentSpec(
            name="unsafe_state",
            description="returns non-serializable state",
            model_ref="reasoning-model",
            input_type=Request,
            output_type=Output,
        )

        async def run_turn(self, request, *, context, call_model):
            return AgentTurnResult(
                status="completed",
                output=Output("done"),
                state_updates={"unsafe": object()},
            )

    lifecycle = []
    runner = AgentRunner(_gateway([]), event_sink=lifecycle.append)

    with pytest.raises(AgentContractError, match="可序列化 JSON"):
        await runner.run(
            UnsafeStateAgent(),
            Request("x"),
            context=AgentContext.create("run-unsafe-state"),
        )

    assert lifecycle[-1].event_type == "agent_failed"
    assert lifecycle[-1].details["checkpoint"]["state_discarded"] is True
