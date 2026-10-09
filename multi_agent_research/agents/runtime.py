"""Bounded Agent runner; production roles migrate onto it in later phases."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Generic, Literal, Mapping, Protocol, TypeVar
from uuid import uuid4

from ..core.budget import RunControlError
from .context import AgentContext
from .contracts import ModelCall, ModelCost
from .events import AgentEvent, AgentEventSink, AgentEventType
from .spec import AgentSpec, AgentSpecAny


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class AgentRuntimeError(RuntimeError):
    """Base error for configuration, contract, or bounded execution failures."""


class AgentConfigurationError(AgentRuntimeError):
    pass


class AgentContractError(AgentRuntimeError):
    pass


class AgentTurnLimitError(AgentRuntimeError):
    pass


class AgentTimeoutError(AgentRuntimeError):
    pass


class AgentToolExecutionError(AgentRuntimeError):
    """Safe wrapper that prevents tool inputs or provider messages entering traces."""


AgentTurnStatus = Literal["continue", "completed", "paused"]
AgentRunStatus = Literal["completed", "paused", "failed"]


@dataclass(frozen=True, slots=True)
class AgentTurnResult(Generic[OutputT]):
    status: AgentTurnStatus
    output: OutputT | None = None
    state_updates: dict[str, Any] = field(default_factory=dict)
    unresolved: tuple[str, ...] = ()
    handoff: dict[str, Any] | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AgentResult(Generic[OutputT]):
    status: AgentRunStatus
    output: OutputT | None
    usage: ModelCost
    turns: int
    local_state: dict[str, Any]
    unresolved: tuple[str, ...] = ()
    handoff: dict[str, Any] | None = None
    error_type: str | None = None
    error_message: str | None = None


class RuntimeAgent(Protocol[InputT, OutputT]):
    spec: AgentSpec[InputT, OutputT]

    async def run_turn(
        self,
        request: InputT,
        *,
        context: AgentContext,
        call_model: ModelCall,
    ) -> AgentTurnResult[OutputT]: ...


ModelResolver = Callable[[AgentSpecAny], ModelCall]


def _add_cost(total: ModelCost, cost: ModelCost) -> ModelCost:
    return {
        "tokens": total["tokens"] + int(cost.get("tokens", 0)),
        "unknown": total["unknown"] + int(cost.get("unknown", 0)),
        # A completed gateway invocation represents at least one model attempt.
        # Older adapters omit this optional field and historically counted as one.
        "attempts": total.get("attempts", 0) + int(cost.get("attempts", 1)),
    }


MAX_AGENT_CHECKPOINT_BYTES = 64 * 1024
MAX_AGENT_UNRESOLVED_ITEMS = 50
MAX_AGENT_UNRESOLVED_CHARS = 1000


def _checkpoint_payload(
    usage: ModelCost,
    local_state: dict[str, Any],
    unresolved: tuple[str, ...] = (),
    handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bounded JSON snapshot suitable for durable storage."""
    if len(unresolved) > MAX_AGENT_UNRESOLVED_ITEMS or any(
        not isinstance(item, str) or len(item) > MAX_AGENT_UNRESOLVED_CHARS
        for item in unresolved
    ):
        raise AgentContractError("Agent unresolved 列表超出安全持久化限制")
    payload = {
        "usage": dict(usage),
        "local_state": local_state,
        "handoff": handoff,
        "unresolved": list(unresolved),
    }
    try:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise AgentContractError("Agent Checkpoint 必须是可序列化 JSON") from exc
    if len(encoded.encode("utf-8")) > MAX_AGENT_CHECKPOINT_BYTES:
        raise AgentContractError(
            f"Agent Checkpoint 超过 {MAX_AGENT_CHECKPOINT_BYTES} 字节限制"
        )
    return json.loads(encoded)


def _failure_checkpoint(
    usage: ModelCost,
    local_state: dict[str, Any],
) -> dict[str, Any]:
    try:
        return _checkpoint_payload(usage, local_state)
    except AgentContractError:
        return {
            "usage": dict(usage),
            "local_state": {},
            "handoff": None,
            "unresolved": [],
            "state_discarded": True,
        }


class AgentRunner:
    """Run one Agent with isolated context, bounded turns, tools, and events."""

    def __init__(
        self,
        resolve_model: ModelResolver,
        *,
        tools: Mapping[str, Any] | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> None:
        self._resolve_model = resolve_model
        self._tools = dict(tools or {})
        self._event_sink = event_sink

    async def _emit(
        self,
        event_type: AgentEventType,
        spec: AgentSpecAny,
        context: AgentContext,
        **details: Any,
    ) -> None:
        if self._event_sink is None:
            return
        value = self._event_sink(
            AgentEvent(
                event_type=event_type,
                agent_name=spec.name,
                agent_version=spec.version,
                run_id=context.run_id,
                agent_run_id=context.agent_run_id,
                parent_agent_run_id=context.parent_agent_run_id,
                section_id=context.section_id,
                turn=context.turn,
                details=details,
            )
        )
        if inspect.isawaitable(value):
            await value

    def _validate_tools(self, spec: AgentSpecAny) -> None:
        missing = sorted(spec.allowed_tools.difference(self._tools))
        if missing:
            raise AgentConfigurationError(
                f"Agent '{spec.name}' 缺少已声明工具：{', '.join(missing)}"
            )
        synchronous = sorted(
            name for name in spec.allowed_tools
            if not (
                inspect.iscoroutinefunction(self._tools[name])
                or inspect.iscoroutinefunction(getattr(self._tools[name], "__call__", None))
            )
        )
        if synchronous:
            raise AgentConfigurationError(
                "Agent Runtime 工具必须是异步可调用对象，以保证生命周期事件可持久化："
                + ", ".join(synchronous)
            )

    def _observed_tools(
        self,
        spec: AgentSpecAny,
        context: AgentContext,
    ) -> dict[str, Any]:
        observed: dict[str, Any] = {}
        for name in spec.allowed_tools:
            tool = self._tools[name]

            async def invoke(*args: Any, _name=name, _tool=tool, **kwargs: Any) -> Any:
                call_id = uuid4().hex
                started = perf_counter()
                await self._emit(
                    "agent_tool_started",
                    spec,
                    context,
                    tool_name=_name,
                    tool_call_id=call_id,
                )
                try:
                    value = await _tool(*args, **kwargs)
                except (Exception, asyncio.CancelledError) as exc:
                    await self._emit(
                        "agent_tool_failed",
                        spec,
                        context,
                        tool_name=_name,
                        tool_call_id=call_id,
                        duration_ms=max(0, round((perf_counter() - started) * 1000)),
                        error_type=type(exc).__name__,
                    )
                    if isinstance(exc, (RunControlError, asyncio.CancelledError)):
                        raise
                    raise AgentToolExecutionError(
                        f"Agent 工具 '{_name}' 调用失败：{type(exc).__name__}"
                    ) from None
                await self._emit(
                    "agent_tool_completed",
                    spec,
                    context,
                    tool_name=_name,
                    tool_call_id=call_id,
                    duration_ms=max(0, round((perf_counter() - started) * 1000)),
                )
                return value

            observed[name] = invoke
        return observed

    async def run(
        self,
        agent: RuntimeAgent[InputT, OutputT],
        request: InputT,
        *,
        context: AgentContext,
    ) -> AgentResult[OutputT]:
        spec = agent.spec
        if not isinstance(request, spec.input_type):
            raise AgentContractError(
                f"Agent '{spec.name}' 输入应为 {spec.input_type.__name__}，"
                f"实际为 {type(request).__name__}"
            )
        self._validate_tools(spec)
        execution_context = context.with_tools({})
        call_model = self._resolve_model(spec)
        if not callable(call_model):
            raise AgentConfigurationError(f"Agent '{spec.name}' 未解析到可调用模型")

        usage: ModelCost = {"tokens": 0, "unknown": 0, "attempts": 0}
        local_state = dict(execution_context.local_state)
        await self._emit("agent_started", spec, execution_context, model_ref=spec.model_ref)

        async def observed_model_call(*args: Any, **kwargs: Any) -> tuple[Any, ModelCost]:
            nonlocal usage
            await self._emit(
                "agent_model_called",
                spec,
                current_context,
                model_ref=spec.model_ref,
            )
            value, cost = await call_model(*args, **kwargs)
            usage = _add_cost(usage, cost)
            return value, cost

        current_context = execution_context
        try:
            async with asyncio.timeout(spec.timeout_seconds):
                for turn in range(1, spec.max_turns + 1):
                    current_context = execution_context.for_turn(turn)
                    current_context.local_state.clear()
                    current_context.local_state.update(local_state)
                    current_context = current_context.with_tools(
                        self._observed_tools(spec, current_context)
                    )
                    await self._emit("agent_turn_started", spec, current_context)
                    outcome = await agent.run_turn(
                        request,
                        context=current_context,
                        call_model=observed_model_call,
                    )
                    if not isinstance(outcome, AgentTurnResult):
                        raise AgentContractError(
                            f"Agent '{spec.name}' 必须返回 AgentTurnResult"
                        )
                    local_state.update(outcome.state_updates)

                    if outcome.status == "completed":
                        if not isinstance(outcome.output, spec.output_type):
                            raise AgentContractError(
                                f"Agent '{spec.name}' 输出应为 {spec.output_type.__name__}"
                            )
                        checkpoint = _checkpoint_payload(
                            usage,
                            local_state,
                            outcome.unresolved,
                            outcome.handoff,
                        )
                        result = AgentResult(
                            status="completed",
                            output=outcome.output,
                            usage=usage,
                            turns=turn,
                            local_state=dict(local_state),
                            unresolved=outcome.unresolved,
                            handoff=outcome.handoff,
                        )
                        await self._emit(
                            "agent_completed",
                            spec,
                            current_context,
                            turns=turn,
                            checkpoint=checkpoint,
                        )
                        return result

                    if outcome.status == "paused":
                        checkpoint = _checkpoint_payload(
                            usage,
                            local_state,
                            outcome.unresolved,
                            outcome.handoff,
                        )
                        result = AgentResult(
                            status="paused",
                            output=outcome.output,
                            usage=usage,
                            turns=turn,
                            local_state=dict(local_state),
                            unresolved=outcome.unresolved,
                            handoff=outcome.handoff,
                        )
                        await self._emit(
                            "agent_paused",
                            spec,
                            current_context,
                            reason=outcome.reason,
                            checkpoint=checkpoint,
                        )
                        return result

                    if outcome.status != "continue":
                        raise AgentContractError(
                            f"Agent '{spec.name}' 返回了未知状态：{outcome.status}"
                        )
                    checkpoint = _checkpoint_payload(
                        usage,
                        local_state,
                        outcome.unresolved,
                        outcome.handoff,
                    )
                    if turn == spec.max_turns:
                        raise AgentTurnLimitError(
                            f"Agent '{spec.name}' 已用完 {spec.max_turns} 个 turn 仍未完成"
                        )
                    await self._emit(
                        "agent_retrying",
                        spec,
                        current_context,
                        reason=outcome.reason,
                        checkpoint=checkpoint,
                    )
                raise AssertionError("Agent turn loop exited unexpectedly")
        except TimeoutError as exc:
            error = AgentTimeoutError(
                f"Agent '{spec.name}' 超过 {spec.timeout_seconds:g} 秒运行上限"
            )
            await self._emit(
                "agent_failed",
                spec,
                current_context,
                error_type=type(error).__name__,
                error_message=str(error)[:1000],
                checkpoint=_failure_checkpoint(usage, local_state),
            )
            raise error from exc
        except RunControlError as exc:
            await self._emit(
                "agent_paused",
                spec,
                current_context,
                reason=type(exc).__name__,
                error_message=str(exc)[:1000],
                checkpoint=_failure_checkpoint(usage, local_state),
            )
            raise
        except Exception as exc:
            await self._emit(
                "agent_failed",
                spec,
                current_context,
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
                checkpoint=_failure_checkpoint(usage, local_state),
            )
            raise
