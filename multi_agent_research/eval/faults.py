"""Reusable fault injection at model, tool, and lifecycle persistence boundaries."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from ..agents.events import AgentEvent, AgentEventSink


FaultPhase = Literal["before", "after"]
AsyncOperation = Callable[..., Awaitable[Any]]


class InjectedProcessCrash(BaseException):
    """Simulate abrupt process loss without entering normal Agent failure handling."""


@dataclass(frozen=True, slots=True)
class FaultRule:
    """Trigger one delay and/or exception at a named async boundary."""

    boundary: str
    occurrence: int = 1
    phase: FaultPhase = "before"
    delay_seconds: float = 0.0
    exception_factory: Callable[[], BaseException] | None = None

    def __post_init__(self) -> None:
        if not self.boundary.strip():
            raise ValueError("FaultRule.boundary 不能为空")
        if self.occurrence < 1:
            raise ValueError("FaultRule.occurrence 必须至少为 1")
        if self.delay_seconds < 0:
            raise ValueError("FaultRule.delay_seconds 不能为负数")
        if self.delay_seconds == 0 and self.exception_factory is None:
            raise ValueError("FaultRule 必须配置延迟或异常")


class FaultInjector:
    """Apply deterministic one-shot faults without changing production code paths."""

    def __init__(self, *rules: FaultRule) -> None:
        self._rules = tuple(rules)
        self._counts: dict[str, int] = {}
        self._triggered: set[int] = set()

    def call_count(self, boundary: str) -> int:
        return self._counts.get(boundary, 0)

    @property
    def triggered_rules(self) -> int:
        return len(self._triggered)

    async def _apply(self, boundary: str, occurrence: int, phase: FaultPhase) -> None:
        for index, rule in enumerate(self._rules):
            if (
                index in self._triggered
                or rule.boundary != boundary
                or rule.occurrence != occurrence
                or rule.phase != phase
            ):
                continue
            self._triggered.add(index)
            if rule.delay_seconds:
                await asyncio.sleep(rule.delay_seconds)
            if rule.exception_factory is not None:
                raise rule.exception_factory()

    def wrap_async(self, boundary: str, operation: AsyncOperation) -> AsyncOperation:
        if not inspect.iscoroutinefunction(operation) and not inspect.iscoroutinefunction(
            getattr(operation, "__call__", None)
        ):
            raise TypeError("FaultInjector 只能包装异步调用边界")

        async def invoke(*args: Any, **kwargs: Any) -> Any:
            occurrence = self._counts.get(boundary, 0) + 1
            self._counts[boundary] = occurrence
            await self._apply(boundary, occurrence, "before")
            result = await operation(*args, **kwargs)
            await self._apply(boundary, occurrence, "after")
            return result

        return invoke

    def wrap_event_sink(
        self,
        sink: AgentEventSink,
        *,
        boundary: Callable[[AgentEvent], str] = lambda event: event.event_type,
    ) -> AgentEventSink:
        async def observed(event: AgentEvent) -> None:
            name = boundary(event)
            occurrence = self._counts.get(name, 0) + 1
            self._counts[name] = occurrence
            await self._apply(name, occurrence, "before")
            value = sink(event)
            if inspect.isawaitable(value):
                await value
            await self._apply(name, occurrence, "after")

        return observed
