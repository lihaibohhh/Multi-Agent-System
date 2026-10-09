"""Agent-scoped lifecycle events without prompt or source payloads."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal
from uuid import uuid4


AgentEventType = Literal[
    "agent_started",
    "agent_turn_started",
    "agent_model_called",
    "agent_tool_started",
    "agent_tool_completed",
    "agent_tool_failed",
    "agent_retrying",
    "agent_paused",
    "agent_completed",
    "agent_failed",
]


@dataclass(frozen=True, slots=True)
class AgentEvent:
    event_type: AgentEventType
    agent_name: str
    agent_version: str
    run_id: str
    agent_run_id: str
    parent_agent_run_id: str | None
    section_id: str | None
    turn: int
    event_id: str = field(default_factory=lambda: uuid4().hex)
    occurred_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    details: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "run_id": self.run_id,
            "agent_run_id": self.agent_run_id,
            "parent_agent_run_id": self.parent_agent_run_id,
            "section_id": self.section_id,
            "turn": self.turn,
            "occurred_at": self.occurred_at,
            "details": self.details,
        }


AgentEventSink = Callable[[AgentEvent], Awaitable[None] | None]


agent_event_sink: ContextVar[AgentEventSink | None] = ContextVar(
    "agent_event_sink",
    default=None,
)


def dispatch_agent_event(event: AgentEvent) -> Awaitable[None] | None:
    """Forward one event to the current Run execution; standalone calls are no-ops."""
    sink = agent_event_sink.get()
    return sink(event) if sink is not None else None
