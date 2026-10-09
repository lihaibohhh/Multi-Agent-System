"""Run-local Agent context, isolated from the full LangGraph state."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Mapping
from uuid import uuid4


class AgentToolUnavailableError(LookupError):
    """The Agent requested a tool outside its declared capability boundary."""


AgentCheckpoint = dict[str, Any]
AgentCheckpointLoader = Callable[
    [str, str | None], Awaitable[AgentCheckpoint | None]
]

agent_checkpoint_loader: ContextVar[AgentCheckpointLoader | None] = ContextVar(
    "agent_checkpoint_loader",
    default=None,
)


@dataclass(frozen=True, slots=True)
class AgentContext:
    run_id: str
    agent_run_id: str
    parent_agent_run_id: str | None = None
    section_id: str | None = None
    turn: int = 0
    local_state: dict[str, Any] = field(default_factory=dict)
    tools: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("AgentContext.run_id 不能为空")
        if not self.agent_run_id.strip():
            raise ValueError("AgentContext.agent_run_id 不能为空")
        if self.turn < 0:
            raise ValueError("AgentContext.turn 不能为负数")

    @classmethod
    def create(
        cls,
        run_id: str,
        *,
        parent_agent_run_id: str | None = None,
        section_id: str | None = None,
        local_state: dict[str, Any] | None = None,
    ) -> AgentContext:
        return cls(
            run_id=run_id,
            agent_run_id=uuid4().hex,
            parent_agent_run_id=parent_agent_run_id,
            section_id=section_id,
            local_state=dict(local_state or {}),
        )

    def for_turn(self, turn: int) -> AgentContext:
        return replace(self, turn=turn, local_state=dict(self.local_state))

    def with_tools(self, tools: Mapping[str, Any]) -> AgentContext:
        return replace(self, tools=MappingProxyType(dict(tools)))

    def require_tool(self, name: str) -> Any:
        try:
            return self.tools[name]
        except KeyError as exc:
            raise AgentToolUnavailableError(
                f"Agent 工具 '{name}' 未授权或未配置"
            ) from exc


async def create_resumable_agent_context(
    run_id: str,
    agent_name: str,
    *,
    section_id: str | None = None,
) -> AgentContext:
    """Create a new Agent execution linked to its latest durable checkpoint."""
    loader = agent_checkpoint_loader.get()
    checkpoint = await loader(agent_name, section_id) if loader is not None else None
    return AgentContext.create(
        run_id,
        parent_agent_run_id=(checkpoint or {}).get("agent_run_id"),
        section_id=section_id,
        local_state=(checkpoint or {}).get("local_state") or {},
    )
