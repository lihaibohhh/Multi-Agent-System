"""Declarative identity and execution limits for independently run agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class AgentSpec(Generic[InputT, OutputT]):
    """Static Agent configuration; mutable run state belongs in AgentContext."""

    name: str
    description: str
    model_ref: str
    input_type: type[InputT]
    output_type: type[OutputT]
    version: str = "1"
    max_turns: int = 1
    timeout_seconds: float = 120.0
    allowed_tools: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for field_name in ("name", "description", "model_ref", "version"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"AgentSpec.{field_name} 不能为空")
        if self.max_turns < 1:
            raise ValueError("AgentSpec.max_turns 必须至少为 1")
        if self.timeout_seconds <= 0:
            raise ValueError("AgentSpec.timeout_seconds 必须大于 0")
        if not isinstance(self.input_type, type) or not isinstance(self.output_type, type):
            raise TypeError("AgentSpec 输入输出契约必须是可运行时检查的类型")
        tools = frozenset(str(name).strip() for name in self.allowed_tools)
        if "" in tools:
            raise ValueError("AgentSpec.allowed_tools 不得包含空名称")
        object.__setattr__(self, "allowed_tools", tools)


AgentSpecAny = AgentSpec[Any, Any]
