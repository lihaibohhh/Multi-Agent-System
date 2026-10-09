"""Typed contracts for the non-Agent retrieval boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


class SearchResult(TypedDict):
    query: str
    source: Literal["knowledge", "web"]
    content: str
    score: float
    metadata: dict
    iteration: int


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Inputs selected by orchestration and analysis, without graph state."""

    question: str
    gaps: tuple[str, ...] = ()
    parent_question: str = ""
    iteration: int = 0
    scope: dict[str, Any] = field(default_factory=dict)
    existing_count: int = 0
