"""Execution-local access to the current Run's coordination snapshot."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Awaitable, Callable

from .models import CoordinationSnapshot


SnapshotLoader = Callable[[], Awaitable[CoordinationSnapshot | None]]

coordination_snapshot_loader: ContextVar[SnapshotLoader | None] = ContextVar(
    "coordination_snapshot_loader",
    default=None,
)


async def load_coordination_snapshot() -> CoordinationSnapshot | None:
    loader = coordination_snapshot_loader.get()
    return await loader() if loader is not None else None
