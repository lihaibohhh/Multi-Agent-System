"""Run identity helpers for LangGraph checkpoint isolation.

The public identifier is ``run_id``.  LangGraph still requires a
``thread_id`` internally, so a run maps one-to-one to a checkpoint thread.
"""

from __future__ import annotations

import re
import uuid
from typing import Any


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SESSION_ID_PATTERN = _RUN_ID_PATTERN


class RunAlreadyExistsError(RuntimeError):
    """Raised when a create operation tries to reuse an existing run."""


def create_run_id() -> str:
    """Create a globally unique identifier for one research execution."""
    return f"run_{uuid.uuid4().hex}"


def create_session_id() -> str:
    """Create a globally unique identifier for one research conversation."""
    return f"session_{uuid.uuid4().hex}"


def normalize_run_id(run_id: str | None) -> str:
    """Return a validated caller-supplied ID or generate a new one."""
    normalized = (run_id or "").strip()
    if not normalized:
        return create_run_id()
    if not _RUN_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "run_id must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )
    return normalized


def normalize_session_id(session_id: str | None) -> str:
    """Return a validated caller-supplied session ID or generate a new one."""
    normalized = (session_id or "").strip()
    if not normalized:
        return create_session_id()
    if not _SESSION_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "session_id must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )
    return normalized


def checkpoint_config(run_id: str) -> dict[str, dict[str, str]]:
    """Map the public run identity to LangGraph's internal thread identity."""
    return {"configurable": {"thread_id": run_id}}


async def ensure_new_run(app: Any, config: dict) -> None:
    """Reject accidental reuse of a checkpoint thread for a new task."""
    snapshot = await app.aget_state(config)
    if snapshot.values:
        run_id = config.get("configurable", {}).get("thread_id", "")
        raise RunAlreadyExistsError(
            f"run_id '{run_id}' already exists; create a new run_id for a new research task"
        )
