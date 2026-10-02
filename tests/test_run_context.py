from __future__ import annotations

from types import SimpleNamespace

import pytest

from multi_agent_research.core.config import DatabaseConfig
from multi_agent_research.core.run_context import (
    RunAlreadyExistsError,
    checkpoint_config,
    create_session_id,
    ensure_new_run,
    normalize_run_id,
    normalize_session_id,
)


def test_normalize_run_id_generates_unique_ids() -> None:
    first = normalize_run_id(None)
    second = normalize_run_id("")

    assert first.startswith("run_")
    assert second.startswith("run_")
    assert first != second


def test_normalize_run_id_preserves_valid_caller_id() -> None:
    assert normalize_run_id("  session-1:run-2  ") == "session-1:run-2"


@pytest.mark.parametrize("run_id", ["has space", "/path", "x" * 129])
def test_normalize_run_id_rejects_invalid_values(run_id: str) -> None:
    with pytest.raises(ValueError):
        normalize_run_id(run_id)


def test_checkpoint_config_maps_run_to_langgraph_thread() -> None:
    assert checkpoint_config("run-123") == {
        "configurable": {"thread_id": "run-123"}
    }


def test_normalize_session_id_generates_and_validates_ids() -> None:
    generated = create_session_id()
    assert generated.startswith("session_")
    assert normalize_session_id("  session-1  ") == "session-1"
    with pytest.raises(ValueError):
        normalize_session_id("invalid session")


@pytest.mark.asyncio
async def test_ensure_new_run_rejects_existing_checkpoint() -> None:
    class FakeApp:
        async def aget_state(self, config: dict):
            thread_id = config["configurable"]["thread_id"]
            values = {"research_question": "旧任务"} if thread_id == "existing" else {}
            return SimpleNamespace(values=values)

    app = FakeApp()
    await ensure_new_run(app, checkpoint_config("new-run"))

    with pytest.raises(RunAlreadyExistsError, match="already exists"):
        await ensure_new_run(app, checkpoint_config("existing"))


def test_postgres_db_url_is_the_primary_database_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_DB_URL", "postgresql://primary.test/db")
    monkeypatch.setenv("DATABASE_URL", "postgresql://legacy.test/db")

    assert DatabaseConfig().url == "postgresql://primary.test/db"


def test_database_url_remains_backward_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_DB_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://legacy.test/db")

    assert DatabaseConfig().url == "postgresql://legacy.test/db"
