from multi_agent_research.core.checkpointer import CheckpointerFactory


def test_lifecycle_cache_key_is_redacted_for_logs() -> None:
    dsn_key = "postgres:postgresql://research_user:secret@127.0.0.1:5432/research"

    label = CheckpointerFactory._safe_key_label(dsn_key)

    assert label == "postgres"
    assert "secret" not in label
    assert "research_user" not in label
