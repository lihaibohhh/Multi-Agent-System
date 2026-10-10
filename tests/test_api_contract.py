from fastapi.testclient import TestClient

from multi_agent_research.api.server import app
from multi_agent_research.api.demo import INDEX_FILE, STATIC_DIR


def test_session_and_run_lifecycle_routes_are_exposed() -> None:
    paths = app.openapi()["paths"]
    expected = {
        "/api/sessions",
        "/api/sessions/{session_id}",
        "/api/sessions/{session_id}/runs",
        "/api/runs",
        "/api/runs/{run_id}",
        "/api/runs/{run_id}/start",
        "/api/runs/{run_id}/resume",
        "/api/runs/{run_id}/pause",
        "/api/runs/{run_id}/budget/migrate",
        "/api/runs/{run_id}/stream",
        "/api/runs/{run_id}/snapshot",
        "/api/runs/{run_id}/agent-trace",
        "/api/runs/{run_id}/usage",
        "/api/runs/{run_id}/report/reassemble",
        "/api/runs/{run_id}/sections/{section_id}/revisions",
    }
    assert expected <= set(paths)


def test_legacy_one_step_stream_is_marked_deprecated() -> None:
    operation = app.openapi()["paths"]["/api/research/stream"]["get"]
    assert operation["deprecated"] is True


def test_demo_is_served_from_standalone_static_assets() -> None:
    assert INDEX_FILE.is_file()
    assert (STATIC_DIR / "styles.css").is_file()
    assert (STATIC_DIR / "app.js").is_file()
    assert 'id="usage-panel"' in INDEX_FILE.read_text(encoding="utf-8")
    assert 'id="internal-audit"' in INDEX_FILE.read_text(encoding="utf-8")
    assert any(getattr(route, "path", None) == "/" for route in app.routes)
    assert any(getattr(route, "path", None) == "/static" for route in app.routes)

    client = TestClient(app)
    assert client.get("/").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
