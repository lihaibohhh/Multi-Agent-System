"""Current graph structure and state-schema tests."""

from multi_agent_research.core.graph import _build_state_graph
from multi_agent_research.core.state import CURRENT_WORKFLOW_VERSION, initial_state


def test_graph_contains_only_current_chapter_nodes() -> None:
    graph = _build_state_graph()
    assert set(graph.nodes) == {
        "plan_sections",
        "section_search",
        "section_analyze",
        "section_write",
        "section_review",
        "section_advance",
        "section_claims",
        "section_claim_gate",
        "report_review",
        "assemble_report",
    }
    assert {"supervisor", "search_agent", "analyst_agent", "writer_agent"}.isdisjoint(
        graph.nodes
    )


def test_initial_state_contains_only_current_workflow_fields() -> None:
    state = initial_state("测试研究问题")
    assert state["workflow_version"] == CURRENT_WORKFLOW_VERSION
    assert state["research_question"] == "测试研究问题"
    assert state["iteration_count"] == 0
    assert state["writer_status"] == "not_started"
    legacy = {
        "task_plan",
        "task_status",
        "next_agent",
        "supervisor_reason",
        "messages",
        "search_results",
        "analyst_verdict",
        "events",
    }
    assert legacy.isdisjoint(state)
