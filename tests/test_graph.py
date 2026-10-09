"""Current graph structure and state-schema tests."""

from multi_agent_research.core.graph import _build_state_graph
from multi_agent_research.core.state import CURRENT_WORKFLOW_VERSION, initial_state
from multi_agent_research.sections.subgraph import (
    SECTION_SUBGRAPH_NODES,
    build_section_subgraph,
)


def test_parent_graph_contains_one_chapter_subgraph_boundary() -> None:
    graph = _build_state_graph()
    assert set(graph.nodes) == {
        "plan_sections",
        "section_cycle",
        "report_review",
        "assemble_report",
    }
    assert {"supervisor", "search_agent", "analyst_agent", "writer_agent"}.isdisjoint(
        graph.nodes
    )


def test_section_subgraph_owns_only_chapter_execution_nodes() -> None:
    graph = build_section_subgraph()
    assert set(graph.nodes) == set(SECTION_SUBGRAPH_NODES)
    assert {"plan_sections", "report_review", "assemble_report"}.isdisjoint(graph.nodes)


def test_initial_state_contains_only_current_workflow_fields() -> None:
    state = initial_state("测试研究问题")
    assert state["workflow_version"] == CURRENT_WORKFLOW_VERSION
    assert CURRENT_WORKFLOW_VERSION == 5
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
