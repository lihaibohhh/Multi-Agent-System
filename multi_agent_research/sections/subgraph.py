"""Checkpointed serial execution unit for one selected chapter."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from . import workflow


class SectionSubgraphState(TypedDict):
    """The chapter-local projection shared with the parent research graph.

    The subgraph owns chapter artifacts and its local routing cursor. Report review,
    final assembly, run budget persistence, and the immutable research request remain
    parent/runtime responsibilities.
    """

    research_question: str
    parent_context: dict | None
    workflow_version: int
    sections: list[dict]
    active_section: int
    section_policy: dict
    section_step: str
    model_calls: int
    usage_unknown_calls: int
    iteration_count: int
    token_budget_used: int


SECTION_SUBGRAPH_NODES = (
    "section_dispatch",
    "section_search",
    "section_write",
    "section_review",
    "section_claims",
    "section_claim_gate",
    "section_advance",
)

_LOCAL_ROUTES = {
    "research": "section_search",
    "write": "section_write",
    "review": "section_review",
    "claims": "section_claims",
    "claim_gate": "section_claim_gate",
    "advance": "section_advance",
    "report_review": "exit",
    "assemble": "exit",
}


def _dispatch_section(state: SectionSubgraphState) -> dict:
    """Create an explicit checkpoint boundary before dispatching resumed local work."""
    return {}


def route_section_subgraph(state: SectionSubgraphState) -> str:
    try:
        return _LOCAL_ROUTES[state["section_step"]]
    except KeyError as exc:
        raise ValueError(
            f"章节子图无法处理 section_step={state.get('section_step')!r}"
        ) from exc


def build_section_subgraph():
    """Build an uncompiled chapter graph so callers can inspect or embed it."""
    graph = StateGraph(SectionSubgraphState)
    graph.add_node("section_dispatch", _dispatch_section)
    graph.add_node("section_search", workflow.research_section)
    graph.add_node("section_write", workflow.write_section)
    graph.add_node("section_review", workflow.review_section)
    graph.add_node("section_claims", workflow.extract_claims)
    graph.add_node("section_claim_gate", workflow.claim_gate)
    graph.add_node("section_advance", workflow.advance_section)
    graph.set_entry_point("section_dispatch")

    routes = {
        "section_search": "section_search",
        "section_write": "section_write",
        "section_review": "section_review",
        "section_claims": "section_claims",
        "section_claim_gate": "section_claim_gate",
        "section_advance": "section_advance",
        "exit": END,
    }
    for node in SECTION_SUBGRAPH_NODES:
        if node == "section_advance":
            graph.add_edge(node, END)
            continue
        graph.add_conditional_edges(node, route_section_subgraph, routes)
    return graph


def compile_section_subgraph():
    """Compile without a saver; the embedded graph inherits the parent's saver."""
    return build_section_subgraph().compile()
