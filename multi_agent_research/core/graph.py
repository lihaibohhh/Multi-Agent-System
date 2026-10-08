"""
graph.py — 章节研究图与旧版 Supervisor 图的组装入口
把所有节点和边连接成完整的 LangGraph StateGraph。
"""

from __future__ import annotations
import asyncio
import logging
from langgraph.graph import StateGraph, END

from .state import ResearchState, initial_state
from .run_context import checkpoint_config, ensure_new_run, normalize_run_id
from .supervisor import supervisor_node, route_from_supervisor, route_from_search
from ..agents.search_agent import search_agent_node
from ..agents.analyst_agent import analyst_agent_node
from ..agents.writer_agent import writer_agent_node
from ..sections import workflow as chapters


logger = logging.getLogger(__name__)


def _build_state_graph() -> StateGraph:
    """
    构建图定义。新任务按章节串行检索、分析、写作、审校，再确定性装配。
    旧版节点和边保留，以支持没有 workflow_version=2 的历史 Checkpoint。
    """
    graph = StateGraph(ResearchState)

    # ── 注册节点 ─────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("analyst_agent", analyst_agent_node)
    graph.add_node("writer_agent", writer_agent_node)

    # ── 入口点 ───────────────────────────────────
    graph.add_node("plan_sections", chapters.plan_sections)
    graph.add_node("section_search", chapters.research_section)
    graph.add_node("section_analyze", chapters.analyze_section)
    graph.add_node("section_write", chapters.write_section)
    graph.add_node("section_review", chapters.review_section)
    graph.add_node("section_advance", chapters.advance_section)
    graph.add_node("section_claims", chapters.extract_claims)
    graph.add_node("section_claim_gate", chapters.claim_gate)
    graph.add_node("report_review", chapters.review_report)
    graph.add_node("assemble_report", chapters.assemble_sections)
    # Missing version denotes an old checkpoint. Keep its node names and edges intact.
    graph.set_conditional_entry_point(
        lambda state: "plan_sections" if state.get("workflow_version") in {2, 3} else "supervisor",
        {"plan_sections": "plan_sections", "supervisor": "supervisor"},
    )
    chapter_routes = {name: name for name in (
        "section_search", "section_analyze", "section_write", "section_review",
        "section_advance", "assemble_report",
        "section_claims", "report_review",
        "section_claim_gate",
    )}
    for node in ("plan_sections", "section_search", "section_analyze", "section_write",
                 "section_review", "section_advance", "section_claims", "section_claim_gate", "report_review"):
        graph.add_conditional_edges(node, chapters.route_section, chapter_routes)
    graph.add_edge("assemble_report", END)

    # ── Supervisor 的条件路由 ─────────────────────
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,          # 返回 "search_agent" / "analyst_agent" / "writer_agent" / END
        {
            "search_agent": "search_agent",
            "analyst_agent": "analyst_agent",
            "writer_agent": "writer_agent",
            END: END,
        },
    )

    # ── Search / Analyst 进入控制环；Writer 完成后结束 ──
    graph.add_conditional_edges(
        "search_agent",
        route_from_search,  # 新增路由函数
        {
            "analyst_agent": "analyst_agent",
            "supervisor": "supervisor",
        }
    )
    graph.add_edge("analyst_agent", "supervisor")
    graph.add_edge("writer_agent", END)

    return graph


def build_graph(checkpointer=None):
    """
        同步编译图。

        checkpointer=None 时自动使用 MemorySaver（适合 CLI / 冒烟测试）。
        FastAPI 生产路径请勿直接调用此函数，改用 streaming._get_app()（异步编译）。
        """
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        logger.warning(
            "[Graph] 使用 MemorySaver（仅调试用，无持久化）"
        )
    from .execution_fence import FencedCheckpointer
    return _build_state_graph().compile(checkpointer=FencedCheckpointer(checkpointer))


# ─────────────────────────────────────────────
# 便捷运行函数
# ─────────────────────────────────────────────
async def run_research(question: str, run_id: str | None = None) -> str:
    """
    端到端运行一次研究任务，返回最终报告。
    
    CLI 用法：
        report = asyncio.run(run_research("量子计算对密码学的影响？"))
        print(report)

    会复用 CheckpointerFactory 单例，与 FastAPI 生产路径共享同一 backend。
    """
    from .checkpointer import CheckpointerFactory

    checkpointer = await CheckpointerFactory.create()

    app = _build_state_graph().compile(checkpointer=checkpointer)

    resolved_run_id = normalize_run_id(run_id)
    config = checkpoint_config(resolved_run_id)
    await ensure_new_run(app, config)
    state = initial_state(question)
    from .streaming import research_config
    config = research_config(resolved_run_id, state)

    print(f"\n{'='*60}")
    print(f"开始研究：{question}")
    print(f"Run ID：{resolved_run_id}")
    print(f"{'='*60}\n")

    async for step in app.astream(state, config=config):
        node_name = list(step.keys())[0]
        logger.info(f"[Graph] 执行节点：{node_name}")

    final_state = await app.aget_state(config)
    report = final_state.values.get("final_report", "未生成报告")

    print(f"\n{'='*60}")
    print("研究完成")
    print(f"{'='*60}\n")
    return report


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # 快速冒烟测试（Day 5-7 用）
    result = asyncio.run(run_research("2026年以来国内AI教育行业有哪些重大进展？结合市场格局分析当前投资价值"))
    print(result)
