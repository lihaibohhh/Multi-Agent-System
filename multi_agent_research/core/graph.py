"""Build the single supported, checkpointed chapter research graph."""

from __future__ import annotations
import asyncio
import logging
from langgraph.graph import StateGraph, END

from .state import ResearchState, initial_state
from .run_context import checkpoint_config, ensure_new_run, normalize_run_id
from ..sections import workflow as chapters
from ..sections.subgraph import compile_section_subgraph


logger = logging.getLogger(__name__)


def _build_state_graph() -> StateGraph:
    """Build the parent workflow around one checkpointed serial chapter subgraph."""
    graph = StateGraph(ResearchState)

    graph.add_node("plan_sections", chapters.plan_sections)
    graph.add_node("section_cycle", compile_section_subgraph())
    graph.add_node("report_review", chapters.review_report)
    graph.add_node("assemble_report", chapters.assemble_sections)
    graph.set_entry_point("plan_sections")

    parent_routes = {
        "section_cycle": "section_cycle",
        "report_review": "report_review",
        "assemble_report": "assemble_report",
    }
    graph.add_conditional_edges("plan_sections", chapters.route_parent, parent_routes)
    graph.add_conditional_edges("section_cycle", chapters.route_parent, parent_routes)
    graph.add_conditional_edges("report_review", chapters.route_parent, parent_routes)
    graph.add_edge("assemble_report", END)

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
