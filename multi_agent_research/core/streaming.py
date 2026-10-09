"""Map the current chapter graph to persisted SSE events."""

from __future__ import annotations
import asyncio
import logging
from typing import AsyncGenerator

from .graph import build_graph
from .checkpointer import CheckpointerFactory
from .run_context import checkpoint_config, ensure_new_run, normalize_run_id
from .state import CURRENT_WORKFLOW_VERSION, initial_state


logger = logging.getLogger(__name__)


def research_config(run_id: str, state: dict) -> dict:
    config = checkpoint_config(run_id)
    # Hard schema limits: four chapters, four searches and four drafts each.
    # Each search/draft has a review node, plus planner/advance/assembly.
    config["recursion_limit"] = 170  # Includes bounded, checkpointed Claim repair/gate pairs.
    return config


def _chapter_event(node: str, output: dict, run_id: str):
    if not isinstance(output, dict):
        return None
    if node == "report_review":
        return "report_review", {"run_id": run_id, "report_review": output["report_review"]}
    if node == "assemble_report":
        return "report_ready", {"run_id": run_id, **_parse_writer(output)}
    if node not in {
        "plan_sections", "section_search", "section_write", "section_review",
        "section_claims",
    }:
        return None
    sections = output.get("sections", [])
    # Serial updates carry the whole snapshot, so repository projection is idempotent.
    return ("section_plan" if node == "plan_sections" else "section_progress"), {
        "run_id": run_id, "stage": node, "sections": sections,
    }


def _stream_update(step):
    """Normalize parent and nested-subgraph update stream records."""
    if isinstance(step, tuple) and len(step) == 2:
        namespace, output = step
    else:
        namespace, output = (), step
    node_name = list(output.keys())[0]
    return namespace, node_name, output[node_name]


def _section_summary(values: dict) -> dict:
    return {
        "sections": values.get("sections", []),
        "report_quality": values.get("report_quality", "pending"),
        "model_calls": values.get("model_calls", 0),
        "usage_unknown_calls": values.get("usage_unknown_calls", 0),
        "report_review": values.get("report_review"),
    }


def _effective_snapshot_values(snapshot) -> dict:
    """Use the deepest persisted child state when a chapter subgraph is pending."""
    current = snapshot
    values = dict(snapshot.values)
    while True:
        nested = [
            task.state
            for task in getattr(current, "tasks", ())
            if hasattr(task.state, "values")
        ]
        if not nested:
            return values
        current = nested[0]
        values.update(current.values)


# ─────────────────────────────────────────────────────────────────────────────
# Graph 单例（跨请求复用，避免重复编译）
# ─────────────────────────────────────────────────────────────────────────────
_compiled_app = None
_app_lock: asyncio.Lock | None = None


def _get_app_lock() -> asyncio.Lock:
    global _app_lock
    if _app_lock is None:
        _app_lock = asyncio.Lock()
    return _app_lock


async def _get_app():
    """
    编译后的 Graph 单例。

    CheckpointerFactory 根据配置创建 Memory / SQLite / Postgres 后端；
    异步锁保证并发启动时只编译一次图。
    """
    global _compiled_app
    if _compiled_app is not None:
        return _compiled_app

    async with _get_app_lock():
        if _compiled_app is None:
            checkpointer = await CheckpointerFactory.create(require_durable=True)
            _compiled_app = build_graph(checkpointer=checkpointer)
    return _compiled_app


async def reset_app():
    """After shutdown, do not reuse a graph with an already closed saver."""
    global _compiled_app, _app_lock
    async with _get_app_lock():
        _compiled_app = None
    _app_lock = None


def _parse_writer(output: dict) -> dict:
    """Project deterministic report assembly output into an SSE payload."""
    report = output.get("final_report", "")
    return {
        "char_count": len(report),
        "preview":    report[:500],
        "writer_status": output.get("writer_status", "")
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主生成器
# ─────────────────────────────────────────────────────────────────────────────
async def astream_research(
    question: str,
    run_id: str | None = None,
    *,
    parent_context: dict | None = None,
) -> AsyncGenerator[tuple[str, dict], None]:
    """
    异步生成器 — 将 graph.astream() 映射为 (event_name, event_data) 序列。

    用法（FastAPI 路由中）：
        async for name, data in astream_research(question, run_id):
            yield _sse(name, data)

    节点到事件的映射：章节规划/进度、全篇审校、报告完成和最终快照。
    """
    resolved_run_id = normalize_run_id(run_id)
    app = await _get_app()
    config = checkpoint_config(resolved_run_id)
    await ensure_new_run(app, config)
    state = initial_state(question, parent_context=parent_context)
    config = research_config(resolved_run_id, state)

    # ── 开始事件 ──────────────────────────────────────────────────────────────
    yield "start", {
        "question": question,
        "run_id": resolved_run_id,
        "parent_run_id": (
            parent_context.get("source_run_id") if parent_context else None
        ),
        "resumed": False,
    }

    # ── 图执行流 ──────────────────────────────────────────────────────────────
    steps = app.astream(state, config=config, subgraphs=True)
    try:
        async for step in steps:
            _, node_name, node_output = _stream_update(step)

            # 跳过 LangGraph 内部节点（__start__ / __end__ 等）
            if node_name.startswith("__"):
                continue

            logger.debug(
                "[Stream] 节点完成：%s | keys=%s",
                node_name,
                list(node_output.keys()) if isinstance(node_output, dict) else [],
            )

            chapter_event = _chapter_event(node_name, node_output, resolved_run_id)
            if chapter_event:
                yield chapter_event
    finally:
        # Close nested graph generators before publishing a paused/terminal state.
        await steps.aclose()

    # ── 读取最终 state，组装 done 事件 ───────────────────────────────────────
    final_state = await app.aget_state(config)
    vals = final_state.values
    if vals.get("writer_status") != "complete":
        raise RuntimeError("chapter workflow ended without assembling a report")

    report = vals.get("final_report", "")
    yield "done", {
        "run_id":            resolved_run_id,
        "report":            report,
        "writer_status":     vals.get("writer_status", "not_started"),
        "char_count":        len(report),
        "total_iterations":  vals.get("iteration_count", 0),
        "total_results":     sum(len(s["results"]) for s in vals.get("sections", [])),
        "token_budget_used": vals.get("token_budget_used", 0),
        **_section_summary(vals),
    }


async def aresume_research(
    run_id: str,
    *, initial_input: dict | None = None,
) -> AsyncGenerator[tuple[str, dict], None]:
    """Continue a failed or interrupted run from its LangGraph checkpoint."""
    resolved_run_id = normalize_run_id(run_id)
    app = await _get_app()
    config = checkpoint_config(resolved_run_id)
    snapshot = await app.aget_state(config)
    if snapshot.values:
        has_pending_subgraph = any(
            getattr(task, "name", "") == "section_cycle"
            for task in getattr(snapshot, "tasks", ())
        )
        if has_pending_subgraph:
            expanded = await app.aget_state(config, subgraphs=True)
            values = (
                _effective_snapshot_values(expanded)
                if expanded.values
                else snapshot.values
            )
        else:
            values = snapshot.values
    else:
        values = {}
    if not values:
        if initial_input is not None:
            # The service proved no model call, node progress or artifact exists.
            # This is startup recovery, never a silent rerun of lost research.
            async for name, data in astream_research(initial_input["question"], run_id,
                                                    parent_context=initial_input.get("parent_context")):
                if name == "start":
                    data = {**data, "resumed": True, "reinitialized": True}
                yield name, data
            return
        raise LookupError(
            f"run '{resolved_run_id}' 缺少 Checkpoint；已有执行记录，拒绝静默重做，请恢复 Checkpoint 备份"
        )
    if values.get("workflow_version") != CURRENT_WORKFLOW_VERSION:
        raise RuntimeError(
            "Checkpoint 属于已移除的旧工作流，不能在当前图中恢复；请重新创建研究任务"
        )
    config = research_config(resolved_run_id, values)

    question = str(
        values.get(
            "research_question",
            values.get("question", ""),
        )
    )
    parent_context = values.get("parent_context")
    yield "start", {
        "question": question,
        "run_id": resolved_run_id,
        "parent_run_id": (
            parent_context.get("source_run_id") if parent_context else None
        ),
        "resumed": True,
    }

    # Repair a crash between checkpoint commit and the business/event store update.
    yield "section_snapshot", {
        "run_id": resolved_run_id, "stage": "resume",
        "sections": values.get("sections", []),
        "report_review": values.get("report_review"),
    }

    # A None input tells LangGraph to continue from the pending checkpoint
    # instead of creating a second state history for this business run.
    steps = app.astream(None, config=config, subgraphs=True)
    try:
        async for step in steps:
            _, node_name, node_output = _stream_update(step)
            if node_name.startswith("__"):
                continue

            logger.debug("[Stream] 恢复节点完成：%s", node_name)
            chapter_event = _chapter_event(node_name, node_output, resolved_run_id)
            if chapter_event:
                yield chapter_event
    finally:
        # Close nested graph generators before publishing a paused/terminal state.
        await steps.aclose()

    final_state = await app.aget_state(config)
    vals = final_state.values
    if vals.get("writer_status") != "complete":
        raise RuntimeError("chapter workflow ended without assembling a report")
    report = vals.get("final_report", "")
    yield "done", {
        "run_id": resolved_run_id,
        "report": report,
        "writer_status": vals.get("writer_status", "not_started"),
        "char_count": len(report),
        "total_iterations": vals.get("iteration_count", 0),
        "total_results": sum(len(s["results"]) for s in vals.get("sections", [])),
        "token_budget_used": vals.get("token_budget_used", 0),
        **_section_summary(vals),
    }
