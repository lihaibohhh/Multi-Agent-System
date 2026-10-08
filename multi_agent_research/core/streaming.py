"""
streaming.py — 多 Agent 研究系统的 SSE 事件流封装

将 graph.astream() 的原始节点输出转换为语义化事件序列，供 FastAPI SSE 端点消费。

新章节流程：start → section_plan → section_progress → report_ready → done。
恢复时增加 section_snapshot；异常由 RunService 写入 error 事件。

旧版事件序列：
  start              → 任务启动确认
  supervisor_decision → Supervisor 路由决策（迭代轮次 / 下一节点 / 指令摘要）
  search_complete    → 检索完成（新增条数 / 累计条数 / 查询列表）
  analyst_verdict    → Analyst 审查结论（verdict / confidence / gaps）
  report_ready       → 报告生成完成（字数 / 内容预览）
  done               → 全流程结束（完整报告 / 统计数据）
  error              → 异常事件（由 server.py 的 generate() 捕获后发出）
"""

from __future__ import annotations
import asyncio
import logging
from typing import AsyncGenerator

from .graph import build_graph
from .checkpointer import CheckpointerFactory
from .run_context import checkpoint_config, ensure_new_run, normalize_run_id
from .state import initial_state


logger = logging.getLogger(__name__)


def research_config(run_id: str, state: dict) -> dict:
    config = checkpoint_config(run_id)
    if state.get("workflow_version") in {2, 3}:
        # Hard schema limits: four chapters, four searches and four drafts each.
        # Each search/draft has a review node, plus planner/advance/assembly.
        config["recursion_limit"] = 170  # Includes bounded, checkpointed Claim repair/gate pairs.
    return config


def _chapter_event(node: str, output: dict, run_id: str):
    if node == "report_review":
        return "report_review", {"run_id": run_id, "report_review": output["report_review"]}
    if node == "assemble_report":
        return "report_ready", {"run_id": run_id, **_parse_writer(output)}
    if node not in {
        "plan_sections", "section_search", "section_analyze", "section_write", "section_review",
        "section_claims",
    }:
        return None
    sections = output.get("sections", [])
    # Serial updates carry the whole snapshot, so repository projection is idempotent.
    return ("section_plan" if node == "plan_sections" else "section_progress"), {
        "run_id": run_id, "stage": node, "sections": sections,
    }


def _section_summary(values: dict) -> dict:
    if values.get("workflow_version") not in {2, 3}:
        return {}
    return {
        "sections": values.get("sections", []),
        "report_quality": values.get("report_quality", "pending"),
        "model_calls": values.get("model_calls", 0),
        "usage_unknown_calls": values.get("usage_unknown_calls", 0),
        "report_review": values.get("report_review"),
    }


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


# ─────────────────────────────────────────────────────────────────────────────
# 节点输出解析器
# ─────────────────────────────────────────────────────────────────────────────
def _parse_supervisor(output: dict) -> dict:
    """
    解析 supervisor_node_llm_async 的返回 dict。

    supervisor 返回字段（直接读取，无需字符串解析）：
      next_agent             : str   下一个节点名
      iteration_count        : int   本轮迭代编号（已 +1）
      supervisor_instruction : str   给下一 Agent 的具体指令
      supervisor_reason      : str   本轮决策原因（调试 / 前端展示用）
    """
    return {
        "iteration":   output.get("iteration_count", 0),
        "next":        output.get("next_agent", ""),
        "reason":      output.get("supervisor_reason", ""),
    }


def _parse_search(output: dict) -> dict:
    """
    解析 search_agent_node 的返回 dict。

    search_agent 写入 events 列表（Reducer 追加）：
      {"type": "SearchCompleted", "payload": {"new_count": N, "total_count": M, "queries": [...]}}

    astream 在 updates 模式下返回本轮新写入的 events（delta），不含历史。
    """
    payload: dict = {}
    for e in reversed(output.get("events", [])):
        if isinstance(e, dict) and e.get("type") == "SearchCompleted":
            payload = e.get("payload", {})
            break

    return {
        "new_count":   payload.get("new_count", 0),
        "total_count": payload.get("total_count", 0),
        "queries":     payload.get("queries_used", []),
    }


def _parse_analyst(output: dict) -> dict | None:
    """
    解析 analyst_agent_node 的返回 dict。

    analyst_agent 写入 analyst_verdict，可能是 Pydantic model 或 dict：
      verdict          : "pass" | "revise" | "reject"
      reason           : str
      specific_gaps    : list[str]
      confidence_score : float
    """
    verdict = output.get("analyst_verdict")
    if not verdict:
        return None

    # 兼容 Pydantic model（with_structured_output 返回）和 dict 两种形式
    _get = verdict.get if isinstance(verdict, dict) else lambda k, d=None: getattr(verdict, k, d)

    if _get("verdict", "") == "not_reviewed":
        return None

    return {
        "verdict":    _get("verdict", ""),
        "confidence": _get("confidence_score", 0.0),
        "gaps":       _get("specific_gaps", []),
        "reason":     _get("reason", ""),
    }


def _parse_writer(output: dict) -> dict:
    """
    解析 writer_agent_node 的返回 dict。

    writer_agent 写入：
      final_report  : str  LLM 正文 + 代码生成的参考来源节
    """
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

    节点到事件的映射：
        supervisor    → supervisor_decision
        search_agent  → search_complete
        analyst_agent → analyst_verdict（analyst_verdict 为空时跳过）
        writer_agent  → report_ready
        图终止后      → done（读取最终 state 后发出）
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
    steps = app.astream(state, config=config)
    try:
        async for step in steps:
            node_name = list(step.keys())[0]

            # 跳过 LangGraph 内部节点（__start__ / __end__ 等）
            if node_name.startswith("__"):
                continue

            node_output = step[node_name]
            logger.debug("[Stream] 节点完成：%s | keys=%s", node_name, list(node_output.keys()))

            chapter_event = _chapter_event(node_name, node_output, resolved_run_id)
            if chapter_event:
                yield chapter_event
                continue

            if node_name == "supervisor":
                yield "supervisor_decision", {
                    "run_id": resolved_run_id,
                    **_parse_supervisor(node_output),
                }

            elif node_name == "search_agent":
                yield "search_complete", {
                    "run_id": resolved_run_id,
                    **_parse_search(node_output),
                }

            elif node_name == "analyst_agent":
                parsed = _parse_analyst(node_output)
                if parsed:
                    yield "analyst_verdict", {"run_id": resolved_run_id, **parsed}

            elif node_name == "writer_agent":
                yield "report_ready", {
                    "run_id": resolved_run_id,
                    **_parse_writer(node_output),
                }
    finally:
        # Close nested graph generators before publishing a paused/terminal state.
        await steps.aclose()

    # ── 读取最终 state，组装 done 事件 ───────────────────────────────────────
    final_state = await app.aget_state(config)
    vals = final_state.values
    if vals.get("workflow_version") in {2, 3} and vals.get("writer_status") != "complete":
        raise RuntimeError("chapter workflow ended without assembling a report")

    report = vals.get("final_report", "")
    yield "done", {
        "run_id":            resolved_run_id,
        "report":            report,
        "writer_status":     vals.get("writer_status", "not_started"),
        "char_count":        len(report),
        "total_iterations":  vals.get("iteration_count", 0),
        "total_results":     sum(len(s["results"]) for s in vals.get("sections", []))
                             if vals.get("workflow_version") in {2, 3}
                             else len(vals.get("search_results", [])),
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
    if not snapshot.values:
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
    config = research_config(resolved_run_id, snapshot.values)

    question = str(
        snapshot.values.get(
            "research_question",
            snapshot.values.get("question", ""),
        )
    )
    parent_context = snapshot.values.get("parent_context")
    yield "start", {
        "question": question,
        "run_id": resolved_run_id,
        "parent_run_id": (
            parent_context.get("source_run_id") if parent_context else None
        ),
        "resumed": True,
    }

    if snapshot.values.get("workflow_version") in {2, 3}:
        # Repair a crash between checkpoint commit and the business/event store update.
        yield "section_snapshot", {
            "run_id": resolved_run_id, "stage": "resume",
            "sections": snapshot.values.get("sections", []),
            "report_review": snapshot.values.get("report_review"),
        }

    # A None input tells LangGraph to continue from the pending checkpoint
    # instead of creating a second state history for this business run.
    steps = app.astream(None, config=config)
    try:
        async for step in steps:
            node_name = list(step.keys())[0]
            if node_name.startswith("__"):
                continue

            node_output = step[node_name]
            logger.debug("[Stream] 恢复节点完成：%s", node_name)
            chapter_event = _chapter_event(node_name, node_output, resolved_run_id)
            if chapter_event:
                yield chapter_event
                continue
            if node_name == "supervisor":
                yield "supervisor_decision", {
                    "run_id": resolved_run_id,
                    **_parse_supervisor(node_output),
                }
            elif node_name == "search_agent":
                yield "search_complete", {
                    "run_id": resolved_run_id,
                    **_parse_search(node_output),
                }
            elif node_name == "analyst_agent":
                parsed = _parse_analyst(node_output)
                if parsed:
                    yield "analyst_verdict", {"run_id": resolved_run_id, **parsed}
            elif node_name == "writer_agent":
                yield "report_ready", {
                    "run_id": resolved_run_id,
                    **_parse_writer(node_output),
                }
    finally:
        # Close nested graph generators before publishing a paused/terminal state.
        await steps.aclose()

    final_state = await app.aget_state(config)
    vals = final_state.values
    if vals.get("workflow_version") in {2, 3} and vals.get("writer_status") != "complete":
        raise RuntimeError("chapter workflow ended without assembling a report")
    report = vals.get("final_report", "")
    yield "done", {
        "run_id": resolved_run_id,
        "report": report,
        "writer_status": vals.get("writer_status", "not_started"),
        "char_count": len(report),
        "total_iterations": vals.get("iteration_count", 0),
        "total_results": sum(len(s["results"]) for s in vals.get("sections", []))
                         if vals.get("workflow_version") in {2, 3}
                         else len(vals.get("search_results", [])),
        "token_budget_used": vals.get("token_budget_used", 0),
        **_section_summary(vals),
    }
