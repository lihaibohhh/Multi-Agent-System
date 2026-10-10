"""Run lifecycle orchestration independent from SSE client connections."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from ..agents.context import agent_checkpoint_loader
from ..agents.events import agent_event_sink
from ..core.run_context import normalize_run_id, normalize_session_id
from ..core.execution_fence import execution_fence
from ..coordination.runtime import coordination_snapshot_loader
from ..core.streaming import aresume_research, astream_research
from ..sections.artifacts import dependency_issues, parent_handoff, revision_sections
from ..sections.model_output import attempt_sink
from ..sections.rendering import assemble_report
from .models import (
    AgentExecutionRecord,
    AgentLifecycleEventRecord,
    AgentTraceEvent,
    ParentContextSnapshot,
    RunEventRecord,
    RunRecord,
    RunUsageSummary,
    RunStatus,
    SessionRecord,
    SessionTimeline,
    TERMINAL_RUN_STATUSES,
)
from .repository import RunConflictError, RunNotFoundError, RunStore, StaleExecutionError
from .runtime import InstanceUnavailableError
from ..core.budget import (BudgetExceeded, RunBudget, RunControlError, current_budget, remaining_seconds,
                          ExecutionPaused, ExecutionTimeLimit, check_available)
from ..core.config import settings


logger = logging.getLogger(__name__)


def _public_agent_event_details(event: AgentLifecycleEventRecord) -> dict:
    """Allow-list trace metadata; never copy arbitrary event details."""
    details = event.details
    if event.event_type in {"agent_started", "agent_model_called"}:
        return {"model_ref": details.get("model_ref")}
    if event.event_type.startswith("agent_tool_"):
        safe = {
            "tool_name": details.get("tool_name"),
            "tool_call_id": details.get("tool_call_id"),
        }
        if event.event_type != "agent_tool_started":
            safe["duration_ms"] = max(0, int(details.get("duration_ms", 0)))
        if event.event_type == "agent_tool_failed":
            safe["error_type"] = details.get("error_type")
        return safe
    safe = {}
    if "turns" in details:
        safe["turns"] = max(0, int(details["turns"]))
    if "error_type" in details:
        safe["error_type"] = details.get("error_type")
    checkpoint = details.get("checkpoint")
    if isinstance(checkpoint, dict):
        usage = checkpoint.get("usage") if isinstance(checkpoint.get("usage"), dict) else {}
        safe["checkpoint"] = {
            "usage": {
                key: max(0, int(usage.get(key, 0)))
                for key in ("tokens", "unknown", "attempts")
            },
            "has_local_state": bool(checkpoint.get("local_state")),
            "has_handoff": checkpoint.get("handoff") is not None,
            "unresolved_count": len(checkpoint.get("unresolved") or []),
            "state_discarded": bool(checkpoint.get("state_discarded", False)),
        }
    return safe
_PARENT_REPORT_LIMIT = 12_000
_PARENT_REFERENCES_LIMIT = 4_000


def _extract_reference_excerpt(report: str) -> str:
    markers = ("## 参考来源", "## 参考资料", "## References")
    positions = [report.rfind(marker) for marker in markers]
    start = max(positions)
    if start < 0:
        return ""
    return report[start:start + _PARENT_REFERENCES_LIMIT]


class RunService:
    def __init__(self, repository: RunStore, *, poll_interval: float = 0.25, heartbeat_interval: float = 5.0) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        self._repository = repository
        self._poll_interval = poll_interval
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._executions: dict[str, str] = {}
        self._heartbeat_interval = heartbeat_interval
        self._monitor: asyncio.Task | None = None
        self._closing = False
        self._ownership_lost = False
        self._reconcile_ok = True
        self._search_slots = asyncio.Semaphore(settings.agent.retrieval_concurrency)
        self._pause_signals: dict[str, asyncio.Event] = {}

    async def start_runtime(self):
        if self._monitor and not self._monitor.done():
            raise RuntimeError("runtime monitor is already running")
        await self._repository.assert_instance_owner()
        await self.recover_stale_runs()
        self._closing = False
        self._ownership_lost = False
        self._monitor = asyncio.create_task(self._watch_runtime(), name="research-runtime-monitor")

    async def runtime_health(self) -> dict:
        owned = False
        if not self._ownership_lost and not self._closing:
            try:
                await self._repository.assert_instance_owner()
                owned = True
            except Exception:
                pass
        return {"mode": "single_instance", "ownership": "ok" if owned else "unavailable",
                "reconciliation": "ok" if self._reconcile_ok else "error",
                "monitor": "ok" if self._monitor and not self._monitor.done() else "stopped"}

    async def _watch_runtime(self):
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            try:
                await self._repository.assert_instance_owner()
            except Exception:
                self._ownership_lost = True
                for task in list(self._tasks.values()):
                    task.cancel()
                logger.error("[RunService] 实例所有权失去，已停止执行；需检查数据库并重启")
                return
            try:
                async with asyncio.timeout(15):
                    await self.recover_stale_runs()
                self._reconcile_ok = True
            except Exception as exc:
                self._reconcile_ok = False
                logger.error("[RunService] 对账失败，将在下次心跳重试：%s", type(exc).__name__)

    async def recover_stale_runs(self) -> list[str]:
        async with self._lock:
            active = {key: self._executions[key] for key, task in self._tasks.items()
                      if not task.done() and key in self._executions}
            return await self._repository.reconcile_running(active)

    async def create_session(
        self,
        *,
        title: str,
        session_id: str | None = None,
    ) -> SessionRecord:
        resolved_session_id = normalize_session_id(session_id)
        return await self._repository.create_session(
            session_id=resolved_session_id,
            title=title.strip(),
        )

    async def get_session(self, session_id: str) -> SessionRecord:
        resolved_session_id = normalize_session_id(session_id)
        record = await self._repository.get_session(resolved_session_id)
        if record is None:
            raise RunNotFoundError(resolved_session_id)
        return record

    async def get_session_timeline(self, session_id: str) -> SessionTimeline:
        session = await self.get_session(session_id)
        runs = await self._repository.list_session_runs(session.session_id)
        return SessionTimeline(session=session, runs=runs)

    async def create_run(
        self,
        *,
        question: str,
        session_id: str | None = None,
        parent_run_id: str | None = None,
        run_id: str | None = None,
        parent_section_ids: list[str] | None = None,
        _revision_target: str | None = None,
        _revision_instruction: str = "",
    ) -> RunRecord:
        resolved_run_id = normalize_run_id(run_id)
        resolved_session_id: str
        parent_context: ParentContextSnapshot | None = None

        if parent_run_id:
            parent = await self.get_run(parent_run_id)
            if parent.status != RunStatus.COMPLETED or not parent.final_report:
                raise RunConflictError(
                    f"parent run '{parent_run_id}' must be completed before it can be inherited"
                )
            if any(s.status not in {"complete", "limited"} for s in parent.sections):
                raise RunConflictError("父 Run 是阶段产物，尚有未完成章节；请使用选章继续/补证据操作")
            if not parent.session_id:
                raise RunConflictError(
                    f"parent run '{parent_run_id}' is not attached to a session"
                )
            resolved_session_id = normalize_session_id(session_id or parent.session_id)
            if resolved_session_id != parent.session_id:
                raise RunConflictError("parent and child runs must belong to the same session")
            report = parent.final_report
            parent_context = ParentContextSnapshot(
                schema_version=2 if parent.sections else 1,
                source_run_id=parent.run_id,
                source_question=parent.question,
                report_excerpt=report[:_PARENT_REPORT_LIMIT],
                reference_excerpt=_extract_reference_excerpt(report),
                report_truncated=len(report) > _PARENT_REPORT_LIMIT,
                captured_at=datetime.now(timezone.utc),
                handoff=parent_handoff(parent.sections, parent_section_ids),
                report_review=parent.report_review.model_copy(deep=True) if parent.report_review else None,
            )
            if _revision_target:
                parent_context.revision_sections = revision_sections(
                    parent.sections, _revision_target, _revision_instruction,
                )
                parent_context.revision_target = _revision_target
                # Never feed the superseded whole report/claims back as current evidence.
                parent_context.handoff = []
                parent_context.report_excerpt = ""
                parent_context.reference_excerpt = ""
                parent_context.report_review = None
        else:
            if parent_section_ids is not None or _revision_target:
                raise ValueError("parent selection requires parent_run_id")
            resolved_session_id = normalize_session_id(session_id)

        session = await self._repository.get_session(resolved_session_id)
        if session is None:
            if session_id or parent_run_id:
                raise RunNotFoundError(resolved_session_id)
            session = await self.create_session(
                session_id=resolved_session_id,
                title=question.strip()[:200],
            )

        record = await self._repository.create_run(
            run_id=resolved_run_id,
            session_id=session.session_id,
            parent_run_id=parent_run_id,
            parent_context=parent_context,
            question=question,
        )
        await self._repository.append_event(
            resolved_run_id,
            "run_created",
            {
                "run_id": resolved_run_id,
                "session_id": session.session_id,
                "parent_run_id": parent_run_id,
                "status": RunStatus.CREATED.value,
            },
        )
        return record

    async def create_section_revision(
        self, run_id: str, section_id: str, *, instruction: str, new_run_id: str | None = None,
    ) -> RunRecord:
        """Create, but do not execute, an immutable report revision as a child Run."""
        if not 5 <= len(instruction.strip()) <= 2000:
            raise ValueError("revision instruction must contain 5–2000 characters")
        parent = await self.get_run(run_id)
        return await self.create_run(
            question=parent.question, parent_run_id=parent.run_id,
            run_id=new_run_id, _revision_target=section_id,
            _revision_instruction=instruction.strip(),
        )

    async def create_section_operation(self, run_id, section_id, *, mode, instruction, new_run_id=None):
        from ..sections.operations import prepare_operation
        if not 5 <= len(instruction.strip()) <= 2000:
            raise ValueError("操作说明需包含 5–2000 个字符")
        async with self._lock:
            await self._repository.assert_instance_owner()
            snapshot = await self._repository.get_snapshot(run_id)
            parent = snapshot.run
            if parent.status not in {RunStatus.COMPLETED, RunStatus.PAUSED, RunStatus.FAILED,
                                     RunStatus.INTERRUPTED, RunStatus.BUDGET_LIMITED}:
                raise RunConflictError("请先等待任务停止；不允许从执行中的章节创建局部操作")
            if not parent.session_id:
                raise RunConflictError("父研究缺少 Session")
            # A failed projection write can lag behind the graph checkpoint.
            # For unfinished Runs use the stopped checkpoint, not a stale UI view.
            from ..core import streaming
            source_state = {}
            if parent.status != RunStatus.COMPLETED:
                app = await streaming._get_app()
                saved = await app.aget_state(streaming.research_config(parent.run_id, {}))
                source_state = saved.values or {}
                if not source_state.get("sections"):
                    raise RunConflictError("未找到可用章节 Checkpoint；不能从不完整展示快照创建继续操作")
            sections, operation = prepare_operation(source_state.get("sections") or parent.sections,
                                                     section_id, mode, instruction.strip())
            operation["source_cursor"] = snapshot.cursor
            from ..sections.models import SectionPolicy
            previous = source_state.get("parent_context") or (parent.parent_context.model_dump() if parent.parent_context else {})
            previous_op = previous.get("section_operation") or {}
            saved_policy = (source_state.get("section_policy") or previous_op.get("section_policy")) if mode == "continue" else None
            operation["section_policy"] = SectionPolicy.model_validate(saved_policy or {
                "max_search_rounds": settings.agent.section_max_search_rounds,
                "max_revisions": settings.agent.section_max_revisions,
            }).model_dump()
            if mode == "continue":
                operation["retrieval_context"] = (previous_op.get("retrieval_context") if previous_op.get("mode") == "continue"
                    else {"source_question": previous.get("source_question", "")} if previous else None)
            context = ParentContextSnapshot(schema_version=3, source_run_id=parent.run_id,
                source_question=parent.question, report_excerpt="", report_truncated=False,
                captured_at=datetime.now(timezone.utc), revision_target=section_id,
                revision_sections=sections, section_operation=operation)
            child = await self._repository.create_run(run_id=normalize_run_id(new_run_id),
                session_id=parent.session_id, parent_run_id=parent.run_id, parent_context=context,
                question=parent.question, parent_snapshot_cursor=snapshot.cursor)
            await self._repository.append_event(child.run_id, "run_created", {
                "run_id": child.run_id, "parent_run_id": parent.run_id, "status": "created",
                "section_operation": operation, "budget_id": child.budget_id})
            return child

    async def get_run(self, run_id: str) -> RunRecord:
        record = await self._repository.get_run(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        return record

    async def get_snapshot(self, run_id: str):
        return await self._repository.get_snapshot(run_id)

    async def get_run_usage(self, run_id: str) -> RunUsageSummary:
        await self.get_run(run_id)
        return await self._repository.get_run_usage(run_id)

    async def reassemble_report(self, run_id: str) -> RunRecord:
        """Rebuild the public report from persisted, reviewed artifacts without AI calls."""
        run = await self.get_run(run_id)
        if run.status != RunStatus.COMPLETED:
            raise RunConflictError("only a completed run can be reassembled")
        if not run.sections or run.report_review is None:
            raise RunConflictError("run has no reviewed section artifacts to reassemble")
        if any(section.status not in {"complete", "limited"} for section in run.sections):
            raise RunConflictError("run contains unfinished sections")
        if dependency_issues(run.sections):
            raise RunConflictError("run contains stale section dependencies")

        limited = (
            any(section.status == "limited" for section in run.sections)
            or run.report_review.verdict != "pass"
        )
        report = assemble_report(run.question, run.sections, limited=limited)
        previous = run.final_report or ""
        return await self._repository.replace_final_report(
            run_id,
            report,
            previous_sha256=hashlib.sha256(previous.encode("utf-8")).hexdigest(),
            new_sha256=hashlib.sha256(report.encode("utf-8")).hexdigest(),
            report_quality="limited" if limited else "reviewed",
        )

    async def list_agent_executions(self, run_id: str) -> list[AgentExecutionRecord]:
        await self.get_run(run_id)
        return await self._repository.list_agent_executions(run_id)

    async def list_agent_events(
        self,
        run_id: str,
        *,
        after: int = 0,
    ) -> list[AgentLifecycleEventRecord]:
        await self.get_run(run_id)
        return await self._repository.list_agent_events(run_id, after)

    async def list_agent_trace(
        self,
        run_id: str,
        *,
        after: int = 0,
    ) -> list[AgentTraceEvent]:
        """Return a public trace without prompts, tool payloads, or checkpoints."""
        await self.get_run(run_id)
        executions, events = await asyncio.gather(
            self._repository.list_agent_executions(run_id),
            self._repository.list_agent_events(run_id, after),
        )
        identities = {record.agent_run_id: record for record in executions}
        trace = []
        for event in events:
            identity = identities[event.agent_run_id]
            trace.append(AgentTraceEvent(
                sequence=event.sequence,
                event_id=event.event_id,
                agent_run_id=event.agent_run_id,
                parent_agent_run_id=identity.parent_agent_run_id,
                agent_name=identity.agent_name,
                agent_version=identity.agent_version,
                section_id=identity.section_id,
                event_type=event.event_type,
                turn=event.turn,
                details=_public_agent_event_details(event),
                occurred_at=event.occurred_at,
            ))
        return trace

    async def pause_run(self, run_id: str):
        async with self._lock:
            await self._repository.assert_instance_owner()
            record = await self._repository.request_pause(run_id)
            if run_id in self._pause_signals:
                self._pause_signals[run_id].set()
            return record

    async def migrate_run_budget(self, run_id: str, *, confirm: bool, reason: str):
        if not confirm or len(reason.strip()) < 5:
            raise ValueError("迁移需明确确认并提供原因；不会清空费用或自动执行")
        async with self._lock:
            await self._repository.assert_instance_owner()
            return await self._repository.migrate_run_budget(run_id, reason.strip())

    async def increase_run_budget(self, run_id: str, **kwargs):
        from .models import BudgetIncreaseRequest
        request = BudgetIncreaseRequest.model_validate(kwargs)
        async with self._lock:
            await self._repository.assert_instance_owner()
            return await self._repository.increase_run_budget(run_id, request)

    async def start_run(self, run_id: str, *, resume: bool = False) -> RunRecord:
        async with self._lock:
            if self._closing or self._ownership_lost:
                raise InstanceUnavailableError("服务正在关闭或已失去执行所有权，请重启服务")
            await self._repository.assert_instance_owner()
            existing_task = self._tasks.get(run_id)
            if existing_task is not None and not existing_task.done():
                return await self.get_run(run_id)

            current = await self.get_run(run_id)
            if not current.budget_id or current.budget.get("version") != 2:
                raise RunConflictError("旧版预算需显式迁移后继续；历史消耗与未知预留不会清零")
            try:
                check_available(current.budget)
            except BudgetExceeded as exc:
                raise RunConflictError(str(exc)) from exc

            expected = (
                (RunStatus.INTERRUPTED, RunStatus.FAILED, RunStatus.PAUSED, RunStatus.BUDGET_LIMITED)
                if resume
                else (RunStatus.CREATED,)
            )
            record = await self._repository.begin_execution(run_id, expected, resume=resume)
            self._pause_signals[run_id] = asyncio.Event()
            coroutine = self._execute(record, resume=resume)
            try:
                task = asyncio.create_task(coroutine, name=f"research-run:{run_id}")
            except BaseException:
                coroutine.close()
                self._pause_signals.pop(run_id, None)
                try:
                    await self._repository.finish_execution(run_id, record.execution_id, RunStatus.INTERRUPTED,
                        {"reason": "background_task_creation_failed"})
                except Exception:
                    logger.error("[RunService] 启动补偿未持久化，等待后台对账")
                raise
            self._tasks[run_id] = task
            self._executions[run_id] = record.execution_id
            task.add_done_callback(lambda finished: self._task_finished(run_id, finished))
        return record

    def _task_finished(self, run_id, task):
        if self._tasks.get(run_id) is task:
            self._tasks.pop(run_id, None)
            self._executions.pop(run_id, None)
            self._pause_signals.pop(run_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("[RunService] 后台执行未能保存终态，将由对账恢复：%s", run_id)

    async def _execute(self, record: RunRecord, *, resume: bool) -> None:
        run_id = record.run_id
        async def save_attempt(data):
            await self._repository.record_model_attempt(run_id, record.execution_id, data)
        async def save_agent_event(event):
            await self._repository.record_agent_event(
                run_id,
                record.execution_id,
                event,
            )
        async def load_agent_checkpoint(agent_name, section_id):
            previous = await self._repository.get_latest_agent_execution(
                run_id,
                agent_name,
                section_id,
            )
            if previous is None:
                return None
            return {
                "agent_run_id": previous.agent_run_id,
                "status": previous.status.value,
                "local_state": previous.local_state,
                "handoff": previous.handoff,
                "unresolved": previous.unresolved,
            }
        audit_token = attempt_sink.set(save_attempt)
        agent_event_token = agent_event_sink.set(save_agent_event)
        checkpoint_loader_token = agent_checkpoint_loader.set(load_agent_checkpoint)
        fence_token = execution_fence.set((self._repository, run_id, record.execution_id))
        async def load_coordination_snapshot():
            method = getattr(self._repository, "get_coordination_snapshot", None)
            return await method(run_id) if method is not None else None

        coordination_token = coordination_snapshot_loader.set(
            load_coordination_snapshot
        )
        budget_token = None
        try:
            budget_token = current_budget.set(RunBudget(self._repository, record, self._search_slots))
            timer = asyncio.timeout(remaining_seconds(record.budget))
            try:
                async with timer:
                    completed_payload = await self._run_graph(record, resume=resume)
            except TimeoutError:
                if timer.expired():
                    raise ExecutionTimeLimit("本次执行时限已到，可继续；累计消耗不会重置") from None
                raise
            await self._repository.finish_execution(
                run_id, record.execution_id, RunStatus.COMPLETED, completed_payload,
            )
        except ExecutionPaused as exc:
            from ..core.retrieval import RetrievalDeferred
            from ..sections.claim_repair import ClaimsPending
            reason = ("claims_pending" if isinstance(exc, ClaimsPending) else
                      "dependency_unavailable" if isinstance(exc, RetrievalDeferred) else
                      "execution_timeout" if isinstance(exc, ExecutionTimeLimit) else "user_pause")
            await self._repository.finish_execution(run_id, record.execution_id, RunStatus.PAUSED,
                {"message": str(exc), "reason": reason,
                 **({"stage": "section_claims", "section_id": exc.section_id} if isinstance(exc, ClaimsPending) else {})})
        except RunControlError as exc:
            await self._repository.finish_execution(
                run_id, record.execution_id, RunStatus.BUDGET_LIMITED if isinstance(exc, BudgetExceeded) else RunStatus.FAILED,
                {"run_id": run_id, "type": type(exc).__name__, "message": str(exc),
                 "reason": "budget_exhausted" if isinstance(exc, BudgetExceeded) else "call_stopped",
                 **({"budget_block": exc.details} if isinstance(exc, BudgetExceeded) else {})},
            )
        except StaleExecutionError:
            logger.warning("[RunService] 旧执行已失效，停止写入：%s", run_id)
        except asyncio.CancelledError:
            await self._repository.finish_execution(
                run_id, record.execution_id, RunStatus.INTERRUPTED,
                {"run_id": run_id, "reason": "ownership_lost" if self._ownership_lost else "service_shutdown"},
            )
            raise
        except Exception as exc:
            logger.error("[RunService] run %s failed: %s", run_id, exc, exc_info=True)
            await self._repository.finish_execution(
                run_id, record.execution_id, RunStatus.FAILED,
                {
                    "run_id": run_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        finally:
            if budget_token is not None:
                current_budget.reset(budget_token)
            agent_event_sink.reset(agent_event_token)
            agent_checkpoint_loader.reset(checkpoint_loader_token)
            attempt_sink.reset(audit_token)
            execution_fence.reset(fence_token)
            coordination_snapshot_loader.reset(coordination_token)

    async def _run_graph(self, record, *, resume):
        run_id = record.run_id
        parent = record.parent_context.model_dump(mode="json") if record.parent_context else None
        initial_input = None
        if resume and await self._repository.can_initialize_missing_checkpoint(run_id):
            initial_input = {"question": record.question, "parent_context": parent}
        stream = (aresume_research(run_id, initial_input=initial_input) if resume
                  else astream_research(record.question, run_id, parent_context=parent))
        completed_payload = None
        try:
            async for event_type, payload in stream:
                payload = {**payload, "execution_id": record.execution_id}
                if event_type == "done":
                    completed_payload = payload
                else:
                    await self._repository.publish_execution_event(run_id, record.execution_id, event_type, payload)
                    signal = self._pause_signals.get(run_id)
                    if signal is not None and signal.is_set():
                        raise ExecutionPaused("已在步骤边界暂停，累计预算与已保存章节保留")
        finally:
            await stream.aclose()
        if completed_payload is None:
            raise RuntimeError("graph finished without a done event")
        return completed_payload

    async def iter_events(
        self,
        run_id: str,
        *,
        after: int = 0,
    ) -> AsyncGenerator[RunEventRecord, None]:
        await self.get_run(run_id)
        cursor = after
        while True:
            events = await self._repository.list_events(run_id, after=cursor)
            for event in events:
                cursor = event.sequence
                yield event

            record = await self.get_run(run_id)
            if record.status in TERMINAL_RUN_STATUSES or record.status in {
                RunStatus.FAILED,
                RunStatus.INTERRUPTED,
                RunStatus.PAUSED,
                RunStatus.BUDGET_LIMITED,
            }:
                trailing = await self._repository.list_events(run_id, after=cursor)
                for event in trailing:
                    cursor = event.sequence
                    yield event
                return
            await asyncio.sleep(self._poll_interval)

    async def shutdown(self) -> None:
        self._closing = True
        had_monitor = self._monitor is not None
        if self._monitor is not None:
            self._monitor.cancel()
            await asyncio.gather(self._monitor, return_exceptions=True)
            self._monitor = None
        async with self._lock:
            tasks = [task for task in self._tasks.values() if not task.done()]
            should_reconcile = had_monitor or bool(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._executions.clear()
        # Failed startup (e.g. missing durable storage) must not alter old Runs.
        if should_reconcile and not self._ownership_lost:
            try:
                async with asyncio.timeout(15):
                    await self.recover_stale_runs()
            except Exception as exc:
                logger.warning("[RunService] 关闭时未能完成对账；下次成功启动后继续：%s", type(exc).__name__)
