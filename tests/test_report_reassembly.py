from datetime import datetime, timezone

from multi_agent_research.runs.models import RunRecord, RunStatus
from multi_agent_research.runs.service import RunService


class ReportStore:
    def __init__(self, run: RunRecord) -> None:
        self.run = run
        self.events: list[dict] = []

    async def get_run(self, run_id: str) -> RunRecord | None:
        return self.run if run_id == self.run.run_id else None

    async def replace_final_report(
        self,
        run_id: str,
        final_report: str,
        *,
        previous_sha256: str,
        new_sha256: str,
        report_quality: str,
    ) -> RunRecord:
        if self.run.final_report == final_report:
            return self.run
        self.run = self.run.model_copy(update={"final_report": final_report})
        self.events.append({
            "previous_sha256": previous_sha256,
            "new_sha256": new_sha256,
            "report_quality": report_quality,
            "model_calls": 0,
            "retrieval_calls": 0,
        })
        return self.run


def _run_immediate(coroutine):
    """Drive coroutines whose test doubles never suspend, avoiding a Windows loop fixture."""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("test coroutine unexpectedly suspended")


def test_reassemble_report_is_public_only_idempotent_and_zero_cost():
    now = datetime.now(timezone.utc)
    run = RunRecord(
        run_id="public-report",
        thread_id="public-report",
        question="研究新能源零部件变化",
        status=RunStatus.COMPLETED,
        created_at=now,
        updated_at=now,
        final_report="旧报告\n\n## 全篇审校待解决问题\n内部冲突清单",
        model_usage={"attempts": 9, "tokens": 81450, "unknown": 0},
        sections=[{
            "section_id": "section_1",
            "title": "需求变化",
            "question": "需求结构发生了什么变化",
            "status": "limited",
            "revision": 1,
            "draft": "需求结构正在改变，但全国外推仍需谨慎[来源1]。",
            "sources": [{
                "query": "需求变化",
                "source": "knowledge",
                "content": "样本显示需求结构变化",
                "score": 0.9,
                "iteration": 0,
                "metadata": {
                    "source": "D:/internal/行业报告.pdf",
                    "page": 8,
                    "chunk_id": "D:/internal/行业报告.pdf::8",
                },
            }],
            "limitations": ["section_1:v1:c1 修复预算已用完"],
        }],
        report_review={
            "verdict": "revise",
            "summary": "内部全篇审校摘要",
            "issues": [{
                "kind": "scope",
                "section_ids": ["section_1"],
                "detail": "内部冲突清单不应进入公开报告",
            }],
        },
    )
    store = ReportStore(run)
    service = RunService(store)  # type: ignore[arg-type]
    before_usage = dict(run.model_usage)

    updated = _run_immediate(service.reassemble_report(run.run_id))
    again = _run_immediate(service.reassemble_report(run.run_id))

    assert updated.final_report == again.final_report
    assert "需求结构正在改变" in updated.final_report
    assert "阅读提示" in updated.final_report
    assert "全篇审校待解决问题" not in updated.final_report
    assert "内部冲突清单" not in updated.final_report
    assert "section_1:v1:c1" not in updated.final_report
    assert "D:/internal" not in updated.final_report
    assert "行业报告 | p.8" in updated.final_report
    assert store.run.model_usage == before_usage
    assert len(store.events) == 1
    assert store.events[0]["model_calls"] == 0
    assert store.events[0]["retrieval_calls"] == 0
