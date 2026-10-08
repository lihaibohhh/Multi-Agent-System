"""Explicit local work plans. No mutation of the source Run/checkpoint."""

from .models import SectionRecord
from .artifacts import validate_dependencies
from .rendering import merge_results


FINISHED = {"complete", "limited"}


def prepare_operation(sections, target, mode, instruction):
    if mode not in {"continue", "supplement", "refresh"}:
        raise ValueError("unknown section operation")
    copied = [SectionRecord.model_validate(s).model_copy(deep=True) for s in sections]
    if not copied or len(copied) > 4:
        raise ValueError("需要已保存的章节计划（1–4章）")
    by_id = {s.section_id: s for s in copied}
    if target not in by_id:
        raise ValueError("unknown section_id")
    for i, section in enumerate(copied):
        if section.artifact_version < 2:
            section.depends_on = [s.section_id for s in copied[:i]]
            section.dependency_revisions = {s.section_id: s.revision for s in copied[:i]}
    validate_dependencies(copied)
    selected = by_id[target]
    if mode == "continue" and selected.status in FINISHED:
        raise ValueError("该章已完成；请选择补证据或刷新来源，不重复继续已完成章节")
    unfinished = [dep for dep in selected.depends_on if by_id[dep].status not in FINISHED]
    if unfinished:
        raise ValueError("请先完成前置章节：" + ", ".join(unfinished))
    affected = {target}
    work = [target]
    for section in copied:
        if section.section_id == target:
            if mode != "continue":
                section.revision_base = section.revision
                section.search_rounds = 0
                section.results = (merge_results(section.results, section.sources) if mode == "supplement" else [])
                section.gaps = [instruction[:500]]
                section.analyst = {}
                section.review = None
                section.limitations = []
                section.evidence_update = {}
                section.status = "stale"
            section.revision_instruction = instruction
        elif affected.intersection(section.depends_on):
            affected.add(section.section_id)
            section.status = "stale"
            section.invalidated_by = [target]
            section.revision_instruction = f"依赖章节 {target} 更新，须重新核查本章结论及证据。"
            section.revision_base = section.revision
            section.search_rounds = 0
            section.results = []
            section.gaps = []
            section.analyst = {}
            section.review = None
            section.limitations = []
            section.dependency_revisions = {}
            # Never pull an unrelated unfinished prerequisite into this task.
            if mode != "supplement" and all(by_id[d].status in FINISHED or d in work for d in section.depends_on):
                work.append(section.section_id)
    return copied, {"mode": mode, "target": target, "work_ids": work,
                    "affected_ids": [s.section_id for s in copied if s.section_id in affected]}


def continue_step(section, policy):
    if section.status == "claims_pending" and section.claim_work:
        return "claims"
    if section.status == "drafted" and section.draft:
        return "review"
    if section.results:
        return "analyze"
    if section.search_rounds >= policy.max_search_rounds:
        return "write"
    return "research"


def operation_summary(state):
    """A finished operation is not necessarily a finished research report."""
    operation = state["parent_context"]["section_operation"]
    target = next(s for s in state["sections"] if s["section_id"] == operation["target"])
    lines = ["# 章节研究阶段产物", "", "本次局部操作已结束，不是完整研究报告。原 Run 保持不变。", "",
             f"操作：{operation['mode']}；目标：{target['title']}", "",
             "## 章节状态", *[f"- {s['title']}：{s['status']}" for s in state["sections"]]]
    if operation["mode"] == "supplement":
        lines += ["", "## 补证据结果（未改正文、未重新绑定旧 Claim）", str(target.get("analyst", {}).get("summary", "")),
                  *[f"- {issue}" for issue in target.get("analyst", {}).get("issues", [])],
                  f"已保存候选来源 {len(target['results'])} 条；检索时间不是文档发布日期。",
                  "下一步请选择本章继续以核验、写作和重新绑定引用；旧正文仅供历史参考。"]
    else:
        lines += ["", "## 本次完成的章节", *[f"### {s['title']}\n\n{s['draft']}" for s in state["sections"]
                    if s["section_id"] in operation["work_ids"] and s["status"] in FINISHED],
                  "", "来源编号为各章本地编号，请在章节产物中查看对应来源。未完成章节需另行选择继续。"]
    return {"final_report": "\n".join(lines), "writer_status": "complete", "section_step": "done",
            "report_quality": "partial"}
