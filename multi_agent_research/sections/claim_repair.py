"""Per-Claim acceptance and typed, program-owned excerpt references for repair."""
import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ..core.budget import ExecutionPaused, current_budget
from .artifacts import bind_claims
from .models import Claim, ClaimExtraction
from .quotes import locate_quote
from .validation import BusinessValidationError


class ClaimsPending(ExecutionPaused):
    def __init__(self, section):
        self.section_id = section.section_id
        pending = len(section.claim_work.get('pending', []))
        reason = f"{pending} 条待修复" if pending else "抽取格式仍待修复"
        super().__init__(f"章节 {section.section_id}：已保存 {len(section.claims)} 条通过校验的关联，"
                         f"{reason}；本次纠正额度已用完，可继续局部修复")


class RepairEvidence(BaseModel):
    source_span_id: str
    relation: Literal["supports", "contradicts", "context"]


class ClaimPatch(BaseModel):
    slot: int = Field(ge=1, le=12)
    statement: str = Field(min_length=1, max_length=1000)
    draft_span_id: str
    assessment: Literal["supported", "uncertain", "unsupported"]
    evidence: list[RepairEvidence] = Field(default_factory=list, max_length=8)
    caveat: str = Field(default="", max_length=1000)


class ClaimRepairs(BaseModel):
    repairs: list[ClaimPatch] = Field(min_length=1, max_length=12)


def epoch():
    scope = current_budget.get()
    return scope.execution_id if scope else "standalone"


def fingerprint(section):
    return hashlib.sha256(json.dumps([section.revision, section.draft, section.sources],
                                     ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def new_work(section):
    return {"fingerprint": fingerprint(section), "accepted": {}, "pending": [],
            "batch_errors": [], "attempts": 0, "total_attempts": 0, "epoch": epoch()}


def accept_one(section, raw, slot):
    try:
        candidate = Claim.model_validate(raw)
        bound = bind_claims(section, ClaimExtraction(claims=[candidate]))[0]
        bound.claim_id = f"{section.section_id}:v{section.revision}:c{slot}"
        return bound.model_dump(mode="json"), None
    except (ValidationError, BusinessValidationError) as exc:
        errors = (exc.details() if isinstance(exc, BusinessValidationError) else
                  [{"field": ".".join(map(str, e["loc"])), "type": e["type"], "message": e["msg"]}
                   for e in exc.errors(include_input=False, include_context=False)])
        # Diagnose the observed failure directly: source text used as body text.
        quote = raw.get("draft_quote", "") if isinstance(raw, dict) else ""
        if isinstance(quote, str) and quote and not locate_quote(section.draft, quote):
            found = [i for i, s in enumerate(section.sources, 1) if locate_quote(s["content"][:1000], quote)]
            if found:
                errors.append({"field": "draft_quote", "type": "source_used_as_draft",
                               "message": f"这句话来自来源 {found}，不在正文中；请从 D 前缀正文片段选择，不可填 S 前缀来源。"})
        return None, {"slot": slot, "candidate": deepcopy(raw), "errors": errors}


def split_extraction(section, candidates, work=None):
    work = deepcopy(work or new_work(section))
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 12:
        work["batch_errors"] = [{"type": "invalid_batch", "message": "需返回 1–12 条 Claim，不可隐式截断结果"}]
        return work
    work["batch_errors"] = []
    work["accepted"], work["pending"] = {}, []
    for slot, raw in enumerate(candidates, 1):
        bound, pending = accept_one(section, raw, slot)
        if pending:
            work["pending"].append(pending)
        else:
            work["accepted"][str(slot)] = bound
    return work


def excerpt_catalog(section):
    catalog = {}
    def add(text, prefix, source_number=None):
        # Exact non-overlapping spans; no model-created offsets, no fuzzy acceptance.
        for match in re.finditer(r"[^。！？\n]+[。！？]?|\n", text):
            for start in range(match.start(), match.end(), 450):
                end = min(start + 450, match.end())
                if text[start:end].strip():
                    key = f"{prefix}:{start}:{end}"
                    catalog[key] = {"text": text[start:end], "source_number": source_number}
    add(section.draft, "D")
    for i, source in enumerate(section.sources, 1):
        add(source["content"][:1000], f"S{i}", i)
    return catalog


def repair_prompt(section, work):
    catalog = excerpt_catalog(section)
    numbers = set()
    all_sources = False
    for pending in work["pending"]:
        raw = pending["candidate"]
        evidence = raw.get("evidence", []) if isinstance(raw, dict) else []
        if not isinstance(evidence, list) or not evidence:
            all_sources = True
        else:
            for item in evidence:
                number = item.get("source_number") if isinstance(item, dict) else None
                if isinstance(number, int) and 1 <= number <= len(section.sources):
                    numbers.add(number)
                else:
                    all_sources = True
    view = {key: value for key, value in catalog.items()
            if value["source_number"] is None or all_sources or value["source_number"] in numbers}
    return json.dumps({"section_id": section.section_id, "pending": work["pending"],
                       "previous_patch_errors": work.get("batch_errors", []), "excerpts": view}, ensure_ascii=False)


def apply_repairs(section, work, patches):
    updated = deepcopy(work)
    catalog = excerpt_catalog(section)
    wanted = {p["slot"] for p in work["pending"]}
    counts = Counter(p.slot for p in patches.repairs)
    updated["ignored_slots"] = [p.slot for p in patches.repairs if p.slot not in wanted]
    updated["batch_errors"] = []  # Extra slots cannot change previously accepted items.
    by_slot = {p.slot: p for p in patches.repairs if p.slot in wanted and counts[p.slot] == 1}
    remaining = []
    for pending in work["pending"]:
        item = deepcopy(pending)
        slot = item["slot"]
        patch = by_slot.get(slot)
        try:
            if patch is None:
                raise ValueError("必须恰好修复一次此 slot，不能删除未解决项")
            draft = catalog.get(patch.draft_span_id)
            if not draft or draft["source_number"] is not None:
                raise ValueError("draft_span_id 必须是当前 D 前缀正文片段，不是来源 ID")
            links = []
            for link in patch.evidence:
                source = catalog.get(link.source_span_id)
                if not source or source["source_number"] is None:
                    raise ValueError("source_span_id 必须是当前 S 前缀来源片段")
                links.append({"source_number": source["source_number"], "quote": source["text"], "relation": link.relation})
            old = item["candidate"] if isinstance(item["candidate"], dict) else {}
            statement = old.get("statement")
            if isinstance(statement, str) and statement.strip() and len(statement) <= 1000 and patch.statement != statement:
                raise ValueError("局部修复不能替换原结论 statement；只修定位，原结论不成立时保留并标注 unsupported/caveat")
            # Never silently drop a known counter-evidence link to pass validation.
            for link in old.get("evidence", []) if isinstance(old.get("evidence"), list) else []:
                if isinstance(link, dict) and link.get("relation") == "contradicts" and not any(
                    x["relation"] == "contradicts" and x["source_number"] == link.get("source_number") for x in links):
                    raise ValueError("不得删除原反证；无法定位则保留待修复")
            assessment = patch.assessment
            if old.get("assessment") in ("uncertain", "unsupported") and assessment == "supported":
                assessment = old["assessment"]
            old_caveat = old.get("caveat", "")
            if not isinstance(old_caveat, str):
                old_caveat = json.dumps(old_caveat, ensure_ascii=False)
            caveat = "；".join(dict.fromkeys(x for x in (old_caveat, patch.caveat) if x))
            if len(caveat) > 1000:
                raise ValueError("原有限制与新 caveat 合计超过上限，保留待修复，不截断重大不确定性")
            raw = {"statement": patch.statement, "draft_quote": draft["text"], "assessment": assessment,
                   "evidence": links, "caveat": caveat}
            bound, failed = accept_one(section, raw, slot)
            if failed:
                item["errors"] = failed["errors"]
                remaining.append(item)
            else:
                updated["accepted"][str(slot)] = bound
        except ValueError as exc:
            item["errors"] = [{"type": "invalid_patch", "message": str(exc)}]
            remaining.append(item)
    updated["pending"] = remaining
    return updated


def salvage_patches(section, work, raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("repairs"), list) or not 1 <= len(raw["repairs"]) <= 12:
        return work
    patches = []
    slots = [p.get("slot") for p in raw["repairs"] if isinstance(p, dict)]
    for value in raw["repairs"]:
        try:
            patch = ClaimPatch.model_validate(value)
            if slots.count(patch.slot) == 1:
                patches.append(patch)
        except ValidationError:
            pass  # Invalid items remain pending, valid siblings can be committed.
    return apply_repairs(section, work, ClaimRepairs(repairs=patches)) if patches else work
