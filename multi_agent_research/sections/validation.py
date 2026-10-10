"""Context-dependent model checks, separate from infrastructure failures."""

from dataclasses import asdict, dataclass

from .models import (
    ChiefEditorResult,
    EditedSectionArtifact,
    EditorialBlueprint,
    EditorialFraming,
    EditorialSectionPlan,
    ReportReview,
    SectionPlan,
    SectionRecord,
    SectionReview,
)
from .rendering import citation_issues


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    type: str
    message: str
    section_id: str | None = None
    source_number: int | None = None


class BusinessValidationError(ValueError):
    """Only explicitly classified model mistakes may spend correction attempts."""

    def __init__(self, issues: list[ValidationIssue], *, retryable: bool = True,
                 repair_hints: list[dict] | None = None):
        if not issues:
            raise ValueError("business validation requires at least one issue")
        self.issues = issues
        self.retryable = retryable
        self.repair_hints = repair_hints or []  # Private source context, never public errors.
        super().__init__("; ".join(f"{i.field}: {i.message}" for i in issues))

    def details(self) -> list[dict]:
        return [asdict(issue) for issue in self.issues]


def validate_plan(plan: SectionPlan, maximum: int, available: set[str]) -> SectionPlan:
    issues = []
    if len(plan.sections) > maximum:
        issues.append(ValidationIssue("sections", "chapter_limit", f"本次最多允许 {maximum} 个章节"))
    for index, section in enumerate(plan.sections):
        if set(section.parent_section_ids) - available:
            issues.append(ValidationIssue(
                f"sections[{index}].parent_section_ids", "unknown_parent_section",
                "只能引用提供的父章节 ID；没有相关父章节时返回空数组",
            ))
    if issues:
        raise BusinessValidationError(issues)
    return plan


def validate_section_review(review: SectionReview) -> SectionReview:
    # Never erase a gap to reconcile a contradictory pass. Preserve all objections.
    if review.verdict == "pass" and (review.issues or review.search_queries):
        return review.model_copy(update={"verdict": "revise"})
    return review


def validate_draft(draft: str, sources: list[dict], section_id: str) -> str:
    issues = [ValidationIssue("draft.citations", "invalid_citation", message, section_id=section_id)
              for message in citation_issues(draft, sources)]
    if issues:
        raise BusinessValidationError(issues)
    return draft


def validate_report_review(review: ReportReview, known: set[str]) -> ReportReview:
    issues = [ValidationIssue(
        f"issues[{index}].section_ids", "unknown_section",
        "只能引用输入中已有的章节 ID；全局问题可使用空数组",
    ) for index, issue in enumerate(review.issues) if set(issue.section_ids) - known]
    if issues:
        raise BusinessValidationError(issues)
    if review.issues and review.verdict == "pass":
        return review.model_copy(update={"verdict": "revise"})
    return review


def validate_chief_editor_result(
    result: ChiefEditorResult,
    sections: list[SectionRecord],
    report_review: ReportReview,
    known_evidence_ids: set[str],
) -> ChiefEditorResult:
    """Reject provenance loss, unknown IDs, and unhandled review issues."""

    from .editorial import evidence_tokens

    known_sections = {section.section_id for section in sections}
    known_claims = {
        claim.claim_id
        for section in sections
        for claim in section.claims
        if claim.assessment != "unsupported"
    }
    issues: list[ValidationIssue] = []

    referenced_sections = {
        section_id
        for edited in result.sections
        for section_id in edited.source_section_ids
    }
    unknown_sections = referenced_sections - known_sections
    if unknown_sections:
        issues.append(ValidationIssue(
            "sections.source_section_ids",
            "unknown_section",
            f"编辑结果引用了未知章节：{sorted(unknown_sections)}",
        ))
    missing_sections = known_sections - referenced_sections
    if missing_sections:
        issues.append(ValidationIssue(
            "sections.source_section_ids",
            "missing_section",
            f"编辑结果遗漏了章节：{sorted(missing_sections)}",
        ))

    declared_claims = set(result.used_claim_ids)
    declared_claims.update(
        claim_id for edited in result.sections for claim_id in edited.claim_ids
    )
    unknown_claims = declared_claims - known_claims
    if unknown_claims:
        issues.append(ValidationIssue(
            "used_claim_ids",
            "unknown_claim",
            f"编辑结果引用了未知或不受支持的 Claim：{sorted(unknown_claims)}",
        ))

    report_text = "\n".join([
        result.executive_summary,
        *(edited.body for edited in result.sections),
        result.conclusion,
    ])
    tokens = evidence_tokens(report_text)
    unknown_evidence = tokens - known_evidence_ids
    if unknown_evidence:
        issues.append(ValidationIssue(
            "report.evidence",
            "unknown_evidence",
            f"编辑结果引用了未知 Evidence：{sorted(unknown_evidence)}",
        ))
    if known_evidence_ids and not tokens:
        issues.append(ValidationIssue(
            "report.evidence",
            "missing_evidence",
            "编辑后的报告没有保留任何证据标记",
        ))

    expected_issue_indexes = set(range(len(report_review.issues)))
    actual_issue_indexes = [item.issue_index for item in result.issue_resolutions]
    if len(actual_issue_indexes) != len(set(actual_issue_indexes)):
        issues.append(ValidationIssue(
            "issue_resolutions",
            "duplicate_issue",
            "同一个全篇审校问题只能处理一次",
        ))
    if set(actual_issue_indexes) != expected_issue_indexes:
        issues.append(ValidationIssue(
            "issue_resolutions",
            "unhandled_issue",
            "必须逐项处理全篇审校问题",
        ))
    if result.verdict == "ready" and result.unresolved_issues:
        issues.append(ValidationIssue(
            "verdict",
            "contradictory_ready",
            "存在未解决问题时不能返回 ready",
        ))
    if report_review.verdict == "revise" and result.verdict == "ready" and any(
        item.action == "preserved_as_limitation" for item in result.issue_resolutions
    ):
        issues.append(ValidationIssue(
            "verdict",
            "limitation_not_reflected",
            "保留了审校限制时必须返回 limited",
        ))

    if issues:
        raise BusinessValidationError(issues)
    return result


def _editorial_issue_checks(
    *,
    verdict: str,
    resolutions,
    unresolved_issues: list[str],
    report_review: ReportReview,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    expected = set(range(len(report_review.issues)))
    actual = [item.issue_index for item in resolutions]
    if len(actual) != len(set(actual)):
        issues.append(ValidationIssue(
            "issue_resolutions", "duplicate_issue", "同一个全篇审校问题只能处理一次"
        ))
    if set(actual) != expected:
        issues.append(ValidationIssue(
            "issue_resolutions", "unhandled_issue", "必须逐项处理全篇审校问题"
        ))
    if verdict == "ready" and unresolved_issues:
        issues.append(ValidationIssue(
            "verdict", "contradictory_ready", "存在未解决问题时不能返回 ready"
        ))
    if verdict == "ready" and any(
        item.action == "preserved_as_limitation" for item in resolutions
    ):
        issues.append(ValidationIssue(
            "verdict", "limitation_not_reflected", "保留了审校限制时必须返回 limited"
        ))
    return issues


def validate_editorial_blueprint(
    blueprint: EditorialBlueprint,
    sections: list[SectionRecord],
    report_review: ReportReview,
    known_evidence_ids: set[str],
) -> EditorialBlueprint:
    """Ensure the plan covers every chapter and only allocates known facts."""

    known_sections = {section.section_id for section in sections}
    actual_sections = [plan.source_section_id for plan in blueprint.section_plans]
    issues = _editorial_issue_checks(
        verdict=blueprint.verdict,
        resolutions=blueprint.issue_resolutions,
        unresolved_issues=blueprint.unresolved_issues,
        report_review=report_review,
    )
    if len(actual_sections) != len(set(actual_sections)):
        issues.append(ValidationIssue(
            "section_plans", "duplicate_section", "每个来源章节只能在编辑蓝图中出现一次"
        ))
    if set(actual_sections) != known_sections:
        issues.append(ValidationIssue(
            "section_plans", "chapter_coverage", "编辑蓝图必须恰好覆盖全部来源章节"
        ))

    from .editorial import editorial_capacity_floor

    sections_by_id = {section.section_id: section for section in sections}
    normalized_plans = []
    for index, plan in enumerate(blueprint.section_plans):
        source = sections_by_id.get(plan.source_section_id)
        if source is None:
            continue
        local_claims = {
            claim.claim_id for claim in source.claims if claim.assessment != "unsupported"
        }
        local_evidence = {
            link.evidence_id
            for claim in source.claims
            if claim.assessment != "unsupported"
            for link in claim.evidence
            if link.evidence_id
        }
        unknown_claims = set(plan.claim_ids) - local_claims
        unknown_evidence = set(plan.evidence_ids) - local_evidence
        if unknown_claims:
            issues.append(ValidationIssue(
                f"section_plans[{index}].claim_ids",
                "unknown_claim",
                f"章节分配了未知 Claim：{sorted(unknown_claims)}",
                section_id=source.section_id,
            ))
        if unknown_evidence or set(plan.evidence_ids) - known_evidence_ids:
            issues.append(ValidationIssue(
                f"section_plans[{index}].evidence_ids",
                "unknown_evidence",
                f"章节分配了未知 Evidence：{sorted(unknown_evidence)}",
                section_id=source.section_id,
            ))
        capacity_floor = editorial_capacity_floor(plan)
        if capacity_floor > 3000:
            issues.append(ValidationIssue(
                f"section_plans[{index}]",
                "infeasible_length_budget",
                "当前 Claim/Evidence 分配无法在单章发布篇幅内清晰表达；请减少次要分配",
                section_id=source.section_id,
            ))
            normalized_plans.append(plan)
        elif plan.target_chars < capacity_floor:
            normalized_plans.append(plan.model_copy(update={"target_chars": capacity_floor}))
        else:
            normalized_plans.append(plan)
    if issues:
        raise BusinessValidationError(issues)
    return blueprint.model_copy(update={"section_plans": normalized_plans})


def validate_edited_section_artifact(
    artifact: EditedSectionArtifact,
    plan: EditorialSectionPlan,
    *,
    required_claim_ids: set[str] | None = None,
    required_evidence_ids: set[str] | None = None,
) -> EditedSectionArtifact:
    """Fence one chapter edit to its blueprint allocation and stable evidence IDs."""

    from .editorial import evidence_tokens

    section = artifact.section
    issues: list[ValidationIssue] = []
    if section.source_section_ids != [plan.source_section_id]:
        issues.append(ValidationIssue(
            "section.source_section_ids",
            "section_boundary",
            "逐章编辑只能声明当前蓝图章节",
            section_id=plan.source_section_id,
        ))
    if section.title != plan.title:
        issues.append(ValidationIssue(
            "section.title", "title_drift", "输出标题必须与编辑蓝图一致"
        ))
    unknown_claims = set(section.claim_ids) - set(plan.claim_ids)
    if unknown_claims:
        issues.append(ValidationIssue(
            "section.claim_ids", "unknown_claim", f"使用了蓝图未分配的 Claim：{sorted(unknown_claims)}"
        ))
    must_keep_claims = (
        set(plan.claim_ids) if required_claim_ids is None else required_claim_ids
    )
    missing_claims = must_keep_claims - set(section.claim_ids)
    if missing_claims:
        issues.append(ValidationIssue(
            "section.claim_ids",
            "dropped_claim",
            f"压缩稿删除了必须保留的 Claim：{sorted(missing_claims)}",
        ))
    tokens = evidence_tokens(section.body)
    handoff_tokens = evidence_tokens(artifact.summary + "\n" + artifact.handoff)
    unknown_evidence = tokens - set(plan.evidence_ids)
    if unknown_evidence:
        issues.append(ValidationIssue(
            "section.body", "unknown_evidence", f"使用了蓝图未分配的 Evidence：{sorted(unknown_evidence)}"
        ))
    must_keep_evidence = (
        set(plan.evidence_ids) if required_evidence_ids is None else required_evidence_ids
    )
    missing_evidence = must_keep_evidence - tokens
    if missing_evidence:
        issues.append(ValidationIssue(
            "section.body",
            "dropped_evidence",
            f"压缩稿删除了必须保留的 Evidence：{sorted(missing_evidence)}",
        ))
    unknown_handoff = handoff_tokens - set(plan.evidence_ids)
    if unknown_handoff:
        issues.append(ValidationIssue(
            "summary", "unknown_evidence", f"摘要或衔接使用了未分配 Evidence：{sorted(unknown_handoff)}"
        ))
    if plan.evidence_ids and not tokens:
        issues.append(ValidationIssue(
            "section.body", "missing_evidence", "有可用证据的编辑章节必须保留稳定证据标记"
        ))
    if "[来源" in (section.body + artifact.summary + artifact.handoff):
        issues.append(ValidationIssue(
            "section.body", "unstable_citation", "主编阶段只能使用稳定 Evidence 标记"
        ))
    if issues:
        raise BusinessValidationError(issues)
    return artifact


def validate_editorial_framing(
    framing: EditorialFraming,
    known_evidence_ids: set[str],
) -> EditorialFraming:
    """Prevent front/back matter from inventing citations outside edited chapters."""

    from .editorial import evidence_tokens

    text = framing.executive_summary + "\n" + framing.conclusion
    unknown = evidence_tokens(text) - known_evidence_ids
    issues: list[ValidationIssue] = []
    if unknown:
        issues.append(ValidationIssue(
            "framing", "unknown_evidence", f"摘要或结论引用了未知 Evidence：{sorted(unknown)}"
        ))
    if "[来源" in text:
        issues.append(ValidationIssue(
            "framing", "unstable_citation", "摘要和结论只能使用稳定 Evidence 标记"
        ))
    if issues:
        raise BusinessValidationError(issues)
    return framing
