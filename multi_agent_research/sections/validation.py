"""Context-dependent model checks, separate from infrastructure failures."""

from dataclasses import asdict, dataclass

from .models import ChiefEditorResult, ReportReview, SectionPlan, SectionRecord, SectionReview
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
