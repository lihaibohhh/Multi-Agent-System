"""Context-dependent model checks, separate from infrastructure failures."""

from dataclasses import asdict, dataclass

from .models import ReportReview, SectionPlan, SectionReview
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
