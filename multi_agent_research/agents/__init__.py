"""Agent roles use ``*_agent.py``; contracts and registry are support modules."""

from .contracts import (
    ClaimExtractionRequest,
    ClaimExtractionResult,
    EvidenceAnalysisRequest,
    EvidenceAnalysisResult,
    ReportReviewRequest,
    ReportReviewResult,
    SectionPlanningRequest,
    SectionReviewRequest,
    SectionReviewResult,
    SectionWritingRequest,
    SectionWritingResult,
)
from .claim_extractor_agent import ClaimExtractorAgent
from .evidence_analyst_agent import EvidenceAnalystAgent
from .planner_agent import PlannerAgent
from .report_reviewer_agent import ReportReviewerAgent
from .registry import AgentRegistry, agent_registry
from .section_reviewer_agent import SectionReviewerAgent
from .section_writer_agent import SectionWriterAgent

__all__ = [
    "AgentRegistry",
    "ClaimExtractionRequest",
    "ClaimExtractionResult",
    "ClaimExtractorAgent",
    "EvidenceAnalysisRequest",
    "EvidenceAnalysisResult",
    "EvidenceAnalystAgent",
    "PlannerAgent",
    "ReportReviewRequest",
    "ReportReviewResult",
    "ReportReviewerAgent",
    "SectionPlanningRequest",
    "SectionReviewRequest",
    "SectionReviewResult",
    "SectionReviewerAgent",
    "SectionWriterAgent",
    "SectionWritingRequest",
    "SectionWritingResult",
    "agent_registry",
]
