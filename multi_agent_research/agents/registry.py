"""Central registry for independently managed agent capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from .evidence_research_agent import EvidenceResearchAgent
from .planner_agent import PlannerAgent
from .report_reviewer_agent import ReportReviewerAgent
from .section_reviewer_agent import SectionReviewerAgent
from .section_writer_agent import SectionWriterAgent


@dataclass(slots=True)
class AgentRegistry:
    planner: PlannerAgent = field(default_factory=PlannerAgent)
    evidence_research: EvidenceResearchAgent = field(default_factory=EvidenceResearchAgent)
    section_writer: SectionWriterAgent = field(default_factory=SectionWriterAgent)
    section_reviewer: SectionReviewerAgent = field(default_factory=SectionReviewerAgent)
    report_reviewer: ReportReviewerAgent = field(default_factory=ReportReviewerAgent)


agent_registry = AgentRegistry()
