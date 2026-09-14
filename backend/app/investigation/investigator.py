from app.investigation.analyzer import EvidenceGapAnalyzer
from app.investigation.models import (
    EvidenceAnalysis,
    Investigation,
    InvestigationReport,
)
from app.investigation.planner import InvestigationPlan, InvestigationPlanner
from app.investigation.reasoner import InvestigationReasoner
from app.investigation.retriever import InvestigationRetriever
from app.investigation.timeline import InvestigationTimeline, TimelineBuilder


class Investigator:
    """
    Orchestrates the complete TRACE investigation workflow.

    Each stage is injected as a dependency so the workflow can be
    tested without calling external services.
    """

    def __init__(
        self,
        planner: InvestigationPlanner,
        retriever: InvestigationRetriever,
        timeline_builder: TimelineBuilder,
        analyzer: EvidenceGapAnalyzer,
        reasoner: InvestigationReasoner,
    ):
        self.planner = planner
        self.retriever = retriever
        self.timeline_builder = timeline_builder
        self.analyzer = analyzer
        self.reasoner = reasoner

    def run(self, question: str) -> "InvestigationResult":
        if not question or not question.strip():
            raise ValueError("Investigation question cannot be empty.")

        question = question.strip()

        plan = self.planner.create_plan(question)

        investigation = Investigation(
            question=question,
            subquestions=plan.subquestions,
        )

        investigation = self.retriever.collect(
            investigation,
            plan,
        )

        timeline = self.timeline_builder.build(
            investigation.evidence,
        )

        evidence_analysis = self.analyzer.analyze(
            investigation,
        )

        report = self.reasoner.reason(
            investigation,
        )

        return InvestigationResult(
            plan=plan,
            investigation=investigation,
            timeline=timeline,
            evidence_analysis=evidence_analysis,
            report=report,
        )


class InvestigationResult:
    """
    Complete output of one TRACE investigation.
    """

    def __init__(
        self,
        plan: InvestigationPlan,
        investigation: Investigation,
        timeline: InvestigationTimeline,
        evidence_analysis: EvidenceAnalysis,
        report: InvestigationReport,
    ):
        self.plan = plan
        self.investigation = investigation
        self.timeline = timeline
        self.evidence_analysis = evidence_analysis
        self.report = report