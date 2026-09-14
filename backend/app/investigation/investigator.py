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


class InvestigationResult:
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


class Investigator:
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

    def run(self, question: str) -> InvestigationResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty")

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
    evidence_analysis=evidence_analysis,
    timeline=timeline,
)

        return InvestigationResult(
            plan=plan,
            investigation=investigation,
            timeline=timeline,
            evidence_analysis=evidence_analysis,
            report=report,
        )