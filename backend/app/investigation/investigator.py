from app.investigation.analyzer import EvidenceGapAnalyzer
from app.investigation.confidence import ConfidenceCalibrator
from app.investigation.models import (
    EvidenceAnalysis,
    Investigation,
    InvestigationReport,
)
from app.investigation.planner import InvestigationPlan, InvestigationPlanner
from app.investigation.reasoner import InvestigationReasoner
from app.investigation.retriever import InvestigationRetriever
from app.investigation.timeline import InvestigationTimeline, TimelineBuilder
from app.investigation.claim_verifier import ClaimVerifier


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
        claim_verifier: ClaimVerifier | None = None,
        confidence_calibrator: ConfidenceCalibrator | None = None,
    ):
        self.planner = planner
        self.retriever = retriever
        self.timeline_builder = timeline_builder
        self.analyzer = analyzer
        self.reasoner = reasoner

        self.claim_verifier = (
            claim_verifier
            if claim_verifier is not None
            else ClaimVerifier()
        )

        self.confidence_calibrator = (
            confidence_calibrator
            if confidence_calibrator is not None
            else ConfidenceCalibrator()
        )

    def run(self, question: str) -> InvestigationResult:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty")

        question = question.strip()

        # 1. Plan the investigation.
        plan = self.planner.create_plan(question)

        # 2. Create investigation state.
        investigation = Investigation(
            question=question,
            subquestions=plan.subquestions,
        )

        # 3. Retrieve evidence.
        investigation = self.retriever.collect(
            investigation,
            plan,
        )

        # 4. Reconstruct chronology.
        timeline = self.timeline_builder.build(
            investigation.evidence,
        )

        # 5. Deterministic evidence-gap analysis.
        evidence_analysis = self.analyzer.analyze(
            investigation,
        )

        # 6. Gemini performs evidence reasoning.
        report = self.reasoner.reason(
            investigation,
            evidence_analysis=evidence_analysis,
            timeline=timeline,
        )

        # 7. Deterministic post-hoc claim verification.
        report = self.claim_verifier.verify(
            report,
            investigation,
        )

        # 8. Deterministic confidence calibration.
        report = self.confidence_calibrator.calibrate(
            report,
            investigation,
        )

        return InvestigationResult(
            plan=plan,
            investigation=investigation,
            timeline=timeline,
            evidence_analysis=evidence_analysis,
            report=report,
        )