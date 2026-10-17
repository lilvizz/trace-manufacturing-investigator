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


# Real pipeline stages, in execution order. Investigator.run() reports the
# start of each one to an optional progress callback.
PIPELINE_STAGES = (
    ("plan", "Planning investigation"),
    ("retrieve", "Retrieving evidence"),
    ("timeline", "Reconstructing timeline"),
    ("analyze", "Analyzing evidence gaps"),
    ("reason", "Gemini reasoning"),
    ("verify", "Verifying claims"),
    ("calibrate", "Calibrating confidence"),
)


class InvestigationResult:
    def __init__(
        self,
        plan: InvestigationPlan,
        investigation: Investigation,
        timeline: InvestigationTimeline,
        evidence_analysis: EvidenceAnalysis,
        report: InvestigationReport,
        conversation=None,
        resolution=None,
    ):
        self.plan = plan
        self.investigation = investigation
        self.timeline = timeline
        self.evidence_analysis = evidence_analysis
        self.report = report
        # Conversation context used for this turn (None for a fresh
        # investigation) and how the question was resolved against it.
        self.conversation = conversation
        self.resolution = resolution


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

    def run(
        self,
        question: str,
        context=None,
        progress_callback=None,
    ) -> InvestigationResult:
        """
        Run the investigation pipeline.

        Synchronous. Both optional arguments default to the original
        behaviour:

        context
            A ConversationContext for a follow-up question in the same
            conversation. The full pipeline still runs; context is used
            only to resolve references for planning/retrieval and is shown
            to the reasoner as not-evidence. Passages cited by the previous
            turn may be re-added as "carried-forward" corpus evidence and
            are re-verified like any other evidence.
        progress_callback
            Called as progress_callback(stage_key, stage_label) at the start
            of each real stage (see PIPELINE_STAGES) and ("done", ...) at
            the end. Callback errors never affect the investigation.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty")

        question = question.strip()
        labels = dict(PIPELINE_STAGES)

        def notify(stage: str) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(stage, labels.get(stage, stage))
            except Exception:
                pass

        resolution = context.resolve(question) if context is not None else None
        planning_question = (
            resolution.retrieval_question
            if resolution is not None and resolution.used_context
            else question
        )

        # 1. Plan the investigation.
        notify("plan")
        plan = self.planner.create_plan(planning_question)

        # A follow-up's reference-resolved question is also searched
        # directly, so "that study" retrieves the study even when the
        # plan's subquestions are generic.
        if (
            resolution is not None
            and resolution.used_context
            and planning_question not in plan.subquestions
        ):
            plan = InvestigationPlan(
                question=plan.question,
                subquestions=list(plan.subquestions) + [planning_question],
                source=getattr(plan, "source", ""),
                fallback_reason=getattr(plan, "fallback_reason", ""),
            )

        # 2. Create investigation state.
        investigation = Investigation(
            question=question,
            subquestions=plan.subquestions,
        )

        # 3. Retrieve evidence (always fresh).
        notify("retrieve")
        investigation = self.retriever.collect(
            investigation,
            plan,
        )

        # 3b. Follow-ups: re-add the corpus passages the previous turn
        # cited, marked carried-forward, so they can be re-verified.
        if context is not None:
            for item in getattr(context, "carried_evidence", ()) or ():
                investigation.add_evidence(item)

        # 4. Reconstruct chronology.
        notify("timeline")
        timeline = self.timeline_builder.build(
            investigation.evidence,
        )

        # 5. Deterministic evidence-gap analysis.
        notify("analyze")
        evidence_analysis = self.analyzer.analyze(
            investigation,
        )

        # 6. Gemini performs evidence reasoning.
        notify("reason")
        reason_kwargs = {
            "evidence_analysis": evidence_analysis,
            "timeline": timeline,
        }
        if context is not None:
            reason_kwargs["conversation"] = context
        report = self.reasoner.reason(
            investigation,
            **reason_kwargs,
        )

        # 7. Deterministic post-hoc claim verification.
        notify("verify")
        report = self.claim_verifier.verify(
            report,
            investigation,
        )

        # 8. Deterministic confidence calibration.
        notify("calibrate")
        report = self.confidence_calibrator.calibrate(
            report,
            investigation,
        )

        notify("done")

        return InvestigationResult(
            plan=plan,
            investigation=investigation,
            timeline=timeline,
            evidence_analysis=evidence_analysis,
            report=report,
            conversation=context,
            resolution=resolution,
        )