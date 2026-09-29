from app.investigation.analyzer import EvidenceGapAnalyzer
from app.investigation.gemini_client import GeminiReasoningClient
from app.investigation.investigator import Investigator
from app.investigation.planner import InvestigationPlanner
from app.investigation.reasoner import InvestigationReasoner
from app.investigation.retriever import InvestigationRetriever
from app.investigation.timeline import TimelineBuilder


def build_investigator(
    retriever,
    gemini_client=None,
) -> Investigator:
    """
    Build the production TRACE investigation pipeline.

    gemini_client is optional so the UI can supply a client with
    tighter retry behavior. Existing callers that omit it retain
    the original backend behavior.
    """

    if gemini_client is None:
        gemini_client = GeminiReasoningClient()

    planner = InvestigationPlanner(
        planning_client=gemini_client,
    )

    investigator_retriever = InvestigationRetriever(
        retriever=retriever,
        top_k_per_subquestion=8,
        max_evidence=30,
        max_per_document=4,
        max_per_subquestion_document=2,
    )

    timeline_builder = TimelineBuilder()
    analyzer = EvidenceGapAnalyzer()
    reasoner = InvestigationReasoner(client=gemini_client)

    return Investigator(
        planner=planner,
        retriever=investigator_retriever,
        timeline_builder=timeline_builder,
        analyzer=analyzer,
        reasoner=reasoner,
    )
