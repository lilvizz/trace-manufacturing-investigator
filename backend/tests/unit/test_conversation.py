"""Contextual follow-up tests: context reaches the reasoner, fresh retrieval
and verification still run, and earlier AI output never becomes evidence."""

import json
from types import SimpleNamespace

from app.investigation.analyzer import EvidenceGapAnalyzer
from app.investigation.conversation import (
    CARRIED_FORWARD_CHANNEL,
    MAX_CARRIED_EVIDENCE,
    MAX_TURNS,
    ConversationContext,
    resolve_question,
    subject_anchors,
)
from app.investigation.investigator import PIPELINE_STAGES, Investigator
from app.investigation.models import (
    ClaimType,
    EvidenceItem,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)
from app.investigation.planner import InvestigationPlan
from app.investigation.reasoner import InvestigationReasoner
from app.investigation.retriever import InvestigationRetriever, RetrievalResult
from app.investigation.timeline import TimelineBuilder


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class RecordingPlanner:
    def __init__(self):
        self.questions = []

    def create_plan(self, question):
        self.questions.append(question)
        return InvestigationPlan(question=question, subquestions=["What established the condition?"])


class RecordingRetriever:
    """Returns the 2014 requalification chunk only when the query names the
    subject, so tests can see whether reference resolution reached retrieval."""

    def __init__(self):
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append(query)
        results = [
            RetrievalResult("chunk-2001", "doc-2001", "07_Thermal_Validation_2001.txt",
                            "B/07.txt", "Press-07 limit set to 42.3 C on 2001-07-05.", 0.9)
        ]
        if "Press-07" in query and "2014" in query:
            results.append(
                RetrievalResult("chunk-2014", "doc-2014", "13_Process_Requalification_2014.txt",
                                "B/13.txt", "2014 study is a characterization, not a revalidated specification.", 0.8)
            )
        return results


class RecordingClient:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return json.dumps(self.payload)


def make(payload):
    planner, retriever, client = RecordingPlanner(), RecordingRetriever(), RecordingClient(payload)
    investigator = Investigator(
        planner=planner,
        retriever=InvestigationRetriever(retriever, top_k_per_subquestion=4),
        timeline_builder=TimelineBuilder(),
        analyzer=EvidenceGapAnalyzer(),
        reasoner=InvestigationReasoner(client),
    )
    return investigator, planner, retriever, client


def prior_result(question="Why is Press-07 limited to 42.3°C?", answer="Earlier answer text.",
                 claims=None, evidence=None, cited=("chunk-2001",)):
    evidence = evidence or [
        EvidenceItem("chunk-2001", "doc-2001", "07_Thermal_Validation_2001.txt", "B/07.txt",
                     "Press-07 limit set to 42.3 C on 2001-07-05.", 0.9, ["q"], retrieval_channels=["bm25"]),
        EvidenceItem("chunk-old", "doc-old", "old.txt", "B/old.txt", "Old passage.", 0.5, ["q"]),
    ]
    claims = claims if claims is not None else [
        ReasoningClaim("Limit derived in 2001.", EvidenceStance.SUPPORTS, ["chunk-2001"], 0.9, ClaimType.ORIGINAL_BASIS),
        ReasoningClaim("Uncited speculation.", EvidenceStance.UNKNOWN, [], 0.0, ClaimType.ORIGINAL_BASIS),
    ]
    return SimpleNamespace(
        investigation=Investigation(question=question, evidence=evidence),
        report=InvestigationReport(question=question, claims=claims, direct_answer=answer,
                                   direct_answer_evidence_chunk_ids=list(cited)),
    )


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------

def test_subject_anchors_finds_equipment_ids_and_temperatures():
    assert subject_anchors("Why is Press-07 limited to 42.3°C after the PX-400 swap?") == [
        "Press-07", "PX-400", "42.3°C"
    ]


def test_referential_question_without_context_asks_for_clarification():
    resolution = resolve_question("Why isn't that study enough?")
    assert resolution.needs_clarification
    assert not resolution.used_context


def test_self_contained_question_without_context_runs_as_asked():
    resolution = resolve_question("What changed when PX-200 was replaced by PX-400?")
    assert not resolution.needs_clarification
    assert resolution.retrieval_question == "What changed when PX-200 was replaced by PX-400?"


def test_followup_is_anchored_to_subjects_from_earlier_user_questions():
    context = ConversationContext.from_results("conv-1", [prior_result()])
    resolution = context.resolve("Why is the 2014 qualification insufficient?")
    assert resolution.used_context and not resolution.needs_clarification
    assert "Press-07" in resolution.retrieval_question
    assert "42.3°C" in resolution.retrieval_question
    assert resolution.retrieval_question.startswith("Why is the 2014 qualification insufficient?")


def test_anchors_come_from_user_questions_not_ai_answers():
    result = prior_result(answer="The PX-999 controller was installed.")
    context = ConversationContext.from_results("conv-1", [result])
    assert "PX-999" not in context.anchors()


# ---------------------------------------------------------------------------
# Bounding and filtering
# ---------------------------------------------------------------------------

def test_context_is_bounded_to_recent_turns_and_verified_claims():
    results = [prior_result(question=f"Question {i} about Press-07?") for i in range(MAX_TURNS + 3)]
    context = ConversationContext.from_results("conv-1", results)
    assert len(context.turns) == MAX_TURNS
    assert context.turns[-1].question == f"Question {MAX_TURNS + 2} about Press-07?"
    statements = [c.statement for c in context.turns[-1].claims]
    assert "Limit derived in 2001." in statements
    assert "Uncited speculation." not in statements  # unverified claims are not carried


def test_carried_evidence_is_only_previously_cited_corpus_passages():
    evidence = [
        EvidenceItem(f"chunk-{i}", f"doc-{i}", f"f{i}.txt", f"B/f{i}.txt", "text", 0.5, ["q"])
        for i in range(10)
    ]
    claims = [ReasoningClaim("c", EvidenceStance.SUPPORTS, [f"chunk-{i}" for i in range(9)], 0.9,
                             ClaimType.ORIGINAL_BASIS)]
    context = ConversationContext.from_results(
        "conv-1", [prior_result(evidence=evidence, claims=claims, cited=())]
    )
    ids = [item.chunk_id for item in context.carried_evidence]
    assert len(ids) == MAX_CARRIED_EVIDENCE
    assert "chunk-9" not in ids  # never cited -> never carried
    assert all(item.retrieval_channels == [CARRIED_FORWARD_CHANNEL] for item in context.carried_evidence)
    # Original corpus text and provenance are kept.
    assert context.carried_evidence[0].relative_path == "B/f0.txt"


def test_prompt_labels_earlier_answers_as_not_evidence():
    text = ConversationContext.from_results("conv-1", [prior_result()]).to_prompt()
    assert "conversation_id: conv-1" in text
    assert "NOT evidence" in text
    assert "Earlier answer text." in text


# ---------------------------------------------------------------------------
# Investigator integration
# ---------------------------------------------------------------------------

SUPPORT_PAYLOAD = {
    "direct_answer": {"text": "Because the 2014 study was a characterization.", "evidence_chunk_ids": ["chunk-2014"]},
    "claims": [
        {"statement": "The 2014 study is not a revalidated specification.", "stance": "supports",
         "claim_type": "current_technical_justification", "evidence_chunk_ids": ["chunk-2014"], "confidence": 0.95},
        {"statement": "Repeated from earlier turn without current evidence.", "stance": "supports",
         "claim_type": "original_basis", "evidence_chunk_ids": ["chunk-old"], "confidence": 0.9},
    ],
    "unresolved_questions": [],
}


def test_followup_context_reaches_planner_retrieval_and_reasoner():
    investigator, planner, retriever, client = make(SUPPORT_PAYLOAD)
    context = ConversationContext.from_results("conv-1", [prior_result()])

    result = investigator.run("Why is the 2014 qualification insufficient?", context=context)

    # Planner and retrieval received the reference-resolved question.
    assert "Press-07" in planner.questions[0]
    assert any("Press-07" in q and "2014" in q for q in retriever.queries)
    # Fresh retrieval found the 2014 passage the follow-up was about.
    ids = {item.chunk_id for item in result.investigation.evidence}
    assert "chunk-2014" in ids
    # The investigation question shown to the user is unchanged.
    assert result.investigation.question == "Why is the 2014 qualification insufficient?"
    # The reasoner saw the labelled context.
    prompt = client.prompts[0]
    assert "<CONVERSATION_CONTEXT>" in prompt and "NOT evidence" in prompt
    assert "Earlier answer text." in prompt
    assert result.conversation is context
    assert result.resolution.used_context


def test_same_question_without_context_does_not_find_the_resolved_passage():
    investigator, planner, retriever, client = make(SUPPORT_PAYLOAD)
    result = investigator.run("Why is the 2014 qualification insufficient?")
    assert "chunk-2014" not in {item.chunk_id for item in result.investigation.evidence}
    assert "CONVERSATION_CONTEXT" not in client.prompts[0]
    assert result.conversation is None


def test_earlier_turn_chunks_cannot_be_cited_unless_retrieved_again():
    investigator, *_ = make(SUPPORT_PAYLOAD)
    context = ConversationContext.from_results("conv-1", [prior_result()])
    result = investigator.run("Why is the 2014 qualification insufficient?", context=context)

    by_statement = {c.statement: c for c in result.report.claims}
    repeated = by_statement["Repeated from earlier turn without current evidence."]
    # chunk-old was in the earlier turn but was neither re-retrieved nor
    # cited by the earlier answer, so it is not valid evidence now.
    assert repeated.evidence_chunk_ids == []
    assert repeated.stance == EvidenceStance.UNKNOWN
    assert repeated.confidence == 0.0


def test_carried_forward_evidence_is_reverified_and_calibrated():
    investigator, *_ = make(SUPPORT_PAYLOAD)
    context = ConversationContext.from_results("conv-1", [prior_result()])
    result = investigator.run("Why is the 2014 qualification insufficient?", context=context)

    carried = [i for i in result.investigation.evidence if CARRIED_FORWARD_CHANNEL in i.retrieval_channels]
    assert [i.chunk_id for i in carried] == ["chunk-2001"]
    justification = next(c for c in result.report.claims
                          if c.claim_type == ClaimType.CURRENT_TECHNICAL_JUSTIFICATION)
    # Deterministic ceiling still applies to follow-ups.
    assert justification.confidence <= 0.70


def test_new_investigation_reasoner_prompt_is_unchanged_without_context():
    investigator, planner, retriever, client = make(SUPPORT_PAYLOAD)
    result = investigator.run("Why is Press-07 limited to 42.3°C?")
    reasoner = InvestigationReasoner(client)
    expected = reasoner._build_prompt(
        result.investigation, evidence_analysis=result.evidence_analysis, timeline=result.timeline
    )
    assert client.prompts[0] == expected


def test_progress_callback_reports_real_stage_order_and_is_isolated():
    investigator, *_ = make(SUPPORT_PAYLOAD)
    stages = []

    def callback(stage, label):
        stages.append(stage)
        raise RuntimeError("UI failure must not break the investigation")

    result = investigator.run("Why is Press-07 limited to 42.3°C?", progress_callback=callback)
    assert stages == [key for key, _ in PIPELINE_STAGES] + ["done"]
    assert result.report is not None


def test_not_established_claims_are_worded_as_not_established_in_context():
    claims = [ReasoningClaim("The current limit is technically justified.", EvidenceStance.NOT_ESTABLISHED,
                             ["chunk-2001"], 0.3, ClaimType.CURRENT_TECHNICAL_JUSTIFICATION)]
    text = ConversationContext.from_results("conv-1", [prior_result(claims=claims)]).to_prompt()
    assert "NOT ESTABLISHED by the evidence: The current limit is technically justified." in text
