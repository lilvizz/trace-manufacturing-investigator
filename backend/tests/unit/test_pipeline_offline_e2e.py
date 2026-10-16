"""End-to-end pipeline checks on the REAL corpus with real BM25 + entity
retrieval, real timeline, analyzer, verifier and calibrator. Only the
Gemini calls are scripted (planner call fails -> deterministic fallback;
reasoning returns fixed JSON). Vector retrieval is exercised separately
(live check: python -m app.retrieval.vector_health)."""

import json
from pathlib import Path

import pytest

from app.config import CORPUS_PATH
from app.ingestion.chunker import chunk_corpus
from app.ingestion.loader import load_corpus
from app.investigation.build import build_investigator
from app.investigation.conversation import ConversationContext
from app.investigation.models import ClaimType, EvidenceStance
from app.retrieval.bm25_search import BM25Searcher
from app.retrieval.entity_matcher import EntityMatcher
from app.retrieval.hybrid_retriever import HybridRetriever

PRESS07 = (
    "Why is Press-07 limited to 42.3°C, what changed since that limit was "
    "established, and is its current technical basis still supported?"
)
FOLLOW_UP = "Why is the 2014 qualification insufficient to establish the current technical justification?"


class ScriptedGemini:
    """Optimistic on purpose: claims current justification is SUPPORTED at
    0.95 and cites a real historical passage, to prove the deterministic
    layers still refuse to establish it."""

    def __init__(self, cite_from_prompt=True):
        self.prompts = []

    def generate(self, prompt):
        if "investigation planner" in prompt:
            raise RuntimeError("planner offline")
        self.prompts.append(prompt)
        import re
        ids = re.findall(r"EVIDENCE CHUNK: (\S+)", prompt)
        historical = next((i for i in ids if i.startswith("07_")), ids[0] if ids else "none")
        return json.dumps({
            "direct_answer": {"text": "The limit dates from the 2001 study.", "evidence_chunk_ids": [historical]},
            "claims": [
                {"statement": "The 2001 study established the limit.", "stance": "supports",
                 "claim_type": "original_basis", "evidence_chunk_ids": [historical], "confidence": 0.95},
                {"statement": "The limit is therefore technically justified today.", "stance": "supports",
                 "claim_type": "current_technical_justification", "evidence_chunk_ids": [historical],
                 "confidence": 0.95},
                {"statement": "No re-justification record exists.", "stance": "not_established",
                 "claim_type": "evidence_gap", "evidence_chunk_ids": [historical], "confidence": 0.9},
                {"statement": "Cites a fabricated source.", "stance": "supports",
                 "claim_type": "original_basis", "evidence_chunk_ids": ["99_Invented_Document_0"], "confidence": 0.9},
            ],
            "unresolved_questions": [],
        })


@pytest.fixture(scope="module")
def corpus_chunks():
    return chunk_corpus(load_corpus(str(CORPUS_PATH)))


@pytest.fixture
def pipeline(corpus_chunks):
    client = ScriptedGemini()
    retriever = HybridRetriever(BM25Searcher(corpus_chunks), None, EntityMatcher(corpus_chunks))
    return build_investigator(retriever, gemini_client=client), client


def test_press07_investigation_on_real_corpus(pipeline, corpus_chunks):
    investigator, client = pipeline
    result = investigator.run(PRESS07)

    known = {c.chunk_id: c for c in corpus_chunks}
    evidence = result.investigation.evidence
    assert evidence, "real retrieval must return evidence"
    # Retrieval never fabricates sources: every passage is a real corpus chunk
    # with its real path and current text.
    for item in evidence:
        assert item.chunk_id in known
        assert item.relative_path == known[item.chunk_id].relative_path
        assert item.content == known[item.chunk_id].content
        assert (Path(CORPUS_PATH) / item.relative_path).exists()
        assert "bm25" in item.retrieval_channels or "entity" in item.retrieval_channels

    # The deterministic gap analysis reports the missing re-justification.
    assert any(f.finding_type.value == "evidence-gap" for f in result.evidence_analysis.findings)

    claims = {c.statement: c for c in result.report.claims}
    # Historical support can be strong...
    assert claims["The 2001 study established the limit."].confidence >= 0.9
    # ...but cannot carry current technical justification past its ceiling.
    assert claims["The limit is therefore technically justified today."].confidence <= 0.70
    gap = claims["No re-justification record exists."]
    assert gap.stance == EvidenceStance.NOT_ESTABLISHED and gap.confidence <= 0.30
    # A citation to a document that was not retrieved is stripped.
    fabricated = claims["Cites a fabricated source."]
    assert fabricated.evidence_chunk_ids == [] and fabricated.confidence == 0.0
    assert result.report.direct_answer_evidence_chunk_ids[0] in known


def test_followup_context_changes_what_is_retrieved_on_real_corpus(pipeline):
    investigator, client = pipeline
    first = investigator.run(PRESS07)

    plain = investigator.run(FOLLOW_UP)
    context = ConversationContext.from_results("inv-1", [first])
    followup = investigator.run(FOLLOW_UP, context=context)

    def ids(result):
        return {e.chunk_id for e in result.investigation.evidence}

    # Reference resolution reaches retrieval: the follow-up finds the 2014
    # requalification section that states it is not an approval/revalidation.
    assert "13_Process_Requalification_2014_0" in ids(followup)
    assert "13_Process_Requalification_2014_0" not in ids(plain)
    # Fresh retrieval still happened (not just carried-forward passages).
    fresh = [e for e in followup.investigation.evidence if e.retrieval_channels != ["carried-forward"]]
    assert len(fresh) >= 10
    # Context reached the reasoner, labelled as not evidence.
    assert "<CONVERSATION_CONTEXT>" in client.prompts[-1]
    assert "<CONVERSATION_CONTEXT>" not in client.prompts[-2]
    # Calibration still applies to the follow-up.
    ctj = [c for c in followup.report.claims if c.claim_type == ClaimType.CURRENT_TECHNICAL_JUSTIFICATION]
    assert ctj and all(c.confidence <= 0.70 for c in ctj)


def test_question_with_no_matching_evidence_is_not_answered(corpus_chunks):
    class NoCitation:
        def generate(self, prompt):
            if "investigation planner" in prompt:
                raise RuntimeError("planner offline")
            return json.dumps({"direct_answer": {"text": "The torque is 45 Nm.", "evidence_chunk_ids": []},
                               "claims": [], "unresolved_questions": []})

    retriever = HybridRetriever(BM25Searcher(corpus_chunks), None, EntityMatcher(corpus_chunks))
    result = build_investigator(retriever, gemini_client=NoCitation()).run(
        "What torque specification applies to the Line 9 conveyor gearbox bolts?"
    )
    # An uncited model answer never becomes the answer.
    assert "45 Nm" not in result.report.direct_answer
    assert "does not establish" in result.report.direct_answer
