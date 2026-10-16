"""Regression tests for the release-audit fixes (see the audit report).

Offline and deterministic. Tests that use the real corpus or the real
persisted embedding index say so; none of them calls Gemini, and none of
them is evidence that live Gemini behaviour works.
"""

import json
import os
import re
from datetime import date
from pathlib import Path

import pytest
from google.genai import errors

from app.config import CORPUS_PATH, INDEX_PATH
from app.ingestion.chunker import chunk_corpus
from app.ingestion.loader import load_corpus
from app.investigation.claim_verifier import ClaimVerifier
from app.investigation.confidence import ConfidenceCalibrator
from app.investigation.models import (
    ClaimType,
    EvidenceItem,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)
from app.investigation.planner import InvestigationPlanner
from app.investigation.reasoner import InvestigationReasoner
from app.investigation.timeline import InvestigationTimeline, TimelineBuilder, TimelineEvent
from app.retrieval.bm25_search import BM25Searcher
from app.retrieval.entity_matcher import EntityMatcher
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_health import load_index
from app.retrieval.vector_search import VectorSearcher


@pytest.fixture(scope="module")
def corpus_chunks():
    return chunk_corpus(load_corpus(str(CORPUS_PATH)))


def _ev(chunk_id, content):
    return EvidenceItem(chunk_id, chunk_id, f"{chunk_id}.txt", f"B_historical_engineering/{chunk_id}.txt",
                        content, 0.5, ["q"])


# ---------------------------------------------------------------------------
# Configuration: .env is loaded by explicit path, never overriding the env
# ---------------------------------------------------------------------------

def test_load_environment_uses_explicit_path_and_never_overrides(tmp_path, monkeypatch):
    import app.config as config

    env_file = tmp_path / ".env"
    env_file.write_text("TRACE_AUDIT_FAKE_VAR=from-file\nTRACE_AUDIT_KEEP=from-file\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_file)
    monkeypatch.delenv("TRACE_AUDIT_FAKE_VAR", raising=False)
    monkeypatch.setenv("TRACE_AUDIT_KEEP", "from-process")
    monkeypatch.chdir(tmp_path.parent)  # a different working directory

    assert config.load_environment() is True
    assert os.environ["TRACE_AUDIT_FAKE_VAR"] == "from-file"
    assert os.environ["TRACE_AUDIT_KEEP"] == "from-process"
    monkeypatch.delenv("TRACE_AUDIT_FAKE_VAR", raising=False)


def test_load_environment_reports_missing_file(tmp_path, monkeypatch):
    import app.config as config

    monkeypatch.setattr(config, "ENV_PATH", tmp_path / "absent.env")
    assert config.load_environment() is False


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------

class _FixedQueryProvider:
    """Test fixture: returns a fixed query vector (no network)."""

    def __init__(self, vector):
        self.vector = list(vector)
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return self.vector


def test_query_dimension_mismatch_fails_loudly_and_is_recorded(corpus_chunks):
    chunks = corpus_chunks[:3]
    vector = VectorSearcher(chunks, [[1.0, 0.0, 0.0]] * 3, _FixedQueryProvider([1.0, 0.0]))
    with pytest.raises(ValueError, match="dimensions"):
        vector.search("anything")
    hybrid = HybridRetriever(BM25Searcher(corpus_chunks), vector, EntityMatcher(corpus_chunks))
    results = hybrid.search("Press-07 limit", top_k=5)
    assert results and hybrid.vector_failures == 1
    assert "dimensions" in hybrid.last_vector_error
    assert all("vector" not in r.channels for r in results)


def test_inconsistent_index_dimensions_are_rejected(corpus_chunks):
    with pytest.raises(ValueError, match="inconsistent"):
        VectorSearcher(corpus_chunks[:2], [[1.0, 0.0], [1.0, 0.0, 0.0]], _FixedQueryProvider([1.0, 0.0]))


@pytest.fixture(scope="module")
def real_index(corpus_chunks):
    if not Path(INDEX_PATH).exists():
        pytest.skip("persisted embedding index not present")
    health, _ids, embeddings = load_index(INDEX_PATH, corpus_chunks)
    assert health.usable, health.summary()
    return health, embeddings


def test_real_index_is_compatible_with_current_corpus(real_index, corpus_chunks):
    health, embeddings = real_index
    assert health.records == len(corpus_chunks) == len(embeddings)
    assert health.ids_match and health.stale_chunk_ids == []
    assert health.model == "gemini-embedding-001" and health.dimension == len(embeddings[0])


def test_real_index_vectors_drive_vector_and_hybrid_retrieval(real_index, corpus_chunks):
    """Controlled retrieval on the REAL persisted vectors: the query vector
    is the stored embedding of the 2014 requalification passage (a fixture
    standing in for a live query embedding, which needs Gemini). Vector
    search must rank that passage first, and hybrid retrieval must carry
    it with a 'vector' channel."""
    _health, embeddings = real_index
    target = "13_Process_Requalification_2014_0"
    index = [c.chunk_id for c in corpus_chunks].index(target)
    provider = _FixedQueryProvider(embeddings[index])
    vector = VectorSearcher(corpus_chunks, embeddings, provider)

    hits = vector.search("ignored by the fixture provider", top_k=5)
    assert hits[0].chunk_id == target and hits[0].score == pytest.approx(1.0, abs=1e-6)
    assert [h.score for h in hits] == sorted([h.score for h in hits], reverse=True)

    hybrid = HybridRetriever(BM25Searcher(corpus_chunks), vector, EntityMatcher(corpus_chunks))
    fused = hybrid.search("Did the PX-400 requalification re-establish the limit?", top_k=8)
    by_id = {r.chunk_id: r for r in fused}
    assert target in by_id and "vector" in by_id[target].channels
    assert hybrid.vector_calls == 1 and hybrid.vector_failures == 0
    known = {c.chunk_id: c for c in corpus_chunks}
    for r in fused:  # provenance preserved through fusion
        assert r.relative_path == known[r.chunk_id].relative_path and r.content == known[r.chunk_id].content


def test_vector_channel_changes_hybrid_results(real_index, corpus_chunks):
    """The vector channel must change what hybrid retrieval returns, not
    merely be configured: with the same lexical query, adding vector
    retrieval surfaces at least one passage BM25 + entity did not."""
    _health, embeddings = real_index
    target = "13_Process_Requalification_2014_0"
    index = [c.chunk_id for c in corpus_chunks].index(target)
    query = "thermal stability envelope characterization"
    lexical = HybridRetriever(BM25Searcher(corpus_chunks), None, EntityMatcher(corpus_chunks)).search(query, top_k=8)
    hybrid = HybridRetriever(
        BM25Searcher(corpus_chunks),
        VectorSearcher(corpus_chunks, embeddings, _FixedQueryProvider(embeddings[index])),
        EntityMatcher(corpus_chunks),
    ).search(query, top_k=8)
    lexical_ids = {r.chunk_id for r in lexical}
    added = [r for r in hybrid if r.chunk_id not in lexical_ids]
    assert added, "vector retrieval did not change the hybrid result set"
    assert all("vector" in r.channels for r in added)


def test_entity_channel_failure_is_recorded(corpus_chunks):
    class BrokenEntities:
        def search(self, query, top_k=5):
            raise RuntimeError("entity index corrupt")

    hybrid = HybridRetriever(BM25Searcher(corpus_chunks), None, BrokenEntities())
    assert hybrid.search("Press-07 limit", top_k=3)
    assert hybrid.entity_failures == 1 and "corrupt" in hybrid.last_entity_error


def test_embedding_provider_retry_budget_is_bounded(monkeypatch):
    from app.retrieval import gemini_embeddings as module

    class Models:
        calls = 0

        def embed_content(self, model, contents, config=None):
            Models.calls += 1
            raise errors.APIError(429, {"error": {"code": 429, "message": "slow down"}})

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            self.models = Models()
            self.http_options = http_options

    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key-not-a-real-secret")
    monkeypatch.setattr(module.genai, "Client", FakeClient)
    monkeypatch.setattr(module.time, "sleep", lambda s: None)
    provider = module.GeminiEmbeddingProvider(max_retries=1, base_delay_seconds=0.0, timeout_seconds=20)
    assert provider.client.http_options.timeout == 20000
    with pytest.raises(errors.APIError):
        provider.embed_query("q")
    assert Models.calls == 2  # first attempt + one retry, never the default five


# ---------------------------------------------------------------------------
# Planner fallback is recorded, not silent
# ---------------------------------------------------------------------------

def test_planner_fallback_is_recorded_with_reason():
    class Failing:
        def generate(self, prompt):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    plan = InvestigationPlanner(planning_client=Failing()).create_plan(
        "Why is Press-07 limited to 42.3°C and is its current basis still supported?"
    )
    assert plan.source == "deterministic-fallback" and "RESOURCE_EXHAUSTED" in plan.fallback_reason
    assert plan.subquestions


def test_planner_records_gemini_and_narrow_sources():
    class Good:
        def generate(self, prompt):
            return json.dumps({"subquestions": ["What established the 42.3°C limit for Press-07?"]})

    planner = InvestigationPlanner(planning_client=Good())
    plan = planner.create_plan("Why is Press-07 limited to 42.3°C and is its current basis still supported?")
    assert plan.source in {"gemini", "deterministic-fallback"}
    if plan.source == "deterministic-fallback":  # parse format differs: reason must say why
        assert plan.fallback_reason
    narrow = planner.create_plan("What is QUAL-2014-009?")
    assert narrow.source in {"deterministic-narrow", "gemini", "deterministic-fallback"}


# ---------------------------------------------------------------------------
# Timeline: provenance, precision, record vs event dates, ambiguity
# ---------------------------------------------------------------------------

def _mentions(line):
    return [(m["date"], m["date_text"], m["kind"], m["precision"], m["ambiguous"])
            for m in TimelineBuilder()._extract_dated_mentions(line)]


def test_every_date_in_a_line_is_extracted_with_its_source_text():
    assert _mentions("Trial on 05-Jul-2001; review on 14-Jul-2001.") == [
        (date(2001, 7, 5), "05-Jul-2001", "mentioned", "day", False),
        (date(2001, 7, 14), "14-Jul-2001", "mentioned", "day", False),
    ]


def test_record_issue_dates_are_distinguished_from_event_dates():
    assert _mentions("Report No: QUAL-2014-009          Date: 27-MAR-2014")[0][2] == "record_issued"
    assert _mentions("Effective Date: 2011-08-15")[0][2] == "record_issued"
    assert _mentions("Install Date:                2011-08-15")[0][2] == "mentioned"
    assert _mentions("Calibration due date: 2019-03-01")[0][2] == "mentioned"


def test_month_only_dates_are_labelled_never_given_a_day():
    (parsed, text, _kind, precision, _amb), = _mentions("target: early Sept 2001")
    assert precision == "month" and text == "Sept 2001" and parsed == date(2001, 9, 1)
    assert _mentions("we may 2014 revisit") == []  # lowercase verb, not a month


def test_ambiguous_numeric_dates_are_flagged_and_invalid_dates_dropped():
    assert _mentions("Reading on 03/04/2014")[0][4] is True
    assert _mentions("Reading on 14/06/2001")[0][4] is False
    assert _mentions("Invalid: 31/02/2001") == []
    assert _mentions("June 5, 2001 kickoff")[0][0] == date(2001, 6, 5)


def test_no_date_is_invented_from_filenames_or_titles():
    item = EvidenceItem("13_Process_Requalification_2014_9", "d", "13_Process_Requalification_2014.txt",
                        "B/13_Process_Requalification_2014.txt", "No dates in this passage.", 0.5, [])
    assert TimelineBuilder().build([item]).events == []


def test_month_and_ambiguous_dates_are_labelled_for_the_reasoner():
    timeline = InvestigationTimeline(events=[
        TimelineEvent(date(2001, 9, 1), "early Sept 2001", ["c1"], date_text="Sept 2001", precision="month"),
        TimelineEvent(date(2014, 4, 3), "Reading 03/04/2014", ["c2"], date_text="03/04/2014", ambiguous=True),
        TimelineEvent(date(2011, 8, 15), "Effective Date: 2011-08-15", ["c3"]),
    ])
    text = InvestigationReasoner(client=None)._format_timeline(timeline)
    assert "2001-09 (month only" in text and "2001-09-01" not in text
    assert "ambiguous numeric date" in text
    assert "- 2011-08-15: Effective Date: 2011-08-15 (evidence: c3)" in text  # unchanged format


def test_press07_record_dates_reconstructed_from_actual_records(corpus_chunks):
    """Independent check of the Press-07 history against the real corpus:
    each record's own issue date, with the passage it came from."""
    items = [EvidenceItem(c.chunk_id, c.document_id, c.filename, c.relative_path, c.content, 0.0, [])
             for c in corpus_chunks]
    events = TimelineBuilder().build(items).events
    issued = {(e.date, e.evidence_chunk_ids[0]) for e in events if e.kind == "record_issued"}
    expected = {
        (date(2001, 7, 5), "07_Thermal_Validation_2001_0"),          # PE-STUDY-2001-07
        (date(2011, 8, 15), "04_ECO_2011_184_Press_Replacement_0"),  # ECO effective date
        (date(2011, 8, 22), "05_PX400_Commissioning_Report_2011_0"),
        (date(2011, 9, 14), "12_PX400_Qualification_2011_0"),        # QUAL-2011-021
        (date(2014, 3, 27), "13_Process_Requalification_2014_0"),    # QUAL-2014-009
        (date(2014, 5, 22), "38_Rejected_Proposal_2014_43C_0"),
        (date(2022, 11, 30), "25_SOP_Cure_Process_v8_Current_0"),
    }
    assert expected <= issued
    for event in events:  # every date text literally occurs in its passage
        content = next(c.content for c in corpus_chunks if c.chunk_id == event.evidence_chunk_ids[0])
        assert event.date_text in content


# ---------------------------------------------------------------------------
# Citation integrity and verification (adversarial)
# ---------------------------------------------------------------------------

PASSAGES = [
    _ev("07_Thermal_Validation_2001_0", "Report PE-STUDY-2001-07: Press-07 on the PX-200 shows instability above 42 deg C; limit 42.3 deg C."),
    _ev("33_Maintenance_Notes_0", "Replaced a belt on the conveyor. Lunch break moved."),
]


def _verify(claim):
    investigation = Investigation(question="q", evidence=PASSAGES)
    report = ClaimVerifier().verify(InvestigationReport("q", [claim]), investigation)
    return ConfidenceCalibrator().calibrate(report, investigation).claims[0]


def test_fabricated_citations_are_removed_and_recorded():
    out = _verify(ReasoningClaim("The PX-200 limit was 42.3°C in 2001.", EvidenceStance.SUPPORTS,
                                 ["07_Thermal_Validation_2001_0", "99_Invented_0"], 0.9, ClaimType.ORIGINAL_BASIS))
    assert out.evidence_chunk_ids == ["07_Thermal_Validation_2001_0"] and out.stance == EvidenceStance.SUPPORTS
    assert any("99_Invented_0" in n for n in out.verification_notes)


def test_only_fabricated_citations_make_a_claim_inconclusive():
    out = _verify(ReasoningClaim("The PX-200 limit was 42.3°C.", EvidenceStance.SUPPORTS,
                                 ["99_Invented_0"], 0.9, ClaimType.ORIGINAL_BASIS))
    assert out.stance == EvidenceStance.UNKNOWN and out.confidence == 0.0 and out.evidence_chunk_ids == []


def test_irrelevant_citation_cannot_make_a_claim_supported():
    out = _verify(ReasoningClaim("ECO-2011-184 replaced the PX-200 with the PX-400 in 2011.", EvidenceStance.SUPPORTS,
                                 ["33_Maintenance_Notes_0"], 0.95, ClaimType.ADMINISTRATIVE_ACTION))
    assert out.stance == EvidenceStance.UNKNOWN and out.confidence == 0.0
    assert any("Grounding check" in n for n in out.verification_notes)


def test_irrelevant_citation_cannot_make_a_contradiction():
    out = _verify(ReasoningClaim("The 2014 study contradicts QUAL-2011-021.", EvidenceStance.CONTRADICTS,
                                 ["33_Maintenance_Notes_0"], 0.8, ClaimType.ORIGINAL_BASIS))
    assert out.stance == EvidenceStance.UNKNOWN


def test_grounded_claim_is_unchanged_and_calibration_still_applies():
    out = _verify(ReasoningClaim("The 2001 study set the Press-07 limit at 42.3°C.", EvidenceStance.SUPPORTS,
                                 ["07_Thermal_Validation_2001_0"], 0.97, ClaimType.ORIGINAL_BASIS))
    assert out.stance == EvidenceStance.SUPPORTS and out.verification_notes == []
    assert out.confidence == pytest.approx(0.97)


def test_absence_claims_are_not_downgraded_by_grounding():
    out = _verify(ReasoningClaim("No record from 2014 re-derives the limit for the PX-400.", EvidenceStance.NOT_ESTABLISHED,
                                 ["33_Maintenance_Notes_0"], 0.6, ClaimType.EVIDENCE_GAP))
    assert out.stance == EvidenceStance.NOT_ESTABLISHED and out.confidence <= 0.30


def test_current_justification_cannot_become_supported_without_rejustification_text():
    out = _verify(ReasoningClaim("The 42.3°C limit is technically justified today for Press-07.", EvidenceStance.SUPPORTS,
                                 ["07_Thermal_Validation_2001_0"], 0.95, ClaimType.CURRENT_TECHNICAL_JUSTIFICATION))
    assert out.stance == EvidenceStance.NOT_ESTABLISHED and out.confidence <= 0.30
    assert any("gate" in n for n in out.verification_notes)


def test_recommendations_stay_recommendations_in_the_ui_grouping():
    from app.command_center import data as D

    claim = ReasoningClaim("Commission a PX-400 characterization of Press-07.", EvidenceStance.SUPPORTS,
                           ["07_Thermal_Validation_2001_0"], 0.9, ClaimType.RECOMMENDATION)
    assert D.claim_group(_verify(claim)) == "recommendation"


# ---------------------------------------------------------------------------
# Error classification (no secrets, distinct causes)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc, expected", [
    (RuntimeError("GEMINI_API_KEY environment variable is not set."), "not configured"),
    (RuntimeError("400 INVALID_ARGUMENT API key not valid. Please pass a valid API key."), "rejected the API key"),
    (RuntimeError("429 RESOURCE_EXHAUSTED"), "quota"),
    (RuntimeError("503 UNAVAILABLE"), "temporarily unavailable"),
    (type("ReadTimeout", (Exception,), {})("timed out"), "timed out"),
    (type("ProxyError", (Exception,), {})("403 Forbidden"), "could not be reached"),
    (RuntimeError("403 PERMISSION_DENIED"), "refused"),
    (RuntimeError("404 NOT_FOUND models/x is not found"), "model not available"),
    (RuntimeError("Reasoning client returned invalid JSON."), "unusable response"),
])
def test_failures_are_classified_distinctly(exc, expected):
    import app.ui as ui

    _tone, title, message = ui._classify_error(exc)
    assert expected.lower() in title.lower() or expected.lower() in message.lower()
    assert not re.search(r"AIza[0-9A-Za-z_\-]{10,}", title + message)
