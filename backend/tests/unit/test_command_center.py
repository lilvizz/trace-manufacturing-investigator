"""Tests for the Command Center data and export layers (app/command_center).

These guard the evidence-integrity rules of the multi-page UI:
- displayed confidence caps match ConfidenceCalibrator's real behaviour
- nothing (claim, edge, event, gap) appears that the result does not record
- unverified items (open questions, recommendations) are never labelled verified
- exports contain only values present on the result
and run the whole data layer once over the REAL corpus with real retrieval,
timeline, analyzer, verifier and calibrator (only Gemini is scripted).
"""

import json
import re
from datetime import date
from types import SimpleNamespace

import pytest

from app.command_center import data as D
from app.command_center import export as X
from app.investigation.confidence import ConfidenceCalibrator
from app.investigation.models import (
    ClaimType,
    EvidenceFinding,
    EvidenceItem,
    EvidenceStance,
    FindingType,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)
from app.investigation.timeline import InvestigationTimeline, TimelineEvent


def _ev(chunk_id, content, path):
    return EvidenceItem(chunk_id, chunk_id.rsplit("_", 1)[0], path.split("/")[-1], path, content, 0.5,
                        ["q1"], retrieval_channels=["bm25"])


def _result(claims=(), evidence=(), findings=(), events=(), unresolved=(), answer="", answer_ids=(),
            question="Why is Press-07 limited to 42.3°C on the PX-400?"):
    return SimpleNamespace(
        investigation=Investigation(question=question, evidence=list(evidence)),
        report=InvestigationReport(question=question, claims=list(claims), unresolved_questions=list(unresolved),
                                   direct_answer=answer, direct_answer_evidence_chunk_ids=list(answer_ids)),
        timeline=InvestigationTimeline(events=list(events)),
        evidence_analysis=SimpleNamespace(findings=list(findings)),
        plan=SimpleNamespace(subquestions=["q1"]),
    )


EVIDENCE = [
    _ev("07_Thermal_Validation_2001_0", "Report No: PE-STUDY-2001-07  Date: 05-JUL-2001\nPress-07 PX-200 limit 42.3 C.",
        "B_historical_engineering/07_Thermal_Validation_2001.txt"),
    _ev("05_PX400_Commissioning_Report_2011_0", "Commissioning of the PX-400 on 22-AUG-2011.",
        "A_asset_equipment_history/05_PX400_Commissioning_Report_2011.txt"),
    _ev("38_Rejected_Proposal_2014_43C_0", "Proposal to raise the limit, no date line.",
        "G_distractors/38_Rejected_Proposal_2014_43C.txt"),
]


def test_stance_caps_match_the_calibrator():
    evidence = [EVIDENCE[0]]
    investigation = Investigation(question="q", evidence=evidence)
    for stance, cap in D.STANCE_CAPS.items():
        claim = ReasoningClaim("s", stance, [evidence[0].chunk_id], 1.0, ClaimType.ORIGINAL_BASIS)
        report = ConfidenceCalibrator().calibrate(InvestigationReport("q", [claim]), investigation)
        assert report.claims[0].confidence == pytest.approx(cap)


def test_calibration_limits_name_the_real_ceilings():
    claim = ReasoningClaim("s", EvidenceStance.NOT_ESTABLISHED, ["c"], 0.3,
                           ClaimType.CURRENT_TECHNICAL_JUSTIFICATION)
    notes = " ".join(D.calibration_limits(claim))
    assert "70%" in notes and "30%" in notes
    uncited = ReasoningClaim("s", EvidenceStance.UNKNOWN, [], 0.0, ClaimType.ORIGINAL_BASIS)
    assert any("0%" in n for n in D.calibration_limits(uncited))


def test_claim_groups_follow_backend_stance_and_type():
    S, T = EvidenceStance, ClaimType
    assert D.claim_group(ReasoningClaim("", S.SUPPORTS, [], 0, T.RECOMMENDATION)) == "recommendation"
    assert D.claim_group(ReasoningClaim("", S.SUPPORTS, [], 0, T.EVIDENCE_GAP)) == "not_established"
    assert D.claim_group(ReasoningClaim("", S.NOT_ESTABLISHED, [], 0, T.ORIGINAL_BASIS)) == "not_established"
    assert D.claim_group(ReasoningClaim("", S.CONTRADICTS, [], 0, T.ORIGINAL_BASIS)) == "contradicted"
    assert D.claim_group(ReasoningClaim("", S.CONTEXT_ONLY, [], 0, T.ORIGINAL_BASIS)) == "context"
    assert D.claim_group(ReasoningClaim("", S.UNKNOWN, [], 0, None)) == "inconclusive"


def test_gap_register_labels_provenance_and_never_invents_items():
    assert D.gap_register(_result()) == []
    result = _result(
        claims=[
            ReasoningClaim("No re-derivation found.", EvidenceStance.NOT_ESTABLISHED, [EVIDENCE[0].chunk_id], 0.3, ClaimType.EVIDENCE_GAP),
            ReasoningClaim("Commission a study.", EvidenceStance.SUPPORTS, [EVIDENCE[1].chunk_id], 0.7, ClaimType.RECOMMENDATION),
        ],
        evidence=EVIDENCE,
        findings=[EvidenceFinding(FindingType.EVIDENCE_GAP, "Analyzer gap.", [EVIDENCE[0].chunk_id])],
        unresolved=["Who approved SOP v8?"],
    )
    by_kind = {i.kind: i for i in D.gap_register(result)}
    assert set(by_kind) == {"verified_gap", "claimed_gap", "recommendation", "open_question"}
    assert by_kind["verified_gap"].verified and by_kind["claimed_gap"].verified
    assert not by_kind["open_question"].verified and not by_kind["recommendation"].verified
    assert by_kind["claimed_gap"].claim_idx == 0


def test_timeline_rows_use_extracted_dates_and_neutral_lane_for_unlisted_folders():
    events = [
        TimelineEvent(date(2001, 7, 5), "Report No: PE-STUDY-2001-07  Date: 05-JUL-2001", [EVIDENCE[0].chunk_id]),
        TimelineEvent(date(2011, 8, 22), "Commissioning of the PX-400 on 22-AUG-2011.", [EVIDENCE[1].chunk_id]),
    ]
    claim = ReasoningClaim("PX-400 commissioned.", EvidenceStance.SUPPORTS, [EVIDENCE[1].chunk_id], 0.9, ClaimType.CURRENT_CAPABILITY)
    result = _result(claims=[claim], evidence=EVIDENCE, events=events)
    rows, total = D.timeline_rows(result)
    assert total == 2 and {r.date for r in rows} == {date(2001, 7, 5), date(2011, 8, 22)}
    px400 = next(r for r in rows if r.date.year == 2011)
    assert px400.cited and px400.claim_idxs == [0] and px400.lane == "Asset & equipment history"
    assert D.lane_for_path("G_distractors/x.txt") == D.UNCATEGORISED_LANE
    assert [d["filename"] for d in D.undated_documents(result)] == ["38_Rejected_Proposal_2014_43C.txt"]


def test_relationship_graph_only_draws_recorded_links():
    claims = [
        ReasoningClaim("Limit from 2001.", EvidenceStance.SUPPORTS, [EVIDENCE[0].chunk_id], 0.9, ClaimType.ORIGINAL_BASIS),
        ReasoningClaim("Fabricated.", EvidenceStance.SUPPORTS, ["99_Invented_0"], 0.9, ClaimType.ORIGINAL_BASIS),
    ]
    events = [TimelineEvent(date(2011, 8, 22), "Commissioning of the PX-400 on 22-AUG-2011.", [EVIDENCE[1].chunk_id])]
    nodes, edges = D.relationship_graph(_result(claims=claims, evidence=EVIDENCE, events=events))
    ids = {n.id for n in nodes}
    assert "claim:0" in ids and "claim:1" not in ids  # no node for an unresolvable citation
    valid_chunks = {e.chunk_id for e in EVIDENCE}
    for edge in edges:
        assert edge.chunk_ids and set(edge.chunk_ids) <= valid_chunks
        assert edge.source in ids and edge.target in ids
    # Equipment links only where the identifier literally appears in a passage.
    for edge in edges:
        if edge.relation == "mentioned in":
            key = edge.source.split(":", 1)[1]
            assert all(key in D.equipment_mentions(next(e.content for e in EVIDENCE if e.chunk_id == c)) for c in edge.chunk_ids)
    assert all(0.0 <= n.y <= 1.0 for n in nodes)


def test_exports_contain_only_result_values():
    claim = ReasoningClaim("Limit from 2001.", EvidenceStance.SUPPORTS, [EVIDENCE[0].chunk_id], 0.9, ClaimType.ORIGINAL_BASIS)
    result = _result(claims=[claim], evidence=EVIDENCE, answer="The limit dates from 2001.",
                     answer_ids=[EVIDENCE[0].chunk_id], unresolved=["Open?"])
    conversation = {"id": "inv-1", "title": "Q", "turns": [
        {"id": 1, "kind": "investigation", "question": "Q", "result": result, "meta": {}, "created_at": "t"},
        {"id": 2, "kind": "clarification", "question": "that?", "result": None, "message": "Which one?"},
    ]}
    md = X.case_to_markdown(conversation)
    assert "The limit dates from 2001." in md and EVIDENCE[0].relative_path in md and "Which one?" in md
    payload = json.loads(X.case_to_json(conversation))
    turn = payload["turns"][0]["result"]
    assert turn["claims"][0]["calibrated_confidence"] == 0.9
    assert [e["chunk_id"] for e in turn["evidence"]] == [e.chunk_id for e in EVIDENCE]
    assert payload["turns"][1]["message"] == "Which one?"


# ---------------------------------------------------------------------------
# Real corpus, real deterministic pipeline, scripted Gemini only.
# ---------------------------------------------------------------------------

class _ScriptedGemini:
    def generate(self, prompt):
        if "investigation planner" in prompt:
            raise RuntimeError("planner offline")
        ids = re.findall(r"EVIDENCE CHUNK: (\S+)", prompt)
        first = next((i for i in ids if i.startswith("07_")), ids[0])
        return json.dumps({
            "direct_answer": {"text": "The limit dates from 2001.", "evidence_chunk_ids": [first]},
            "claims": [
                {"statement": "The 2001 study set the limit.", "stance": "supports", "claim_type": "original_basis",
                 "evidence_chunk_ids": [first], "confidence": 0.9},
                {"statement": "Uses an invented source.", "stance": "supports", "claim_type": "original_basis",
                 "evidence_chunk_ids": ["99_Invented_Document_0"], "confidence": 0.9},
            ],
            "unresolved_questions": ["Was the PX-400 characterised?"],
        })


def test_data_layer_over_real_pipeline_result():
    from app.config import CORPUS_PATH
    from app.ingestion.chunker import chunk_corpus
    from app.ingestion.loader import load_corpus
    from app.investigation.build import build_investigator
    from app.retrieval.bm25_search import BM25Searcher
    from app.retrieval.entity_matcher import EntityMatcher
    from app.retrieval.hybrid_retriever import HybridRetriever

    chunks = chunk_corpus(load_corpus(str(CORPUS_PATH)))
    retriever = HybridRetriever(BM25Searcher(chunks), None, EntityMatcher(chunks))
    result = build_investigator(retriever, gemini_client=_ScriptedGemini()).run(
        "Why is Press-07 limited to 42.3°C, and is its current technical basis still supported?"
    )
    evidence_ids = {e.chunk_id for e in result.investigation.evidence}

    rows, total = D.timeline_rows(result)
    extracted = {(e.date, c) for e in result.timeline.events for c in e.evidence_chunk_ids}
    assert rows and total == len(result.timeline.events)
    for row in rows:
        assert all((row.date, c) in extracted for c in row.chunk_ids)

    nodes, edges = D.relationship_graph(result)
    assert edges
    assert all(set(e.chunk_ids) <= evidence_ids for e in edges)
    claim_nodes = [n for n in nodes if n.kind == "claim"]
    assert claim_nodes and all(set(n.chunk_ids) <= evidence_ids for n in claim_nodes)

    corpus = D.corpus_rows(chunks, result)
    assert len(corpus) == len(chunks)
    assert sum(r["retrieved"] for r in corpus) == len(evidence_ids)

    conversation = {"id": "inv-1", "title": "t", "turns": [
        {"id": 1, "kind": "investigation", "question": "t", "result": result, "meta": {}}]}
    payload = json.loads(X.case_to_json(conversation))
    cited = {c for claim in payload["turns"][0]["result"]["claims"] for c in claim["evidence_chunk_ids"]}
    assert "99_Invented_Document_0" not in cited
    assert cited <= evidence_ids


def test_completed_run_records_timing_timestamp_and_next_page(monkeypatch):
    """_run_pending additions used by the Command Center shell and history."""
    import app.ui as ui

    class Rerun(Exception):
        pass

    class Placeholder:
        def markdown(self, *a, **k):
            pass

        def empty(self):
            pass

    state = {}
    monkeypatch.setattr(ui.st, "session_state", state)
    monkeypatch.setattr(ui.st, "rerun", lambda: (_ for _ in ()).throw(Rerun()))
    monkeypatch.setattr(ui.st, "empty", lambda: Placeholder())
    result = _result(claims=[], evidence=EVIDENCE)

    class Investigator:
        def run(self, question, context=None, progress_callback=None):
            progress_callback("plan", "Planning investigation")
            progress_callback("done", "")
            return result

    monkeypatch.setattr(ui, "load_pipeline", lambda: (Investigator(), {"documents": 1, "chunks": 1}))
    assert ui._request_run("Why is Press-07 limited to 42.3°C?")
    with pytest.raises(Rerun):
        ui._run_pending(state["pending_run"])
    turn = state["conversations"][0]["turns"][0]
    assert state["cc_after_run"] == "new"
    assert "plan" in turn["meta"]["stage_seconds"] and turn["meta"]["total_seconds"] >= 0
    assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", turn["created_at"])


def test_classic_ui_remains_available(monkeypatch):
    import app.ui as ui

    called = []
    monkeypatch.setenv("TRACE_CLASSIC_UI", "1")
    monkeypatch.setattr(ui, "main_classic", lambda: called.append(True))
    ui.main()
    assert called == [True]
