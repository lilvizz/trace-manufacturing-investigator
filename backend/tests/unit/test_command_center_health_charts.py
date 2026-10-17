"""Health states and chart specs for the Command Center (offline).

Health rows must come from real checks and never conflate: index present vs
loaded, loaded vs query embedding working, embedding working vs vector
contribution, retrieval vs Gemini. Charts must build valid Vega-Lite for
real data, including with a focused node/event (this caught a runtime
error in nested Altair conditions during the audit)."""

from types import SimpleNamespace

import pytest

from app.command_center import charts as CH
from app.command_center import components as C
from app.command_center import data as D


@pytest.fixture
def session(monkeypatch):
    state = {}
    monkeypatch.setattr(C.st, "session_state", state)
    return state


def _meta(**index):
    base = {"present": True, "loaded": True, "ids_match": True, "dimension": 3072, "records": 70,
            "stale_chunks": 0, "model": "gemini-embedding-001", "error": ""}
    base.update(index)
    return {"documents": 40, "chunks": 70,
            "vector_status": {"state": "on", "reason": "", "api_key_present": True, "index": base, "probe_dimension": 3072}}


def _states(rows):
    return {name: state for name, state, _why in rows}


def test_healthy_index_but_no_run_is_not_verified(session):
    states = _states(C.health_rows(_meta()))
    assert states["Embedding index"] == "HEALTHY" and states["Query embedding"] == "HEALTHY"
    assert states["Gemini planning & reasoning"] == "NOT VERIFIED"
    assert states["Vector contribution"] == "NOT VERIFIED"


def test_missing_index_fails_with_recovery_command(session):
    rows = C.health_rows(_meta(present=False, loaded=False))
    name, state, why = next(r for r in rows if r[0] == "Embedding index")
    assert state == "FAILED" and "python -m app.retrieval.indexer" in why


def test_stale_index_is_degraded(session):
    assert _states(C.health_rows(_meta(stale_chunks=3)))["Embedding index"] == "DEGRADED"


def test_missing_key_and_failed_probe_are_failures(session):
    meta = _meta()
    meta["vector_status"].update(state="failed", reason="Live query embedding failed: ProxyError", api_key_present=False)
    states = _states(C.health_rows(meta))
    assert states["Query embedding"] == "FAILED" and states["Gemini API key"] == "FAILED"


def test_vector_contribution_comes_from_the_last_run(session):
    session["last_pipeline_run"] = {"ok": True, "at": "t", "plan_source": "gemini", "vector_calls": 10,
                                    "vector_failures": 0, "vector_passages": 6, "total_seconds": 12.0}
    states = _states(C.health_rows(_meta()))
    assert states["Vector contribution"] == "HEALTHY" and states["Gemini planning & reasoning"] == "HEALTHY"
    session["last_pipeline_run"].update(vector_failures=10, vector_passages=0)
    assert _states(C.health_rows(_meta()))["Vector contribution"] == "DEGRADED"
    session["last_pipeline_run"].update(plan_source="deterministic-fallback", plan_fallback_reason="429")
    assert _states(C.health_rows(_meta()))["Gemini planning & reasoning"] == "DEGRADED"


def test_failed_run_and_startup_failure_are_reported(session):
    session["last_pipeline_run"] = {"ok": False, "at": "t", "title": "Gemini could not be reached"}
    assert _states(C.health_rows(_meta()))["Gemini planning & reasoning"] == "FAILED"
    rows = C.health_rows(None, RuntimeError("GEMINI_API_KEY environment variable is not set."))
    assert rows[0][1] == "FAILED" and "not configured" in rows[0][2]


def _real_docs():
    from app.config import CORPUS_PATH
    from app.ingestion.loader import load_corpus

    return load_corpus(str(CORPUS_PATH))


def test_corpus_graph_edges_are_textual_facts():
    docs = _real_docs()
    nodes, edges = D.corpus_graph(docs)
    by_file = {d["filename"]: d for d in docs}
    ids = D.document_record_ids(docs)
    for edge in edges:
        source_text = by_file[edge.source[4:]]["content"]
        if edge.relation == "references":
            assert ids[edge.target[4:]] in source_text  # B really contains A's header record ID
        else:
            assert edge.target.split(":", 1)[1] in D.equipment_mentions(source_text)
    assert "SOP-CURE-07" not in ids.values()  # shared by every SOP revision: never linked


def test_charts_build_with_and_without_focus():
    docs = _real_docs()
    nodes, edges = D.corpus_graph(docs)
    for focus in (None, nodes[0].id, "equip:px400"):
        CH.graph_chart(nodes, edges, focus_id=focus, label_kinds=("equipment",),
                       x_domain=(-1.75, 1.75), y_domain=(-1.2, 1.2)).to_dict()
    assert CH.neighbours(edges, nodes[0].id)
    from datetime import date

    rows = [D.TimelineRow(0, date(2001, 7, 5), "05 Jul 2001", "2001", "h", "d", [], "event", ["c"], "Engineering record",
                          "f.txt", "B/f.txt", True, [], precision="day"),
            D.TimelineRow(1, date(2001, 9, 1), "Sep 2001", "2001", "h", "d", [], "event", ["c"], "Engineering record",
                          "f.txt", "B/f.txt", False, [], precision="month")]
    for focus in (None, 0, 1):
        CH.timeline_chart(rows, focus_idx=focus).to_dict()
    frame = CH.timeline_frame(rows)
    assert list(frame["kind"]) == ["Dated statement", "Month-only date"]
    assert "(month only)" in frame["date_label"][1]
