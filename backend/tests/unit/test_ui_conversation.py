"""UI session-state tests for the investigation conversation: queued runs,
duplicate guards, context isolation between investigations, clarification,
failure/retry, reset, vector status wording and clickable citations."""

from types import SimpleNamespace

import pytest

import app.ui as ui
from app.investigation.models import (
    ClaimType,
    EvidenceItem,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)


class Rerun(Exception):
    pass


class FakePlaceholder:
    def __init__(self):
        self.calls = []

    def markdown(self, content, **kwargs):
        self.calls.append(content)

    def empty(self):
        self.calls.append("<cleared>")


@pytest.fixture
def session(monkeypatch):
    state = {}
    placeholder = FakePlaceholder()
    monkeypatch.setattr(ui.st, "session_state", state)
    monkeypatch.setattr(ui.st, "rerun", lambda: (_ for _ in ()).throw(Rerun()))
    monkeypatch.setattr(ui.st, "empty", lambda: placeholder)
    monkeypatch.setattr(ui.st, "markdown", lambda *a, **k: None)
    return SimpleNamespace(state=state, placeholder=placeholder)


def _result(question, answer="Answer.", cited=("c1",)):
    evidence = [EvidenceItem("c1", "d1", "07_Thermal_Validation_2001.txt", "B/07.txt",
                             "Press-07 limit text.", 0.9, ["q"], retrieval_channels=["bm25", "vector"])]
    claims = [ReasoningClaim("Limit derived in 2001.", EvidenceStance.SUPPORTS, ["c1"], 0.9, ClaimType.ORIGINAL_BASIS)]
    return SimpleNamespace(
        investigation=Investigation(question=question, evidence=evidence),
        report=InvestigationReport(question=question, claims=claims, direct_answer=answer,
                                   direct_answer_evidence_chunk_ids=list(cited)),
        timeline=SimpleNamespace(events=[]),
        evidence_analysis=SimpleNamespace(findings=[]),
        plan=SimpleNamespace(subquestions=[]),
    )


class FakeRetriever:
    vector_calls = 0
    vector_failures = 0
    last_vector_error = None


class FakeInvestigator:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def run(self, question, context=None, progress_callback=None):
        self.calls.append((question, context))
        if progress_callback:
            progress_callback("plan", "Planning investigation")
            progress_callback("done", "done")
        if self.error:
            raise self.error
        result = _result(question)
        result.conversation = context
        return result


def _pipeline(monkeypatch, investigator):
    meta = {"documents": 40, "chunks": 70, "vector_enabled": False,
            "vector_status": {"state": "off", "reason": "x"}, "retriever": FakeRetriever()}
    monkeypatch.setattr(ui, "load_pipeline", lambda: (investigator, meta))


# ---------------------------------------------------------------------------

def test_request_run_blocks_duplicates_while_pending(session):
    assert ui._request_run("Why is Press-07 limited?")
    assert not ui._request_run("Another question while running")
    assert session.state["pending_run"]["question"] == "Why is Press-07 limited?"


def test_followup_request_does_not_overwrite_the_composer(session):
    assert ui._request_run("Why is the 2014 study insufficient?", mode="follow-up", conv_id="inv-1")
    assert "pending_question" not in session.state
    assert session.state["pending_run"] == {
        "question": "Why is the 2014 study insufficient?", "mode": "follow-up", "conv_id": "inv-1"
    }


def test_new_investigation_creates_conversation_without_context(session, monkeypatch):
    investigator = FakeInvestigator()
    _pipeline(monkeypatch, investigator)
    ui._request_run("Why is Press-07 limited to 42.3°C?")
    with pytest.raises(Rerun):
        ui._run_pending(session.state["pending_run"])

    assert investigator.calls[0][1] is None  # no context for a new investigation
    conversation = session.state["conversations"][0]
    assert [t["kind"] for t in conversation["turns"]] == ["investigation"]
    assert "pending_run" not in session.state
    assert session.state["active_conv"] == conversation["id"]
    # Real stage progress was rendered from the pipeline callback.
    assert any("Planning investigation" in c for c in session.placeholder.calls)
    # Per-run retrieval facts come from evidence channels.
    assert conversation["turns"][0]["meta"]["retrieval_run"]["vector_passages"] == 1
    assert "retriever" not in conversation["turns"][0]["meta"]


def test_followup_context_comes_only_from_its_own_conversation(session, monkeypatch):
    investigator = FakeInvestigator()
    _pipeline(monkeypatch, investigator)
    session.state["conversations"] = [
        {"id": "inv-a", "title": "A", "turns": [
            {"id": 1, "kind": "investigation", "question": "Why is PX-999 set to 50°C?",
             "result": _result("Why is PX-999 set to 50°C?", answer="Other investigation."), "meta": {}}]},
        {"id": "inv-b", "title": "B", "turns": [
            {"id": 2, "kind": "investigation", "question": "Why is Press-07 limited to 42.3°C?",
             "result": _result("Why is Press-07 limited to 42.3°C?"), "meta": {}}]},
    ]
    ui._request_run("Why isn't the 2014 study enough?", mode="follow-up", conv_id="inv-b")
    with pytest.raises(Rerun):
        ui._run_pending(session.state["pending_run"])

    question, context = investigator.calls[0]
    assert context is not None and context.conversation_id == "inv-b"
    prompt = context.to_prompt()
    assert "Press-07" in prompt
    assert "PX-999" not in prompt and "Other investigation." not in prompt
    assert [t["kind"] for t in session.state["conversations"][1]["turns"]] == ["investigation", "follow-up"]
    assert len(session.state["conversations"][0]["turns"]) == 1


def test_referential_new_question_gets_clarification_and_no_pipeline_call(session, monkeypatch):
    monkeypatch.setattr(ui, "load_pipeline", lambda: pytest.fail("pipeline must not run"))
    ui._request_run("Why isn't that study enough?")
    with pytest.raises(Rerun):
        ui._run_pending(session.state["pending_run"])
    turn = session.state["conversations"][0]["turns"][0]
    assert turn["kind"] == "clarification" and turn["result"] is None
    assert "Which equipment, parameter, or document" in turn["message"]
    assert "pending_run" not in session.state


def test_failed_run_records_error_and_never_auto_retries(session, monkeypatch):
    investigator = FakeInvestigator(error=RuntimeError("429 RESOURCE_EXHAUSTED"))
    _pipeline(monkeypatch, investigator)
    ui._request_run("Why is Press-07 limited to 42.3°C?")
    with pytest.raises(Rerun):
        ui._run_pending(session.state["pending_run"])

    assert "pending_run" not in session.state and "run_question" not in session.state
    error = session.state["last_error"]
    assert error["mode"] == "new" and error["question"] == "Why is Press-07 limited to 42.3°C?"
    assert "result" not in session.state
    assert session.state.get("conversations", []) == []
    # Explicit retry queues exactly one run.
    assert ui._request_run(error["question"], mode=error["mode"], conv_id=error["conv_id"])
    assert "last_error" not in session.state


def test_reset_starts_fresh_but_keeps_listed_investigations(session):
    session.state.update({"conversations": [{"id": "inv-1", "title": "t", "turns": []}],
                          "active_conv": "inv-1", "active_id": 3, "result": object(),
                          "pending_run": {"question": "q"}, "last_error": {"q": 1}})
    ui._reset_workspace()
    for key in ("active_conv", "active_id", "result", "pending_run", "last_error"):
        assert key not in session.state
    assert session.state["conversations"][0]["id"] == "inv-1"
    assert session.state["pending_question"] == ui.HERO_QUESTION


def test_vector_state_is_on_only_after_successful_probe():
    on = ui._vector_state({"vector_enabled": True, "vector_status": {"state": "on"}})
    failed = ui._vector_state({"vector_enabled": False,
                               "vector_status": {"state": "failed", "reason": "Live query embedding failed: 403"}})
    off = ui._vector_state({"vector_enabled": False, "vector_status": {"state": "off", "reason": "no key"}})
    assert on[1].startswith("ON")
    assert failed[1].startswith("FAILED") and "403" in failed[2]
    assert off[1] == "OFF" and off[2] == "no key"


def test_retrieval_chip_reports_actual_run_not_configuration():
    failing = ui._retrieval_chip({"vector_enabled": True, "retrieval_run": {
        "vector_calls": 10, "vector_failures": 10, "vector_last_error": "403", "passages": 27, "vector_passages": 0}})
    assert "Vector failed in 10 of 10 searches" in failing
    working = ui._retrieval_chip({"vector_enabled": True, "retrieval_run": {
        "vector_calls": 10, "vector_failures": 0, "passages": 27, "vector_passages": 14}})
    assert "vector ranked <strong>14</strong> of 27" in working


def test_direct_answer_citations_open_the_real_passage(monkeypatch):
    rendered = []
    monkeypatch.setattr(ui.st, "markdown", lambda content, **kwargs: rendered.append(content))
    result = _result("q", answer="The limit dates from 2001.", cited=("c1", "missing"))
    ui.render_direct_answer(result)
    html = rendered[0]
    assert '<details class="cite">' in html
    assert "07_Thermal_Validation_2001.txt · c1" in html
    assert "B/07.txt" in html and "Press-07 limit text." in html
    assert "missing" not in html
    assert "\n" not in html


def test_classify_error_explains_missing_or_invalid_configuration():
    tone, title, message = ui._classify_error(RuntimeError("GEMINI_API_KEY environment variable is not set."))
    assert "not configured" in title and ".env" in message
    tone, title, message = ui._classify_error(RuntimeError("400 INVALID_ARGUMENT API key not valid"))
    assert "rejected" in title


def test_classify_error_explains_network_failures():
    class ProxyError(Exception):
        pass

    tone, title, message = ui._classify_error(ProxyError("403 Forbidden"))
    assert "could not be reached" in title and "network" in message
