"""Regression tests for the redesigned TRACE workspace (app/ui.py).

These cover the evidence-integrity rules the redesign must keep:
- the headline never shows a stale positive statement under NOT ESTABLISHED
- "not assessed" is never conflated with "not established"
- the current-justification tile always shows the most conservative claim
- derived chips/labels only come from data that actually exists
- markup never contains blank lines (Streamlit raw-HTML block rule)
"""

from datetime import date
from types import SimpleNamespace

import app.ui as ui


def _capture(monkeypatch):
    rendered = []
    monkeypatch.setattr(ui.st, "markdown", lambda content, **kwargs: rendered.append(content))
    return rendered


def _no_blank_lines(html):
    assert not [line for line in html.split("\n") if line.strip() == ""], html


def _claim(stance, claim_type, confidence=0.9, statement="s", ids=("c1",)):
    return SimpleNamespace(
        stance=stance,
        claim_type=claim_type,
        confidence=confidence,
        statement=statement,
        evidence_chunk_ids=list(ids),
    )


def _evidence(chunk_id, content, path="B_historical_engineering/doc.txt", filename="doc.txt"):
    return SimpleNamespace(
        chunk_id=chunk_id,
        document_id=filename.rsplit(".", 1)[0],
        filename=filename,
        relative_path=path,
        content=content,
        retrieval_score=0.5,
        retrieved_for=["q1"],
    )


def _result(claims=(), evidence=(), findings=(), question="Is the current technical basis still supported?", events=(), unresolved=()):
    return SimpleNamespace(
        report=SimpleNamespace(
            claims=list(claims),
            unresolved_questions=list(unresolved),
            direct_answer="",
            direct_answer_evidence_chunk_ids=[],
        ),
        investigation=SimpleNamespace(question=question, evidence=list(evidence)),
        evidence_analysis=SimpleNamespace(findings=list(findings)),
        timeline=SimpleNamespace(events=list(events)),
        plan=SimpleNamespace(subquestions=["q1"]),
    )


def test_assessment_headline_never_shows_stale_positive_statement(monkeypatch):
    claim = _claim(
        ui.EvidenceStance.NOT_ESTABLISHED,
        ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
        confidence=0.3,
        statement="The current 42.3°C limit is technically justified for the PX-400 configuration.",
    )
    rendered = _capture(monkeypatch)
    ui.render_assessment(_result([claim]))

    html = rendered[0]
    assert "is technically justified for the PX-400" not in html
    assert "does not establish the current technical basis" in html
    assert "NOT ESTABLISHED" in html
    _no_blank_lines(html)


def test_basis_claim_prefers_conservative_stance_for_current_justification():
    optimistic = _claim(ui.EvidenceStance.SUPPORTS, ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.7)
    gap = _claim(ui.EvidenceStance.NOT_ESTABLISHED, ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.3)
    chosen, differing = ui._basis_claim(_result([optimistic, gap]), ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION)
    assert chosen is gap
    assert differing == 1


def test_basis_matrix_distinguishes_not_assessed_from_not_established(monkeypatch):
    claims = [
        _claim(ui.EvidenceStance.SUPPORTS, ui.ClaimType.ORIGINAL_BASIS),
        _claim(ui.EvidenceStance.NOT_ESTABLISHED, ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.3),
    ]
    rendered = _capture(monkeypatch)
    ui.render_basis_matrix(_result(claims))

    html = "".join(rendered)
    assert html.count("Not assessed") == 2  # capability + administrative rows
    assert html.count("Not established") == 1
    assert "Supported" in html
    for content in rendered:
        _no_blank_lines(content)


def test_basis_matrix_renders_nothing_without_claims(monkeypatch):
    rendered = _capture(monkeypatch)
    ui.render_basis_matrix(_result([]))
    assert rendered == []


def test_question_subjects_only_lists_tokens_found_in_evidence():
    result = _result(
        question="Why is Press-07 limited to 42.3°C after the PX-900 change?",
        evidence=[_evidence("c1", "Press-07 UCL is 42.3 deg C per SOP.")],
    )
    tokens = [token for token, _count in ui._question_subjects(result)]
    assert "Press-07" in tokens
    assert "42.3°C" in tokens
    assert "PX-900" not in tokens


def test_source_category_is_an_allow_list():
    assert ui._source_category("B_historical_engineering/07.txt") == "Engineering record"
    assert ui._source_category("F_human_informal/35.txt") == "Informal / recollection"
    # Unknown folders get no invented label (real path is still shown).
    assert ui._source_category("G_distractors/36.txt") is None
    assert ui._source_category("") is None


def test_content_html_escapes_and_never_emits_blank_lines():
    text = "Line one <b>\n\n\n\nLine two & more"
    html = ui._content_html(text)
    assert "<b>" not in html and "&lt;b&gt;" in html
    assert "\n" not in html
    assert "&amp;" in html


def test_timeline_entries_turn_record_headers_into_dated_milestones():
    events = [
        SimpleNamespace(
            date=date(2001, 7, 5),
            description="Report No: PE-STUDY-2001-07                            Date: 05-JUL-2001",
            evidence_chunk_ids=["07_Thermal_Validation_2001_0"],
        ),
        SimpleNamespace(
            date=date(2011, 8, 22),
            description="Report No: COMM-2011-057   Date: 2011-08-22",
            evidence_chunk_ids=["05_PX400_Commissioning_Report_2011_0"],
        ),
        SimpleNamespace(
            date=date(2011, 8, 23),
            description="Plant Engineering Mgr: K. Doraiswamy 23-Aug-2011",
            evidence_chunk_ids=["05_PX400_Commissioning_Report_2011_1"],
        ),
    ]
    evidence = [
        _evidence("07_Thermal_Validation_2001_0", "...", filename="07_Thermal_Validation_2001.txt"),
        _evidence("05_PX400_Commissioning_Report_2011_0", "...", filename="05_PX400_Commissioning_Report_2011.txt"),
        _evidence("05_PX400_Commissioning_Report_2011_1", "...", filename="05_PX400_Commissioning_Report_2011.txt"),
    ]
    entries, total = ui._timeline_entries(_result(evidence=evidence, events=events))

    assert total == 3
    assert [e["kind"] for e in entries] == ["record", "record"]
    assert entries[0]["title"] == "Thermal Validation 2001"
    assert entries[0]["ref"] == "PE-STUDY-2001-07"
    # The sign-off line a day later is merged into its record.
    assert entries[1]["lines"] == ["Plant Engineering Mgr: K. Doraiswamy 23-Aug-2011"]


def test_render_timeline_ends_on_reported_gap(monkeypatch):
    finding = SimpleNamespace(
        finding_type=SimpleNamespace(value="evidence-gap"),
        statement="Technical re-justification is not established in the available evidence.",
        evidence_chunk_ids=[],
    )
    events = [
        SimpleNamespace(date=date(2001, 9, 10), description="Limit changed from 45.0°C to 42.3°C following thermal investigation", evidence_chunk_ids=["c1"]),
    ]
    rendered = _capture(monkeypatch)
    ui.render_timeline(_result(events=events, findings=[finding]))
    html = "".join(rendered)
    assert "Unresolved" in html
    assert "not established in the available evidence" in html
    for content in rendered:
        _no_blank_lines(content)


def test_render_claims_explains_not_established_versus_false(monkeypatch):
    claims = [
        _claim(ui.EvidenceStance.NOT_ESTABLISHED, ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.3),
        _claim(ui.EvidenceStance.SUPPORTS, ui.ClaimType.ORIGINAL_BASIS, 0.9),
    ]
    rendered = _capture(monkeypatch)
    ui.render_claims(_result(claims, evidence=[_evidence("c1", "text")]))
    html = "".join(rendered)
    assert "Not established is not the same as false" in html
    assert "No contradictions were identified" in html
    assert "Needs human verification" in html
    assert "30% calibrated confidence" in html
    for content in rendered:
        _no_blank_lines(content)


def test_classify_error_preserves_rate_limit_and_unavailable_wording():
    tone, title, message = ui._classify_error(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert tone == "warn" and "quota is currently exhausted" in message
    tone, title, message = ui._classify_error(RuntimeError("503 UNAVAILABLE"))
    assert "temporarily unavailable" in message
    tone, title, message = ui._classify_error(RuntimeError("boom"))
    assert tone == "" and "backend pipeline returned an error" in message


def test_request_run_queues_question_and_clears_error(monkeypatch):
    state = {"last_error": {"question": "old"}}
    monkeypatch.setattr(ui.st, "session_state", state)
    ui._request_run("  Why is Press-07 limited?  ")
    assert state["run_question"] == "Why is Press-07 limited?"
    assert state["pending_question"] == "Why is Press-07 limited?"
    assert "last_error" not in state

    ui._request_run("   ")
    assert state["run_question"] == "Why is Press-07 limited?"
