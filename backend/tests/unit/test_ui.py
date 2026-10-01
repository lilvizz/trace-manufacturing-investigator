from types import SimpleNamespace

import app.ui as ui


def _fake_result():
    evidence = [
        SimpleNamespace(
            chunk_id="chunk-2011",
            filename="2011_replacement.txt",
        ),
        SimpleNamespace(
            chunk_id="chunk-2014",
            filename="2014_review.txt",
        ),
    ]

    report = SimpleNamespace(
        direct_answer=(
            "The 2011 replacement carried the existing 42.3°C "
            "limit forward rather than establishing a new technical basis."
        ),
        direct_answer_evidence_chunk_ids=[
            "chunk-2011",
            "chunk-2014",
            "chunk-does-not-exist",
        ],
    )

    investigation = SimpleNamespace(
        evidence=evidence,
    )

    return SimpleNamespace(
        report=report,
        investigation=investigation,
    )


def test_render_direct_answer_renders_answer_and_valid_evidence(monkeypatch):
    result = _fake_result()
    rendered = []

    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    ui.render_direct_answer(result)

    assert len(rendered) == 1

    html = rendered[0]

    assert "DIRECT ANSWER" in html
    assert (
        "The 2011 replacement carried the existing 42.3°C "
        "limit forward rather than establishing a new technical basis."
    ) in html

    assert "2011_replacement.txt · chunk-2011" in html
    assert "2014_review.txt · chunk-2014" in html

    # The renderer must not display an evidence ID that isn't
    # actually present in the investigation evidence set.
    assert "chunk-does-not-exist" not in html


def test_render_direct_answer_does_nothing_when_answer_is_empty(monkeypatch):
    result = SimpleNamespace(
        report=SimpleNamespace(
            direct_answer="",
            direct_answer_evidence_chunk_ids=["chunk-1"],
        ),
        investigation=SimpleNamespace(
            evidence=[],
        ),
    )

    rendered = []

    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    ui.render_direct_answer(result)

    assert rendered == []


def test_render_result_calls_direct_answer_before_assessment(monkeypatch):
    calls = []

    monkeypatch.setattr(
        ui,
        "render_direct_answer",
        lambda result: calls.append("direct_answer"),
    )
    monkeypatch.setattr(
        ui,
        "render_assessment",
        lambda result: calls.append("assessment"),
    )
    monkeypatch.setattr(
        ui,
        "render_confidence_context",
        lambda result: calls.append("confidence"),
    )
    monkeypatch.setattr(
        ui,
        "render_finding_summary",
        lambda result: calls.append("findings"),
    )
    monkeypatch.setattr(
        ui,
        "render_investigation_path",
        lambda result: calls.append("path"),
    )
    monkeypatch.setattr(
        ui,
        "render_timeline",
        lambda result: calls.append("timeline"),
    )
    monkeypatch.setattr(
        ui,
        "render_gap_and_action",
        lambda result: calls.append("gap"),
    )
    monkeypatch.setattr(
        ui,
        "render_evidence",
        lambda result: calls.append("evidence"),
    )
    monkeypatch.setattr(
        ui,
        "render_claims",
        lambda result: calls.append("claims"),
    )

    class FakeColumns:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        ui.st,
        "columns",
        lambda *args, **kwargs: (FakeColumns(), FakeColumns()),
    )
    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        ui.st,
        "expander",
        lambda *args, **kwargs: FakeColumns(),
    )
    monkeypatch.setattr(
        ui.st,
        "json",
        lambda *args, **kwargs: None,
    )

    result = SimpleNamespace(
        report=SimpleNamespace(unresolved_questions=[]),
        plan=SimpleNamespace(subquestions=[]),
        investigation=SimpleNamespace(
            evidence=[],
            question="test question",
        ),
        timeline=SimpleNamespace(events=[]),
        evidence_analysis=SimpleNamespace(findings=[]),
    )

    ui.render_result(
        result,
        {
            "documents": 40,
            "chunks": 70,
            "vector_enabled": False,
        },
    )

    assert calls[0] == "direct_answer"
    assert calls[1] == "assessment"