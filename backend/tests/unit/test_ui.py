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


# ---------------------------------------------------------------------------
# Regression tests: _clean_direct_answer
# ---------------------------------------------------------------------------


def test_clean_direct_answer_strips_fence_with_preamble_text():
    """A model response that puts prose before a fenced block must not
    leak the literal fence markers (previously only a fence anchored to
    the very start of the string was stripped)."""
    raw = (
        "Here is the answer:\n"
        "```html\n"
        '<div class="direct-answer-text">Press-07 was limited to 42.3C</div>\n'
        "```"
    )

    cleaned = ui._clean_direct_answer(raw)

    assert "```" not in cleaned
    assert "<div" not in cleaned
    assert "Press-07 was limited to 42.3C" in cleaned


def test_clean_direct_answer_recovers_text_from_unterminated_attribute():
    """An unterminated quoted attribute must not cause the entire answer
    to be silently discarded. (html.parser.HTMLParser treats everything
    after a missing closing quote as still inside the tag and never
    reports it as data; the regex-based cleaner has no such failure mode.)"""
    raw = '<div class="direct-answer-text>Press-07 was limited to 42.3C in 2001.'

    cleaned = ui._clean_direct_answer(raw)

    assert cleaned != ""
    assert "Press-07 was limited to 42.3C in 2001" in cleaned


def test_clean_direct_answer_removes_well_formed_markup():
    raw = (
        '<div class="direct-answer-text">\n'
        "    Press-07 was originally limited to 42.3°C in 2001.\n"
        "</div>"
    )

    cleaned = ui._clean_direct_answer(raw)

    assert "<div" not in cleaned
    assert "</div>" not in cleaned
    assert cleaned == "Press-07 was originally limited to 42.3°C in 2001."


def test_clean_direct_answer_does_not_alter_plain_prose():
    raw = (
        "Press-07 was limited to 42.3°C in 2001 because of thermal "
        "instability above 42°C."
    )

    assert ui._clean_direct_answer(raw) == raw


# ---------------------------------------------------------------------------
# Regression tests: _looks_like_fragment / _condense_timeline_events
# ---------------------------------------------------------------------------


def test_looks_like_fragment_flags_garbage_examples():
    garbage = [
        "Production Supervisor M. Elango, 25-Apr-2001) to limit continuous run",
        "report (INV-2001-014), and Quality review of 14-Jun-2001 (ref QM-2001-06),",
        "A cross-functional review was held (Quality meeting, 14-Jun-2001; see",
        "Ref: ENG-NOTE-2001-09 Date: 05-SEP-2001",
    ]

    for description in garbage:
        assert ui._looks_like_fragment(description), description


def test_looks_like_fragment_accepts_clean_examples():
    clean = [
        "Initial 45.0°C setpoint established",
        "Limit changed from 45.0°C to 42.3°C following thermal investigation",
        "Equipment replaced; 42.3°C value retained",
        "42.3°C confirmed unchanged",
        "Proposal submitted to raise limit to 43.0°C",
    ]

    for description in clean:
        assert not ui._looks_like_fragment(description), description


def test_condense_timeline_events_filters_fragments_for_any_question():
    """The quality filter must not be gated to the flagship HERO_QUESTION:
    it has to apply to every investigation question."""
    events = [
        SimpleNamespace(
            date="2001-04-25",
            description="Production Supervisor M. Elango, 25-Apr-2001) to limit continuous run",
            evidence_chunk_ids=["chunk-a"],
        ),
        SimpleNamespace(
            date="2001-09-10",
            description="Limit changed from 45.0°C to 42.3°C following thermal investigation",
            evidence_chunk_ids=["chunk-b"],
        ),
        SimpleNamespace(
            date="2011-08-20",
            description="Equipment replaced; 42.3°C value retained",
            evidence_chunk_ids=["chunk-c"],
        ),
    ]

    result = SimpleNamespace(
        timeline=SimpleNamespace(events=events),
        investigation=SimpleNamespace(question="What equipment is on Press-05?"),
    )

    selected, total = ui._condense_timeline_events(result)

    assert total == 3
    descriptions = [e.description for e in selected]
    assert "Production Supervisor M. Elango, 25-Apr-2001) to limit continuous run" not in descriptions
    assert "Limit changed from 45.0°C to 42.3°C following thermal investigation" in descriptions
    assert "Equipment replaced; 42.3°C value retained" in descriptions


def test_condense_timeline_events_falls_back_when_all_events_are_fragments():
    """If every extracted event looks like a fragment, do not return an
    empty timeline — fall back to the deduplicated set so the UI never
    hides the whole timeline."""
    events = [
        SimpleNamespace(
            date="2001-09-05",
            description="Ref: ENG-NOTE-2001-09 Date: 05-SEP-2001",
            evidence_chunk_ids=["chunk-a"],
        ),
        SimpleNamespace(
            date="2001-06-14",
            description="report (INV-2001-014), and Quality review of 14-Jun-2001 (ref QM-2001-06),",
            evidence_chunk_ids=["chunk-b"],
        ),
    ]

    result = SimpleNamespace(
        timeline=SimpleNamespace(events=events),
        investigation=SimpleNamespace(question="Some other question"),
    )

    selected, total = ui._condense_timeline_events(result)

    assert total == 2
    assert len(selected) == 2


def test_condense_timeline_events_dedupes_exact_duplicates():
    events = [
        SimpleNamespace(
            date="2014-03-15",
            description="42.3°C confirmed unchanged",
            evidence_chunk_ids=["chunk-a"],
        ),
        SimpleNamespace(
            date="2014-03-15",
            description="42.3°C confirmed unchanged",
            evidence_chunk_ids=["chunk-a"],
        ),
    ]

    result = SimpleNamespace(
        timeline=SimpleNamespace(events=events),
        investigation=SimpleNamespace(question="Unrelated question"),
    )

    selected, total = ui._condense_timeline_events(result)

    assert total == 1
    assert len(selected) == 1


def test_looks_like_fragment_flags_csv_data_row():
    """Raw CSV export rows (e.g. pulled in from an SPC config file) must
    be treated as fragments even though they contain no unbalanced
    parens and don't start lowercase."""
    csv_row = (
        'Cure_Zone_Temp,Press-07,PX-400,36.0,38.5,42.3,42.3,44.0,'
        '"See Engineering file",2011-08-20,T.Balakrishnan'
    )
    assert ui._looks_like_fragment(csv_row)


def test_looks_like_fragment_flags_multi_field_record_line():
    """A line carrying two or more 'Label: value' fields is a form
    excerpt, not a narrative sentence, even when it doesn't start with
    one of the known document-header keywords."""
    record_line = "WO No: WO-18-0322                                     Date Completed: 12-SEP-2018"
    assert ui._looks_like_fragment(record_line)


def test_looks_like_fragment_accepts_clean_examples_with_new_rules():
    """The broadened CSV/multi-field checks must not create false
    positives on ordinary narrative timeline sentences."""
    clean = [
        "Initial 45.0°C setpoint established",
        "Limit changed from 45.0°C to 42.3°C following thermal investigation",
        "Equipment replaced; 42.3°C value retained",
        "42.3°C confirmed unchanged",
        "Proposal submitted to raise limit to 43.0°C",
        (
            "A 2014 study (QUAL-2014-009) showed PX-400 maintains stable "
            "thermal control up to at least 44°C, but a proposal to raise "
            "the limit to 43.0°C (PCP-2014-022) was rejected for commercial "
            "and customer PPAP lead-time reasons rather than technical failure"
        ),
    ]
    for description in clean:
        assert not ui._looks_like_fragment(description), description


# ---------------------------------------------------------------------------
# Regression tests: HTML block must not be broken by blank lines
# ---------------------------------------------------------------------------
#
# Root cause of the literal "<div>...</div>" leak: Streamlit's markdown
# renderer treats a blank (or whitespace-only) line inside a string passed
# to st.markdown(..., unsafe_allow_html=True) as the end of the current
# raw-HTML block. Any deeply-indented markup after that point then gets
# parsed as an indented code block and rendered as literal text instead
# of HTML. These tests assert the rendered markup is a single physical
# line (or at least contains no blank/whitespace-only line), which is
# what prevents that failure mode.


def _assert_no_blank_lines(html: str) -> None:
    lines = html.split("\n")
    blank_lines = [line for line in lines if line.strip() == ""]
    assert not blank_lines, f"HTML markup contains blank line(s): {html!r}"


def test_render_direct_answer_markup_has_no_blank_lines(monkeypatch):
    result = _fake_result()
    rendered = []

    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    ui.render_direct_answer(result)

    assert len(rendered) == 1
    _assert_no_blank_lines(rendered[0])
    # And the markup must still be well-formed, not just blank-line-free.
    assert "<div class=\"direct-answer-card\">" in rendered[0]
    assert "<div class=\"direct-answer-text\">" in rendered[0]


def test_render_timeline_markup_has_no_blank_lines(monkeypatch):
    events = [
        SimpleNamespace(
            date="1999-03-23",
            description="Initial 45.0°C setpoint established",
            evidence_chunk_ids=["chunk-a"],
        ),
        SimpleNamespace(
            date="2001-09-10",
            description="Limit changed from 45.0°C to 42.3°C following thermal investigation",
            evidence_chunk_ids=["chunk-b"],
        ),
        SimpleNamespace(
            date="2011-08-20",
            description="Equipment replaced; 42.3°C value retained",
            evidence_chunk_ids=["chunk-c"],
        ),
    ]

    result = SimpleNamespace(
        timeline=SimpleNamespace(events=events),
        investigation=SimpleNamespace(question="Unrelated question"),
    )

    rendered = []

    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    ui.render_timeline(result)

    # The timeline events are concatenated into one combined markdown
    # call; that call (and every other render_timeline markdown call)
    # must contain no blank line.
    for content in rendered:
        _assert_no_blank_lines(content)

    combined = "".join(rendered)
    assert combined.count('<div class="timeline-event">') == 3
    assert "Equipment replaced; 42.3°C value retained" in combined