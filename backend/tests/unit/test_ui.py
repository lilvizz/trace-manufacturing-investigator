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


# ---------------------------------------------------------------------------
# Regression tests: _primary_claim question-specific selection
# ---------------------------------------------------------------------------
#
# Bug: a question that explicitly asks about the CURRENT technical
# justification of a parameter/limit (e.g. "...the exact technical
# justification for keeping the Press-07 UCL at 42.3°C after the PX-400
# replacement in 2011?") was selecting an unrelated ADMINISTRATIVE_ACTION
# claim as the primary assessment, because the word "replacement" in the
# question happened to match that claim type's keyword list and that
# claim also had a higher raw confidence. The fix only changes claim
# *selection*; no confidence value is read, written, or recalculated.


def _claim(claim_type, confidence, statement="statement"):
    return SimpleNamespace(
        claim_type=claim_type,
        confidence=confidence,
        statement=statement,
    )


def test_primary_claim_technical_justification_question_wins_despite_lower_confidence():
    question = (
        "What was the exact technical justification for keeping the "
        "Press-07 UCL at 42.3°C after the PX-400 replacement in 2011?"
    )
    claims = [
        _claim(ui.ClaimType.ADMINISTRATIVE_ACTION, 0.95, "administrative"),
        _claim(ui.ClaimType.ORIGINAL_BASIS, 0.90, "original"),
        _claim(ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.30, "technical"),
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    assert chosen.claim_type == ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION
    assert chosen.confidence == 0.30


def test_primary_claim_original_basis_question_prefers_original_basis():
    question = "What originally established the 42.3°C boundary for Press-07?"
    claims = [
        _claim(ui.ClaimType.ADMINISTRATIVE_ACTION, 0.95, "administrative"),
        _claim(ui.ClaimType.ORIGINAL_BASIS, 0.90, "original"),
        _claim(ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.30, "technical"),
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    assert chosen.claim_type == ui.ClaimType.ORIGINAL_BASIS


def test_primary_claim_administrative_question_prefers_administrative_action():
    question = "Why was the 43.0°C proposal rejected?"
    claims = [
        _claim(ui.ClaimType.ADMINISTRATIVE_ACTION, 0.95, "administrative"),
        _claim(ui.ClaimType.ORIGINAL_BASIS, 0.90, "original"),
        _claim(ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.30, "technical"),
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    assert chosen.claim_type == ui.ClaimType.ADMINISTRATIVE_ACTION


def test_primary_claim_equipment_question_prefers_current_capability():
    question = "What equipment changed on Press-07 since 2011?"
    claims = [
        _claim(ui.ClaimType.ADMINISTRATIVE_ACTION, 0.95, "administrative"),
        _claim(ui.ClaimType.CURRENT_CAPABILITY, 0.90, "capability"),
        _claim(ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.30, "technical"),
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    assert chosen.claim_type == ui.ClaimType.CURRENT_CAPABILITY


def test_primary_claim_no_semantic_target_keeps_existing_scoring_behavior():
    """With no keyword/trigger overlap at all, score reduces to raw
    confidence, so the highest-confidence claim wins -- exactly the
    pre-existing fallback behavior, untouched by this change."""
    question = "Tell me about Press-07."
    claims = [
        _claim(ui.ClaimType.ADMINISTRATIVE_ACTION, 0.95, "administrative"),
        _claim(ui.ClaimType.ORIGINAL_BASIS, 0.90, "original"),
        _claim(ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.30, "technical"),
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    assert chosen.claim_type == ui.ClaimType.ADMINISTRATIVE_ACTION
    assert chosen.confidence == 0.95


def test_primary_claim_original_technical_basis_question_prefers_original_basis():
    """Bug: 'What was the original technical basis for setting the
    Press-07 UCL at 42.3°C in 2001?' was being forced onto
    CURRENT_TECHNICAL_JUSTIFICATION because bare 'technical basis' is an
    ambiguous trigger. The question explicitly says "original" / is
    asking about the 2001 derivation, so ORIGINAL_BASIS must win, even
    though CURRENT_TECHNICAL_JUSTIFICATION's own keyword-scoring terms
    don't even match this question's wording."""
    question = (
        "What was the original technical basis for setting the "
        "Press-07 UCL at 42.3°C in 2001?"
    )
    claims = [
        _claim(ui.ClaimType.ADMINISTRATIVE_ACTION, 0.95, "administrative"),
        _claim(ui.ClaimType.ORIGINAL_BASIS, 0.95, "original"),
        _claim(ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.30, "technical"),
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    assert chosen.claim_type == ui.ClaimType.ORIGINAL_BASIS
    assert chosen.confidence == 0.95


def test_primary_claim_current_justification_hero_question_still_wins():
    """Regression guard for the hero question, which contains BOTH an
    original-basis-ish word ("established") and strong current-
    justification markers ("current technical basis", "still
    supported") -- the strong markers must still force
    CURRENT_TECHNICAL_JUSTIFICATION, exactly as before this fix."""
    question = (
        "Why is Press-07 limited to 42.3°C, what changed since that "
        "limit was established, and is its current technical basis "
        "still supported?"
    )
    claims = [
        _claim(ui.ClaimType.ADMINISTRATIVE_ACTION, 0.95, "administrative"),
        _claim(ui.ClaimType.ORIGINAL_BASIS, 0.95, "original"),
        _claim(ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.30, "technical"),
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    assert chosen.claim_type == ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION
    assert chosen.confidence == 0.30


def test_primary_claim_current_justification_after_replacement_still_wins():
    """Regression guard for the earlier bug fix: a question mentioning
    'replacement' (an ADMINISTRATIVE_ACTION keyword) in the context of
    asking about the technical justification after that replacement
    must still select CURRENT_TECHNICAL_JUSTIFICATION, not be re-broken
    by the original-basis carve-out added in this round."""
    question = (
        "What was the exact technical justification for keeping the "
        "Press-07 UCL at 42.3°C after the PX-400 replacement in 2011?"
    )
    claims = [
        _claim(ui.ClaimType.ADMINISTRATIVE_ACTION, 0.95, "administrative"),
        _claim(ui.ClaimType.ORIGINAL_BASIS, 0.90, "original"),
        _claim(ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION, 0.30, "technical"),
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    assert chosen.claim_type == ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION
    assert chosen.confidence == 0.30


def test_primary_claim_does_not_hijack_on_bare_evidence_keyword():
    """Bug: 'Does the evidence establish that RTD-7714-2 failed
    calibration, and what impact does that have on measurement
    reliability and the required response?' was selecting a 0%-confidence
    EVIDENCE_GAP claim (about unrelated pre-failure batch records) as the
    hero assessment, making the UI show 'INCONCLUSIVE / Insufficient'
    directly under a Direct Answer that confidently says the opposite.
    Root cause: EVIDENCE_GAP's keyword list included the bare word
    'evidence', which matches almost every question in an evidence-
    investigation tool, regardless of relevance. The fix removes that
    bare keyword, keeping only phrases that actually signal a
    missing-evidence question ('evidence gap', 'missing evidence',
    'what evidence'). This claim set and question are taken verbatim
    from the reported case."""
    question = (
        "Does the evidence establish that RTD-7714-2 failed calibration, "
        "and what impact does that have on measurement reliability and "
        "the required response?"
    )
    calibration_failure = _claim(
        ui.ClaimType.ORIGINAL_BASIS,
        0.95,
        "RTD-7714-2 failed annual calibration on September 12, 2018, with "
        "an as-found deviation of 0.31 °C against the 0.30 °C "
        "acceptance threshold.",
    )
    procedure_requirement = _claim(
        ui.ClaimType.ORIGINAL_BASIS,
        0.95,
        "Metrology Procedure MET-PROC-004 mandates an as-found deviation "
        "limit of <= 0.30 °C for process temperature sensors.",
    )
    drift_history = _claim(
        ui.ClaimType.CURRENT_CAPABILITY,
        0.95,
        "RTD-7714-2 exhibited a history of progressive thermal drift from "
        "2012 through 2017 prior to failing in 2018.",
    )
    replacement_action = _claim(
        ui.ClaimType.ADMINISTRATIVE_ACTION,
        0.95,
        "Work Order WO-18-0322 executed a like-for-like replacement of "
        "RTD-7714-2 with a new sensor, recording an as-left deviation of "
        "0.01 °C.",
    )
    post_replacement_quality = _claim(
        ui.ClaimType.CURRENT_CAPABILITY,
        0.90,
        "Three production batches run post-replacement were verified to "
        "be within normal quality parameters.",
    )
    pre_failure_gap = _claim(
        ui.ClaimType.EVIDENCE_GAP,
        0.0,
        "Individual batch quality records and retrospective impact "
        "assessments for product manufactured immediately prior to the "
        "September 12, 2018 calibration failure are not established in "
        "the available evidence.",
    )
    claims = [
        calibration_failure,
        procedure_requirement,
        drift_history,
        replacement_action,
        post_replacement_quality,
        pre_failure_gap,
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    # The headline must select the calibration-failure finding, not the
    # unrelated evidence-gap claim.
    assert chosen.claim_type != ui.ClaimType.EVIDENCE_GAP
    assert chosen is calibration_failure
    assert chosen.confidence == 0.95

    # The pre-detection product-impact gap must still be present in the
    # full claim list -- this fix only changes which claim is selected as
    # the hero, not what claims exist or their confidence values.
    assert pre_failure_gap in claims
    assert pre_failure_gap.claim_type == ui.ClaimType.EVIDENCE_GAP
    assert pre_failure_gap.confidence == 0.0


def test_primary_claim_current_justification_question_unaffected_by_evidence_gap_fix():
    """Confirms the Press-07 current-technical-justification question
    (the one that SHOULD show NOT_ESTABLISHED) still correctly selects
    CURRENT_TECHNICAL_JUSTIFICATION after the EVIDENCE_GAP keyword change
    above -- that fix only touches the generic keyword-scoring fallback,
    not the strong-trigger override this question relies on."""
    question = (
        "Why is Press-07 limited to 42.3°C, what changed since that "
        "limit was established, and is its current technical basis "
        "still supported?"
    )
    claims = [
        _claim(ui.ClaimType.ADMINISTRATIVE_ACTION, 0.95, "administrative"),
        _claim(ui.ClaimType.ORIGINAL_BASIS, 0.95, "original"),
        _claim(
            ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
            0.30,
            "The available evidence does not establish the current "
            "technical basis for the parameter or limit on the current "
            "equipment configuration.",
        ),
        _claim(ui.ClaimType.EVIDENCE_GAP, 0.85, "evidence gap claim"),
    ]
    result = SimpleNamespace(
        report=SimpleNamespace(claims=claims),
        investigation=SimpleNamespace(question=question),
    )

    chosen = ui._primary_claim(result)

    assert chosen.claim_type == ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION
    assert chosen.confidence == 0.30


# ---------------------------------------------------------------------------
# Regression tests: render_simple_result keeps Investigation coverage
# and Claim Breakdown visible
# ---------------------------------------------------------------------------
#
# Bug: a narrow factual question (e.g. the original-technical-basis
# question above) is classified by _is_simple_factual_question() and
# routed to render_simple_result(), which only rendered Direct Answer,
# Current Assessment, and Supporting Evidence -- silently dropping the
# Investigation Coverage metrics and Claim Breakdown that the full case
# UI (render_result) shows. These are investigation metrics, not part
# of the simple/full presentation choice, so they must appear in both.


def _fake_simple_result():
    claim = SimpleNamespace(
        claim_type=ui.ClaimType.ORIGINAL_BASIS,
        confidence=0.95,
        statement="The 42.3°C limit was established in 2001.",
        stance=ui.EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-2001"],
    )
    return SimpleNamespace(
        report=SimpleNamespace(
            claims=[claim],
            direct_answer="",
            direct_answer_evidence_chunk_ids=[],
            unresolved_questions=[],
        ),
        plan=SimpleNamespace(subquestions=["q1", "q2"]),
        investigation=SimpleNamespace(
            evidence=[],
            question="What was the original technical basis in 2001?",
        ),
        timeline=SimpleNamespace(events=[]),
        evidence_analysis=SimpleNamespace(findings=[]),
    )


def test_render_simple_result_includes_investigation_coverage_and_claims(monkeypatch):
    result = _fake_simple_result()
    rendered = []

    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    ui.render_simple_result(result, {})

    combined = "".join(rendered)
    assert "Investigation coverage" in combined
    assert "Reasoning claims" in combined or "claims" in combined.lower()
    assert "Supporting evidence" in combined


# ---------------------------------------------------------------------------
# Regression tests: hero assessment confidence presentation
# ---------------------------------------------------------------------------
#
# The top-level "Current assessment" hero previously showed a raw
# percentage (e.g. "95%") directly beside the stance badge (e.g.
# "NO ESCALATION"). When a different, lower-confidence claim (e.g.
# "NOT ESTABLISHED — 30%") appeared further down the same page, the two
# numbers next to two different stances read as contradictory on a first
# skim, even though they are both correct and independently calibrated.
# The fix is presentation-only: the hero shows a qualitative label
# instead of a number; every individual claim's own percentage (rendered
# by render_claims, untouched here) is unaffected.


def test_confidence_level_buckets_high_moderate_low():
    assert ui._confidence_level(0.95) == "High confidence"
    assert ui._confidence_level(0.85) == "High confidence"
    assert ui._confidence_level(0.84) == "Moderate confidence"
    assert ui._confidence_level(0.60) == "Moderate confidence"
    assert ui._confidence_level(0.59) == "Low confidence"
    assert ui._confidence_level(0.30) == "Low confidence"
    assert ui._confidence_level(0.0) == "Low confidence"


def test_evidence_strength_buckets_strong_moderate_insufficient():
    assert ui._evidence_strength(0.95) == "Strong"
    assert ui._evidence_strength(0.85) == "Strong"
    assert ui._evidence_strength(0.84) == "Moderate"
    assert ui._evidence_strength(0.60) == "Moderate"
    assert ui._evidence_strength(0.59) == "Insufficient"
    assert ui._evidence_strength(0.30) == "Insufficient"
    assert ui._evidence_strength(0.0) == "Insufficient"


def test_render_assessment_hero_shows_evidence_strength_not_percentage(monkeypatch):
    """The hero's strength display must be a plain evidence-sufficiency
    word ("Strong"/"Moderate"/"Insufficient"), and the raw percentage
    must NOT appear in the hero -- a judge skimming the headline must
    never see a number that could be misread as "there is a 95%/30%
    chance this conclusion is correct"."""
    claim = SimpleNamespace(
        stance=ui.EvidenceStance.SUPPORTS,
        statement="Historical rejection was a scheduling decision, not technical.",
        confidence=0.95,
        claim_type=ui.ClaimType.ADMINISTRATIVE_ACTION,
    )
    result = SimpleNamespace(
        report=SimpleNamespace(claims=[claim]),
        investigation=SimpleNamespace(question="why wasn't it adopted"),
    )

    rendered = []
    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    ui.render_assessment(result)

    assert len(rendered) == 1
    html = rendered[0]

    assert "Strong" in html
    assert "evidence strength" in html
    assert "95%" not in html
    assert "0.95" not in html


def test_render_assessment_hero_label_unchanged_not_established_shows_insufficient(monkeypatch):
    """The stance -> label mapping (SUPPORTS -> NO ESCALATION, etc.) must
    not be touched by the evidence-strength wording change, and a low-
    confidence NOT_ESTABLISHED claim must read as "Insufficient"
    evidence strength, with no raw percentage in the hero."""
    claim = SimpleNamespace(
        stance=ui.EvidenceStance.NOT_ESTABLISHED,
        statement="Current technical basis is not established.",
        confidence=0.30,
        claim_type=ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
    )
    result = SimpleNamespace(
        report=SimpleNamespace(claims=[claim]),
        investigation=SimpleNamespace(question="is the current limit justified"),
    )

    rendered = []
    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    ui.render_assessment(result)

    html = rendered[0]
    label, _tone = ui.STANCE_META[ui.EvidenceStance.NOT_ESTABLISHED]
    assert label in html
    assert "Insufficient" in html
    assert "30%" not in html


def test_render_confidence_context_shows_evidence_strength_not_percentage(monkeypatch):
    """Bug: render_confidence_context()'s "What the confidence applies to"
    caption still hardcoded the raw {confidence:.0%} into its prose (e.g.
    "The 95% score applies to..."), even though the hero above it was
    redesigned to show only the evidence-strength word ("Strong" /
    "Insufficient") and never a raw percentage. The caption must describe
    the same word the hero shows, not reintroduce a number nowhere else
    on screen."""
    claim = SimpleNamespace(
        stance=ui.EvidenceStance.NOT_ESTABLISHED,
        statement="Current technical basis is not established.",
        confidence=0.30,
        claim_type=ui.ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
    )
    result = SimpleNamespace(
        report=SimpleNamespace(claims=[claim]),
        investigation=SimpleNamespace(question="is the current limit justified"),
    )

    rendered = []
    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    ui.render_confidence_context(result)

    assert len(rendered) == 1
    html = rendered[0]

    assert "Insufficient" in html
    assert "evidence-strength rating" in html
    assert "30%" not in html
    assert "0.3" not in html


def test_render_confidence_context_strong_case_shows_strong_word(monkeypatch):
    """Same caption, high-confidence side: must say 'Strong', never a
    raw percentage, for a NO ESCALATION / high-confidence claim."""
    claim = SimpleNamespace(
        stance=ui.EvidenceStance.SUPPORTS,
        statement="Historical rejection was a scheduling decision.",
        confidence=0.95,
        claim_type=ui.ClaimType.ADMINISTRATIVE_ACTION,
    )
    result = SimpleNamespace(
        report=SimpleNamespace(claims=[claim]),
        investigation=SimpleNamespace(question="why wasn't it adopted"),
    )

    rendered = []
    monkeypatch.setattr(
        ui.st,
        "markdown",
        lambda content, **kwargs: rendered.append(content),
    )

    ui.render_confidence_context(result)

    html = rendered[0]
    assert "Strong" in html
    assert "95%" not in html


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