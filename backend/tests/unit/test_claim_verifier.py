"""
Fix 2 regression tests (hostile audit MUST-FIX #2): the rejustification
safety invariant must not be bypassable just because Gemini omitted,
mislabeled, or invalidated a claim's `claim_type`.
"""

import pytest

from app.investigation.claim_verifier import (
    ClaimVerifier,
    _looks_like_current_technical_justification,
)
from app.investigation.models import (
    ClaimType,
    EvidenceItem,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)


def make_evidence(chunk_id: str, content: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        filename=f"{chunk_id}.txt",
        relative_path="B_historical_engineering/example.txt",
        content=content,
        retrieval_score=0.9,
        retrieved_for=["Why is the current limit still supported?"],
    )


def make_report(claim: ReasoningClaim) -> InvestigationReport:
    return InvestigationReport(question="Q", claims=[claim])


# No evidence anywhere contains an explicit re-justification marker, so
# every case below has no rejustification evidence available.
NO_REJUSTIFICATION_EVIDENCE = [
    make_evidence(
        "chunk-1",
        "Press-07 remains configured the same way it was in 2001. "
        "No changes have been made to the equipment.",
    )
]


def test_missing_claim_type_with_justification_language_is_gated():
    """1. claim_type is None, but the statement text itself asserts the
    current technical basis is justified -- must still be gated."""
    claim = ReasoningClaim(
        statement=(
            "The current 42.3 deg C limit remains technically justified "
            "based on ongoing production."
        ),
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.90,
        claim_type=None,
    )
    investigation = Investigation(
        question="Is the current basis still supported?",
        evidence=NO_REJUSTIFICATION_EVIDENCE,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    assert result.stance == EvidenceStance.NOT_ESTABLISHED
    assert result.confidence <= 0.30


def test_invalid_claim_type_with_justification_language_is_gated():
    """2. claim_type holds something that is not a valid ClaimType value
    at all (e.g. a raw string Gemini emitted that failed enum parsing
    upstream and was not normalized to None) -- must still be gated."""
    claim = ReasoningClaim(
        statement=(
            "The limit is currently technically justified given present "
            "operating conditions."
        ),
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.85,
        claim_type="not_a_real_claim_type",  # type: ignore[arg-type]
    )
    investigation = Investigation(
        question="Is the current basis still supported?",
        evidence=NO_REJUSTIFICATION_EVIDENCE,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    assert result.stance == EvidenceStance.NOT_ESTABLISHED
    assert result.confidence <= 0.30


def test_mislabeled_current_technical_justification_is_gated():
    """3. Gemini labels a de facto current-technical-justification
    statement as CURRENT_CAPABILITY -- the deterministic fallback must
    still route it through the gate."""
    claim = ReasoningClaim(
        statement=(
            "The current limit remains technically justified under "
            "present production conditions."
        ),
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.95,
        claim_type=ClaimType.CURRENT_CAPABILITY,
    )
    investigation = Investigation(
        question="Is the current basis still supported?",
        evidence=NO_REJUSTIFICATION_EVIDENCE,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    assert result.stance == EvidenceStance.NOT_ESTABLISHED
    assert result.confidence <= 0.30


def test_ordinary_capability_claim_is_not_misclassified():
    """4. A genuine capability claim must NOT be swept into the
    justification gate just because it discusses the current state of
    the equipment."""
    statement = (
        "Press 7 is currently capable of running stably at 44 deg C "
        "with no reported quality effects."
    )

    assert not _looks_like_current_technical_justification(statement)

    claim = ReasoningClaim(
        statement=statement,
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.80,
        claim_type=ClaimType.CURRENT_CAPABILITY,
    )
    investigation = Investigation(
        question="What is the equipment currently capable of?",
        evidence=NO_REJUSTIFICATION_EVIDENCE,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    # Untouched: capability claims are not subject to the
    # rejustification gate at all.
    assert result.stance == EvidenceStance.SUPPORTS
    assert result.confidence == 0.80


def test_administrative_approval_claim_is_not_misclassified():
    """5. An administrative-approval claim must NOT be swept into the
    justification gate merely because it mentions the limit."""
    statement = (
        "Quality administratively approved continued use of the "
        "existing limit in the 2014 review without new technical data."
    )

    assert not _looks_like_current_technical_justification(statement)

    claim = ReasoningClaim(
        statement=statement,
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.75,
        claim_type=ClaimType.ADMINISTRATIVE_ACTION,
    )
    # The cited passage must actually concern the 2014 review the claim
    # names: since the citation-grounding check, a claim whose specific
    # anchors (here "2014") appear in none of its cited passages is no
    # longer accepted as SUPPORTS. This test is about gate classification,
    # so it cites a passage that grounds the claim (still no re-justification).
    evidence = [
        make_evidence(
            "chunk-1",
            "2014 review: Quality approved continued use of the existing "
            "limit. No new technical data was generated.",
        )
    ]
    investigation = Investigation(
        question="Was the limit administratively reviewed?",
        evidence=evidence,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    assert result.stance == EvidenceStance.SUPPORTS
    assert result.confidence == 0.75


def test_explicit_technical_revalidation_still_recognized_with_evidence():
    """6. A correctly-labeled CURRENT_TECHNICAL_JUSTIFICATION claim
    backed by genuine re-justification evidence must still clear the
    gate (the existing invariant must not be weakened)."""
    evidence = [
        make_evidence(
            "chunk-2014",
            "Following the 2014 requalification study, the limit was "
            "re-approved for PX-400 at 42.3 deg C.",
        )
    ]
    claim = ReasoningClaim(
        statement=(
            "The current limit was technically revalidated in 2014 and "
            "remains justified."
        ),
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-2014"],
        confidence=0.90,
        claim_type=ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
    )
    investigation = Investigation(
        question="Is the current basis still supported?",
        evidence=evidence,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    assert result.stance == EvidenceStance.SUPPORTS
    assert result.confidence == 0.90


def test_explicit_type_still_gated_even_without_matching_statement_text():
    """Defense in depth: an explicitly-labeled CURRENT_TECHNICAL_
    JUSTIFICATION claim is still gated by claim_type alone, exactly as
    before Fix 2, even if its statement text happens not to trip the
    text-based fallback markers. The existing invariant is unchanged for
    correctly-labeled claims."""
    claim = ReasoningClaim(
        statement="Nothing has changed since 2001.",
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.90,
        claim_type=ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
    )
    investigation = Investigation(
        question="Is the current basis still supported?",
        evidence=NO_REJUSTIFICATION_EVIDENCE,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    assert result.stance == EvidenceStance.NOT_ESTABLISHED
    assert result.confidence <= 0.30


# ---------------------------------------------------------------------------
# Semantic-completeness audit of _looks_like_current_technical_justification
# (raised after Phase C review): the single-phrase alternation missed
# constructions where the re-justification ACTION verb (re-approved,
# revalidated, re-established) is separated from, or precedes, the
# CURRENT limit/boundary/basis subject it applies to. These are real
# phrasings an LLM could plausibly produce, and they must not be able to
# bypass the gate by landing outside the classifier's vocabulary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "The current limit was technically revalidated in 2014.",
        "The current 42.3°C boundary was technically re-established for "
        "PX-400.",
        "Engineering technically re-approved the current operating limit "
        "based on study data.",
        "The current limit was re-approved based on the 2014 "
        "qualification study.",
        "The current technical basis remains justified.",
        "The current limit is supported by the 2014 study.",
    ],
)
def test_current_justification_phrasings_are_detected(statement):
    assert _looks_like_current_technical_justification(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "PX-400 was qualified.",
        "PX-400 can operate stably at 44°C.",
        "The SOP remained unchanged.",
        "Quality administratively approved continued use.",
        "No evidence shows the limit was revalidated.",
        "The limit was not technically re-established.",
        # Adversarial: negation directly on the action verb, with the
        # current-limit subject still present, must not flip a denial
        # into a positive justification claim.
        "No current limit was re-approved based on new data.",
        "The current limit was not re-approved in 2014.",
    ],
)
def test_non_justification_phrasings_stay_excluded(statement):
    assert not _looks_like_current_technical_justification(statement)


def test_action_before_subject_mislabeled_claim_is_still_gated():
    """
    End-to-end regression: a claim whose action verb (re-approved)
    precedes its current-limit subject, mislabeled as
    ADMINISTRATIVE_ACTION, with no rejustification evidence available,
    must still be forced to NOT_ESTABLISHED.
    """
    claim = ReasoningClaim(
        statement=(
            "Engineering technically re-approved the current operating "
            "limit based on study data."
        ),
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.9,
        claim_type=ClaimType.ADMINISTRATIVE_ACTION,
    )
    investigation = Investigation(
        question="Is the current basis still supported?",
        evidence=NO_REJUSTIFICATION_EVIDENCE,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    assert result.stance == EvidenceStance.NOT_ESTABLISHED
    assert result.confidence <= 0.30


# ---------------------------------------------------------------------------
# Forward-retraction audit (raised after the second Phase C review): the
# previous _is_negated() only looked BACKWARD from the action marker, so a
# retraction/scope-limiting clause written AFTER the marker ("...was
# re-approved in 1999, but that approval does not apply to the current
# configuration") could still slip through as a false positive. Fixed by
# also checking, within the same sentence as the match, for a contrastive
# conjunction followed by a negation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "The limit was re-approved in 1999, but that approval does not "
        "apply to the current configuration.",
        "The current limit was revalidated in 2014, but that validation "
        "did not cover the current equipment.",
        "Engineering re-established the 42.3°C limit historically; "
        "however, that decision was not re-established for the current "
        "configuration.",
        "Records show the limit was re-approved, although no current "
        "technical approval exists.",
        "The limit was technically revalidated for PX-200, not for the "
        "current PX-400.",
        "The current limit was re-approved administratively, but no "
        "technical revalidation was performed.",
    ],
)
def test_retracted_justification_phrasings_stay_excluded(statement):
    assert not _looks_like_current_technical_justification(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "The current limit was technically revalidated for PX-400 in 2014.",
        "Engineering technically re-established the current operating "
        "boundary for PX-400.",
        "The current technical basis remains supported by the 2014 study.",
    ],
)
def test_unretracted_justification_phrasings_still_trigger(statement):
    assert _looks_like_current_technical_justification(statement)


def test_retracted_justification_mislabeled_claim_is_not_gated():
    """
    End-to-end regression: a claim whose statement retracts its own
    re-approval ("...but no technical revalidation was performed"),
    mislabeled as CURRENT_CAPABILITY, must NOT be swept into the
    rejustification gate -- the statement itself is already saying no
    current technical justification exists, not asserting one.
    """
    claim = ReasoningClaim(
        statement=(
            "The current limit was re-approved administratively, but no "
            "technical revalidation was performed."
        ),
        stance=EvidenceStance.CONTRADICTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.6,
        claim_type=ClaimType.CURRENT_CAPABILITY,
    )
    investigation = Investigation(
        question="Is the current basis still supported?",
        evidence=NO_REJUSTIFICATION_EVIDENCE,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    # Untouched by the rejustification gate: stance/confidence pass
    # through unchanged (this claim type isn't even gated).
    assert result.stance == EvidenceStance.CONTRADICTS
    assert result.confidence == 0.6


# ---------------------------------------------------------------------------
# Over-suppression audit (raised after the third Phase C review): the
# sentence-wide _RETRACTION_MARKER from the previous round was too broad --
# it suppressed genuine current-technical-justification assertions whenever
# ANY "but/however + negation" appeared later in the same sentence, even
# when that clause negated a different, unrelated, earlier thing ("...but
# the PREVIOUS administrative approval was not renewed") rather than
# retracting the justification just asserted. Fixed by waiving the
# retraction when the negated clause explicitly marks its subject as
# distinct and earlier ("previous"/"older"/"prior"/"earlier").
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "The current limit was technically revalidated based on the 2014 "
        "study, but the previous administrative approval was not renewed.",
        "Engineering technically re-established the current operating "
        "boundary from the qualification data, although the older change "
        "request was not approved.",
        "The current limit was re-approved based on new technical data, "
        "but the previous SOP revision was not retained.",
    ],
)
def test_unrelated_prior_item_negation_does_not_suppress(statement):
    assert _looks_like_current_technical_justification(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "The current limit was technically revalidated in 2014, but that "
        "validation did not cover the current equipment.",
        "The current limit was re-approved, but no technical "
        "revalidation was performed.",
        "The current boundary was technically re-established, although "
        "that decision does not apply to the current configuration.",
    ],
)
def test_same_subject_retraction_still_suppresses(statement):
    assert not _looks_like_current_technical_justification(statement)


def test_unrelated_prior_approval_mislabeled_claim_is_still_gated():
    """
    End-to-end regression: a claim whose genuine technical revalidation
    is followed only by an unrelated, explicitly-"previous" approval
    lapsing, mislabeled as ADMINISTRATIVE_ACTION, must still be routed
    through the rejustification gate (the justification assertion was
    never actually retracted).
    """
    claim = ReasoningClaim(
        statement=(
            "The current limit was technically revalidated based on the "
            "2014 study, but the previous administrative approval was "
            "not renewed."
        ),
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.9,
        claim_type=ClaimType.ADMINISTRATIVE_ACTION,
    )
    investigation = Investigation(
        question="Is the current basis still supported?",
        evidence=NO_REJUSTIFICATION_EVIDENCE,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    assert result.stance == EvidenceStance.NOT_ESTABLISHED
    assert result.confidence <= 0.30


# ---------------------------------------------------------------------------
# Final precision round (classifier freeze candidate): the retraction
# clause must be about the technical-justification assertion itself
# (validity/scope/applicability/whether the act occurred) to suppress it.
# A clause about an unrelated operational or administrative-timing fact,
# even with the same "but/however + negation" shape, must not suppress a
# genuine justification assertion.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement,expected",
    [
        (
            "The current limit was technically revalidated based on the "
            "2014 study, but the previous administrative approval was "
            "not renewed.",
            True,
        ),
        (
            "Engineering technically re-established the current "
            "operating boundary from the qualification data, although "
            "the older change request was not approved.",
            True,
        ),
        (
            "The current limit was re-approved based on new technical "
            "data, but the previous SOP revision was not retained.",
            True,
        ),
        (
            "The current limit was technically revalidated in 2014, but "
            "that validation did not cover the current equipment.",
            False,
        ),
        (
            "The current limit was re-approved, but no technical "
            "revalidation was performed.",
            False,
        ),
        (
            "The current boundary was technically re-established, "
            "although that decision does not apply to the current "
            "configuration.",
            False,
        ),
        (
            "The current limit was technically revalidated, but "
            "production output was not increased.",
            True,
        ),
        (
            "The current technical basis remains supported, although "
            "the administrative approval was delayed.",
            True,
        ),
    ],
)
def test_retraction_relevance_cases_1_through_8(statement, expected):
    assert _looks_like_current_technical_justification(statement) is expected


def test_unrelated_operational_fact_mislabeled_claim_is_still_gated():
    """
    End-to-end regression: a genuine technical revalidation followed
    only by an unrelated operational fact (production output), labeled
    CURRENT_CAPABILITY instead of CURRENT_TECHNICAL_JUSTIFICATION, must
    still be routed through the rejustification gate.
    """
    claim = ReasoningClaim(
        statement=(
            "The current limit was technically revalidated, but "
            "production output was not increased."
        ),
        stance=EvidenceStance.SUPPORTS,
        evidence_chunk_ids=["chunk-1"],
        confidence=0.9,
        claim_type=ClaimType.CURRENT_CAPABILITY,
    )
    investigation = Investigation(
        question="Is the current basis still supported?",
        evidence=NO_REJUSTIFICATION_EVIDENCE,
    )

    verified = ClaimVerifier().verify(make_report(claim), investigation)
    result = verified.claims[0]

    assert result.stance == EvidenceStance.NOT_ESTABLISHED
    assert result.confidence <= 0.30
