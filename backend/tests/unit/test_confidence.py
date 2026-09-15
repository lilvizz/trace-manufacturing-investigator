from app.investigation.confidence import ConfidenceCalibrator
from app.investigation.models import (
    ClaimType,
    EvidenceItem,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)


def make_investigation():
    evidence = EvidenceItem(
        chunk_id="chunk-1",
        document_id="doc-1",
        filename="test.txt",
        relative_path="test.txt",
        content="Explicit technical evidence.",
        retrieval_score=1.0,
    )

    return Investigation(
        question="Why is this limit 42.3°C?",
        evidence=[evidence],
    )


def test_current_technical_justification_has_confidence_ceiling():
    investigation = make_investigation()

    report = InvestigationReport(
        question=investigation.question,
        claims=[
            ReasoningClaim(
                statement="The current limit is technically justified.",
                stance=EvidenceStance.SUPPORTS,
                evidence_chunk_ids=["chunk-1"],
                confidence=0.99,
                claim_type=ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
            )
        ],
    )

    result = ConfidenceCalibrator().calibrate(
        report,
        investigation,
    )

    assert result.claims[0].confidence == 0.70


def test_not_established_is_capped_at_030():
    investigation = make_investigation()

    report = InvestigationReport(
        question=investigation.question,
        claims=[
            ReasoningClaim(
                statement="No current technical justification is established.",
                stance=EvidenceStance.NOT_ESTABLISHED,
                evidence_chunk_ids=["chunk-1"],
                confidence=0.95,
                claim_type=ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
            )
        ],
    )

    result = ConfidenceCalibrator().calibrate(
        report,
        investigation,
    )

    assert result.claims[0].confidence == 0.30


def test_missing_evidence_forces_unknown():
    investigation = make_investigation()

    report = InvestigationReport(
        question=investigation.question,
        claims=[
            ReasoningClaim(
                statement="Unsupported claim.",
                stance=EvidenceStance.SUPPORTS,
                evidence_chunk_ids=["does-not-exist"],
                confidence=0.95,
                claim_type=ClaimType.ORIGINAL_BASIS,
            )
        ],
    )

    result = ConfidenceCalibrator().calibrate(
        report,
        investigation,
    )

    claim = result.claims[0]

    assert claim.stance == EvidenceStance.UNKNOWN
    assert claim.confidence == 0.0


def test_context_only_has_confidence_ceiling():
    investigation = make_investigation()

    report = InvestigationReport(
        question=investigation.question,
        claims=[
            ReasoningClaim(
                statement="This document provides contextual information.",
                stance=EvidenceStance.CONTEXT_ONLY,
                evidence_chunk_ids=["chunk-1"],
                confidence=0.95,
                claim_type=ClaimType.CURRENT_CAPABILITY,
            )
        ],
    )

    result = ConfidenceCalibrator().calibrate(
        report,
        investigation,
    )

    assert result.claims[0].confidence == 0.65


def test_confidence_is_clamped():
    investigation = make_investigation()

    report = InvestigationReport(
        question=investigation.question,
        claims=[
            ReasoningClaim(
                statement="Claim.",
                stance=EvidenceStance.SUPPORTS,
                evidence_chunk_ids=["chunk-1"],
                confidence=4.0,
                claim_type=ClaimType.ORIGINAL_BASIS,
            )
        ],
    )

    result = ConfidenceCalibrator().calibrate(
        report,
        investigation,
    )

    assert result.claims[0].confidence == 1.0