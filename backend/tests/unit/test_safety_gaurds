from app.investigation.analyzer import (
    EvidenceGapAnalyzer,
    _contains_explicit_rejustification,
)
from app.investigation.claim_verifier import ClaimVerifier
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


def test_negated_reapproval_is_not_explicit_rejustification():
    assert not _contains_explicit_rejustification(
        "No evidence suggests the limit was re-approved for PX-400."
    )
    assert not _contains_explicit_rejustification(
        "The limit was not re-approved for PX-400."
    )
    assert not _contains_explicit_rejustification(
        "The record was carried forward without being re-approved."
    )


def test_positive_reapproval_is_explicit_rejustification():
    assert _contains_explicit_rejustification(
        "The 42.3°C limit was re-approved for PX-400."
    )


def test_claim_verifier_keeps_negated_reapproval_not_established():
    investigation = Investigation(
        question="Is the current technical basis still supported?",
        evidence=[
            make_evidence(
                "chunk-1",
                "No evidence suggests the limit was re-approved for PX-400.",
            )
        ],
    )
    report = InvestigationReport(
        question=investigation.question,
        claims=[
            ReasoningClaim(
                statement="The current limit is technically justified.",
                stance=EvidenceStance.SUPPORTS,
                evidence_chunk_ids=["chunk-1"],
                confidence=0.90,
                claim_type=ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
            )
        ],
    )

    verified = ClaimVerifier().verify(report, investigation)
    claim = verified.claims[0]

    assert claim.stance == EvidenceStance.NOT_ESTABLISHED
    assert claim.confidence <= 0.30


def test_analyzer_keeps_gap_when_reapproval_is_negated():
    investigation = Investigation(
        question="Why is the current limit still supported?",
        subquestions=["What supports the current limit?"],
        evidence=[
            make_evidence(
                "chunk-change",
                "The PX-200 was replaced by PX-400. No evidence suggests the limit was re-approved for PX-400.",
            )
        ],
    )

    analysis = EvidenceGapAnalyzer().analyze(investigation)

    assert any(
        finding.finding_type.value == "evidence-gap"
        and "Technical re-justification is not established" in finding.statement
        for finding in analysis.findings
    )


def test_analyzer_flags_mismatched_equipment_without_blocking_evidence():
    investigation = Investigation(
        question="Why is Press-07 limited to 42.3°C?",
        subquestions=["What supports the current limit?"],
        evidence=[
            make_evidence(
                "match",
                "Press-07 remains configured at 42.3°C.",
            ),
            make_evidence(
                "distractor",
                "Press-05 thermal investigation evaluated a 44.0°C UCL.",
            ),
        ],
    )

    analysis = EvidenceGapAnalyzer().analyze(investigation)

    consistency_findings = [
        finding
        for finding in analysis.findings
        if "equipment identifiers" in finding.statement
    ]

    assert len(consistency_findings) == 1
    assert consistency_findings[0].evidence_chunk_ids == ["distractor"]
    assert len(investigation.evidence) == 2
