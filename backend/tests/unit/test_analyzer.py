from app.investigation.analyzer import EvidenceGapAnalyzer
from app.investigation.models import (
    EvidenceItem,
    EvidenceStance,
    FindingType,
    Investigation,
)


def evidence(
    chunk_id,
    subquestion,
    stance=EvidenceStance.UNKNOWN,
):
    return EvidenceItem(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        filename=f"{chunk_id}.txt",
        relative_path=f"docs/{chunk_id}.txt",
        content="evidence",
        retrieval_score=0.9,
        retrieved_for=[subquestion],
        stance=stance,
    )


def test_no_evidence_creates_evidence_gap():
    investigation = Investigation(
        question="Why is the limit 42.3°C?",
        subquestions=["What established the limit?"],
    )

    analysis = EvidenceGapAnalyzer().analyze(investigation)

    assert len(analysis.findings) == 1
    assert analysis.findings[0].finding_type == FindingType.EVIDENCE_GAP


def test_all_subquestions_have_evidence():
    investigation = Investigation(
        question="Why is the limit 42.3°C?",
        subquestions=[
            "What established the limit?",
            "What changed later?",
        ],
        evidence=[
            evidence(
                "chunk-1",
                "What established the limit?",
            ),
            evidence(
                "chunk-2",
                "What changed later?",
            ),
        ],
    )

    analysis = EvidenceGapAnalyzer().analyze(investigation)

    gaps = [
        finding
        for finding in analysis.findings
        if finding.finding_type == FindingType.EVIDENCE_GAP
    ]

    assert gaps == []


def test_missing_subquestion_creates_gap():
    investigation = Investigation(
        question="Why is the limit 42.3°C?",
        subquestions=[
            "What established the limit?",
            "What changed later?",
        ],
        evidence=[
            evidence(
                "chunk-1",
                "What established the limit?",
            ),
        ],
    )

    analysis = EvidenceGapAnalyzer().analyze(investigation)

    gaps = [
        finding
        for finding in analysis.findings
        if finding.finding_type == FindingType.EVIDENCE_GAP
    ]

    assert len(gaps) == 1
    assert "What changed later?" in gaps[0].statement


def test_stance_is_not_used_as_ground_truth():
    investigation = Investigation(
        question="Why is the limit 42.3°C?",
        subquestions=["What established the limit?"],
        evidence=[
            evidence(
                "chunk-1",
                "What established the limit?",
                EvidenceStance.SUPPORTS,
            ),
        ],
    )

    analysis = EvidenceGapAnalyzer().analyze(investigation)

    # The analyzer must not blindly convert Gemini/populated stance
    # into a finding. Stance is handled later by the reasoning layer.
    assert all(
        finding.finding_type != FindingType.SUPPORTED
        for finding in analysis.findings
    )


def test_contradiction_is_not_inferred_from_stance():
    investigation = Investigation(
        question="Why is the limit 42.3°C?",
        subquestions=["What changed later?"],
        evidence=[
            evidence(
                "chunk-1",
                "What changed later?",
                EvidenceStance.CONTRADICTS,
            ),
        ],
    )

    analysis = EvidenceGapAnalyzer().analyze(investigation)

    assert all(
        finding.finding_type != FindingType.CONTRADICTION
        for finding in analysis.findings
    )


def test_context_is_not_inferred_from_stance():
    investigation = Investigation(
        question="Why is the limit 42.3°C?",
        subquestions=["What established the limit?"],
        evidence=[
            evidence(
                "chunk-1",
                "What established the limit?",
                EvidenceStance.CONTEXT_ONLY,
            ),
        ],
    )

    analysis = EvidenceGapAnalyzer().analyze(investigation)

    assert all(
        finding.finding_type != FindingType.CONTEXT
        for finding in analysis.findings
    )