from app.investigation.models import (
    EvidenceItem,
    EvidenceStance,
    Investigation,
)


def make_evidence(chunk_id: str) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=chunk_id,
        document_id="doc_01",
        filename="example.txt",
        relative_path="B_historical_engineering/example.txt",
        content="Example evidence.",
        retrieval_score=0.85,
        retrieved_for="What historical event established this limit?",
    )


def test_investigation_stores_question_and_subquestions():
    investigation = Investigation(
        question="Why does this limit exist?",
        subquestions=[
            "What originally established the limit?",
            "What changed later?",
        ],
    )

    assert investigation.question == "Why does this limit exist?"
    assert len(investigation.subquestions) == 2
    assert investigation.evidence == []


def test_evidence_can_be_added():
    investigation = Investigation(
        question="Why does this limit exist?"
    )

    evidence = make_evidence("chunk_01")

    investigation.add_evidence(evidence)

    assert len(investigation.evidence) == 1
    assert investigation.evidence[0] == evidence


def test_duplicate_evidence_is_not_added_twice():
    investigation = Investigation(
        question="Why does this limit exist?"
    )

    investigation.add_evidence(make_evidence("chunk_01"))
    investigation.add_evidence(make_evidence("chunk_01"))

    assert len(investigation.evidence) == 1


def test_multiple_evidence_items_can_be_added():
    investigation = Investigation(
        question="Why does this limit exist?"
    )

    investigation.add_evidence_many(
        [
            make_evidence("chunk_01"),
            make_evidence("chunk_02"),
            make_evidence("chunk_03"),
        ]
    )

    assert len(investigation.evidence) == 3


def test_default_evidence_stance_is_unknown():
    evidence = make_evidence("chunk_01")

    assert evidence.stance == EvidenceStance.UNKNOWN