import json

import pytest

from app.investigation.models import (
    EvidenceItem,
    EvidenceStance,
    Investigation,
)
from app.investigation.reasoner import InvestigationReasoner


class FakeReasoningClient:
    def __init__(self, response):
        self.response = response
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.response


def make_investigation():
    return Investigation(
        question="Why is the limit 42.3°C?",
        subquestions=[
            "What established the limit?",
            "What changed later?",
        ],
        evidence=[
            EvidenceItem(
                chunk_id="chunk-1",
                document_id="doc-1",
                filename="validation.txt",
                relative_path="B_historical_engineering/validation.txt",
                content="Historical validation evidence.",
                retrieval_score=0.95,
                retrieved_for=["What established the limit?"],
            ),
            EvidenceItem(
                chunk_id="chunk-2",
                document_id="doc-2",
                filename="requalification.txt",
                relative_path="B_historical_engineering/requalification.txt",
                content="Later requalification evidence.",
                retrieval_score=0.91,
                retrieved_for=["What changed later?"],
            ),
        ],
    )


def test_reasoner_creates_report():
    response = json.dumps(
        {
            "claims": [
                {
                    "statement": "The historical validation established the limit.",
                    "stance": "supports",
                    "evidence_chunk_ids": ["chunk-1"],
                    "confidence": 0.9,
                }
            ],
            "unresolved_questions": [],
        }
    )

    client = FakeReasoningClient(response)
    investigation = make_investigation()

    report = InvestigationReasoner(client).reason(investigation)

    assert report.question == "Why is the limit 42.3°C?"
    assert len(report.claims) == 1
    assert report.claims[0].stance == EvidenceStance.SUPPORTS
    assert report.claims[0].evidence_chunk_ids == ["chunk-1"]
    assert report.claims[0].confidence == 0.9


def test_invalid_citation_is_removed():
    response = json.dumps(
        {
            "claims": [
                {
                    "statement": "Unsupported claim.",
                    "stance": "supports",
                    "evidence_chunk_ids": [
                        "chunk-1",
                        "DOES-NOT-EXIST",
                    ],
                    "confidence": 0.8,
                }
            ]
        }
    )

    client = FakeReasoningClient(response)

    report = InvestigationReasoner(client).reason(
        make_investigation()
    )

    assert report.claims[0].evidence_chunk_ids == ["chunk-1"]


def test_confidence_is_clamped_when_citation_is_valid():
    response = json.dumps(
        {
            "claims": [
                {
                    "statement": "Claim.",
                    "stance": "supports",
                    "evidence_chunk_ids": ["chunk-1"],
                    "confidence": 4.5,
                },
                {
                    "statement": "Claim 2.",
                    "stance": "supports",
                    "evidence_chunk_ids": ["chunk-2"],
                    "confidence": -2,
                },
            ]
        }
    )

    client = FakeReasoningClient(response)

    report = InvestigationReasoner(client).reason(
        make_investigation()
    )

    assert report.claims[0].confidence == 1.0
    assert report.claims[1].confidence == 0.0


def test_claim_without_valid_citation_becomes_unknown():
    response = json.dumps(
        {
            "claims": [
                {
                    "statement": "Unsupported factual claim.",
                    "stance": "supports",
                    "evidence_chunk_ids": ["DOES-NOT-EXIST"],
                    "confidence": 0.9,
                }
            ]
        }
    )

    client = FakeReasoningClient(response)

    report = InvestigationReasoner(client).reason(
        make_investigation()
    )

    assert len(report.claims) == 1
    assert report.claims[0].stance == EvidenceStance.UNKNOWN
    assert report.claims[0].evidence_chunk_ids == []
    assert report.claims[0].confidence == 0.0


def test_unresolved_questions_are_preserved():
    response = json.dumps(
        {
            "claims": [],
            "unresolved_questions": [
                "Is the current limit still technically justified?"
            ],
        }
    )

    client = FakeReasoningClient(response)

    report = InvestigationReasoner(client).reason(
        make_investigation()
    )

    assert report.unresolved_questions == [
        "Is the current limit still technically justified?"
    ]


def test_invalid_json_raises():
    client = FakeReasoningClient("This is not JSON.")

    with pytest.raises(ValueError, match="invalid JSON"):
        InvestigationReasoner(client).reason(
            make_investigation()
        )


def test_prompt_contains_evidence_and_rules():
    client = FakeReasoningClient(
        json.dumps(
            {
                "claims": [],
                "unresolved_questions": [],
            }
        )
    )

    InvestigationReasoner(client).reason(
        make_investigation()
    )

    prompt = client.prompts[0]

    assert "Why is the limit 42.3°C?" in prompt
    assert "chunk-1" in prompt
    assert "chunk-2" in prompt
    assert "Do not invent facts" in prompt
    assert "administrative persistence" in prompt


def test_prompt_distinguishes_persistence_from_reaffirmation():
    client = FakeReasoningClient(
        json.dumps(
            {
                "claims": [
                    {
                        "statement": "The limit remained unchanged.",
                        "stance": "supports",
                        "evidence_chunk_ids": ["chunk-1"],
                        "confidence": 0.9,
                    }
                ],
                "unresolved_questions": [],
            }
        )
    )

    reasoner = InvestigationReasoner(client=client)

    investigation = Investigation(
        question="Why is the limit 42.3°C?",
        subquestions=["What changed later?"],
        evidence=[
            EvidenceItem(
                chunk_id="chunk-1",
                document_id="doc-1",
                filename="review.txt",
                relative_path="review.txt",
                content=(
                    "The 42.3°C limit remained unchanged during "
                    "the review. No technical reaffirmation was recorded."
                ),
                retrieval_score=1.0,
                retrieved_for=["What changed later?"],
            )
        ],
    )

    reasoner.reason(investigation)

    prompt = client.prompts[0]

    assert "Do not infer reaffirmation from mere persistence" in prompt
    assert "administratively retained" in prompt
    assert "carried forward unchanged" in prompt
    
def test_direct_answer_is_parsed_with_valid_evidence():
    response = json.dumps(
        {
            "direct_answer": {
                "text": "The 2011 replacement was documented through ECO-2011-184.",
                "evidence_chunk_ids": ["chunk-1"],
            },
            "claims": [],
            "unresolved_questions": [],
        }
    )

    client = FakeReasoningClient(response)
    report = InvestigationReasoner(client).reason(make_investigation())

    assert report.direct_answer == (
        "The 2011 replacement was documented through ECO-2011-184."
    )
    assert report.direct_answer_evidence_chunk_ids == ["chunk-1"]


def test_direct_answer_without_valid_citation_becomes_not_established():
    response = json.dumps(
        {
            "direct_answer": {
                "text": "The equipment was replaced because the old system was inadequate.",
                "evidence_chunk_ids": ["DOES-NOT-EXIST"],
            },
            "claims": [],
            "unresolved_questions": [],
        }
    )

    client = FakeReasoningClient(response)
    report = InvestigationReasoner(client).reason(make_investigation())

    assert report.direct_answer == (
        "The available evidence does not establish a direct answer "
        "to the investigation question."
    )
    assert report.direct_answer_evidence_chunk_ids == []


def test_direct_answer_is_not_hard_coded_to_flagship_case():
    response = json.dumps(
        {
            "direct_answer": {
                "text": "The calibration failure was documented in the supplied evidence.",
                "evidence_chunk_ids": ["chunk-1"],
            },
            "claims": [],
            "unresolved_questions": [],
        }
    )

    client = FakeReasoningClient(response)

    investigation = make_investigation()
    investigation.question = (
        "What evidence documents the calibration failure?"
    )

    report = InvestigationReasoner(client).reason(investigation)

    assert "42.3°C" not in report.direct_answer
    assert "calibration failure" in report.direct_answer