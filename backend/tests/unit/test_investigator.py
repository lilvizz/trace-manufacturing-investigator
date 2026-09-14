import json

import pytest

from app.investigation.analyzer import EvidenceGapAnalyzer
from app.investigation.investigator import Investigator
from app.investigation.models import EvidenceItem
from app.investigation.reasoner import InvestigationReasoner
from app.investigation.retriever import (
    InvestigationRetriever,
    RetrievalResult,
)
from app.investigation.timeline import TimelineBuilder


class FakePlanner:
    def create_plan(self, question):
        from app.investigation.planner import InvestigationPlan

        return InvestigationPlan(
            question=question,
            subquestions=[
                "What established the condition?",
                "What changed later?",
            ],
        )


class FakeRetriever:
    def search(self, query, top_k=5):
        if "established" in query:
            return [
                RetrievalResult(
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    filename="historical.txt",
                    relative_path="B/historical.txt",
                    content="The condition was established on 2001-07-05.",
                    score=0.95,
                )
            ]

        return [
            RetrievalResult(
                chunk_id="chunk-2",
                document_id="doc-2",
                filename="requalification.txt",
                relative_path="B/requalification.txt",
                content="The equipment was requalified on 2014-06-18.",
                score=0.90,
            )
        ]


class FakeReasoningClient:
    def generate(self, prompt):
        return json.dumps(
            {
                "claims": [
                    {
                        "statement": (
                            "The historical evidence establishes "
                            "the original condition."
                        ),
                        "stance": "supports",
                        "evidence_chunk_ids": ["chunk-1"],
                        "confidence": 0.9,
                    }
                ],
                "unresolved_questions": [
                    "Is the condition still technically justified?"
                ],
            }
        )


def make_investigator():
    retriever = InvestigationRetriever(
        FakeRetriever(),
        top_k_per_subquestion=3,
    )

    reasoner = InvestigationReasoner(
        FakeReasoningClient()
    )

    return Investigator(
        planner=FakePlanner(),
        retriever=retriever,
        timeline_builder=TimelineBuilder(),
        analyzer=EvidenceGapAnalyzer(),
        reasoner=reasoner,
    )


def test_run_executes_complete_pipeline():
    investigator = make_investigator()

    result = investigator.run(
        "Why is this condition still present?"
    )

    assert result.plan.question == (
        "Why is this condition still present?"
    )

    assert len(result.investigation.evidence) == 2

    assert len(result.timeline.events) == 2

    assert result.report.question == (
        "Why is this condition still present?"
    )

    assert len(result.report.claims) == 1

    assert result.report.claims[0].evidence_chunk_ids == [
        "chunk-1"
    ]


def test_run_preserves_evidence_provenance():
    investigator = make_investigator()

    result = investigator.run(
        "Why is this condition still present?"
    )

    evidence_ids = {
        evidence.chunk_id
        for evidence in result.investigation.evidence
    }

    for claim in result.report.claims:
        for chunk_id in claim.evidence_chunk_ids:
            assert chunk_id in evidence_ids


def test_run_returns_unresolved_questions():
    investigator = make_investigator()

    result = investigator.run(
        "Why is this condition still present?"
    )

    assert result.report.unresolved_questions == [
        "Is the condition still technically justified?"
    ]


def test_empty_question_is_rejected():
    investigator = make_investigator()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        investigator.run("   ")