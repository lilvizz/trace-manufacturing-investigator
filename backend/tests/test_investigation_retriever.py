from app.investigation.models import Investigation
from app.investigation.planner import InvestigationPlan
from app.investigation.retriever import (
    InvestigationRetriever,
    RetrievalResult,
)


class FakeRetriever:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, [])


def result(
    chunk_id,
    document_id,
    score,
    content="evidence",
):
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.txt",
        relative_path=f"B_historical_engineering/{document_id}.txt",
        content=content,
        score=score,
    )


def test_collect_retrieves_for_every_subquestion():
    retriever = FakeRetriever(
        {
            "question one": [
                result("chunk-1", "doc-1", 0.9),
            ],
            "question two": [
                result("chunk-2", "doc-2", 0.8),
            ],
        }
    )

    investigation = Investigation(
        question="main question",
        subquestions=["question one", "question two"],
    )

    plan = InvestigationPlan(
        question="main question",
        subquestions=["question one", "question two"],
    )

    collector = InvestigationRetriever(
        retriever,
        top_k_per_subquestion=3,
    )

    collector.collect(investigation, plan)

    assert len(investigation.evidence) == 2
    assert retriever.calls == [
        ("question one", 3),
        ("question two", 3),
    ]


def test_duplicate_chunks_are_merged():
    shared = result("shared", "doc-1", 0.9)

    retriever = FakeRetriever(
        {
            "question one": [shared],
            "question two": [shared],
        }
    )

    investigation = Investigation(
        question="main question",
        subquestions=["question one", "question two"],
    )

    plan = InvestigationPlan(
        question="main question",
        subquestions=["question one", "question two"],
    )

    collector = InvestigationRetriever(retriever)

    collector.collect(investigation, plan)

    assert len(investigation.evidence) == 1
    assert investigation.evidence[0].chunk_id == "shared"
    assert investigation.evidence[0].retrieved_for == [
        "question one",
        "question two",
    ]


def test_duplicate_chunk_keeps_best_score():
    retriever = FakeRetriever(
        {
            "question one": [
                result("shared", "doc-1", 0.6),
            ],
            "question two": [
                result("shared", "doc-1", 0.95),
            ],
        }
    )

    investigation = Investigation(
        question="main question",
        subquestions=["question one", "question two"],
    )

    plan = InvestigationPlan(
        question="main question",
        subquestions=["question one", "question two"],
    )

    collector = InvestigationRetriever(retriever)

    collector.collect(investigation, plan)

    assert len(investigation.evidence) == 1
    assert investigation.evidence[0].retrieval_score == 0.95


def test_evidence_is_limited_globally():
    retriever = FakeRetriever(
        {
            "question one": [
                result("chunk-1", "doc-1", 0.9),
                result("chunk-2", "doc-2", 0.5),
            ],
            "question two": [
                result("chunk-3", "doc-3", 0.8),
                result("chunk-4", "doc-4", 0.4),
            ],
        }
    )

    investigation = Investigation(
        question="main question",
        subquestions=["question one", "question two"],
    )

    plan = InvestigationPlan(
        question="main question",
        subquestions=["question one", "question two"],
    )

    collector = InvestigationRetriever(
        retriever,
        max_evidence=2,
    )

    collector.collect(investigation, plan)

    assert len(investigation.evidence) == 2
    assert [item.chunk_id for item in investigation.evidence] == [
        "chunk-1",
        "chunk-3",
    ]


def test_empty_subquestion_results_do_not_fail():
    retriever = FakeRetriever(
        {
            "question one": [],
        }
    )

    investigation = Investigation(
        question="main question",
        subquestions=["question one"],
    )

    plan = InvestigationPlan(
        question="main question",
        subquestions=["question one"],
    )

    collector = InvestigationRetriever(retriever)

    collector.collect(investigation, plan)

    assert investigation.evidence == []