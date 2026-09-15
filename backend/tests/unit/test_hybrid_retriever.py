from dataclasses import dataclass

import pytest

from app.retrieval.hybrid_retriever import HybridRetriever


@dataclass(frozen=True)
class FakeResult:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    score: float


class FakeRanker:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k=5):
        return self.results[:top_k]


def result(chunk_id, document_id, score=1.0):
    return FakeResult(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{chunk_id}.txt",
        relative_path=f"{chunk_id}.txt",
        content=f"Content for {chunk_id}",
        score=score,
    )


def test_rrf_combines_multiple_rankers():
    bm25 = FakeRanker(
        [
            result("A", "doc-a"),
            result("B", "doc-b"),
            result("C", "doc-c"),
        ]
    )

    vector = FakeRanker(
        [
            result("B", "doc-b"),
            result("A", "doc-a"),
            result("D", "doc-d"),
        ]
    )

    entity = FakeRanker(
        [
            result("B", "doc-b"),
            result("D", "doc-d"),
            result("A", "doc-a"),
        ]
    )

    retriever = HybridRetriever(
        bm25_searcher=bm25,
        vector_searcher=vector,
        entity_matcher=entity,
    )

    results = retriever.search(
        "PX-400",
        top_k=4,
    )

    assert results[0].chunk_id == "B"


def test_rrf_rewards_consistent_results():
    bm25 = FakeRanker(
        [
            result("A", "doc-a"),
            result("B", "doc-b"),
            result("C", "doc-c"),
        ]
    )

    vector = FakeRanker(
        [
            result("A", "doc-a"),
            result("C", "doc-c"),
            result("B", "doc-b"),
        ]
    )

    entity = FakeRanker(
        [
            result("A", "doc-a"),
            result("B", "doc-b"),
        ]
    )

    retriever = HybridRetriever(
        bm25_searcher=bm25,
        vector_searcher=vector,
        entity_matcher=entity,
    )

    results = retriever.search(
        "PX-400",
        top_k=3,
    )

    assert results[0].chunk_id == "A"


def test_document_diversification_is_applied():
    bm25 = FakeRanker(
        [
            result("A1", "same-doc"),
            result("A2", "same-doc"),
            result("A3", "same-doc"),
            result("B1", "other-doc"),
        ]
    )

    vector = FakeRanker([])

    entity = FakeRanker([])

    retriever = HybridRetriever(
        bm25_searcher=bm25,
        vector_searcher=vector,
        entity_matcher=entity,
        max_per_document=2,
    )

    results = retriever.search(
        "PX-400",
        top_k=4,
    )

    same_doc_count = sum(
        result.document_id == "same-doc"
        for result in results
    )

    assert same_doc_count <= 2


def test_vector_failure_falls_back_to_other_channels():
    bm25 = FakeRanker(
        [
            result("A", "doc-a"),
            result("B", "doc-b"),
        ]
    )

    class FailingVector:
        def search(self, query, top_k=5):
            raise RuntimeError("embedding service unavailable")

    entity = FakeRanker(
        [
            result("B", "doc-b"),
        ]
    )

    retriever = HybridRetriever(
        bm25_searcher=bm25,
        vector_searcher=FailingVector(),
        entity_matcher=entity,
    )

    results = retriever.search(
        "PX-400",
        top_k=2,
    )

    assert results
    assert {result.chunk_id for result in results} == {"A", "B"}


def test_empty_query_raises():
    retriever = HybridRetriever(
        bm25_searcher=FakeRanker([]),
        vector_searcher=FakeRanker([]),
        entity_matcher=FakeRanker([]),
    )

    with pytest.raises(ValueError):
        retriever.search("")


def test_invalid_top_k_raises():
    retriever = HybridRetriever(
        bm25_searcher=FakeRanker([]),
        vector_searcher=FakeRanker([]),
        entity_matcher=FakeRanker([]),
    )

    with pytest.raises(ValueError):
        retriever.search("PX-400", top_k=0)