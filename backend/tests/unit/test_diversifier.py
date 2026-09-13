from app.retrieval.diversifier import diversify_results
from app.retrieval.vector_search import VectorSearchResult


def make_result(
    document_id: str,
    chunk_id: str,
    score: float,
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.txt",
        relative_path=f"{document_id}.txt",
        content="test content",
        score=score,
    )


def test_diversifier_limits_duplicate_documents():
    results = [
        make_result("doc1", "doc1_0", 0.90),
        make_result("doc1", "doc1_1", 0.89),
        make_result("doc1", "doc1_2", 0.88),
        make_result("doc2", "doc2_0", 0.87),
        make_result("doc3", "doc3_0", 0.86),
    ]

    diversified = diversify_results(
        results,
        max_per_document=1,
        top_k=3,
    )

    assert len(diversified) == 3

    document_ids = [
        result.document_id
        for result in diversified
    ]

    assert document_ids == [
        "doc1",
        "doc2",
        "doc3",
    ]


def test_diversifier_can_allow_multiple_chunks():
    results = [
        make_result("doc1", "doc1_0", 0.90),
        make_result("doc1", "doc1_1", 0.89),
        make_result("doc2", "doc2_0", 0.88),
    ]

    diversified = diversify_results(
        results,
        max_per_document=2,
        top_k=3,
    )

    assert len(diversified) == 3

    assert diversified[0].chunk_id == "doc1_0"
    assert diversified[1].chunk_id == "doc1_1"
    assert diversified[2].chunk_id == "doc2_0"