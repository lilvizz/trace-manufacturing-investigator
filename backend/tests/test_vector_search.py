from pathlib import Path

from app.ingestion.loader import load_corpus
from app.ingestion.chunker import chunk_corpus
from app.retrieval.test_embeddings import TestEmbeddingProvider
from app.retrieval.vector_search import VectorSearcher


def build_searcher():
    project_root = Path(__file__).resolve().parents[2]
    corpus_path = project_root / "data" / "apex-corpus"

    documents = load_corpus(str(corpus_path))
    chunks = chunk_corpus(documents)

    provider = TestEmbeddingProvider()

    return VectorSearcher(chunks, provider)


def test_vector_search_returns_results():
    searcher = build_searcher()

    results = searcher.search(
        "Why was the temperature boundary established?",
        top_k=5,
    )

    assert len(results) == 5

    for result in results:
        assert result.content
        assert 0 <= result.score <= 1


def test_vector_search_is_ranked():
    searcher = build_searcher()

    results = searcher.search(
        "thermal instability temperature",
        top_k=5,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)