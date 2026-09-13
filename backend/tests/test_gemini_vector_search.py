from pathlib import Path

from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
from app.retrieval.embedding_store import EmbeddingStore
from app.retrieval.vector_search import VectorSearcher


def build_searcher():
    project_root = Path(__file__).resolve().parents[2]

    index_path = (
        project_root
        / "data"
        / "indexes"
        / "apex_embeddings.json"
    )

    store = EmbeddingStore(index_path)

    chunks, embeddings = store.load()

    provider = GeminiEmbeddingProvider()

    return VectorSearcher(
        chunks,
        embeddings,
        provider,
    )


def test_gemini_vector_search_returns_results():
    searcher = build_searcher()

    results = searcher.search(
        "Why was the Press-07 temperature limit established?",
        top_k=5,
    )

    assert len(results) == 5

    for result in results:
        assert result.content
        assert result.score >= -1
        assert result.score <= 1