"""
Integration test: loads the real persisted embedding index and makes
a real call to the Gemini embedding API to embed a query.

Not run by a plain `pytest` invocation (see backend/pytest.ini,
testpaths = tests/unit). Run explicitly with:

    pytest tests/integration

Requires both a real GEMINI_API_KEY and a prebuilt embedding index at
data/indexes/apex_embeddings.json (see app/retrieval/indexer.py);
skips cleanly if either is unavailable.
"""

import os

import pytest

from app.config import INDEX_PATH
from app.retrieval.embedding_store import EmbeddingStore
from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
from app.retrieval.vector_search import VectorSearcher

pytestmark = pytest.mark.integration

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def build_searcher():
    store = EmbeddingStore(INDEX_PATH)
    chunks, embeddings = store.load()

    provider = GeminiEmbeddingProvider()

    return VectorSearcher(
        chunks,
        embeddings,
        provider,
    )


@pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="GEMINI_API_KEY not set; skipping live Gemini integration test.",
)
@pytest.mark.skipif(
    not INDEX_PATH.exists(),
    reason=(
        "Embedding index not found at "
        f"{INDEX_PATH}; run app/retrieval/indexer.py first."
    ),
)
def test_gemini_vector_search_returns_results():
    searcher = build_searcher()

    results = searcher.search(
        "What historical decision established this equipment limit?",
        top_k=5,
    )

    assert len(results) == 5

    for result in results:
        assert result.content
        assert result.score >= -1
        assert result.score <= 1
