"""
Integration test: makes a real call to the Gemini embedding API.

Not run by a plain `pytest` invocation (see backend/pytest.ini,
testpaths = tests/unit). Run explicitly with:

    pytest tests/integration

Requires a real GEMINI_API_KEY; skips cleanly if one isn't set.
"""

import os

import pytest

from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider

pytestmark = pytest.mark.integration

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="GEMINI_API_KEY not set; skipping live Gemini integration test.",
)
def test_gemini_can_generate_embedding():
    provider = GeminiEmbeddingProvider()

    embedding = provider.embed_query(
        "What historical decision established this equipment limit?"
    )

    assert embedding
    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)
