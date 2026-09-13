from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider


def test_gemini_can_generate_embedding():
    provider = GeminiEmbeddingProvider()

    embedding = provider.embed_query(
        "Why was the 42.3 degree temperature limit established?"
    )

    assert embedding
    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)