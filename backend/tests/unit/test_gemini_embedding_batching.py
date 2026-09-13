from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider


class FakeGeminiEmbeddingProvider(GeminiEmbeddingProvider):
    def __init__(self):
        pass

    def embed_documents(self, texts):
        return [[float(len(text))] for text in texts]


def test_documents_are_embedded_in_batches():
    provider = FakeGeminiEmbeddingProvider()

    texts = [
        "document one",
        "document two",
        "document three",
        "document four",
        "document five",
    ]

    embeddings = provider.embed_documents_batched(
        texts,
        batch_size=2,
    )

    assert len(embeddings) == 5
    assert embeddings[0] == [12.0]
    assert embeddings[-1] == [13.0]