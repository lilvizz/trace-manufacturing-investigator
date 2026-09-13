import hashlib

from app.retrieval.embeddings import EmbeddingProvider


class TestEmbeddingProvider(EmbeddingProvider):
    """
    Deterministic fake embedding provider used only for tests.

    This is NOT used by the real TRACE application.
    """

    def __init__(self, dimensions: int = 64):
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        values = []

        for index in range(self.dimensions):
            digest = hashlib.sha256(
                f"{index}:{text.lower()}".encode("utf-8")
            ).digest()

            value = int.from_bytes(digest[:4], "big") / 2**32
            values.append(value)

        return values

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)