from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for generating vector embeddings."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a query."""
        raise NotImplementedError