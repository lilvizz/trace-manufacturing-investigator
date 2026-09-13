import os

from dotenv import load_dotenv
from google import genai

from app.retrieval.embeddings import EmbeddingProvider

load_dotenv()


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini-based embedding provider."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
        )

        return [
            embedding.values
            for embedding in response.embeddings
        ]

    def embed_documents_batched(
        self,
        texts: list[str],
        batch_size: int = 50,
    ) -> list[list[float]]:
        """
        Embed documents in controlled batches.

        The caller can choose a smaller batch size if necessary.
        """

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            embeddings = self.embed_documents(batch)

            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )

        return response.embeddings[0].values