import os
import random
import time

from google import genai
from google.genai import errors, types

from app.config import load_environment
from app.retrieval.embeddings import EmbeddingProvider

load_environment()


# HTTP status codes worth retrying: 429 (rate limited) and the standard
# transient server-side failures. Anything else (4xx like bad request/
# auth) is treated as non-retryable and raised immediately.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY_SECONDS = 1.0


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini-based embedding provider."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        max_retries: int = DEFAULT_MAX_RETRIES,
        base_delay_seconds: float = DEFAULT_BASE_DELAY_SECONDS,
        timeout_seconds: float | None = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        # Defaults keep the original behaviour for indexing. The UI passes
        # a short retry budget and a timeout so query embedding can never
        # stall an investigation for minutes.
        if timeout_seconds:
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
            )
        else:
            self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max(0, int(max_retries))
        self.base_delay_seconds = max(0.0, float(base_delay_seconds))

    def _embed_with_retry(
        self,
        contents,
        task_type: str,
        max_retries: int | None = None,
        base_delay: float | None = None,
    ):
        """
        Call the Gemini embed_content API, retrying transient failures
        (429 / 5xx) with exponential backoff and jitter. Non-retryable
        errors (bad request, auth, etc.) are raised immediately.
        """

        if max_retries is None:
            max_retries = getattr(self, "max_retries", DEFAULT_MAX_RETRIES)
        if base_delay is None:
            base_delay = getattr(self, "base_delay_seconds", DEFAULT_BASE_DELAY_SECONDS)

        attempt = 0

        while True:
            try:
                return self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
            except errors.APIError as exc:
                attempt += 1

                is_retryable = exc.code in RETRYABLE_STATUS_CODES
                retries_exhausted = attempt > max_retries

                if not is_retryable or retries_exhausted:
                    raise

                delay = base_delay * (2 ** (attempt - 1))
                delay += random.uniform(0, base_delay)

                time.sleep(delay)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        response = self._embed_with_retry(
            contents=texts,
            task_type="RETRIEVAL_DOCUMENT",
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

        The caller can choose a smaller batch size if necessary. Each
        batch already retries transient failures internally (see
        _embed_with_retry); a batch that still fails after retries are
        exhausted raises, leaving whatever the caller has already
        persisted from prior batches intact.
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
        response = self._embed_with_retry(
            contents=text,
            task_type="RETRIEVAL_QUERY",
        )

        return response.embeddings[0].values
