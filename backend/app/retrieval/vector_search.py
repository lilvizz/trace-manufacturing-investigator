from dataclasses import dataclass
import math


@dataclass
class VectorSearchResult:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    score: float


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


class VectorSearcher:
    """Vector similarity search over precomputed embeddings."""

    def __init__(
        self,
        chunks,
        embeddings,
        embedding_provider,
    ):
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        dimensions = {len(e) for e in embeddings}
        if len(dimensions) > 1:
            raise ValueError(
                f"Index embeddings have inconsistent dimensions: {sorted(dimensions)}."
            )

        self.chunks = chunks
        self.embeddings = embeddings
        self.embedding_provider = embedding_provider
        self.dimension = next(iter(dimensions)) if dimensions else None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:

        query_embedding = self.embedding_provider.embed_query(
            query
        )

        # zip() in cosine_similarity would silently truncate a mismatched
        # vector and return meaningless scores; fail loudly instead so the
        # hybrid retriever records a vector failure.
        if self.dimension is not None and len(query_embedding) != self.dimension:
            raise ValueError(
                f"Query embedding has {len(query_embedding)} dimensions but the "
                f"index has {self.dimension}; rebuild the index with the query model "
                "(python -m app.retrieval.indexer)."
            )

        scored = []

        for chunk, embedding in zip(
            self.chunks,
            self.embeddings,
        ):
            score = cosine_similarity(
                query_embedding,
                embedding,
            )

            scored.append((chunk, score))

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for chunk, score in scored[:top_k]:
            results.append(
                VectorSearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    relative_path=chunk.relative_path,
                    content=chunk.content,
                    score=score,
                )
            )

        return results