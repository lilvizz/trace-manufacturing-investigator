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
    """Vector similarity search over document chunks."""

    def __init__(self, chunks, embedding_provider):
        self.chunks = chunks
        self.embedding_provider = embedding_provider

        self.embeddings = embedding_provider.embed_documents(
            [chunk.content for chunk in chunks]
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[VectorSearchResult]:

        query_embedding = self.embedding_provider.embed_query(query)

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