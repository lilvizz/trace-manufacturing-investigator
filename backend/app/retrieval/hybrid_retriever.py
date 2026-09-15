from dataclasses import dataclass

from app.retrieval.diversifier import diversify_results


@dataclass(frozen=True)
class HybridSearchResult:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    score: float


class HybridRetriever:
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF).

    Combines independent retrieval signals without requiring their
    raw scores to be on the same scale.

    Retrieval channels:
        - BM25 lexical retrieval
        - semantic/vector retrieval
        - deterministic entity retrieval
    """

    def __init__(
        self,
        bm25_searcher,
        vector_searcher,
        entity_matcher,
        rrf_k: int = 60,
        candidate_k: int = 30,
        max_per_document: int = 2,
    ):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")

        if max_per_document <= 0:
            raise ValueError(
                "max_per_document must be greater than zero"
            )

        self.bm25_searcher = bm25_searcher
        self.vector_searcher = vector_searcher
        self.entity_matcher = entity_matcher
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.max_per_document = max_per_document

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[HybridSearchResult]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        rankers = []

        # BM25 is expected to be available.
        bm25_results = self.bm25_searcher.search(
            query,
            top_k=self.candidate_k,
        )
        rankers.append(bm25_results)

        # Vector retrieval is allowed to fail gracefully.
        try:
            vector_results = self.vector_searcher.search(
                query,
                top_k=self.candidate_k,
            )
        except Exception:
            vector_results = []

        if vector_results:
            rankers.append(vector_results)

        # Entity matching returns no results when the query contains
        # no recognized technical entities.
        try:
            entity_results = self.entity_matcher.search(
                query,
                top_k=self.candidate_k,
            )
        except Exception:
            entity_results = []

        if entity_results:
            rankers.append(entity_results)

        fused = self._rrf_fuse(rankers)

        diversified = diversify_results(
            fused,
            max_per_document=self.max_per_document,
            top_k=top_k,
        )

        return diversified

    def _rrf_fuse(self, rankers) -> list[HybridSearchResult]:
        """
        Fuse ranked result lists using Reciprocal Rank Fusion.

        RRF score:

            sum(1 / (rrf_k + rank))

        where rank is one-indexed.
        """

        scores = {}
        result_by_chunk_id = {}

        for results in rankers:
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id

                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

                result_by_chunk_id[chunk_id] = result

        ranked_chunk_ids = sorted(
            scores,
            key=lambda chunk_id: (
                scores[chunk_id],
                result_by_chunk_id[chunk_id].score,
            ),
            reverse=True,
        )

        return [
            HybridSearchResult(
                chunk_id=chunk_id,
                document_id=result_by_chunk_id[chunk_id].document_id,
                filename=result_by_chunk_id[chunk_id].filename,
                relative_path=result_by_chunk_id[chunk_id].relative_path,
                content=result_by_chunk_id[chunk_id].content,
                score=scores[chunk_id],
            )
            for chunk_id in ranked_chunk_ids
        ]