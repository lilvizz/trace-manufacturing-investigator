from dataclasses import dataclass
import threading

from app.retrieval.diversifier import diversify_results


@dataclass(frozen=True)
class HybridSearchResult:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    score: float
    # Which retrieval channels ranked this chunk ("bm25", "vector",
    # "entity"). Defaulted so existing constructors keep working; it lets
    # an investigation show, per passage, whether vector retrieval
    # actually contributed rather than inferring it from configuration.
    channels: tuple[str, ...] = ()


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

        # Cumulative channel diagnostics. Vector failures still fall back
        # gracefully to BM25 + entity retrieval (unchanged behaviour), but
        # they are now counted and the last error is kept, so the UI can
        # report a fallback instead of silently claiming hybrid retrieval.
        self._stats_lock = threading.Lock()
        self.vector_calls = 0
        self.vector_failures = 0
        self.last_vector_error: str | None = None
        self.entity_failures = 0
        self.last_entity_error: str | None = None

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

        # Vector retrieval is allowed to fail gracefully. A missing
        # searcher is "disabled", not a failure; a raised exception is
        # recorded so the fallback is visible to callers.
        vector_results = []
        if self.vector_searcher is not None:
            try:
                vector_results = self.vector_searcher.search(
                    query,
                    top_k=self.candidate_k,
                )
                with self._stats_lock:
                    self.vector_calls += 1
            except Exception as exc:
                vector_results = []
                with self._stats_lock:
                    self.vector_calls += 1
                    self.vector_failures += 1
                    self.last_vector_error = (
                        f"{type(exc).__name__}: {exc}"
                    )[:400]

        if vector_results:
            rankers.append(vector_results)

        # Entity matching returns no results when the query contains
        # no recognized technical entities.
        try:
            entity_results = self.entity_matcher.search(
                query,
                top_k=self.candidate_k,
            )
        except Exception as exc:
            # Still a graceful fallback, but recorded (previously silent).
            entity_results = []
            with self._stats_lock:
                self.entity_failures += 1
                self.last_entity_error = f"{type(exc).__name__}: {exc}"[:400]

        if entity_results:
            rankers.append(entity_results)

        fused = self._rrf_fuse(rankers)

        channel_ids = {
            "bm25": {r.chunk_id for r in bm25_results},
            "vector": {r.chunk_id for r in vector_results},
            "entity": {r.chunk_id for r in entity_results},
        }
        fused = [
            HybridSearchResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                filename=r.filename,
                relative_path=r.relative_path,
                content=r.content,
                score=r.score,
                channels=tuple(
                    name for name, ids in channel_ids.items()
                    if r.chunk_id in ids
                ),
            )
            for r in fused
        ]

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