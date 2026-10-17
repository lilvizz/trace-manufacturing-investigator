"""
Vector-retrieval health checks for TRACE.

Separates three states that must never be conflated:

1. index present / loadable       -- the persisted embedding file exists
                                     and is consistent with the corpus;
2. query embedding verified       -- a real query was embedded by the
                                     live provider with the index's
                                     dimensionality;
3. vector retrieval contributed   -- for a given investigation, passages
                                     were actually ranked by the vector
                                     channel (see HybridSearchResult.channels).

Only (2) justifies showing "VECTOR ON". (3) is reported per investigation.

Run locally (from backend/) to validate the live pipeline:

    python -m app.retrieval.vector_health
    python -m app.retrieval.vector_health --query "Why is Press-07 limited to 42.3 C?"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

LEGACY_INDEX_MODEL = "gemini-embedding-001"
PROBE_QUERY = "Press-07 cure zone temperature limit technical basis"


@dataclass
class IndexHealth:
    path: str
    present: bool = False
    loaded: bool = False
    index_format: str = ""
    model: str = ""
    model_declared: bool = False
    records: int = 0
    corpus_chunks: int = 0
    dimensions: list[int] = field(default_factory=list)
    ids_match: bool = False
    stale_chunk_ids: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def dimension(self) -> int | None:
        return self.dimensions[0] if len(self.dimensions) == 1 else None

    @property
    def usable(self) -> bool:
        """Index can back vector search for the current corpus (IDs and
        dimensions consistent). Stale text is a warning, not a blocker:
        the chunk identity is unchanged, only its embedding is older."""
        return self.loaded and self.ids_match and self.dimension is not None

    def summary(self) -> dict:
        return {
            "present": self.present,
            "loaded": self.loaded,
            "format": self.index_format,
            "model": self.model,
            "model_declared": self.model_declared,
            "records": self.records,
            "corpus_chunks": self.corpus_chunks,
            "dimension": self.dimension,
            "ids_match": self.ids_match,
            "stale_chunks": len(self.stale_chunk_ids),
            "error": self.error,
        }


def load_index(index_path, corpus_chunks):
    """Load the persisted index in either supported format and check it
    against the current corpus chunks.

    Returns (health, index_chunk_ids, embeddings). Embeddings are paired
    with the *current* corpus chunks by ID by the caller, so evidence text
    shown to the reasoner always comes from the corpus, never from a
    possibly older copy stored in the index.
    """
    path = Path(index_path)
    health = IndexHealth(path=str(path), corpus_chunks=len(corpus_chunks))
    health.present = path.exists()
    if not health.present:
        health.error = "Index file not found. Run: python -m app.retrieval.indexer"
        return health, [], []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            health.index_format = "legacy-list"
            health.model = LEGACY_INDEX_MODEL
            health.model_declared = False
            records = payload
        elif isinstance(payload, dict):
            health.index_format = f"store-v{payload.get('schema_version', '?')}"
            health.model = str(payload.get("model", ""))
            health.model_declared = bool(payload.get("model"))
            # Keep the store's own validation (schema version, expected
            # model, declared dimensions, internal corpus hash): any
            # EmbeddingIndexError makes the index unusable, as before.
            from app.retrieval.embedding_store import EmbeddingStore

            EmbeddingStore(path).load(expected_model=LEGACY_INDEX_MODEL)
            records = payload.get("chunks", [])
        else:
            raise ValueError(f"Unexpected index format: {type(payload).__name__}")

        ids = [r["chunk_id"] for r in records]
        embeddings = [r["embedding"] for r in records]
        health.records = len(records)
        health.dimensions = sorted({len(e) for e in embeddings})
        health.ids_match = ids == [c.chunk_id for c in corpus_chunks]

        if health.ids_match:
            health.stale_chunk_ids = [
                r["chunk_id"]
                for r, chunk in zip(records, corpus_chunks)
                if str(r.get("content", "")).replace("\r", "")
                != str(chunk.content).replace("\r", "")
            ]
        health.loaded = True
        return health, ids, embeddings
    except Exception as exc:  # pragma: no cover - defensive
        health.error = f"{type(exc).__name__}: {exc}"
        return health, [], []


@dataclass
class ProbeResult:
    ok: bool
    dimension: int | None = None
    error: str = ""


def probe_query_embedding(provider, expected_dimension: int | None) -> ProbeResult:
    """Embed one real query with the live provider and check that its
    dimensionality matches the index. This is the only check that makes
    "VECTOR ON" true."""
    try:
        vector = list(provider.embed_query(PROBE_QUERY))
    except Exception as exc:
        return ProbeResult(ok=False, error=f"{type(exc).__name__}: {exc}"[:400])
    if expected_dimension is not None and len(vector) != expected_dimension:
        return ProbeResult(
            ok=False,
            dimension=len(vector),
            error=(
                f"Query embedding has {len(vector)} dimensions but the index has "
                f"{expected_dimension}; rebuild the index with the current model."
            ),
        )
    return ProbeResult(ok=True, dimension=len(vector))


def _main() -> int:  # pragma: no cover - manual validation entry point
    import argparse
    import os
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.config import CORPUS_PATH, INDEX_PATH, load_environment

    load_environment()  # same explicit .env path the Gemini clients use
    from app.ingestion.chunker import chunk_corpus
    from app.ingestion.loader import load_corpus
    from app.retrieval.bm25_search import BM25Searcher
    from app.retrieval.entity_matcher import EntityMatcher
    from app.retrieval.hybrid_retriever import HybridRetriever

    parser = argparse.ArgumentParser(description="Validate TRACE vector retrieval end to end.")
    parser.add_argument("--query", default="Why is Press-07 limited to 42.3°C and is its current technical basis still supported?")
    args = parser.parse_args()

    documents = load_corpus(str(CORPUS_PATH))
    chunks = chunk_corpus(documents)
    print(f"[1] corpus: {len(documents)} documents, {len(chunks)} chunks")

    health, _ids, embeddings = load_index(INDEX_PATH, chunks)
    print(f"[2] index: {json.dumps(health.summary())}")
    if health.stale_chunk_ids:
        print(f"    WARNING stale embeddings for: {', '.join(health.stale_chunk_ids)}")
        print("    Fix: python -m app.retrieval.indexer  (re-embeds the current corpus)")
    if not health.usable:
        print("FAIL: index not usable"); return 1

    if not os.getenv("GEMINI_API_KEY"):
        print("FAIL: GEMINI_API_KEY is not set (put it in the project .env)"); return 1

    from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
    from app.retrieval.vector_search import VectorSearcher

    provider = GeminiEmbeddingProvider(model=health.model or LEGACY_INDEX_MODEL)
    probe = probe_query_embedding(provider, health.dimension)
    print(f"[3] query embedding probe: ok={probe.ok} dim={probe.dimension} {probe.error}")
    if not probe.ok:
        print("FAIL: live query embedding did not succeed"); return 1

    vector = VectorSearcher(chunks=chunks, embeddings=embeddings, embedding_provider=provider)
    vector_hits = vector.search(args.query, top_k=5)
    print("[4] vector-only top 5:")
    for hit in vector_hits:
        print(f"    {hit.score:.4f}  {hit.chunk_id}")

    hybrid = HybridRetriever(BM25Searcher(chunks), vector, EntityMatcher(chunks))
    fused = hybrid.search(args.query, top_k=8)
    with_vector = [r for r in fused if "vector" in r.channels]
    print("[5] hybrid top 8 (channels):")
    for r in fused:
        print(f"    {r.chunk_id:55s} {'+'.join(r.channels)}")
    print(f"    vector calls={hybrid.vector_calls} failures={hybrid.vector_failures} "
          f"last_error={hybrid.last_vector_error}")
    if not with_vector or hybrid.vector_failures:
        print("FAIL: vector channel did not contribute to hybrid results"); return 1
    print(f"PASS: vector retrieval contributed {len(with_vector)}/{len(fused)} hybrid results")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
