"""Vector retrieval health: channel attribution, visible fallback, index
consistency against the real corpus, and query-probe semantics."""

import json

import pytest

from app.config import CORPUS_PATH, INDEX_PATH
from app.ingestion.chunker import DocumentChunk, chunk_corpus
from app.ingestion.loader import load_corpus
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.vector_health import load_index, probe_query_embedding
from app.retrieval.vector_search import VectorSearchResult


def _r(chunk_id, doc="d"):
    return VectorSearchResult(chunk_id, doc + chunk_id, f"{chunk_id}.txt", f"B/{chunk_id}.txt", "text", 1.0)


class Fixed:
    def __init__(self, results=None, error=None):
        self.results, self.error = results or [], error

    def search(self, query, top_k=5):
        if self.error:
            raise self.error
        return list(self.results)


def test_hybrid_results_record_which_channels_found_them():
    retriever = HybridRetriever(Fixed([_r("a"), _r("b")]), Fixed([_r("b"), _r("c")]), Fixed([_r("c")]))
    channels = {r.chunk_id: r.channels for r in retriever.search("press", top_k=5)}
    assert channels == {"a": ("bm25",), "b": ("bm25", "vector"), "c": ("vector", "entity")}
    assert retriever.vector_calls == 1 and retriever.vector_failures == 0


def test_vector_failure_still_falls_back_but_is_recorded():
    retriever = HybridRetriever(Fixed([_r("a")]), Fixed(error=RuntimeError("403 Forbidden")), Fixed())
    results = retriever.search("press", top_k=5)
    assert [r.chunk_id for r in results] == ["a"]
    assert all("vector" not in r.channels for r in results)
    assert retriever.vector_failures == 1
    assert "403 Forbidden" in retriever.last_vector_error


def test_disabled_vector_searcher_is_not_counted_as_a_failure():
    retriever = HybridRetriever(Fixed([_r("a")]), None, Fixed())
    retriever.search("press", top_k=5)
    assert retriever.vector_calls == 0 and retriever.vector_failures == 0


def _chunk(chunk_id, content):
    return DocumentChunk(document_id="d", filename="f.txt", relative_path="B/f.txt",
                         chunk_id=chunk_id, content=content)


def test_load_index_detects_stale_text_and_dimensions(tmp_path):
    chunks = [_chunk("c1", "current text"), _chunk("c2", "same")]
    records = [
        {"chunk_id": "c1", "document_id": "d", "filename": "f.txt", "relative_path": "B/f.txt",
         "content": "older text", "embedding": [0.1, 0.2, 0.3]},
        {"chunk_id": "c2", "document_id": "d", "filename": "f.txt", "relative_path": "B/f.txt",
         "content": "same", "embedding": [0.3, 0.2, 0.1]},
    ]
    path = tmp_path / "index.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    health, ids, embeddings = load_index(path, chunks)
    assert health.usable and health.ids_match
    assert health.dimension == 3
    assert health.stale_chunk_ids == ["c1"]
    assert health.model_declared is False  # legacy format: model assumed, not declared


def test_store_format_index_keeps_embedding_store_validation(tmp_path):
    from app.retrieval.embedding_store import EmbeddingStore

    chunks = [_chunk("c1", "text one"), _chunk("c2", "text two")]
    path = tmp_path / "index.json"
    EmbeddingStore(path).save(chunks, [[0.1, 0.2], [0.2, 0.1]], model="gemini-embedding-001")
    health, _, _ = load_index(path, chunks)
    assert health.usable and health.model_declared and health.stale_chunk_ids == []

    EmbeddingStore(path).save(chunks, [[0.1, 0.2], [0.2, 0.1]], model="some-other-model")
    health, _, _ = load_index(path, chunks)
    assert not health.usable and "some-other-model" in health.error


def test_load_index_reports_missing_file_and_id_mismatch(tmp_path):
    health, _, _ = load_index(tmp_path / "missing.json", [_chunk("c1", "x")])
    assert not health.present and not health.usable

    path = tmp_path / "index.json"
    path.write_text(json.dumps([{"chunk_id": "other", "content": "x", "embedding": [1.0]}]), encoding="utf-8")
    health, _, _ = load_index(path, [_chunk("c1", "x")])
    assert health.loaded and not health.ids_match and not health.usable


class Provider:
    def __init__(self, vector=None, error=None):
        self.vector, self.error = vector, error

    def embed_query(self, text):
        if self.error:
            raise self.error
        return self.vector


def test_query_probe_is_the_only_path_to_vector_on():
    assert probe_query_embedding(Provider([0.0] * 3), 3).ok
    failed = probe_query_embedding(Provider(error=RuntimeError("API key not valid")), 3)
    assert not failed.ok and "API key not valid" in failed.error
    mismatch = probe_query_embedding(Provider([0.0] * 768), 3072)
    assert not mismatch.ok and "768" in mismatch.error


@pytest.mark.skipif(not INDEX_PATH.exists(), reason="persisted index not present")
def test_real_persisted_index_matches_real_corpus():
    documents = load_corpus(str(CORPUS_PATH))
    chunks = chunk_corpus(documents)
    health, ids, embeddings = load_index(INDEX_PATH, chunks)

    assert len(documents) == 40
    assert len(chunks) == 70
    assert health.records == 70
    assert health.ids_match
    assert health.dimension == 3072
    assert health.usable
    # Stale text is reported (not hidden); rebuilding the index clears it.
    assert isinstance(health.stale_chunk_ids, list)
