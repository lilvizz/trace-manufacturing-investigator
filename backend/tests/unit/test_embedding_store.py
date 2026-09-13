import json

import pytest

from app.ingestion.chunker import DocumentChunk
from app.retrieval.embedding_store import EmbeddingIndexError, EmbeddingStore


def make_chunks():
    return [
        DocumentChunk(
            document_id="doc1",
            filename="test.txt",
            relative_path="test.txt",
            chunk_id="doc1_0",
            content="Thermal instability occurred during extended runs.",
        ),
        DocumentChunk(
            document_id="doc2",
            filename="test2.txt",
            relative_path="test2.txt",
            chunk_id="doc2_0",
            content="PX-400 qualification showed stable operation.",
        ),
    ]


def test_embedding_store_round_trip(tmp_path):
    chunks = make_chunks()

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    index_path = tmp_path / "embeddings.json"

    store = EmbeddingStore(index_path)

    store.save(chunks, embeddings, model="fake-embedding-model")

    loaded_chunks, loaded_embeddings = store.load()

    assert len(loaded_chunks) == 2
    assert len(loaded_embeddings) == 2

    assert loaded_chunks[0].chunk_id == "doc1_0"
    assert loaded_chunks[1].chunk_id == "doc2_0"

    assert loaded_embeddings == embeddings


def test_embedding_store_persists_metadata(tmp_path):
    chunks = make_chunks()
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    index_path = tmp_path / "embeddings.json"
    store = EmbeddingStore(index_path)

    store.save(chunks, embeddings, model="fake-embedding-model")

    payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["model"] == "fake-embedding-model"
    assert payload["dimensions"] == 3
    assert isinstance(payload["corpus_hash"], str)
    assert len(payload["corpus_hash"]) == 64  # sha256 hex digest


def test_load_rejects_mismatched_expected_model(tmp_path):
    chunks = make_chunks()
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    index_path = tmp_path / "embeddings.json"
    store = EmbeddingStore(index_path)

    store.save(chunks, embeddings, model="gemini-embedding-001")

    with pytest.raises(EmbeddingIndexError):
        store.load(expected_model="some-other-model")


def test_load_rejects_unknown_schema_version(tmp_path):
    index_path = tmp_path / "embeddings.json"

    payload = {
        "schema_version": 999,
        "model": "fake-embedding-model",
        "dimensions": 3,
        "corpus_hash": "irrelevant",
        "chunks": [],
    }

    index_path.write_text(json.dumps(payload), encoding="utf-8")

    store = EmbeddingStore(index_path)

    with pytest.raises(EmbeddingIndexError):
        store.load()


def test_load_rejects_tampered_content_hash(tmp_path):
    chunks = make_chunks()
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    index_path = tmp_path / "embeddings.json"
    store = EmbeddingStore(index_path)

    store.save(chunks, embeddings, model="fake-embedding-model")

    # Simulate hand-editing/corruption: change a chunk's content
    # without updating the stored corpus_hash.
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    payload["chunks"][0]["content"] = "tampered content"
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EmbeddingIndexError):
        store.load()


def test_load_raises_for_missing_index(tmp_path):
    store = EmbeddingStore(tmp_path / "does_not_exist.json")

    with pytest.raises(FileNotFoundError):
        store.load()
