from pathlib import Path

from app.ingestion.chunker import DocumentChunk
from app.retrieval.embedding_store import EmbeddingStore


def test_embedding_store_round_trip(tmp_path):
    chunks = [
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

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    index_path = Path(tmp_path) / "embeddings.json"

    store = EmbeddingStore(index_path)

    store.save(chunks, embeddings)

    loaded_chunks, loaded_embeddings = store.load()

    assert len(loaded_chunks) == 2
    assert len(loaded_embeddings) == 2

    assert loaded_chunks[0].chunk_id == "doc1_0"
    assert loaded_chunks[1].chunk_id == "doc2_0"

    assert loaded_embeddings == embeddings