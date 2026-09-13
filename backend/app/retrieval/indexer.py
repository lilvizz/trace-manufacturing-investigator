from pathlib import Path

from app.ingestion.loader import load_corpus
from app.ingestion.chunker import chunk_corpus
from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
from app.retrieval.embedding_store import EmbeddingStore


def build_embedding_index(
    corpus_path: str | Path,
    index_path: str | Path,
) -> int:
    """
    Build and persist embeddings for the entire corpus.

    This should normally be run only when the corpus changes.
    """

    documents = load_corpus(str(corpus_path))
    chunks = chunk_corpus(documents)

    provider = GeminiEmbeddingProvider()

    embeddings = provider.embed_documents_batched(
        [chunk.content for chunk in chunks],
        batch_size=50,
    )

    store = EmbeddingStore(index_path)

    store.save(
        chunks,
        embeddings,
    )

    return len(chunks)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]

    corpus_path = (
        project_root
        / "data"
        / "apex-corpus"
    )

    index_path = (
        project_root
        / "data"
        / "indexes"
        / "apex_embeddings.json"
    )

    chunk_count = build_embedding_index(
        corpus_path,
        index_path,
    )

    print(
        f"Embedding index created successfully: "
        f"{chunk_count} chunks indexed."
    )