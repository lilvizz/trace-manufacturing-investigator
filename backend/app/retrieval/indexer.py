from app.config import CORPUS_PATH, INDEX_PATH
from app.ingestion.loader import load_corpus
from app.ingestion.chunker import chunk_corpus
from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
from app.retrieval.embedding_store import EmbeddingStore


def build_embedding_index(
    corpus_path=CORPUS_PATH,
    index_path=INDEX_PATH,
    batch_size: int = 50,
) -> int:
    """
    Build and persist embeddings for the entire corpus.

    This should normally be run only when the corpus changes.

    Embeddings are persisted after every batch (not only once at the
    end), so a transient failure partway through a large run does not
    discard embeddings already computed for earlier batches. This is
    a deliberately simple form of incremental persistence: the whole
    index file is rewritten each time with everything embedded so
    far, not a database or a diffing system.
    """

    documents = load_corpus(str(corpus_path))
    chunks = chunk_corpus(documents)

    provider = GeminiEmbeddingProvider()
    store = EmbeddingStore(index_path)

    embedded_chunks = []
    embedded_vectors = []

    texts = [chunk.content for chunk in chunks]

    for start in range(0, len(texts), batch_size):
        batch_chunks = chunks[start:start + batch_size]
        batch_texts = texts[start:start + batch_size]

        # A batch that ultimately fails (retries exhausted) raises
        # here, but every prior batch has already been written to
        # disk by the store.save() call below.
        batch_embeddings = provider.embed_documents(batch_texts)

        embedded_chunks.extend(batch_chunks)
        embedded_vectors.extend(batch_embeddings)

        store.save(
            embedded_chunks,
            embedded_vectors,
            model=provider.model,
        )

    return len(embedded_chunks)


if __name__ == "__main__":
    chunk_count = build_embedding_index()

    print(
        f"Embedding index created successfully: "
        f"{chunk_count} chunks indexed."
    )
