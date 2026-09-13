from app.config import CORPUS_PATH
from app.ingestion.loader import load_corpus
from app.ingestion.chunker import chunk_corpus
from app.retrieval.fake_embeddings import TestEmbeddingProvider
from app.retrieval.vector_search import VectorSearcher


def build_searcher():
    documents = load_corpus(str(CORPUS_PATH))
    chunks = chunk_corpus(documents)

    provider = TestEmbeddingProvider()

    embeddings = provider.embed_documents(
        [chunk.content for chunk in chunks]
    )

    return VectorSearcher(
        chunks,
        embeddings,
        provider,
    )


def test_vector_search_returns_results():
    searcher = build_searcher()

    results = searcher.search(
        "What equipment maintenance records are available for the presses?",
        top_k=5,
    )

    assert len(results) == 5

    for result in results:
        assert result.content
        assert result.score >= 0


def test_vector_search_is_ranked():
    searcher = build_searcher()

    results = searcher.search(
        "calibration history and sensor deviation results",
        top_k=5,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)
