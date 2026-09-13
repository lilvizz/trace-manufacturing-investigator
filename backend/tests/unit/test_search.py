from app.config import CORPUS_PATH
from app.ingestion.loader import load_corpus
from app.ingestion.chunker import chunk_corpus
from app.retrieval.search import CorpusSearcher


def build_searcher():
    documents = load_corpus(str(CORPUS_PATH))
    chunks = chunk_corpus(documents)

    return CorpusSearcher(chunks)


def test_search_returns_results():
    searcher = build_searcher()

    results = searcher.search(
        "What equipment maintenance records are available for the presses?",
        top_k=5,
    )

    assert len(results) == 5

    for result in results:
        assert result.content
        assert result.score >= 0


def test_search_results_are_ranked():
    searcher = build_searcher()

    results = searcher.search(
        "calibration history and sensor deviation results",
        top_k=5,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)
