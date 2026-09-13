from pathlib import Path

from app.ingestion.loader import load_corpus
from app.ingestion.chunker import chunk_corpus
from app.retrieval.search import CorpusSearcher


def build_searcher():
    project_root = Path(__file__).resolve().parents[2]
    corpus_path = project_root / "data" / "apex-corpus"

    documents = load_corpus(str(corpus_path))
    chunks = chunk_corpus(documents)

    return CorpusSearcher(chunks)


def test_search_returns_results():
    searcher = build_searcher()

    results = searcher.search(
        "Why is the Press-07 temperature limit 42.3 degrees Celsius?",
        top_k=5,
    )

    assert len(results) == 5

    for result in results:
        assert result.content
        assert result.score >= 0


def test_search_results_are_ranked():
    searcher = build_searcher()

    results = searcher.search(
        "PX-200 thermal instability 42.3 temperature",
        top_k=5,
    )

    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)