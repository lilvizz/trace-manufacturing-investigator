from dataclasses import dataclass

import pytest

from app.retrieval.bm25_search import BM25Searcher


@dataclass
class FakeChunk:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str


@pytest.fixture
def chunks():
    return [
        FakeChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="px400.txt",
            relative_path="px400.txt",
            content=(
                "PX-400 qualification demonstrated thermal stability "
                "through 44 degrees Celsius."
            ),
        ),
        FakeChunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            filename="press05.txt",
            relative_path="press05.txt",
            content=(
                "Press-05 experienced a hydraulic pressure excursion "
                "during production."
            ),
        ),
        FakeChunk(
            chunk_id="chunk-3",
            document_id="doc-3",
            filename="eco.txt",
            relative_path="eco.txt",
            content=(
                "ECO-2011-184 replaced the PX-200 press with PX-400."
            ),
        ),
    ]


def test_search_returns_relevant_result(chunks):
    searcher = BM25Searcher(chunks)

    results = searcher.search(
        "PX-400 thermal stability",
        top_k=2,
    )

    assert results
    assert results[0].chunk_id == "chunk-1"


def test_search_preserves_technical_identifiers(chunks):
    searcher = BM25Searcher(chunks)

    results = searcher.search(
        "ECO-2011-184 PX-400",
        top_k=1,
    )

    assert results[0].chunk_id == "chunk-3"


def test_search_respects_top_k(chunks):
    searcher = BM25Searcher(chunks)

    results = searcher.search(
        "PX",
        top_k=2,
    )

    assert len(results) <= 2


def test_empty_query_raises(chunks):
    searcher = BM25Searcher(chunks)

    with pytest.raises(ValueError):
        searcher.search("")


def test_invalid_top_k_raises(chunks):
    searcher = BM25Searcher(chunks)

    with pytest.raises(ValueError):
        searcher.search("PX-400", top_k=0)