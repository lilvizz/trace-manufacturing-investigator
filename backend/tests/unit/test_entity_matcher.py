from dataclasses import dataclass

import pytest

from app.retrieval.entity_matcher import (
    EntityMatcher,
    extract_entities,
)


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
                "PX-400 qualification for Press-07 "
                "under ECO-2011-184."
            ),
        ),
        FakeChunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            filename="px200.txt",
            relative_path="px200.txt",
            content=(
                "PX-200 thermal investigation established "
                "the original process boundary."
            ),
        ),
        FakeChunk(
            chunk_id="chunk-3",
            document_id="doc-3",
            filename="generic.txt",
            relative_path="generic.txt",
            content="Generic manufacturing information.",
        ),
    ]


def test_extract_entities_preserves_identifiers():
    entities = extract_entities(
        "PX-400 Press-07 ECO-2011-184 at 42.3°C"
    )

    assert "px-400" in entities
    assert "press-07" in entities
    assert "eco-2011-184" in entities
    assert "42.3°c" in entities


def test_search_matches_shared_entities(chunks):
    matcher = EntityMatcher(chunks)

    results = matcher.search(
        "PX-400 ECO-2011-184",
        top_k=2,
    )

    assert results
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 2.0


def test_search_does_not_return_chunks_without_entity_overlap(chunks):
    matcher = EntityMatcher(chunks)

    results = matcher.search(
        "completely unrelated information",
        top_k=5,
    )

    assert results == []


def test_empty_query_raises(chunks):
    matcher = EntityMatcher(chunks)

    with pytest.raises(ValueError):
        matcher.search("")


def test_invalid_top_k_raises(chunks):
    matcher = EntityMatcher(chunks)

    with pytest.raises(ValueError):
        matcher.search("PX-400", top_k=0)