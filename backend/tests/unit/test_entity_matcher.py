from dataclasses import dataclass

import pytest

from app.retrieval.entity_matcher import (
    EntityMatcher,
    extract_entities,
)
from app.investigation.analyzer import (
    _equipment_entities,
    _has_equipment_overlap,
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

    # Hyphens/whitespace between an identifier's letters and digits are
    # not semantically meaningful (see test_entity_matcher_normalization
    # for the full equivalence-class regression coverage), so the
    # canonical forms below drop them rather than preserving them
    # verbatim.
    assert "px400" in entities
    assert "press7" in entities
    assert "eco2011184" in entities
    assert "42.3c" in entities


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


# ---------------------------------------------------------------------------
# Fix 1 regression tests (hostile audit MUST-FIX #1): equipment identifiers
# and temperature values written in different-but-equivalent notations must
# resolve to the same canonical entity. These use the corpus's own real
# notation, not only synthetic clean examples.
# ---------------------------------------------------------------------------


def test_equipment_identifier_variants_normalize_to_same_entity():
    """'Press 7', 'Press-07' and 'Press07' name the same asset."""
    press_space = extract_entities("Press 7 cure zone boundary")
    press_hyphen_padded = extract_entities("Press-07 maintenance log")
    press_no_separator = extract_entities("Press07 asset master")

    assert press_space == press_hyphen_padded == press_no_separator
    assert "press7" in press_space


def test_temperature_variants_normalize_to_same_entity():
    """'42.3 deg C', '42.3°C', '42.3 C' and '42.3C' are the same value."""
    deg_word = extract_entities(
        "Cure zone upper control limit (UCL): 42.3 deg C"
    )
    deg_symbol = extract_entities("the limit is 42.3°C in all SOPs")
    deg_space = extract_entities("not to exceed 42.3 C under any condition")
    deg_bare = extract_entities("recorded as 42.3C on the chart")

    assert deg_word == deg_symbol == deg_space == deg_bare
    assert "42.3c" in deg_word


def test_engineering_note_2001_no_longer_flagged_as_equipment_mismatch():
    """
    Regression for the real bug found in the hostile audit: the 2001
    engineering note that establishes the original 42.3 deg C basis for
    Press 7 was incorrectly flagged as a possible equipment-mismatch
    distractor when the question referred to the same press as
    "Press-07", because the old normalization treated "Press 7" and
    "Press-07" as different entities.
    """
    question_entities = _equipment_entities(
        "Why is Press-07 limited to 42.3°C, what changed since that "
        "limit was established, and is its current technical basis "
        "still supported?"
    )

    # Real excerpt from
    # data/apex-corpus/B_historical_engineering/10_Engineering_Note_2001.txt
    engineering_note_2001_excerpt = (
        "Subject: Press 7 (PX-200) cure zone -- revised upper operating "
        "boundary\n\n"
        "this note confirms the revised upper operating boundary for "
        "the Press 7 cure zone.\n\n"
        "Cure zone upper control limit (UCL):  42.3 deg C"
    )
    evidence_entities = _equipment_entities(engineering_note_2001_excerpt)

    assert _has_equipment_overlap(question_entities, evidence_entities)


def test_known_distractor_press_still_flagged_as_mismatch():
    """
    The fix must not remove the equipment-mismatch signal entirely: a
    genuine distractor about a different press must still be flagged
    as not overlapping with the question's equipment.
    """
    question_entities = _equipment_entities(
        "Why is Press-07 limited to 42.3°C?"
    )

    # Based on data/apex-corpus/G_distractors/36_Press05_Thermal_Investigation_2016.txt
    # (a genuine distractor about a different press/model).
    distractor_excerpt = (
        "Press-05 (PX-300) thermal investigation, 2016. A standalone "
        "study of an unrelated cure zone."
    )
    evidence_entities = _equipment_entities(distractor_excerpt)

    assert not _has_equipment_overlap(question_entities, evidence_entities)


def test_flagship_question_retrieves_original_basis_via_entity_match(chunks):
    """
    Retrieval-behavior check (non-brittle): once normalization treats
    'Press-07' and 'Press 7' as the same entity, a chunk written with the
    corpus's 'Press 7' notation is retrievable by a question written
    with 'Press-07' notation through the entity matcher alone. This does
    not assert any specific rank/score beyond "it is returned at all".
    """
    original_basis_chunk = FakeChunk(
        chunk_id="chunk-2001",
        document_id="doc-2001",
        filename="10_Engineering_Note_2001.txt",
        relative_path="B_historical_engineering/10_Engineering_Note_2001.txt",
        content=(
            "Subject: Press 7 (PX-200) cure zone -- revised upper "
            "operating boundary. Cure zone upper control limit (UCL): "
            "42.3 deg C."
        ),
    )
    matcher = EntityMatcher([original_basis_chunk])

    results = matcher.search(
        "Why is Press-07 limited to 42.3°C?",
        top_k=5,
    )

    assert any(r.chunk_id == "chunk-2001" for r in results)