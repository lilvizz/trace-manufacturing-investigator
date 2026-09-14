from datetime import date

from app.investigation.models import EvidenceItem
from app.investigation.timeline import (
    InvestigationTimeline,
    TimelineBuilder,
    TimelineEvent,
)


def evidence(chunk_id, content):
    return EvidenceItem(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        filename=f"doc-{chunk_id}.txt",
        relative_path=f"docs/doc-{chunk_id}.txt",
        content=content,
        retrieval_score=0.9,
        retrieved_for=["question"],
    )


def test_extracts_iso_date():
    builder = TimelineBuilder()

    timeline = builder.build(
        [
            evidence(
                "chunk-1",
                "Validation completed on 2014-06-18.",
            )
        ]
    )

    assert len(timeline.events) == 1
    assert timeline.events[0].date == date(2014, 6, 18)
    assert "Validation completed" in timeline.events[0].description


def test_extracts_numeric_date():
    builder = TimelineBuilder()

    timeline = builder.build(
        [
            evidence(
                "chunk-1",
                "Review held on 14/06/2001.",
            )
        ]
    )

    assert len(timeline.events) == 1
    assert timeline.events[0].date == date(2001, 6, 14)


def test_extracts_textual_date():
    builder = TimelineBuilder()

    timeline = builder.build(
        [
            evidence(
                "chunk-1",
                "Equipment replaced on 12-Aug-2011.",
            )
        ]
    )

    assert len(timeline.events) == 1
    assert timeline.events[0].date == date(2011, 8, 12)


def test_events_are_sorted_chronologically():
    builder = TimelineBuilder()

    timeline = builder.build(
        [
            evidence(
                "chunk-1",
                "Event occurred on 2014-06-18.",
            ),
            evidence(
                "chunk-2",
                "Event occurred on 2001-07-05.",
            ),
            evidence(
                "chunk-3",
                "Event occurred on 2011-09-20.",
            ),
        ]
    )

    assert [event.date for event in timeline.events] == [
        date(2001, 7, 5),
        date(2011, 9, 20),
        date(2014, 6, 18),
    ]


def test_duplicate_event_merges_evidence():
    timeline = InvestigationTimeline()

    timeline.add_event(
        TimelineEvent(
            date=date(2014, 6, 18),
            description="Validation completed.",
            evidence_chunk_ids=["chunk-1"],
        )
    )

    timeline.add_event(
        TimelineEvent(
            date=date(2014, 6, 18),
            description="Validation completed.",
            evidence_chunk_ids=["chunk-2"],
        )
    )

    assert len(timeline.events) == 1
    assert timeline.events[0].evidence_chunk_ids == [
        "chunk-1",
        "chunk-2",
    ]


def test_multiple_dates_in_same_document_are_preserved():
    builder = TimelineBuilder()

    timeline = builder.build(
        [
            evidence(
                "chunk-1",
                """
                Trial started on 05-Jul-2001.
                Review completed on 14-Jul-2001.
                """,
            )
        ]
    )

    assert len(timeline.events) == 2
    assert timeline.events[0].date == date(2001, 7, 5)
    assert timeline.events[1].date == date(2001, 7, 14)


def test_invalid_dates_are_ignored():
    builder = TimelineBuilder()

    timeline = builder.build(
        [
            evidence(
                "chunk-1",
                "Invalid date: 31/02/2001.",
            )
        ]
    )

    assert timeline.events == []


def test_undated_evidence_does_not_create_fake_event():
    builder = TimelineBuilder()

    timeline = builder.build(
        [
            evidence(
                "chunk-1",
                "The equipment was stable during testing.",
            )
        ]
    )

    assert timeline.events == []