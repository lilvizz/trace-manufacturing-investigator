import csv
import io

import pytest

from app.config import CORPUS_PATH
from app.ingestion.chunker import chunk_document, chunk_corpus
from app.ingestion.loader import load_corpus


def test_document_is_split_into_chunks():
    document = {
        "document_id": "test_document",
        "filename": "test.txt",
        "relative_path": "test.txt",
        "content": "A" * 5000,
    }

    chunks = chunk_document(
        document,
        max_chars=1000,
        overlap=100,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.document_id == "test_document"
        assert chunk.filename == "test.txt"
        assert chunk.content


def test_entire_corpus_can_be_chunked():
    documents = load_corpus(str(CORPUS_PATH))
    chunks = chunk_corpus(documents)

    assert len(documents) == 40
    assert len(chunks) > 40


def test_invalid_chunk_configuration_is_rejected():
    """overlap must always be smaller than max_chars, or the sliding
    window in the text chunker can stall or move backwards."""

    document = {
        "document_id": "test_document",
        "filename": "test.txt",
        "relative_path": "test.txt",
        "extension": ".txt",
        "content": "some content",
    }

    with pytest.raises(ValueError):
        chunk_document(document, max_chars=100, overlap=100)

    with pytest.raises(ValueError):
        chunk_document(document, max_chars=100, overlap=150)


def _make_csv_document(document_id: str, rows: list[list[str]]) -> dict:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")

    for row in rows:
        writer.writerow(row)

    return {
        "document_id": document_id,
        "filename": f"{document_id}.csv",
        "relative_path": f"{document_id}.csv",
        "extension": ".csv",
        "content": buffer.getvalue(),
    }


def test_csv_header_appears_in_every_chunk():
    header = ["CalDate", "SensorID", "AsFound_Dev_C", "Result"]

    rows = [header] + [
        [f"2020-0{i}-01", f"RTD-{i}", "0.05", "PASS"]
        for i in range(1, 40)
    ]

    document = _make_csv_document("calibration_history", rows)

    # A small max_chars forces multiple chunks so we can verify the
    # header shows up in more than just the first one.
    chunks = chunk_document(document, max_chars=200, overlap=0)

    assert len(chunks) > 1

    for chunk in chunks:
        first_line = chunk.content.splitlines()[0]
        assert first_line == ",".join(header)


def test_csv_rows_are_not_cut_in_the_middle():
    header = ["WorkOrder", "Date", "Description", "Notes"]

    rows = [header] + [
        [f"WO-{i:04d}", "2020-01-01", "Routine inspection", "No issues found"]
        for i in range(50)
    ]

    document = _make_csv_document("maintenance_history", rows)

    chunks = chunk_document(document, max_chars=200, overlap=0)

    assert len(chunks) > 1

    reconstructed_rows = []

    for chunk in chunks:
        lines = chunk.content.splitlines()

        # First line of every chunk must be the header.
        assert lines[0] == ",".join(header)

        # Every remaining line must be a complete, well-formed row
        # (csv.reader will raise/produce a malformed row if a value
        # or line was truncated mid-field).
        for line in lines[1:]:
            parsed = next(csv.reader([line]))
            assert len(parsed) == len(header)
            reconstructed_rows.append(parsed)

    assert reconstructed_rows == rows[1:]


def test_csv_with_only_a_header_still_produces_one_chunk():
    document = _make_csv_document(
        "empty_log",
        [["Date", "Value"]],
    )

    chunks = chunk_document(document, max_chars=1800, overlap=200)

    assert len(chunks) == 1
    assert chunks[0].content == "Date,Value"


def test_real_corpus_csv_files_chunk_without_error():
    """Smoke test against the actual frozen corpus: every real .csv
    file in apex-corpus must chunk cleanly and every chunk must start
    with that file's header."""

    documents = load_corpus(str(CORPUS_PATH))
    csv_documents = [doc for doc in documents if doc["extension"] == ".csv"]

    assert csv_documents, "expected at least one .csv file in the corpus"

    for document in csv_documents:
        chunks = chunk_document(document)

        assert chunks, f"no chunks produced for {document['filename']}"

        expected_header = next(
            csv.reader(io.StringIO(document["content"].strip()))
        )
        expected_header_line = ",".join(expected_header)

        for chunk in chunks:
            first_line = chunk.content.splitlines()[0]
            assert first_line == expected_header_line, (
                f"missing/incorrect header in a chunk of "
                f"{document['filename']}"
            )
