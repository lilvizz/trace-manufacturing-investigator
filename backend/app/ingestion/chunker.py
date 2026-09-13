import csv
import io
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    document_id: str
    filename: str
    relative_path: str
    chunk_id: str
    content: str


def _make_chunk(document: dict, chunk_number: int, content: str) -> DocumentChunk:
    return DocumentChunk(
        document_id=document["document_id"],
        filename=document["filename"],
        relative_path=document["relative_path"],
        chunk_id=f"{document['document_id']}_{chunk_number}",
        content=content,
    )


def _render_csv_rows(rows: list[list[str]]) -> str:
    """Render one or more CSV rows back into normalized CSV text."""

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")

    for row in rows:
        writer.writerow(row)

    return buffer.getvalue().strip("\n")


def _chunk_text_document(
    document: dict,
    max_chars: int,
    overlap: int,
) -> list[DocumentChunk]:
    """
    Split free-text (.txt / .md) content into overlapping character-based
    chunks. This is the original TRACE chunking strategy, unchanged.
    """

    content = document["content"].strip()

    if not content:
        return []

    chunks = []
    start = 0
    chunk_number = 0

    while start < len(content):
        end = min(start + max_chars, len(content))

        chunk_text = content[start:end].strip()

        if chunk_text:
            chunks.append(_make_chunk(document, chunk_number, chunk_text))
            chunk_number += 1

        if end >= len(content):
            break

        start = end - overlap

    return chunks


def _chunk_csv_document(
    document: dict,
    max_chars: int,
) -> list[DocumentChunk]:
    """
    Split .csv content into chunks that always contain the header plus
    one or more *complete* data rows. Rows are never cut mid-value, and
    every chunk is independently interpretable because it always carries
    its column header.
    """

    content = document["content"].strip()

    if not content:
        return []

    rows = list(csv.reader(io.StringIO(content)))

    if not rows:
        return []

    header, data_rows = rows[0], rows[1:]
    header_text = _render_csv_rows([header])

    if not data_rows:
        # Header-only CSV: still emit one chunk so the file isn't dropped.
        return [_make_chunk(document, 0, header_text)]

    chunks = []
    chunk_number = 0
    current_rows: list[list[str]] = []
    current_length = len(header_text)

    def flush() -> None:
        nonlocal chunk_number, current_rows, current_length

        if not current_rows:
            return

        chunk_text = header_text + "\n" + _render_csv_rows(current_rows)
        chunks.append(_make_chunk(document, chunk_number, chunk_text))

        chunk_number += 1
        current_rows = []
        current_length = len(header_text)

    for row in data_rows:
        row_text = _render_csv_rows([row])
        # +1 accounts for the newline joining this row to the chunk.
        row_length = len(row_text) + 1

        if current_rows and current_length + row_length > max_chars:
            flush()

        current_rows.append(row)
        current_length += row_length

    flush()

    return chunks


def chunk_document(
    document: dict,
    max_chars: int = 1800,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Split a document into chunks.

    .txt and .md documents use a simple overlapping character-based
    strategy. .csv documents are parsed with the csv module and split
    on row boundaries so that a chunk never separates a data row from
    its header and never cuts a row in half.
    """

    if overlap >= max_chars:
        raise ValueError(
            "overlap must be smaller than max_chars "
            f"(got max_chars={max_chars}, overlap={overlap})."
        )

    extension = document.get("extension", "").lower()

    if extension == ".csv":
        return _chunk_csv_document(document, max_chars=max_chars)

    return _chunk_text_document(document, max_chars=max_chars, overlap=overlap)


def chunk_corpus(
    documents: list[dict],
    max_chars: int = 1800,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Chunk every document in the corpus."""

    all_chunks = []

    for document in documents:
        all_chunks.extend(
            chunk_document(
                document,
                max_chars=max_chars,
                overlap=overlap,
            )
        )

    return all_chunks
