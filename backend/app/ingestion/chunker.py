from dataclasses import dataclass


@dataclass
class DocumentChunk:
    document_id: str
    filename: str
    relative_path: str
    chunk_id: str
    content: str


def chunk_document(
    document: dict,
    max_chars: int = 1800,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Split a document into overlapping text chunks.

    The first version deliberately uses a simple character-based
    strategy. We will improve this once retrieval is working.
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
            chunks.append(
                DocumentChunk(
                    document_id=document["document_id"],
                    filename=document["filename"],
                    relative_path=document["relative_path"],
                    chunk_id=f"{document['document_id']}_{chunk_number}",
                    content=chunk_text,
                )
            )

            chunk_number += 1

        if end >= len(content):
            break

        start = end - overlap

    return chunks


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