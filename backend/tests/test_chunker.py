from app.ingestion.chunker import chunk_document, chunk_corpus


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
    from pathlib import Path
    from app.ingestion.loader import load_corpus

    project_root = Path(__file__).resolve().parents[2]
    corpus_path = project_root / "data" / "apex-corpus"

    documents = load_corpus(str(corpus_path))
    chunks = chunk_corpus(documents)

    assert len(documents) == 40
    assert len(chunks) > 40