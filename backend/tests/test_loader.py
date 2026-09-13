from pathlib import Path

from app.ingestion.loader import load_corpus


def test_corpus_loads_expected_documents():
    project_root = Path(__file__).resolve().parents[2]
    corpus_path = project_root / "data" / "apex-corpus"

    documents = load_corpus(str(corpus_path))

    assert len(documents) == 40


def test_documents_have_required_fields():
    project_root = Path(__file__).resolve().parents[2]
    corpus_path = project_root / "data" / "apex-corpus"

    documents = load_corpus(str(corpus_path))

    required_fields = {
        "document_id",
        "filename",
        "relative_path",
        "extension",
        "content",
    }

    for document in documents:
        assert required_fields.issubset(document.keys())
        assert document["content"].strip()