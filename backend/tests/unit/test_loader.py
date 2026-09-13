from app.config import CORPUS_PATH
from app.ingestion.loader import load_corpus


def test_corpus_loads_expected_documents():
    documents = load_corpus(str(CORPUS_PATH))

    assert len(documents) == 40


def test_documents_have_required_fields():
    documents = load_corpus(str(CORPUS_PATH))

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
