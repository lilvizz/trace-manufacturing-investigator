from pathlib import Path


SUPPORTED_EXTENSIONS = {".txt", ".csv", ".md"}


def load_corpus(corpus_path: str) -> list[dict]:
    """
    Load all supported documents from the frozen TRACE corpus.

    Returns a list of dictionaries containing:
    - document_id
    - filename
    - relative_path
    - extension
    - content
    """

    root = Path(corpus_path)

    if not root.exists():
        raise FileNotFoundError(f"Corpus not found: {root}")

    documents = []

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        content = file_path.read_text(encoding="utf-8")

        relative_path = file_path.relative_to(root)

        documents.append(
            {
                "document_id": file_path.stem,
                "filename": file_path.name,
                "relative_path": str(relative_path),
                "extension": file_path.suffix.lower(),
                "content": content,
            }
        )

    return documents