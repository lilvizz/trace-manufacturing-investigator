import hashlib
import json
from pathlib import Path

from app.ingestion.chunker import DocumentChunk

# Bump this if the on-disk index format changes in a way that makes
# older index files unreadable/unsafe to load.
SCHEMA_VERSION = 1


class EmbeddingIndexError(Exception):
    """Raised when a persisted embedding index is missing, corrupted,
    or incompatible with the code trying to load it."""


class EmbeddingStore:
    """Persist document chunks and their embeddings to disk."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)

    @staticmethod
    def _compute_corpus_hash(chunks: list[DocumentChunk]) -> str:
        """
        Hash of chunk ids + content, independent of embedding values.
        Used to detect when the underlying corpus/chunking changed
        without embeddings being regenerated, or when an index file
        has been hand-edited/corrupted.
        """

        hasher = hashlib.sha256()

        for chunk in sorted(chunks, key=lambda c: c.chunk_id):
            hasher.update(chunk.chunk_id.encode("utf-8"))
            hasher.update(b"\x00")
            hasher.update(chunk.content.encode("utf-8"))
            hasher.update(b"\x00")

        return hasher.hexdigest()

    def save(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
        model: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        dimensions = len(embeddings[0]) if embeddings else 0

        records = []

        for chunk, embedding in zip(chunks, embeddings):
            records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "relative_path": chunk.relative_path,
                    "content": chunk.content,
                    "embedding": embedding,
                }
            )

        payload = {
            "schema_version": SCHEMA_VERSION,
            "model": model,
            "dimensions": dimensions,
            "corpus_hash": self._compute_corpus_hash(chunks),
            "chunks": records,
        }

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def load(
        self,
        expected_model: str | None = None,
    ) -> tuple[list[DocumentChunk], list[list[float]]]:
        """
        Load chunks and embeddings, validating that the index is
        internally consistent (schema version, declared dimensions,
        and a content hash of its own chunks) and, optionally, that it
        was built with a specific embedding model.
        """

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Embedding index not found: {self.index_path}"
            )

        payload = json.loads(
            self.index_path.read_text(
                encoding="utf-8"
            )
        )

        schema_version = payload.get("schema_version")

        if schema_version != SCHEMA_VERSION:
            raise EmbeddingIndexError(
                f"Embedding index schema_version={schema_version!r} is "
                f"incompatible with the expected schema_version="
                f"{SCHEMA_VERSION!r}. Rebuild the index."
            )

        stored_model = payload.get("model")

        if expected_model is not None and stored_model != expected_model:
            raise EmbeddingIndexError(
                f"Embedding index was built with model {stored_model!r}, "
                f"but {expected_model!r} was expected. Rebuild the index "
                "or use a matching embedding provider."
            )

        records = payload.get("chunks", [])

        chunks = []
        embeddings = []

        for record in records:
            chunks.append(
                DocumentChunk(
                    document_id=record["document_id"],
                    filename=record["filename"],
                    relative_path=record["relative_path"],
                    chunk_id=record["chunk_id"],
                    content=record["content"],
                )
            )

            embeddings.append(record["embedding"])

        declared_dimensions = payload.get("dimensions")

        if embeddings and declared_dimensions is not None:
            actual_dimensions = len(embeddings[0])

            if actual_dimensions != declared_dimensions:
                raise EmbeddingIndexError(
                    "Embedding index dimensions do not match stored "
                    f"metadata: expected {declared_dimensions}, found "
                    f"{actual_dimensions}."
                )

        stored_hash = payload.get("corpus_hash")

        if stored_hash is not None:
            current_hash = self._compute_corpus_hash(chunks)

            if current_hash != stored_hash:
                raise EmbeddingIndexError(
                    "Embedding index content hash does not match its "
                    "own stored chunks; the index file may be "
                    "corrupted or hand-edited. Rebuild the index."
                )

        return chunks, embeddings
