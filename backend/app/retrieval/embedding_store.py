import json
from pathlib import Path

from app.ingestion.chunker import DocumentChunk


class EmbeddingStore:
    """Persist document chunks and their embeddings to disk."""

    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)

    def save(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

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

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index_path.write_text(
            json.dumps(records),
            encoding="utf-8",
        )

    def load(self) -> tuple[list[DocumentChunk], list[list[float]]]:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Embedding index not found: {self.index_path}"
            )

        records = json.loads(
            self.index_path.read_text(
                encoding="utf-8"
            )
        )

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

        return chunks, embeddings