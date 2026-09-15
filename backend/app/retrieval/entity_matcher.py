import re
from dataclasses import dataclass


ENTITY_PATTERNS = (
    re.compile(r"\bPX-\d+\b", re.IGNORECASE),
    re.compile(r"\bPress[- ]?\d+\b", re.IGNORECASE),
    re.compile(r"\bECO-\d{4}-\d+\b", re.IGNORECASE),
    re.compile(r"\bQUAL-\d{4}-\d+\b", re.IGNORECASE),
    re.compile(r"\bENG-NOTE-\d{4}-\d+\b", re.IGNORECASE),
    re.compile(r"\bINV-\d{4}-\d+\b", re.IGNORECASE),
    re.compile(r"\bQM-\d{4}-\d+\b", re.IGNORECASE),
    re.compile(r"\bSPC-REV-\d{4}-\d+\b", re.IGNORECASE),
    re.compile(r"\bSOP-[A-Z0-9-]+\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*°?\s*[CF]\b", re.IGNORECASE),
)


def extract_entities(text: str) -> set[str]:
    entities = set()

    for pattern in ENTITY_PATTERNS:
        for match in pattern.findall(text):
            normalized = re.sub(r"\s+", "", match.lower())
            entities.add(normalized)

    return entities


@dataclass(frozen=True)
class EntitySearchResult:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    score: float


class EntityMatcher:
    """
    Exact technical-entity matching for manufacturing evidence.

    Entity matching is deliberately deterministic. It is not intended
    to replace semantic retrieval; it provides a complementary signal
    for identifiers such as PX-400, ECO numbers, qualification records,
    and temperature limits.
    """

    def __init__(self, chunks):
        self.chunks = list(chunks)

        if not self.chunks:
            raise ValueError("chunks cannot be empty")

        self._entities_by_chunk = {
            chunk.chunk_id: extract_entities(chunk.content)
            for chunk in self.chunks
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[EntitySearchResult]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_entities = extract_entities(query)

        if not query_entities:
            return []

        scored = []

        for chunk in self.chunks:
            chunk_entities = self._entities_by_chunk[chunk.chunk_id]
            overlap = query_entities.intersection(chunk_entities)

            if not overlap:
                continue

            scored.append(
                (
                    len(overlap),
                    chunk,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            EntitySearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                relative_path=chunk.relative_path,
                content=chunk.content,
                score=float(overlap_count),
            )
            for overlap_count, chunk in scored[:top_k]
        ]