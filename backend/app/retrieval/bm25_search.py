import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class BM25SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    score: float


def _tokenize(text: str) -> list[str]:
    """
    Tokenize technical manufacturing text while preserving
    identifiers such as PX-400 and ECO-2011-184.
    """
    return re.findall(
        r"[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*",
        text.lower(),
    )


class BM25Searcher:
    """
    BM25 lexical retrieval over investigation chunks.
    """

    def __init__(self, chunks):
        self.chunks = list(chunks)

        if not self.chunks:
            raise ValueError("chunks cannot be empty")

        tokenized_corpus = [
            _tokenize(chunk.content)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[BM25SearchResult]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            BM25SearchResult(
                chunk_id=self.chunks[index].chunk_id,
                document_id=self.chunks[index].document_id,
                filename=self.chunks[index].filename,
                relative_path=self.chunks[index].relative_path,
                content=self.chunks[index].content,
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]