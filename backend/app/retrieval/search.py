from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    score: float


class CorpusSearcher:
    """Simple lexical retrieval baseline for the TRACE corpus."""

    def __init__(self, chunks):
        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )

        self.matrix = self.vectorizer.fit_transform(
            chunk.content for chunk in chunks
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the most relevant chunks for a query."""

        query_vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        )[0]

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:
            chunk = self.chunks[index]

            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    relative_path=chunk.relative_path,
                    content=chunk.content,
                    score=float(scores[index]),
                )
            )

        return results