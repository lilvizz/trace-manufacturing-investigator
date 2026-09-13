from pathlib import Path

from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
from app.retrieval.embedding_store import EmbeddingStore
from app.retrieval.vector_search import VectorSearcher


QUESTIONS = {
    "Q1": "Why is the Press-07 temperature limit 42.3 degrees Celsius?",
    "Q2": (
        "Does the original reason for the 42.3 degree limit still apply "
        "after the PX-200 was replaced by the PX-400?"
    ),
    "Q3": "Why was the 42.3 degree limit retained after the 2014 requalification?",
    "Q4": "What changed over time that affected the justification for the 42.3 degree limit?",
    "Q5": (
        "What evidence and approvals would be needed before changing "
        "the 42.3 degree limit?"
    ),
}


def build_searcher():
    project_root = Path(__file__).resolve().parents[3]

    index_path = (
        project_root
        / "data"
        / "indexes"
        / "apex_embeddings.json"
    )

    store = EmbeddingStore(index_path)

    chunks, embeddings = store.load()

    provider = GeminiEmbeddingProvider()

    return VectorSearcher(
        chunks,
        embeddings,
        provider,
    )


def main():
    searcher = build_searcher()

    for question_id, question in QUESTIONS.items():
        print()
        print("=" * 80)
        print(f"{question_id}: {question}")
        print("=" * 80)

        results = searcher.search(
            question,
            top_k=10,
        )

        for rank, result in enumerate(results, start=1):
            print()
            print(
                f"{rank}. {result.filename} "
                f"(score={result.score:.4f})"
            )
            print(f"   {result.relative_path}")


if __name__ == "__main__":
    main()