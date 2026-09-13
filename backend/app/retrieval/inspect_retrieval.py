import json

from app.config import INDEX_PATH, PROJECT_ROOT
from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
from app.retrieval.embedding_store import EmbeddingStore
from app.retrieval.vector_search import VectorSearcher


BENCHMARK_QUESTIONS_PATH = (
    PROJECT_ROOT / "private_evaluation" / "benchmark_questions.json"
)


def load_benchmark_questions() -> dict:
    """
    Load the private TRACE benchmark questions.

    These are deliberately kept out of tracked source/dev code (see
    .gitignore: private_evaluation/) instead of being hardcoded here,
    so the benchmark question set isn't exposed by the public repo.
    This script is a manual, local-only retrieval-inspection tool.
    """

    if not BENCHMARK_QUESTIONS_PATH.exists():
        raise FileNotFoundError(
            "Benchmark questions not found at "
            f"{BENCHMARK_QUESTIONS_PATH}. This file is intentionally "
            "excluded from the public repository. Restore it from the "
            "private evaluation materials to run this script."
        )

    return json.loads(
        BENCHMARK_QUESTIONS_PATH.read_text(encoding="utf-8")
    )


def build_searcher():
    store = EmbeddingStore(INDEX_PATH)

    chunks, embeddings = store.load()

    provider = GeminiEmbeddingProvider()

    return VectorSearcher(
        chunks,
        embeddings,
        provider,
    )


def main():
    searcher = build_searcher()
    questions = load_benchmark_questions()

    for question_id, question in questions.items():
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
