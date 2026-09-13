from app.retrieval.vector_search import VectorSearchResult


def diversify_results(
    results: list[VectorSearchResult],
    max_per_document: int = 1,
    top_k: int = 10,
) -> list[VectorSearchResult]:
    """
    Keep the strongest result from each document before filling
    the remaining slots with additional chunks.

    This prevents one document from dominating the evidence set.
    """

    selected = []
    document_counts = {}

    for result in results:
        count = document_counts.get(result.document_id, 0)

        if count >= max_per_document:
            continue

        selected.append(result)
        document_counts[result.document_id] = count + 1

        if len(selected) >= top_k:
            break

    return selected