from pathlib import Path

from app.ingestion.chunker import chunk_corpus
from app.ingestion.loader import load_corpus
from app.investigation.build import build_investigator
from app.retrieval.bm25_search import BM25Searcher
from app.retrieval.entity_matcher import EntityMatcher
from app.retrieval.hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_PATH = PROJECT_ROOT / "data" / "apex-corpus"


def main():
    print("=" * 70)
    print("TRACE — REAL GEMINI INVESTIGATION")
    print("=" * 70)

    print("\nLoading corpus...")

    documents = load_corpus(str(CORPUS_PATH))
    chunks = chunk_corpus(documents)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created:   {len(chunks)}")

    # Hybrid local retrieval.
    #
    # BM25 provides strong lexical matching for manufacturing
    # identifiers, terminology, and exact values.
    # Entity matching provides an additional deterministic
    # retrieval signal for important technical identifiers.
    #
    # Vector retrieval will be added once the embedding index
    # is regenerated against the current chunking configuration.
    bm25_searcher = BM25Searcher(chunks)
    entity_matcher = EntityMatcher(chunks)

    hybrid_retriever = HybridRetriever(
        bm25_searcher=bm25_searcher,
        vector_searcher=None,
        entity_matcher=entity_matcher,
    )

    investigator = build_investigator(
        retriever=hybrid_retriever,
    )

    question = "Why is this limit 42.3°C?"

    print("\nQuestion:")
    print(question)

    print("\nRunning TRACE investigation...")

    result = investigator.run(question)

    print("\n" + "=" * 70)
    print("INVESTIGATION PLAN")
    print("=" * 70)

    for number, subquestion in enumerate(
        result.plan.subquestions,
        start=1,
    ):
        print(f"{number}. {subquestion}")

    print("\n" + "=" * 70)
    print("RETRIEVED EVIDENCE")
    print("=" * 70)

    for evidence in result.investigation.evidence:
        print(
            f"\n[{evidence.chunk_id}] "
            f"score={evidence.retrieval_score:.4f}"
        )
        print(f"File: {evidence.filename}")
        print(
            f"Retrieved for: "
            f"{len(evidence.retrieved_for)} subquestion(s)"
        )

    print("\n" + "=" * 70)
    print("TIMELINE")
    print("=" * 70)

    for event in result.timeline.events:
        print(
            f"{event.date}: "
            f"{event.description} "
            f"[{', '.join(event.evidence_chunk_ids)}]"
        )

    print("\n" + "=" * 70)
    print("EVIDENCE ANALYSIS")
    print("=" * 70)

    for finding in result.evidence_analysis.findings:
        print(f"\n[{finding.finding_type.value}]")
        print(finding.statement)

        if finding.evidence_chunk_ids:
            print(
                f"Evidence: "
                f"{', '.join(finding.evidence_chunk_ids)}"
            )

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    for claim in result.report.claims:
        print(f"\n[{claim.stance.value}]")
        print(claim.statement)
        print(
            f"Evidence: "
            f"{', '.join(claim.evidence_chunk_ids)}"
        )
        print(f"Confidence: {claim.confidence:.2f}")

    if result.report.unresolved_questions:
        print("\nUnresolved questions:")

        for unresolved in result.report.unresolved_questions:
            print(f"- {unresolved}")


if __name__ == "__main__":
    main()