from pathlib import Path

from app.ingestion.chunker import chunk_corpus
from app.ingestion.loader import load_corpus
from app.investigation.build import build_investigator
from app.retrieval.search import CorpusSearcher


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

    # Local TF-IDF retrieval baseline.
    # This avoids spending Gemini embedding quota while we test
    # the investigation workflow itself.
    corpus_searcher = CorpusSearcher(chunks)

    investigator = build_investigator(
        retriever=corpus_searcher,
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
        print(f"Retrieved for: {len(evidence.retrieved_for)} subquestion(s)")

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
                f"Evidence: {', '.join(finding.evidence_chunk_ids)}"
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