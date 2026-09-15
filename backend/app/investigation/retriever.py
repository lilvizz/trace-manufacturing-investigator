from dataclasses import dataclass
from typing import Protocol

from app.investigation.models import EvidenceItem, Investigation
from app.investigation.planner import InvestigationPlan
from app.retrieval.diversifier import diversify_results


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    score: float


class Retriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        ...


class InvestigationRetriever:
    """
    Retrieves evidence for each investigation subquestion.

    Retrieval is performed independently for each subquestion so that
    different parts of a long-running investigation can surface
    different documents.
    """

    def __init__(
        self,
        retriever: Retriever,
        top_k_per_subquestion: int = 8,
        max_evidence: int = 30,
        max_per_document: int = 4,
        max_per_subquestion_document: int = 2,
    ):
        if top_k_per_subquestion <= 0:
            raise ValueError(
                "top_k_per_subquestion must be greater than zero"
            )

        if max_evidence <= 0:
            raise ValueError(
                "max_evidence must be greater than zero"
            )

        if max_per_document <= 0:
            raise ValueError(
                "max_per_document must be greater than zero"
            )

        if max_per_subquestion_document <= 0:
            raise ValueError(
                "max_per_subquestion_document must be greater than zero"
            )

        self.retriever = retriever
        self.top_k_per_subquestion = top_k_per_subquestion
        self.max_evidence = max_evidence
        self.max_per_document = max_per_document
        self.max_per_subquestion_document = (
            max_per_subquestion_document
        )

    def collect(
        self,
        investigation: Investigation,
        plan: InvestigationPlan,
    ) -> Investigation:

        for subquestion in plan.subquestions:
            results = self.retriever.search(
                subquestion,
                top_k=self.top_k_per_subquestion,
            )

            diversified_results = diversify_results(
                results,
                max_per_document=self.max_per_subquestion_document,
                top_k=self.top_k_per_subquestion,
            )

            evidence_items = [
                EvidenceItem(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    filename=result.filename,
                    relative_path=result.relative_path,
                    content=result.content,
                    retrieval_score=result.score,
                    retrieved_for=[subquestion],
                )
                for result in diversified_results
            ]

            investigation.add_evidence_many(evidence_items)

        self._limit_evidence(investigation)

        return investigation

    def _limit_evidence(
        self,
        investigation: Investigation,
    ) -> None:
        """
        Apply a global document cap before the final evidence cap.

        This prevents multiple subquestions from collectively flooding
        the investigation with chunks from the same document.
        """

        if len(investigation.evidence) <= self.max_evidence:
            return

        ranked = sorted(
            investigation.evidence,
            key=lambda item: item.retrieval_score,
            reverse=True,
        )

        selected = []
        document_counts = {}

        for item in ranked:
            count = document_counts.get(item.document_id, 0)

            if count >= self.max_per_document:
                continue

            selected.append(item)
            document_counts[item.document_id] = count + 1

            if len(selected) >= self.max_evidence:
                break

        investigation.evidence = selected