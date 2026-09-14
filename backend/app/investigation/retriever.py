from dataclasses import dataclass
from typing import Protocol

from app.investigation.models import EvidenceItem, Investigation
from app.investigation.planner import InvestigationPlan


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
    Collects evidence for every subquestion in an investigation plan.

    Each planner subquestion gets its own retrieval pass. Results are
    then merged into one investigation-level evidence set with duplicate
    chunks collapsed while preserving retrieval provenance.
    """

    def __init__(
        self,
        retriever: Retriever,
        top_k_per_subquestion: int = 5,
        max_evidence: int = 30,
    ):
        if top_k_per_subquestion < 1:
            raise ValueError("top_k_per_subquestion must be at least 1.")

        if max_evidence < 1:
            raise ValueError("max_evidence must be at least 1.")

        self.retriever = retriever
        self.top_k_per_subquestion = top_k_per_subquestion
        self.max_evidence = max_evidence

    def collect(
        self,
        investigation: Investigation,
        plan: InvestigationPlan,
    ) -> Investigation:
        """
        Execute retrieval for every subquestion and add the resulting
        evidence to the investigation.

        The original investigation object is updated and returned.
        """

        for subquestion in plan.subquestions:
            results = self.retriever.search(
                subquestion,
                top_k=self.top_k_per_subquestion,
            )

            for result in results:
                investigation.add_evidence(
                    EvidenceItem(
                        chunk_id=result.chunk_id,
                        document_id=result.document_id,
                        filename=result.filename,
                        relative_path=result.relative_path,
                        content=result.content,
                        retrieval_score=result.score,
                        retrieved_for=[subquestion],
                    )
                )

        self._limit_evidence(investigation)

        return investigation

    def _limit_evidence(self, investigation: Investigation) -> None:
        """
        Keep the strongest evidence globally after all retrieval passes.

        This matters because limiting each query independently can cause
        the final investigation to contain too many low-value duplicates.
        """

        investigation.evidence.sort(
            key=lambda evidence: evidence.retrieval_score,
            reverse=True,
        )

        del investigation.evidence[self.max_evidence :]