from dataclasses import dataclass, field
from enum import Enum


class EvidenceStance(str, Enum):
    UNKNOWN = "unknown"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT_ONLY = "context-only"


@dataclass
class EvidenceItem:
    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    retrieval_score: float
    retrieved_for: list[str] = field(default_factory=list)
    stance: EvidenceStance = EvidenceStance.UNKNOWN


@dataclass
class Investigation:
    question: str
    subquestions: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)

    def add_evidence(self, item: EvidenceItem) -> None:
        existing = next(
            (
                evidence
                for evidence in self.evidence
                if evidence.chunk_id == item.chunk_id
            ),
            None,
        )

        if existing is None:
            self.evidence.append(item)
            return

        for subquestion in item.retrieved_for:
            if subquestion not in existing.retrieved_for:
                existing.retrieved_for.append(subquestion)

        existing.retrieval_score = max(
            existing.retrieval_score,
            item.retrieval_score,
        )

    def add_evidence_many(self, items: list[EvidenceItem]) -> None:
        for item in items:
            self.add_evidence(item)