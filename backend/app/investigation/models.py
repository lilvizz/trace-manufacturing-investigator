from dataclasses import dataclass, field
from enum import Enum


class EvidenceStance(str, Enum):
    """How evidence relates to the investigation."""

    UNKNOWN = "unknown"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT_ONLY = "context-only"


@dataclass
class EvidenceItem:
    """A piece of evidence gathered during an investigation."""

    chunk_id: str
    document_id: str
    filename: str
    relative_path: str
    content: str
    retrieval_score: float
    retrieved_for: str
    stance: EvidenceStance = EvidenceStance.UNKNOWN


@dataclass
class Investigation:
    """
    State accumulated during a TRACE investigation.

    An investigation is deliberately mutable: each retrieval stage
    can add evidence to the same investigation.
    """

    question: str
    subquestions: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)

    def add_evidence(self, item: EvidenceItem) -> None:
        """Add evidence unless the same chunk is already present."""

        if any(
            existing.chunk_id == item.chunk_id
            for existing in self.evidence
        ):
            return

        self.evidence.append(item)

    def add_evidence_many(
        self,
        items: list[EvidenceItem],
    ) -> None:
        """Add multiple evidence items while preventing duplicates."""

        for item in items:
            self.add_evidence(item)