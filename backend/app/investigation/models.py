from dataclasses import dataclass, field
from enum import Enum


class EvidenceStance(str, Enum):
    UNKNOWN = "unknown"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT_ONLY = "context-only"
    NOT_ESTABLISHED = "not_established"


class ClaimType(str, Enum):
    ORIGINAL_BASIS = "original_basis"
    CURRENT_CAPABILITY = "current_capability"
    ADMINISTRATIVE_ACTION = "administrative_action"
    CURRENT_TECHNICAL_JUSTIFICATION = "current_technical_justification"
    EVIDENCE_GAP = "evidence_gap"
    RECOMMENDATION = "recommendation"


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
    # Retrieval channels that found this chunk ("bm25", "vector",
    # "entity", or "carried-forward" for evidence reused from an earlier
    # turn of the same conversation). Optional; empty for older callers.
    retrieval_channels: list[str] = field(default_factory=list)


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

        for channel in item.retrieval_channels:
            if channel not in existing.retrieval_channels:
                existing.retrieval_channels.append(channel)

    def add_evidence_many(self, items: list[EvidenceItem]) -> None:
        for item in items:
            self.add_evidence(item)


class FindingType(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTION = "contradiction"
    EVIDENCE_GAP = "evidence-gap"
    CONTEXT = "context"


@dataclass
class EvidenceFinding:
    finding_type: FindingType
    statement: str
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceAnalysis:
    findings: list[EvidenceFinding] = field(default_factory=list)

    def add_finding(self, finding: EvidenceFinding) -> None:
        self.findings.append(finding)


@dataclass
class ReasoningClaim:
    statement: str
    stance: EvidenceStance
    evidence_chunk_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    claim_type: ClaimType | None = None
    # Deterministic verification steps that changed this claim (citations
    # removed, stance downgraded, ...), in plain words. Empty when the
    # verifier left the claim unchanged. Default keeps older construction
    # working.
    verification_notes: list[str] = field(default_factory=list)
    # Result of ClaimVerifier's deterministic identifier check, recorded so
    # the UI can say exactly what was checked and nothing more. It is a
    # LEXICAL check (identifiers appearing in cited text), never semantic
    # support. Values: "" (check not applicable / not run), "procedure_
    # referenced" (a recommendation's named QP / CQ-POL identifier appears
    # in a cited passage), "identifier_overlap" (another of the claim's
    # identifiers appears in a cited passage), "no_identifiers" (nothing
    # checkable in the claim), "failed" (none appears).
    citation_check: str = ""
    citation_check_anchors: list[str] = field(default_factory=list)
    citation_check_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class InvestigationReport:
    question: str
    claims: list[ReasoningClaim] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)

    # Question-specific answer. Kept after the existing fields so older
    # positional construction remains backward-compatible.
    direct_answer: str = ""
    direct_answer_evidence_chunk_ids: list[str] = field(default_factory=list)
