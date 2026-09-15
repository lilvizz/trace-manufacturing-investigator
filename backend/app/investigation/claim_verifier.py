from dataclasses import replace

from app.investigation.analyzer import _REJUSTIFICATION_MARKERS
from app.investigation.models import (
    ClaimType,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)


class ClaimVerifier:
    """
    Deterministic post-hoc verification of Gemini-generated claims.

    The verifier does not call an LLM.

    Its job is to make sure that:
    1. citations actually exist,
    2. claims that require explicit technical re-justification
       cannot be marked as technically established without such evidence,
    3. unverifiable claims are downgraded rather than silently removed.

    The verifier is intentionally conservative.
    """

    def verify(
        self,
        report: InvestigationReport,
        investigation: Investigation,
    ) -> InvestigationReport:
        evidence_by_id = {
            evidence.chunk_id: evidence
            for evidence in investigation.evidence
        }

        verified_claims: list[ReasoningClaim] = []

        for claim in report.claims:
            verified_claim = self._verify_claim(
                claim,
                evidence_by_id,
            )

            verified_claims.append(verified_claim)

        return replace(
            report,
            claims=verified_claims,
        )

    def _verify_claim(
        self,
        claim: ReasoningClaim,
        evidence_by_id: dict,
    ) -> ReasoningClaim:
        """
        Apply all deterministic verification rules to one claim.
        """

        claim = self._enforce_citation_existence(
            claim,
            evidence_by_id,
        )

        claim = self._enforce_rejustification_invariant(
            claim,
            evidence_by_id,
        )

        return claim

    def _enforce_citation_existence(
        self,
        claim: ReasoningClaim,
        evidence_by_id: dict,
    ) -> ReasoningClaim:
        """
        Remove nonexistent evidence IDs.

        If no valid evidence remains, the claim becomes UNKNOWN with
        zero confidence.

        We deliberately keep the claim instead of silently deleting it.
        """

        valid_ids = [
            chunk_id
            for chunk_id in claim.evidence_chunk_ids
            if chunk_id in evidence_by_id
        ]

        if len(valid_ids) != len(claim.evidence_chunk_ids):
            claim = replace(
                claim,
                evidence_chunk_ids=valid_ids,
            )

        if not valid_ids:
            return replace(
                claim,
                stance=EvidenceStance.UNKNOWN,
                confidence=0.0,
            )

        return claim

    def _enforce_rejustification_invariant(
        self,
        claim: ReasoningClaim,
        evidence_by_id: dict,
    ) -> ReasoningClaim:
        """
        Enforce the most important TRACE safety rule:

            current_technical_justification
                         +
            no explicit re-justification evidence
                         ↓
                  NOT_ESTABLISHED

        A qualification, stability result, unchanged SOP, or current
        configuration reference is NOT automatically technical
        re-justification.
        """

        if claim.claim_type != ClaimType.CURRENT_TECHNICAL_JUSTIFICATION:
            return claim

        rejustification_evidence = []

        for chunk_id in claim.evidence_chunk_ids:
            evidence = evidence_by_id.get(chunk_id)

            if evidence is None:
                continue

            if _REJUSTIFICATION_MARKERS.search(
                evidence.content
            ):
                rejustification_evidence.append(
                    evidence
                )

        if rejustification_evidence:
            return claim

        return replace(
            claim,
            stance=EvidenceStance.NOT_ESTABLISHED,
            confidence=min(
                claim.confidence,
                0.30,
            ),
        )