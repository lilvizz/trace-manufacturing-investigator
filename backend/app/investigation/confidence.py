from dataclasses import replace

from app.investigation.models import (
    ClaimType,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)


class ConfidenceCalibrator:
    """
    Applies deterministic confidence ceilings to Gemini-generated claims.

    Gemini's confidence is treated as an input signal, not as ground truth.

    The calibrator lowers confidence when the evidence situation makes
    high confidence inappropriate.
    """

    # Hard ceilings for claim categories that TRACE must treat conservatively.
    _CLAIM_TYPE_CEILINGS = {
        ClaimType.CURRENT_TECHNICAL_JUSTIFICATION: 0.70,
        ClaimType.EVIDENCE_GAP: 0.85,
        ClaimType.RECOMMENDATION: 0.80,
    }

    def calibrate(
        self,
        report: InvestigationReport,
        investigation: Investigation,
    ) -> InvestigationReport:
        evidence_by_id = {
            evidence.chunk_id: evidence
            for evidence in investigation.evidence
        }

        calibrated_claims = [
            self._calibrate_claim(
                claim,
                evidence_by_id,
            )
            for claim in report.claims
        ]

        return replace(
            report,
            claims=calibrated_claims,
        )

    def _calibrate_claim(
        self,
        claim: ReasoningClaim,
        evidence_by_id: dict,
    ) -> ReasoningClaim:
        confidence = self._clamp(claim.confidence)

        # No evidence means no meaningful confidence.
        valid_evidence = [
            evidence_by_id[chunk_id]
            for chunk_id in claim.evidence_chunk_ids
            if chunk_id in evidence_by_id
        ]

        if not valid_evidence:
            return replace(
                claim,
                stance=EvidenceStance.UNKNOWN,
                confidence=0.0,
            )

        # Hard ceiling based on the semantic type of the claim.
        ceiling = self._CLAIM_TYPE_CEILINGS.get(
            claim.claim_type,
            1.0,
        )

        confidence = min(
            confidence,
            ceiling,
        )

        # NOT_ESTABLISHED is intentionally conservative.
        if claim.stance == EvidenceStance.NOT_ESTABLISHED:
            confidence = min(
                confidence,
                0.30,
            )

        # CONTRADICTS should not receive extreme confidence merely because
        # Gemini used a high number. Contradictions require explicit evidence.
        if claim.stance == EvidenceStance.CONTRADICTS:
            confidence = min(
                confidence,
                0.85,
            )

        # CONTEXT_ONLY is not proof of the main claim.
        if claim.stance == EvidenceStance.CONTEXT_ONLY:
            confidence = min(
                confidence,
                0.65,
            )

        return replace(
            claim,
            confidence=confidence,
        )

    @staticmethod
    def _clamp(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0

        return max(
            0.0,
            min(1.0, value),
        )