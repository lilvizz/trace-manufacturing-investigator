from app.investigation.models import (
    EvidenceAnalysis,
    EvidenceFinding,
    EvidenceStance,
    FindingType,
    Investigation,
)


class EvidenceGapAnalyzer:
    """
    Performs deterministic evidence-coverage analysis.

    This component deliberately does NOT infer contradictions from prose.
    Semantic support/contradiction judgments will be supplied by the
    reasoning layer later.
    """

    def analyze(self, investigation: Investigation) -> EvidenceAnalysis:
        analysis = EvidenceAnalysis()

        if not investigation.evidence:
            analysis.add_finding(
                EvidenceFinding(
                    finding_type=FindingType.EVIDENCE_GAP,
                    statement=(
                        "No evidence was retrieved for this investigation."
                    ),
                )
            )
            return analysis

        self._analyze_subquestion_coverage(
            investigation,
            analysis,
        )

        self._analyze_explicit_stances(
            investigation,
            analysis,
        )

        return analysis

    def _analyze_subquestion_coverage(
        self,
        investigation: Investigation,
        analysis: EvidenceAnalysis,
    ) -> None:
        covered = {
            subquestion
            for evidence in investigation.evidence
            for subquestion in evidence.retrieved_for
        }

        for subquestion in investigation.subquestions:
            if subquestion not in covered:
                analysis.add_finding(
                    EvidenceFinding(
                        finding_type=FindingType.EVIDENCE_GAP,
                        statement=(
                            "No retrieved evidence is associated with "
                            f"subquestion: {subquestion}"
                        ),
                    )
                )

    def _analyze_explicit_stances(
        self,
        investigation: Investigation,
        analysis: EvidenceAnalysis,
    ) -> None:
        for evidence in investigation.evidence:
            if evidence.stance == EvidenceStance.SUPPORTS:
                analysis.add_finding(
                    EvidenceFinding(
                        finding_type=FindingType.SUPPORTED,
                        statement=(
                            f"Evidence {evidence.chunk_id} is marked "
                            "as supporting its retrieval context."
                        ),
                        evidence_chunk_ids=[evidence.chunk_id],
                    )
                )

            elif evidence.stance == EvidenceStance.CONTRADICTS:
                analysis.add_finding(
                    EvidenceFinding(
                        finding_type=FindingType.CONTRADICTION,
                        statement=(
                            f"Evidence {evidence.chunk_id} is marked "
                            "as contradicting its retrieval context."
                        ),
                        evidence_chunk_ids=[evidence.chunk_id],
                    )
                )

            elif evidence.stance == EvidenceStance.CONTEXT_ONLY:
                analysis.add_finding(
                    EvidenceFinding(
                        finding_type=FindingType.CONTEXT,
                        statement=(
                            f"Evidence {evidence.chunk_id} provides "
                            "context but is not marked as supporting "
                            "or contradicting the investigation."
                        ),
                        evidence_chunk_ids=[evidence.chunk_id],
                    )
                )