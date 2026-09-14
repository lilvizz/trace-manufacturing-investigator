import json
from typing import Protocol

from app.investigation.models import (
    EvidenceAnalysis,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)
from app.investigation.timeline import InvestigationTimeline


class ReasoningClient(Protocol):
    def generate(self, prompt: str) -> str:
        ...


class InvestigationReasoner:
    """
    Converts retrieved evidence into evidence-backed claims.

    The model is explicitly constrained to cite retrieved chunk IDs.
    """

    def __init__(self, client: ReasoningClient):
        self.client = client

    def reason(
        self,
        investigation: Investigation,
        evidence_analysis: EvidenceAnalysis | None = None,
        timeline: InvestigationTimeline | None = None,
    ) -> InvestigationReport:
        prompt = self._build_prompt(
            investigation,
            evidence_analysis=evidence_analysis,
            timeline=timeline,
        )

        raw_response = self.client.generate(prompt)

        claims, unresolved_questions = self._parse_response(
            raw_response,
            investigation,
        )

        return InvestigationReport(
            question=investigation.question,
            claims=claims,
            unresolved_questions=unresolved_questions,
        )

    def _build_prompt(
        self,
        investigation: Investigation,
        evidence_analysis: EvidenceAnalysis | None = None,
        timeline: InvestigationTimeline | None = None,
    ) -> str:
        structured_analysis = self._format_evidence_analysis(
            evidence_analysis
        )
        timeline_text = self._format_timeline(timeline)

        prompt = f"""
You are the final evidence reasoning engine for TRACE,
a manufacturing evidence investigation system.

Your job is to answer the user's investigation question using
ONLY the supplied evidence and structured investigation context.

USER QUESTION:
{investigation.question}

INVESTIGATION SUBQUESTIONS:
{self._format_subquestions(investigation)}

STRUCTURED EVIDENCE ANALYSIS:
{structured_analysis}

INVESTIGATION TIMELINE:
{timeline_text}

RETRIEVED EVIDENCE:
{self._format_evidence(investigation)}

IMPORTANT REASONING RULES:

1. Use only the supplied evidence.
2. Do not invent facts, dates, causes, decisions, or missing records.
3. Every factual claim MUST cite one or more supplied evidence chunk IDs.
4. If evidence is insufficient, use "unknown" rather than guessing.
5. Distinguish supports, contradicts, context-only, and unknown.
6. Do NOT treat administrative persistence as technical justification.
7. The fact that a value remained unchanged does NOT prove it was technically reaffirmed.
8. Do not infer reaffirmation from mere persistence.
   Do not infer reaffirmation from repeated appearance in SOPs, SPC
   configuration, QMS records, or operating data alone.
9. Only say "reaffirmed", "validated", "confirmed", or "re-approved"
   when the evidence explicitly documents that action.
10. Distinguish between:
    - technically justified
    - technically reviewed
    - administratively retained
    - carried forward unchanged
    - observed in operation
    - unknown
11. Process or quality stability data without an explicit technical
    conclusion does NOT establish technical justification.
12. Absence of evidence does not prove that something never happened.
    Say "not established in the available evidence" when appropriate.
13. Never strengthen the wording of source evidence.
14. When evidence conflicts, explicitly identify the contradiction.
15. If a historical technical reason existed but the equipment/process
    later changed, investigate whether the original reason still applies.
16. For the 42.3°C question specifically, distinguish:
    a) why the value was originally created,
    b) why it remained administratively in place,
    c) whether its technical justification was re-established after
       the equipment change.
17. Do not recommend changing a production control limit merely because
    later evidence suggests the original failure mechanism may no longer
    apply. Identify what further engineering validation is required.

Every claim must use wording no stronger than the evidence supports.
When multiple pieces of evidence differ in strength, preserve the
uncertainty rather than choosing the stronger interpretation.

Return ONLY valid JSON:

{{
  "claims": [
    {{
      "statement": "...",
      "stance": "supports|contradicts|unknown|context-only",
      "evidence_chunk_ids": ["..."],
      "confidence": 0.0
    }}
  ],
  "unresolved_questions": ["..."]
}}
"""
        return prompt.strip()

    def _format_subquestions(
        self,
        investigation: Investigation,
    ) -> str:
        if not investigation.subquestions:
            return "None."

        return "\n".join(
            f"- {question}"
            for question in investigation.subquestions
        )

    def _format_evidence_analysis(
        self,
        evidence_analysis: EvidenceAnalysis | None,
    ) -> str:
        if evidence_analysis is None or not evidence_analysis.findings:
            return "No structured evidence findings were generated."

        lines = []

        for finding in evidence_analysis.findings:
            evidence_ids = ", ".join(
                finding.evidence_chunk_ids
            )

            if evidence_ids:
                lines.append(
                    f"- [{finding.finding_type.value}] "
                    f"{finding.statement} "
                    f"(evidence: {evidence_ids})"
                )
            else:
                lines.append(
                    f"- [{finding.finding_type.value}] "
                    f"{finding.statement}"
                )

        return "\n".join(lines)

    def _format_timeline(
        self,
        timeline: InvestigationTimeline | None,
    ) -> str:
        if timeline is None or not timeline.events:
            return "No structured timeline events were extracted."

        lines = []

        for event in timeline.events:
            evidence_ids = ", ".join(
                event.evidence_chunk_ids
            )

            if evidence_ids:
                lines.append(
                    f"- {event.date}: {event.description} "
                    f"(evidence: {evidence_ids})"
                )
            else:
                lines.append(
                    f"- {event.date}: {event.description}"
                )

        return "\n".join(lines)

    def _format_evidence(
        self,
        investigation: Investigation,
    ) -> str:
        if not investigation.evidence:
            return "No evidence was retrieved."

        return "\n\n".join(
            (
                f"EVIDENCE CHUNK: {item.chunk_id}\n"
                f"DOCUMENT: {item.filename}\n"
                f"CONTENT:\n{item.content}"
            )
            for item in investigation.evidence
        )

    def _parse_response(
        self,
        raw_response: str,
        investigation: Investigation,
    ) -> tuple[list[ReasoningClaim], list[str]]:

        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Reasoning client returned invalid JSON."
            ) from exc

        evidence_ids = {
            evidence.chunk_id
            for evidence in investigation.evidence
        }

        claims = []

        for raw_claim in payload.get("claims", []):
            cited_ids = [
                chunk_id
                for chunk_id in raw_claim.get(
                    "evidence_chunk_ids",
                    [],
                )
                if chunk_id in evidence_ids
            ]

            raw_stance = raw_claim.get(
                "stance",
                EvidenceStance.UNKNOWN.value,
            )

            try:
                stance = EvidenceStance(raw_stance)
            except ValueError:
                stance = EvidenceStance.UNKNOWN

            confidence = float(
                raw_claim.get("confidence", 0.0)
            )

            confidence = max(0.0, min(1.0, confidence))

            statement = str(
                raw_claim.get("statement", "")
            ).strip()

            if not statement:
                continue

            if not cited_ids:
                claims.append(
                    ReasoningClaim(
                        statement=statement,
                        stance=EvidenceStance.UNKNOWN,
                        evidence_chunk_ids=[],
                        confidence=0.0,
                    )
                )
                continue

            claims.append(
                ReasoningClaim(
                    statement=statement,
                    stance=stance,
                    evidence_chunk_ids=cited_ids,
                    confidence=confidence,
                )
            )

        unresolved = [
            str(question)
            for question in payload.get(
                "unresolved_questions",
                [],
            )
        ]

        return claims, unresolved