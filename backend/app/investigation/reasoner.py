import json
from typing import Protocol

from app.investigation.models import (
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)


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

    def reason(self, investigation: Investigation) -> InvestigationReport:
        prompt = self._build_prompt(investigation)

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

    def _build_prompt(self, investigation: Investigation) -> str:
        evidence_text = "\n\n".join(
            (
                f"EVIDENCE CHUNK: {item.chunk_id}\n"
                f"DOCUMENT: {item.filename}\n"
                f"CONTENT:\n{item.content}"
            )
            for item in investigation.evidence
        )

        subquestions = "\n".join(
            f"- {question}"
            for question in investigation.subquestions
        )

        return f"""
You are the evidence reasoning engine for TRACE,
a manufacturing evidence investigation system.

USER QUESTION:
{investigation.question}

INVESTIGATION SUBQUESTIONS:
{subquestions}

RETRIEVED EVIDENCE:
{evidence_text}

Rules:

1. Use ONLY the supplied evidence.
2. Do not invent facts, dates, causes, or decisions.
3. Every factual claim MUST cite one or more supplied evidence chunk IDs.
4. Use "unknown" when the evidence does not establish something.
5. Distinguish:
   - evidence that supports a claim
   - evidence that contradicts a claim
   - information that is only contextual
   - genuine evidence gaps
6. Do not treat administrative persistence as technical justification.
7. Do not assume that absence of evidence proves that something never happened.
8. Be explicit about uncertainty.

Return ONLY valid JSON:

{{
  "claims": [
    {{
      "statement": "factual statement",
      "stance": "supports|contradicts|unknown|context-only",
      "evidence_chunk_ids": ["chunk-id"],
      "confidence": 0.0
    }}
  ],
  "unresolved_questions": [
    "question that the evidence cannot establish"
  ]
}}
""".strip()

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
                for chunk_id in raw_claim.get("evidence_chunk_ids", [])
                if chunk_id in evidence_ids
            ]

            stance = EvidenceStance(
                raw_claim.get(
                    "stance",
                    EvidenceStance.UNKNOWN.value,
                )
            )

            confidence = float(
                raw_claim.get("confidence", 0.0)
            )

            confidence = max(0.0, min(1.0, confidence))

            claims.append(
                ReasoningClaim(
                    statement=str(
                        raw_claim.get("statement", "")
                    ),
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