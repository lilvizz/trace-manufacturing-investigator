import json
from typing import Protocol

from app.investigation.models import (
    ClaimType,
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

    The model is explicitly constrained to:
    - cite retrieved chunk IDs
    - distinguish historical technical evidence from current justification
    - distinguish administrative persistence from technical validation
    - identify evidence gaps explicitly
    - avoid unsupported engineering conclusions
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

        (
            direct_answer,
            direct_answer_evidence_ids,
            claims,
            unresolved_questions,
        ) = self._parse_response(
            raw_response,
            investigation,
        )

        return InvestigationReport(
            question=investigation.question,
            direct_answer=direct_answer,
            direct_answer_evidence_chunk_ids=direct_answer_evidence_ids,
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

2. Do not invent facts, dates, causes, decisions, calculations,
   approvals, validations, or missing records.

3. Every factual claim MUST cite one or more supplied evidence chunk IDs.

4. If evidence is insufficient, explicitly represent the claim as
   "not_established" or "unknown" rather than guessing.

5. Distinguish:
   - supports
   - contradicts
   - context-only
   - unknown
   - not_established

6. Do NOT treat administrative persistence as technical justification.

7. The fact that a value remained unchanged does NOT prove it was
   technically reaffirmed.

8. Do not infer technical reaffirmation from mere persistence.

   Repeated appearance in:
   - SOPs
   - SPC configuration
   - QMS records
   - operating data
   - current configuration records

   is NOT sufficient by itself to establish technical revalidation
   or technical reaffirmation.
   
   - Do not infer reaffirmation from mere persistence.

9. Only say:
   - reaffirmed
   - validated
   - confirmed
   - re-approved
   - technically re-justified
   - technically re-established

   when the evidence explicitly documents the relevant action or
   conclusion.

10. Distinguish between:

    - original technical basis
    - current equipment/process capability
    - technically justified
    - technically reviewed
    - technically revalidated
    - administratively retained
    - carried forward unchanged
    - observed in operation
    - evidence gap
    - not established
    - unknown

11. Process or quality stability data without an explicit technical
    conclusion does NOT establish technical justification of a
    production control limit.

12. Qualification or requalification of equipment does NOT automatically
    mean that an existing production limit was revalidated.

13. Demonstrated stability above an existing limit does NOT automatically
    establish a new approved production limit.

14. Absence of evidence does not prove that something never happened.

    Use wording such as:
    "not established in the available evidence"

    when the available evidence does not establish the claim.

15. Never strengthen the wording of source evidence.

16. When evidence conflicts, explicitly identify the contradiction
    and explain whether it may instead reflect:
    - different equipment
    - different product
    - different time period
    - different process scope
    - different document purpose

17. If a historical technical reason existed but the equipment/process
    later changed, investigate whether the original reason was
    technically re-established for the changed equipment.

18. For any claim about CURRENT technical justification of an existing
    parameter, threshold, specification, control limit, or operating rule:

    Use claim_type:
        "current_technical_justification"

    and stance:
        "not_established"

    UNLESS the cited evidence explicitly establishes that the exact
    current parameter/limit was technically reviewed, tested,
    revalidated, re-approved, or otherwise technically re-established
    for the current equipment/process.

    The following are NOT sufficient by themselves:
    - broader equipment capability
    - stable operation above the limit
    - unchanged SOP/SPC references
    - continued use of the old value
    - administrative retention
    - absence of a later rejection
    - a generic equipment qualification

19. Tailor the reasoning to the USER QUESTION.

    Do NOT assume the user is asking about the flagship 42.3°C case,
    Press-07, PX-200, PX-400, or any other specific historical case
    unless those entities actually appear in the user's question or
    supplied evidence.

    Identify the specific subject, decision, parameter, equipment,
    process, event, or document the user asked about and answer that
    question directly.

    The investigation subquestions and retrieved evidence are supporting
    context, not a replacement for the user's actual question.

20. If the investigation concerns a historical parameter, limit, or
    decision that later changed equipment/process/context, distinguish
    as applicable:
    - original basis
    - what changed
    - later capability or observations
    - administrative persistence or decisions
    - current technical justification
    - evidence gaps

21. If an administrative/commercial proposal was rejected or deferred,
    do NOT characterize that as a technical rejection unless the
    evidence explicitly states a technical reason.

21. If a proposed change was rejected for commercial, scheduling,
    customer-approval, PPAP, or administrative reasons, represent that
    as an administrative_action claim.

22. Do not recommend changing a production control limit merely because
    later evidence suggests the original failure mechanism may no longer
    apply.

    If the evidence suggests a possible change may deserve investigation,
    identify the required engineering validation/change-control process
    instead of asserting a new safe limit.

23. A recommendation is NOT a technical fact.

    Recommendations must be represented with:
        claim_type = "recommendation"

    and must clearly distinguish what the evidence establishes from
    what should be investigated or validated next.

24. Evidence gaps are first-class findings.

    When the available evidence does not establish an important part
    of the user's question, explicitly produce an evidence_gap claim
    and/or a relevant claim with:
        stance = "not_established"

25. Do not use "supports" merely because a cited document is related
    to the topic.

    The cited evidence must actually support the specific claim being made.

26. Preserve uncertainty.

    If evidence is incomplete, conflicting, indirect, informal, or
    limited in scope, reduce the strength of the claim rather than
    converting uncertainty into certainty.

27. Claim type definitions:

    "original_basis"
        The historical technical reason or evidence that originally
        established a parameter/limit.

    "current_capability"
        What later testing, qualification, operation, or evidence shows
        the current equipment/process can do. This does NOT automatically
        establish an approved production limit.

    "administrative_action"
        Approval, rejection, deferral, scheduling, customer, PPAP,
        procedural, or other administrative decisions.

    "current_technical_justification"
        Whether the exact current production parameter/limit is
        technically justified for the current equipment/process.

    "evidence_gap"
        A material piece of evidence needed to establish an important
        conclusion is not established in the available evidence.

    "recommendation"
        A proposed next action, validation, review, or change-control
        step. Recommendations must not be presented as established facts.

28. For current_technical_justification claims, do not infer a positive
    technical justification from the absence of contradictory evidence.

29. A claim may cite several documents, but multiple citations do not
    automatically make a claim stronger. Evidence must independently
    support the specific claim.

30. If the evidence establishes that a historical technical justification
    existed but does not establish that it was re-established for the
    current equipment, state BOTH facts.

    Do not collapse them into a single "supported" conclusion.

Every claim must use wording no stronger than the evidence supports.

Return ONLY valid JSON.

The JSON schema is:

{{
  "direct_answer": {{
    "text": "A concise answer to the USER QUESTION using only the supplied evidence.",
    "evidence_chunk_ids": ["..."]
  }},
  "claims": [
    {{
      "statement": "...",
      "claim_type": "original_basis|current_capability|administrative_action|current_technical_justification|evidence_gap|recommendation",
      "stance": "supports|contradicts|unknown|context-only|not_established",
      "evidence_chunk_ids": ["..."],
      "confidence": 0.0
    }}
  ],
  "unresolved_questions": ["..."]
}}

Additional output requirements:

- direct_answer.text MUST directly answer the USER QUESTION.
- direct_answer MUST NOT simply repeat the overall evidence-gap conclusion
  unless that conclusion is actually the answer to the user's question.
- direct_answer.evidence_chunk_ids MUST contain only supplied evidence chunk IDs.
- The direct answer should cite the evidence needed to support its factual content.
- If the available evidence does not establish the answer, say so explicitly
  rather than substituting an answer to a different question.
- claim_type MUST be one of the allowed values above.
- stance MUST be one of the allowed values above.
- evidence_chunk_ids MUST contain only supplied evidence chunk IDs.
- Do not invent chunk IDs.
- Every factual claim should have at least one evidence_chunk_id.
- If no valid evidence supports a factual claim, use:
      stance = "not_established"
      evidence_chunk_ids = []
      confidence = 0.0
- For current_technical_justification claims, use
  stance = "not_established" unless explicit technical
  re-justification evidence exists.
- Confidence must reflect the strength of the evidence, not simply
  confidence in your own generated answer.
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
    ) -> tuple[str, list[str], list[ReasoningClaim], list[str]]:

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

        # ---------------------------------------------------------
        # Direct answer
        # ---------------------------------------------------------
        # This is deliberately separate from the global claim set.
        # It forces the model to answer the user's actual question
        # instead of allowing the flagship evidence-gap conclusion to
        # become the answer to every query.
        raw_direct_answer = payload.get("direct_answer", {})

        if not isinstance(raw_direct_answer, dict):
            raw_direct_answer = {}

        direct_answer_text = str(
            raw_direct_answer.get("text", "")
        ).strip()

        raw_direct_answer_ids = raw_direct_answer.get(
            "evidence_chunk_ids",
            [],
        )

        if not isinstance(raw_direct_answer_ids, list):
            raw_direct_answer_ids = []

        direct_answer_evidence_ids = [
            chunk_id
            for chunk_id in raw_direct_answer_ids
            if isinstance(chunk_id, str)
            and chunk_id in evidence_ids
        ]

        # Never let an uncited model answer become a trusted answer.
        # A direct answer with no valid evidence is replaced by an explicit
        # evidence-gap response. This is intentionally conservative.
        if not direct_answer_text or not direct_answer_evidence_ids:
            direct_answer_text = (
                "The available evidence does not establish a direct "
                "answer to the investigation question."
            )
            direct_answer_evidence_ids = []

        claims = []

        for raw_claim in payload.get("claims", []):
            if not isinstance(raw_claim, dict):
                continue

            raw_citations = raw_claim.get(
                "evidence_chunk_ids",
                [],
            )

            if not isinstance(raw_citations, list):
                raw_citations = []

            cited_ids = [
                chunk_id
                for chunk_id in raw_citations
                if isinstance(chunk_id, str)
                and chunk_id in evidence_ids
            ]

            raw_stance = raw_claim.get(
                "stance",
                EvidenceStance.UNKNOWN.value,
            )

            try:
                stance = EvidenceStance(raw_stance)
            except ValueError:
                stance = EvidenceStance.UNKNOWN

            raw_claim_type = raw_claim.get("claim_type")

            claim_type = None

            if raw_claim_type is not None:
                try:
                    claim_type = ClaimType(raw_claim_type)
                except ValueError:
                    claim_type = None

            try:
                confidence = float(
                    raw_claim.get("confidence", 0.0)
                )
            except (TypeError, ValueError):
                confidence = 0.0

            confidence = max(
                0.0,
                min(1.0, confidence),
            )

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
                        claim_type=claim_type,
                    )
                )
                continue

            claims.append(
                ReasoningClaim(
                    statement=statement,
                    stance=stance,
                    evidence_chunk_ids=cited_ids,
                    confidence=confidence,
                    claim_type=claim_type,
                )
            )

        raw_unresolved = payload.get(
            "unresolved_questions",
            [],
        )

        if not isinstance(raw_unresolved, list):
            raw_unresolved = []

        unresolved = [
            str(question).strip()
            for question in raw_unresolved
            if str(question).strip()
        ]

        return (
            direct_answer_text,
            direct_answer_evidence_ids,
            claims,
            unresolved,
        )