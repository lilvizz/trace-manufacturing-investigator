"""
Bounded conversation context for TRACE follow-up questions.

Design rules (enforced here and in Investigator / InvestigationReasoner):

* A follow-up always runs the full pipeline: fresh planning, fresh
  retrieval, fresh timeline/gap analysis, Gemini reasoning, deterministic
  claim verification and confidence calibration.
* Earlier turns are used only to RESOLVE REFERENCES ("that study", "the
  replacement", "why isn't it enough?"). They are passed to the reasoner
  in a section explicitly labelled as not-evidence.
* Earlier AI answers never become evidence. The reasoner can only cite
  chunk IDs present in the current turn's evidence set; that filter
  already exists in InvestigationReasoner._parse_response and is
  re-checked by ClaimVerifier.
* Earlier evidence may be re-used only as the original corpus passages
  (same chunk ID, path and text), marked "carried-forward", and only the
  passages the previous turn's verified claims or answer actually cited.
* Context is bounded (MAX_TURNS, character budgets) and keyed by an
  explicit conversation_id; a new investigation starts with none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from app.investigation.models import EvidenceItem, EvidenceStance

MAX_TURNS = 3
MAX_CLAIMS_PER_TURN = 5
MAX_ANSWER_CHARS = 500
MAX_STATEMENT_CHARS = 240
MAX_CARRIED_EVIDENCE = 6
MAX_ANCHORS = 6

CARRIED_FORWARD_CHANNEL = "carried-forward"

# Subject anchors a user names explicitly: equipment / document IDs
# (Press-07, PX-400, RTD-7714-2, QUAL-2014-009) and temperatures.
_ID_TOKEN = re.compile(r"\b[A-Z][A-Za-z]{1,12}-\d[\w\-]*\b")
_TEMP_TOKEN = re.compile(r"\b\d{2,3}(?:\.\d+)?\s?(?:°\s?C|deg\s?C|degC)\b", re.IGNORECASE)

# Words that point back at something said earlier.
_REFERENTIAL = re.compile(
    r"\b(it|its|it's|that|this|those|these|they|them|their|"
    r"the same|the study|the report|the replacement|the qualification|"
    r"the requalification|the limit|the change|the document|the record|"
    r"the proposal|above|earlier|previous|previously|instead)\b",
    re.IGNORECASE,
)


def subject_anchors(text: str) -> list[str]:
    """Explicit subject tokens in a piece of user text, in order."""
    found: list[str] = []
    for match in list(_ID_TOKEN.finditer(text or "")) + list(_TEMP_TOKEN.finditer(text or "")):
        token = match.group(0).strip()
        normalised = re.sub(r"\s+", " ", token)
        if normalised.lower() not in {f.lower() for f in found}:
            found.append(normalised)
    return found


def is_referential(question: str) -> bool:
    return bool(_REFERENTIAL.search(question or ""))


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass(frozen=True)
class PriorClaim:
    statement: str
    stance: str
    claim_type: str | None
    evidence_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class PriorTurn:
    question: str
    direct_answer: str
    claims: tuple[PriorClaim, ...] = ()
    cited_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FollowupResolution:
    question: str
    retrieval_question: str
    anchors: tuple[str, ...] = ()
    used_context: bool = False
    clarification: str = ""

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarification)


CLARIFICATION_NO_CONTEXT = (
    "This question refers to something earlier (for example \"it\" or \"that "
    "study\"), but there is no earlier turn in this investigation to resolve it "
    "against. Which equipment, parameter, or document do you mean? For example: "
    "\"Why is the 2014 Press-07 requalification not enough to justify the "
    "42.3°C limit?\""
)


def resolve_question(question: str, context: "ConversationContext | None" = None) -> FollowupResolution:
    """Decide how a question should be investigated.

    * Names its own subject           -> investigated as asked.
    * Follow-up without its own subject -> anchored to subjects from
      earlier questions in the SAME conversation (and the previous
      question), so planning and retrieval target the right records.
    * Refers back with no conversation -> ask for clarification instead
      of guessing a connection.
    """
    question = " ".join(str(question or "").split())
    own = subject_anchors(question)

    if context is not None and context.turns:
        if own:
            return FollowupResolution(question, question, tuple(own), used_context=True)
        anchors = context.anchors()
        previous = context.turns[-1].question
        hint = "; ".join(anchors) if anchors else ""
        retrieval = (
            f"{question} (follow-up regarding {hint}; previous question: {_clip(previous, 160)})"
            if hint
            else f"{question} (follow-up to: {_clip(previous, 200)})"
        )
        return FollowupResolution(question, retrieval, tuple(anchors), used_context=True)

    if not own and is_referential(question):
        return FollowupResolution(question, question, (), used_context=False,
                                  clarification=CLARIFICATION_NO_CONTEXT)

    return FollowupResolution(question, question, tuple(own), used_context=False)


@dataclass(frozen=True)
class ConversationContext:
    conversation_id: str
    turns: tuple[PriorTurn, ...] = ()
    carried_evidence: tuple[EvidenceItem, ...] = field(default=(), compare=False)

    # ------------------------------------------------------------------
    @classmethod
    def from_results(cls, conversation_id: str, results) -> "ConversationContext":
        """Build bounded context from earlier InvestigationResults of the
        same conversation (oldest first). Only claims that survived
        verification (a stance other than unknown AND at least one valid
        citation) are carried, and only as statements to re-check."""
        results = [r for r in list(results or []) if r is not None][-MAX_TURNS:]
        turns: list[PriorTurn] = []
        for result in results:
            report = getattr(result, "report", None)
            claims = []
            for claim in list(getattr(report, "claims", []) or []):
                stance = getattr(claim, "stance", EvidenceStance.UNKNOWN)
                ids = tuple(getattr(claim, "evidence_chunk_ids", []) or ())
                if stance == EvidenceStance.UNKNOWN or not ids:
                    continue
                ctype = getattr(claim, "claim_type", None)
                statement = _clip(claim.statement, MAX_STATEMENT_CHARS)
                # Never pass a positively worded statement that the
                # evidence did NOT establish without saying so in words.
                if stance == EvidenceStance.NOT_ESTABLISHED:
                    statement = f"NOT ESTABLISHED by the evidence: {statement}"
                elif stance == EvidenceStance.CONTRADICTS:
                    statement = f"CONTRADICTED by the evidence: {statement}"
                claims.append(
                    PriorClaim(
                        statement=statement,
                        stance=str(getattr(stance, "value", stance)),
                        claim_type=str(getattr(ctype, "value", ctype)) if ctype else None,
                        evidence_chunk_ids=ids,
                    )
                )
            cited = list(getattr(report, "direct_answer_evidence_chunk_ids", []) or [])
            for claim in claims:
                cited.extend(claim.evidence_chunk_ids)
            turns.append(
                PriorTurn(
                    question=_clip(getattr(getattr(result, "investigation", None), "question", ""), 400),
                    direct_answer=_clip(getattr(report, "direct_answer", ""), MAX_ANSWER_CHARS),
                    claims=tuple(claims[:MAX_CLAIMS_PER_TURN]),
                    cited_chunk_ids=tuple(dict.fromkeys(cited)),
                )
            )

        carried: list[EvidenceItem] = []
        if results:
            last = results[-1]
            evidence = {
                item.chunk_id: item
                for item in list(getattr(getattr(last, "investigation", None), "evidence", []) or [])
            }
            note = f"Carried forward: cited in the previous turn ({_clip(turns[-1].question, 90)})"
            for chunk_id in turns[-1].cited_chunk_ids:
                item = evidence.get(chunk_id)
                if item is None:
                    continue
                carried.append(
                    replace(
                        item,
                        retrieved_for=[note],
                        retrieval_channels=[CARRIED_FORWARD_CHANNEL],
                    )
                )
                if len(carried) >= MAX_CARRIED_EVIDENCE:
                    break

        return cls(conversation_id=str(conversation_id), turns=tuple(turns), carried_evidence=tuple(carried))

    # ------------------------------------------------------------------
    def anchors(self) -> list[str]:
        """Subject anchors from earlier USER questions, most recent first.
        Taken from what the user typed, never from AI answers."""
        found: list[str] = []
        for turn in reversed(self.turns):
            for token in subject_anchors(turn.question):
                if token.lower() not in {f.lower() for f in found}:
                    found.append(token)
        return found[:MAX_ANCHORS]

    def resolve(self, question: str) -> FollowupResolution:
        return resolve_question(question, self)

    def to_prompt(self) -> str:
        """Bounded, clearly-labelled text block for the reasoner."""
        lines = [f"conversation_id: {self.conversation_id}"]
        anchors = self.anchors()
        if anchors:
            lines.append(f"Subjects named in earlier user questions: {', '.join(anchors)}")
        for index, turn in enumerate(self.turns, start=1):
            lines.append(f"Turn {index}")
            lines.append(f"  User question: {turn.question}")
            if turn.direct_answer:
                lines.append(
                    "  Earlier TRACE answer (AI output from a previous turn; NOT evidence): "
                    f"{turn.direct_answer}"
                )
            if turn.claims:
                lines.append(
                    "  Earlier verified claims (re-establish from the CURRENT evidence before relying on them):"
                )
                for claim in turn.claims:
                    tag = f"{claim.stance}" + (f" · {claim.claim_type}" if claim.claim_type else "")
                    lines.append(
                        f"    - [{tag}] {claim.statement} (cited then: {', '.join(claim.evidence_chunk_ids)})"
                    )
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "turns": len(self.turns),
            "anchors": self.anchors(),
            "carried_evidence": [item.chunk_id for item in self.carried_evidence],
        }
