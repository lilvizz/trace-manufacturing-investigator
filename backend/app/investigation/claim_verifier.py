import re
from dataclasses import replace

from app.investigation.analyzer import _contains_explicit_rejustification
from app.investigation.models import (
    ClaimType,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)


# Fix 2 (hostile-audit MUST-FIX): the rejustification safety gate below
# was previously keyed ONLY on `claim.claim_type == CURRENT_TECHNICAL_
# JUSTIFICATION`, a field the LLM supplies. If Gemini omits claim_type,
# supplies an invalid value, or mislabels a de-facto current-technical-
# justification statement as something else (e.g. CURRENT_CAPABILITY),
# the gate could be silently bypassed.
#
# This marker set is a deterministic, text-only fallback: it looks at
# what the claim's OWN statement actually asserts, independent of
# whatever claim_type label it was given. It is deliberately narrow --
# it must fire only on language that explicitly asserts the CURRENT
# technical basis/limit is justified/supported/valid right now, not on
# every mention of qualification, capability, SOP, "unchanged", or
# review. Those remain distinct claim categories and must not collapse
# into this one.
_CURRENT_JUSTIFICATION_STATEMENT_MARKERS = re.compile(
    r"\b("
    r"re-?justif\w*|"
    r"current(?:ly)?\s+technical(?:ly)?\s+justif\w*|"
    r"technical(?:ly)?\s+justif\w*\s+(?:currently|today|now|at\s+present)|"
    r"current\s+(?:technical\s+)?basis\s+(?:is|remains|continues\s+to\s+be)\s+"
    r"(?:supported|valid|justified|established)|"
    r"(?:is|remains|continues\s+to\s+be)\s+technically\s+"
    r"(?:justified|supported|valid)|"
    r"current\s+limit\s+(?:is|remains)\s+(?:technically\s+)?"
    r"(?:justified|supported|valid|appropriate)"
    r")\b",
    re.IGNORECASE,
)

_STATEMENT_NEGATION_WORDS = {
    "no", "not", "never", "without", "none", "lacks", "lacking", "unsupported",
}

# Second detection path: engineering prose frequently separates "what was
# re-justified" (the current limit/boundary/basis) from "what action
# re-justified it" (re-approved / revalidated / re-established), often
# with the verb coming BEFORE the subject ("Engineering technically
# re-approved the current operating limit..."). The single-phrase
# alternation above misses that word order. This path instead requires
# BOTH, independent of order:
#   (a) the statement names the CURRENT limit/boundary/basis as its
#       subject, and
#   (b) the statement asserts an explicit re-justification ACTION
#       (re-approved, revalidated, re-established, technically
#       justified, ...) somewhere in the same statement.
# Neither half alone is enough -- "PX-400 was qualified" and "the limit
# was revalidated" (no "current") both fail (a); "Quality
# administratively approved continued use" fails both. This mirrors
# _contains_explicit_rejustification's marker vocabulary in analyzer.py,
# applied to the claim's own statement text instead of evidence content.
_CURRENT_LIMIT_SUBJECT = re.compile(
    r"\bcurrent\s+"
    r"(?:\d+(?:\.\d+)?\s*(?:°|deg(?:rees)?\.?)?\s*[cf]\s+)?"
    r"(?:operating\s+)?(?:technical\s+)?"
    r"(?:limit|boundary|basis)\b",
    re.IGNORECASE,
)

_JUSTIFICATION_ACTION = re.compile(
    r"\b("
    r"re-?justif\w*|"
    r"re-?approv\w*|"
    r"re-?validat\w*|"
    r"re-?establish\w*|"
    r"technically\s+justif\w*"
    r")\b",
    re.IGNORECASE,
)


# A contrastive clause ("but"/"however"/"although"/"though"/"yet")
# followed, later in the SAME sentence, by a negation retracts whatever
# positive assertion came before it -- e.g. "The limit was re-approved
# in 1999, but that approval does not apply to the current
# configuration." or "...re-approved administratively, but no technical
# revalidation was performed." _is_negated alone cannot catch this: it
# only looks backward from the action marker, and here the negation
# comes AFTER it. Scoped to the sentence (not the whole statement) so a
# retraction in one sentence cannot suppress a genuine, unrelated
# positive assertion made in another sentence of the same statement.
_RETRACTION_MARKER = re.compile(
    r"\b(?:but|however|although|though|yet)\b"
    r"[^.;]{0,160}?"
    r"\b(?:not|no|never|without|does\s+not|did\s+not|was\s+not|is\s+not|"
    r"were\s+not|are\s+not)\b",
    re.IGNORECASE,
)

# A retraction clause that explicitly marks its own subject as a
# different, earlier thing ("the PREVIOUS administrative approval was
# not renewed", "the OLDER change request was not approved") is negating
# something distinct from the technical-justification assertion just
# made, not retracting that assertion itself -- e.g. "The current limit
# was technically revalidated..., but the previous administrative
# approval was not renewed." The revalidation stands; only the separate,
# earlier approval lapsed. Without this marker, the same clause shape
# ("but/although ... was not ...") is indistinguishable from a genuine
# retraction (see _RETRACTION_MARKER above), so this check only waives
# the retraction when that distinctness is stated explicitly.
_DISTINCT_PRIOR_SUBJECT_MARKER = re.compile(
    r"\b(?:previous|older|prior|earlier)\b",
    re.IGNORECASE,
)

# A retraction clause only retracts the TECHNICAL-JUSTIFICATION assertion
# when it is actually talking about that assertion -- its validity, its
# scope/applicability, or whether the justification action itself
# occurred ("that validation did not COVER the current equipment", "no
# TECHNICAL REVALIDATION was performed", "that decision does not APPLY
# to the current configuration"). A clause about an unrelated
# operational or administrative/timing fact ("production output was not
# increased", "the administrative approval was delayed") does not touch
# the justification just asserted and must not suppress it, even though
# it has the same "but/however + negation" shape. This check runs only
# when _DISTINCT_PRIOR_SUBJECT_MARKER above did not already waive the
# retraction.
_JUSTIFICATION_RETRACTION_VOCAB = re.compile(
    r"\b(?:validat\w*|approv\w*|justif\w*|establish\w*|apply|appli\w*|"
    r"cover\w*|technical\w*|current|decision\w*)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+")


def _sentence_containing(text: str, index: int) -> str:
    offset = 0
    for sentence in _SENTENCE_SPLIT.split(text):
        end = offset + len(sentence)
        if offset <= index < end or (index == end and sentence):
            return sentence
        offset = end + 1  # +1 for the split whitespace consumed
    return text


def _is_negated(text: str, match_start: int) -> bool:
    preceding = text[max(0, match_start - 120):match_start]
    words = re.findall(r"\b[\w']+\b", preceding.lower())

    # "no re-justification occurred" / "lacks current technical
    # justification" etc. describe an evidence GAP, not a positive
    # assertion -- do not let the local negation flip this into a
    # justification claim.
    if any(word in _STATEMENT_NEGATION_WORDS for word in words[-7:]):
        return True

    # Forward-looking: a retraction later in the same sentence (see
    # _RETRACTION_MARKER above) also means this is not a positive
    # current-technical-justification assertion -- UNLESS that clause
    # explicitly marks what it's negating as a different, earlier thing
    # (see _DISTINCT_PRIOR_SUBJECT_MARKER above), or isn't actually
    # about the justification assertion at all (see
    # _JUSTIFICATION_RETRACTION_VOCAB above), in either of which cases
    # the justification just asserted stands untouched.
    sentence = _sentence_containing(text, match_start)
    retraction = _RETRACTION_MARKER.search(sentence)
    if not retraction:
        return False

    retracted_clause = sentence[retraction.start():]

    if _DISTINCT_PRIOR_SUBJECT_MARKER.search(retracted_clause):
        return False

    return bool(_JUSTIFICATION_RETRACTION_VOCAB.search(retracted_clause))


def _looks_like_current_technical_justification(statement) -> bool:
    """
    Conservative, deterministic check: does this claim's own statement
    text assert that the CURRENT technical basis/limit is justified,
    supported, or valid right now?

    Used only as a fallback for claims whose claim_type is missing,
    invalid, or does not already equal CURRENT_TECHNICAL_JUSTIFICATION.
    A correctly-labeled claim always goes through the gate regardless
    of this check; this function exists so a mislabeled or unlabeled
    one cannot escape it.
    """

    if not isinstance(statement, str) or not statement.strip():
        return False

    for match in _CURRENT_JUSTIFICATION_STATEMENT_MARKERS.finditer(statement):
        if _is_negated(statement, match.start()):
            continue
        return True

    if _CURRENT_LIMIT_SUBJECT.search(statement):
        for match in _JUSTIFICATION_ACTION.finditer(statement):
            if _is_negated(statement, match.start()):
                continue
            return True

    return False


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

        Fix 2 (hostile-audit MUST-FIX): this gate must not depend only
        on Gemini's self-reported `claim_type`. A claim is routed
        through the gate when EITHER:
          - claim_type is explicitly CURRENT_TECHNICAL_JUSTIFICATION
            (the existing, unchanged behavior), OR
          - claim_type is missing, invalid, or something else, but the
            claim's own statement text is a de facto current-technical-
            justification assertion (the deterministic text fallback).
        A claim whose type is merely something other than
        CURRENT_TECHNICAL_JUSTIFICATION *and* whose text does not read
        that way is left untouched -- ordinary capability, qualification,
        and administrative claims must not be swept into this gate.
        """

        is_explicit_type = (
            claim.claim_type == ClaimType.CURRENT_TECHNICAL_JUSTIFICATION
        )
        is_semantic_justification = (
            not is_explicit_type
            and _looks_like_current_technical_justification(claim.statement)
        )

        if not (is_explicit_type or is_semantic_justification):
            return claim

        rejustification_evidence = []

        for chunk_id in claim.evidence_chunk_ids:
            evidence = evidence_by_id.get(chunk_id)

            if evidence is None:
                continue

            if _contains_explicit_rejustification(evidence.content):
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