import re

from app.retrieval.entity_matcher import extract_entities

from app.investigation.models import (
    EvidenceAnalysis,
    EvidenceFinding,
    FindingType,
    Investigation,
    EvidenceItem,
)


# These are broad corpus categories, not document-specific rules.
# The analyzer uses provenance as supporting context, not as proof of a claim.
_TECHNICAL_SECTIONS = {
    "A_asset_equipment_history",
    "B_historical_engineering",
}

_ADMINISTRATIVE_SECTIONS = {
    "C_spc_quality",
    "D_sop_evolution",
}

_HUMAN_SECTIONS = {
    "F_human_informal",
}

_DISTRACTOR_SECTIONS = {
    "G_distractors",
}


# Indicates that equipment/process capability or configuration changed.
_CHANGE_MARKERS = re.compile(
    r"\b("
    r"replaced|replacement|"
    r"commissioned|commissioning|"
    r"decommissioned|decommissioning|"
    r"requalif\w*|"
    r"qualif\w*|"
    r"upgrad\w*|"
    r"retrofit\w*"
    r")\b",
    re.IGNORECASE,
)


# Indicates explicit technical re-establishment of a limit.
#
# IMPORTANT:
# Mere words such as "qualified", "stable", "validated", or
# "requalified" are deliberately NOT enough.
#
# The analyzer must not confuse:
#
#     equipment qualification
#             with
#     technical re-justification of the current limit
#
_REJUSTIFICATION_MARKERS = re.compile(
    r"\b("
    r"re-?approved|"
    r"revalidated\s+for|"
    r"technically\s+justified\s+for|"
    r"technically\s+justified\s+the|"
    r"confirmed\s+applicable\s+to|"
    r"requalified\s+(?:production\s+)?limit\s+of|"
    r"new\s+(?:production\s+)?limit\s+of|"
    r"new\s+(?:production\s+)?limit\s+(?:was\s+)?set\s+(?:to|at)|"
    r"limit\s+(?:was|has\s+been)\s+(?:raised|lowered|changed)\s+to|"
    r"limit\s+(?:was|has\s+been)\s+re-?established|"
    r"upper\s+(?:control\s+)?limit\s+(?:was|has\s+been)\s+re-?established"
    r")\b",
    re.IGNORECASE,
)


_NEGATION_WORDS = {"no", "not", "never", "without", "none"}


def _contains_explicit_rejustification(text: str) -> bool:
    """
    Return True only when a re-justification marker is asserted, not when
    the evidence explicitly denies, lacks, or excludes that action.

    The marker regex remains broad for recall; this sentence/word-window
    guard prevents phrases such as "not re-approved" or "without
    re-approval" from being mistaken for positive evidence.
    """

    if not isinstance(text, str) or not text.strip():
        return False

    for match in _REJUSTIFICATION_MARKERS.finditer(text):
        preceding = text[max(0, match.start() - 120):match.start()]
        words = re.findall(r"\b[\w']+\b", preceding.lower())

        # Only inspect the local clause immediately preceding the marker.
        # This catches common constructions without allowing a distant
        # unrelated "not" to suppress a positive statement.
        if any(word in _NEGATION_WORDS for word in words[-7:]):
            continue

        # "without re-approval" is also commonly written with the
        # negation directly adjacent to the marker.
        if re.search(r"\bwithout\s+(?:being\s+)?$", preceding, re.IGNORECASE):
            continue

        return True

    return False


def _equipment_entities(text: str) -> set[str]:
    """Extract only equipment identifiers used for consistency checks."""
    entities = extract_entities(text)
    return {
        entity
        for entity in entities
        if entity.startswith("px") or entity.startswith("press")
    }


def _has_equipment_overlap(question_entities: set[str], evidence_entities: set[str]) -> bool:
    """
    Compare like-for-like equipment identifiers.

    A Press identifier is compared with Press identifiers and a PX model
    identifier with PX identifiers. We intentionally do not infer a
    Press↔PX mapping here because that relationship is corpus-specific.
    """
    press_q = {e for e in question_entities if e.startswith("press")}
    press_e = {e for e in evidence_entities if e.startswith("press")}
    px_q = {e for e in question_entities if e.startswith("px")}
    px_e = {e for e in evidence_entities if e.startswith("px")}

    if press_q and press_e and press_q.intersection(press_e):
        return True
    if px_q and px_e and px_q.intersection(px_e):
        return True

    # If both sides expose a comparable equipment namespace but none match,
    # they are inconsistent. If only one side exposes an identifier, do not
    # assume a mismatch.
    comparable_namespace_present = (press_q and press_e) or (px_q and px_e)
    return not comparable_namespace_present


# Administrative/commercial decision language.
_ADMIN_DECISION_MARKERS = re.compile(
    r"\b("
    r"deferred|"
    r"rejected|"
    r"tabled|"
    r"postponed|"
    r"not\s+approved|"
    r"commercial(?:ly)?|"
    r"scheduling|"
    r"customer\s+(?:re)?approval|"
    r"PPAP|"
    r"customer\s+notification"
    r")\b",
    re.IGNORECASE,
)


# Explicit technical-safety/failure language.
_TECHNICAL_SAFETY_MARKERS = re.compile(
    r"\b("
    r"unsafe|"
    r"thermal\s+instability|"
    r"failure\s+mode|"
    r"out\s+of\s+spec(?:ification)?|"
    r"exceeded\s+tolerance|"
    r"non-?conforming|"
    r"equipment\s+failure|"
    r"process\s+failure|"
    r"failed\s+validation"
    r")\b",
    re.IGNORECASE,
)


# Current/persistence language. This is deliberately NOT treated as
# technical justification. It is only evidence that a value continued
# to appear in operational/configuration records.
_PERSISTENCE_MARKERS = re.compile(
    r"\b("
    r"remained|"
    r"remains|"
    r"retained|"
    r"continues?\s+to\s+be|"
    r"carried\s+forward|"
    r"unchanged|"
    r"still\s+used|"
    r"current(?:ly)?\s+configured|"
    r"current\s+(?:limit|setting|configuration)"
    r")\b",
    re.IGNORECASE,
)


def _section_of(item: EvidenceItem) -> str:
    """
    Return the top-level corpus section from relative_path.

    Example:
        B_historical_engineering/10_Engineering_Note.txt
        -> B_historical_engineering
    """
    normalized = item.relative_path.replace("\\", "/")

    known_sections = (
        _TECHNICAL_SECTIONS
        | _ADMINISTRATIVE_SECTIONS
        | _HUMAN_SECTIONS
        | _DISTRACTOR_SECTIONS
    )

    for section in known_sections:
        if (
            normalized == section
            or normalized.startswith(f"{section}/")
            or f"/{section}/" in normalized
        ):
            return section

    return ""


class EvidenceGapAnalyzer:
    """
    Deterministic evidence analysis for the investigation pipeline.

    The analyzer is intentionally conservative.

    It does NOT attempt to answer the user's question itself.
    Instead, it extracts structured evidence states that the Gemini
    reasoner can use when constructing the final report.

    In particular, it distinguishes:

        historical technical evidence
        administrative evidence
        equipment/process changes
        current/persistence evidence
        explicit technical re-justification
        evidence gaps
        administrative decisions vs technical-safety conclusions

    The analyzer must never treat:
        persistence == technical reaffirmation
        qualification == limit re-justification
        stability == approval of a new production limit
        administrative rejection == technical rejection
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

        self._analyze_source_provenance(
            investigation,
            analysis,
        )

        # Re-justification analysis is only relevant when the user's
        # question actually asks about a limit, parameter, current basis,
        # technical justification, or continued support. It must not leak
        # into ordinary factual questions such as "what equipment changed?"
        if self._requires_technical_justification_analysis(investigation.question):
            self._analyze_rejustification_gap(
                investigation,
                analysis,
            )

        # Administrative-decision context is useful when the question asks
        # about approval/rejection/decision history. Avoid injecting it into
        # unrelated factual lookups.
        if self._requires_decision_analysis(investigation.question):
            self._analyze_administrative_decisions(
                investigation,
                analysis,
            )

        self._analyze_distractor_only_coverage(
            investigation,
            analysis,
        )

        self._analyze_equipment_consistency(
            investigation,
            analysis,
        )

        return analysis

    def _analyze_subquestion_coverage(
        self,
        investigation: Investigation,
        analysis: EvidenceAnalysis,
    ) -> None:
        """
        Identify subquestions for which no retrieved evidence exists.

        This is intentionally a coverage check, not a relevance judgment.
        """

        covered_subquestions = {
            subquestion
            for evidence in investigation.evidence
            for subquestion in evidence.retrieved_for
        }

        for subquestion in investigation.subquestions:
            if subquestion not in covered_subquestions:
                analysis.add_finding(
                    EvidenceFinding(
                        finding_type=FindingType.EVIDENCE_GAP,
                        statement=(
                            "No retrieved evidence is associated with "
                            f"subquestion: {subquestion}"
                        ),
                    )
                )

    def _analyze_source_provenance(
        self,
        investigation: Investigation,
        analysis: EvidenceAnalysis,
    ) -> None:
        """
        Tell the reasoner what broad kinds of sources are represented.

        Provenance is contextual information. It does not itself establish
        whether a claim is technically correct.
        """

        technical_ids: list[str] = []
        administrative_ids: list[str] = []
        human_ids: list[str] = []

        for item in investigation.evidence:
            section = _section_of(item)

            if section in _TECHNICAL_SECTIONS:
                technical_ids.append(item.chunk_id)

            elif section in _ADMINISTRATIVE_SECTIONS:
                administrative_ids.append(item.chunk_id)

            elif section in _HUMAN_SECTIONS:
                human_ids.append(item.chunk_id)

        if technical_ids:
            analysis.add_finding(
                EvidenceFinding(
                    finding_type=FindingType.CONTEXT,
                    statement=(
                        f"{len(technical_ids)} retrieved chunk(s) originate "
                        "from technical/engineering source sections. "
                        "These sources may provide technical basis or "
                        "equipment/process history."
                    ),
                    evidence_chunk_ids=technical_ids,
                )
            )

        if administrative_ids:
            analysis.add_finding(
                EvidenceFinding(
                    finding_type=FindingType.CONTEXT,
                    statement=(
                        f"{len(administrative_ids)} retrieved chunk(s) "
                        "originate from administrative/procedural source "
                        "sections such as SPC configuration or SOP records. "
                        "Their persistence does not by itself establish "
                        "technical re-justification."
                    ),
                    evidence_chunk_ids=administrative_ids,
                )
            )

        if human_ids:
            analysis.add_finding(
                EvidenceFinding(
                    finding_type=FindingType.CONTEXT,
                    statement=(
                        f"{len(human_ids)} retrieved chunk(s) originate "
                        "from informal or human-recollection sources. "
                        "Treat these as lower-authority context unless "
                        "corroborated by formal records."
                    ),
                    evidence_chunk_ids=human_ids,
                )
            )

    @staticmethod
    def _requires_technical_justification_analysis(question: str) -> bool:
        q = " ".join(str(question or "").lower().split())
        return any(term in q for term in (
            "why",
            "limit",
            "threshold",
            "parameter",
            "setting",
            "technical justification",
            "technical basis",
            "justification",
            "rationale",
            "revalidated",
            "revalidation",
            "re-validated",
            "re-justif",
            "re-derived",
            "still supported",
            "current technical",
            "current limit",
            "current parameter",
            "original basis",
        ))

    @staticmethod
    def _requires_decision_analysis(question: str) -> bool:
        q = " ".join(str(question or "").lower().split())
        return any(term in q for term in (
            "approved",
            "approval",
            "rejected",
            "reject",
            "proposal",
            "decision",
            "why was",
            "reason for",
        ))

    def _analyze_rejustification_gap(
        self,
        investigation: Investigation,
        analysis: EvidenceAnalysis,
    ) -> None:
        """
        Detect the important pattern:

            change/requalification evidence
                    +
            persistence/current configuration evidence
                    +
            no explicit technical re-justification evidence
                    =
            evidence gap

        This is deliberately a pattern detector rather than a benchmark-
        specific answer generator.

        A qualification/requalification record by itself is NOT considered
        sufficient evidence of technical re-justification.
        """

        change_ids: list[str] = []
        persistence_ids: list[str] = []
        rejustification_ids: list[str] = []

        for item in investigation.evidence:
            content = item.content

            if _CHANGE_MARKERS.search(content):
                change_ids.append(item.chunk_id)

            if _PERSISTENCE_MARKERS.search(content):
                persistence_ids.append(item.chunk_id)

            if _contains_explicit_rejustification(content):
                rejustification_ids.append(item.chunk_id)

        # We need evidence that something changed before a "re-justification"
        # gap makes sense.
        if not change_ids:
            return

        # If explicit re-justification evidence exists, do not call the
        # current technical justification an evidence gap based only on
        # lexical absence.
        if rejustification_ids:
            analysis.add_finding(
                EvidenceFinding(
                    finding_type=FindingType.CONTEXT,
                    statement=(
                        "Retrieved evidence contains an explicit "
                        "re-justification or limit-change marker. "
                        "The reasoner must inspect the cited evidence and "
                        "determine whether it actually applies to the "
                        "current equipment and exact parameter."
                    ),
                    evidence_chunk_ids=rejustification_ids,
                )
            )
            return

        # Persistence strengthens the inference that the old value was
        # carried forward, but absence of persistence should not prevent us
        # from flagging a possible re-justification gap when a major change
        # occurred.
        supporting_ids = list(dict.fromkeys(change_ids + persistence_ids))

        analysis.add_finding(
            EvidenceFinding(
                finding_type=FindingType.EVIDENCE_GAP,
                statement=(
                    "Retrieved evidence documents an equipment or process "
                    "change, but does not contain an explicit technical "
                    "re-justification of the current parameter/limit for "
                    "the changed configuration. Qualification, stability, "
                    "unchanged configuration, or administrative persistence "
                    "alone does not establish technical re-justification. "
                    "Technical re-justification is not established in the "
                    "available evidence."
                ),
                evidence_chunk_ids=supporting_ids,
            )
        )

    def _analyze_administrative_decisions(
        self,
        investigation: Investigation,
        analysis: EvidenceAnalysis,
    ) -> None:
        """
        Identify administrative/commercial decisions that should not be
        automatically interpreted as technical safety conclusions.
        """

        administrative_only_ids: list[str] = []

        for item in investigation.evidence:
            content = item.content

            has_admin_decision = bool(
                _ADMIN_DECISION_MARKERS.search(content)
            )

            has_technical_safety = bool(
                _TECHNICAL_SAFETY_MARKERS.search(content)
            )

            if has_admin_decision and not has_technical_safety:
                administrative_only_ids.append(item.chunk_id)

        if administrative_only_ids:
            analysis.add_finding(
                EvidenceFinding(
                    finding_type=FindingType.CONTEXT,
                    statement=(
                        "Retrieved evidence contains administrative, "
                        "commercial, scheduling, or approval-decision "
                        "language without an explicit technical-safety "
                        "conclusion in the same evidence. Do not interpret "
                        "the administrative decision as proof that the "
                        "parameter or proposed value was technically "
                        "unsafe unless separate evidence explicitly "
                        "establishes that conclusion."
                    ),
                    evidence_chunk_ids=administrative_only_ids,
                )
            )

    def _analyze_equipment_consistency(
        self,
        investigation: Investigation,
        analysis: EvidenceAnalysis,
    ) -> None:
        """
        Flag retrieved chunks whose explicit equipment identifier conflicts
        with the equipment identifier in the user's question.

        This is a context/evidence-quality warning, not a hard retrieval
        filter. Evidence without a comparable equipment identifier is left
        untouched because the analyzer cannot safely infer an asset mismatch.
        """

        question_entities = _equipment_entities(investigation.question)
        if not question_entities:
            return

        mismatched_ids: list[str] = []

        for item in investigation.evidence:
            evidence_entities = _equipment_entities(item.content)
            if not evidence_entities:
                continue

            if not _has_equipment_overlap(question_entities, evidence_entities):
                mismatched_ids.append(item.chunk_id)

        if mismatched_ids:
            analysis.add_finding(
                EvidenceFinding(
                    finding_type=FindingType.CONTEXT,
                    statement=(
                        "Some retrieved evidence contains equipment "
                        "identifiers that do not match the equipment "
                        "identified in the investigation question. Treat "
                        "those chunks as potential distractors unless the "
                        "records explicitly establish their relationship "
                        "to the investigated equipment."
                    ),
                    evidence_chunk_ids=mismatched_ids,
                )
            )

    def _analyze_distractor_only_coverage(
        self,
        investigation: Investigation,
        analysis: EvidenceAnalysis,
    ) -> None:
        """
        Detect subquestions for which every associated retrieved chunk
        comes from the known distractor section.

        This is a warning about evidence quality, not proof that the
        retrieved evidence is false.
        """

        by_subquestion: dict[str, list[EvidenceItem]] = {}

        for item in investigation.evidence:
            for subquestion in item.retrieved_for:
                by_subquestion.setdefault(
                    subquestion,
                    [],
                ).append(item)

        for subquestion, items in by_subquestion.items():
            if not items:
                continue

            sections = {
                _section_of(item)
                for item in items
            }

            if sections and sections.issubset(_DISTRACTOR_SECTIONS):
                analysis.add_finding(
                    EvidenceFinding(
                        finding_type=FindingType.EVIDENCE_GAP,
                        statement=(
                            "All retrieved evidence associated with "
                            f'subquestion "{subquestion}" originates '
                            "from distractor corpus material. Treat the "
                            "subquestion as weakly supported until "
                            "corroborating evidence is found."
                        ),
                        evidence_chunk_ids=[
                            item.chunk_id
                            for item in items
                        ],
                    )
                )