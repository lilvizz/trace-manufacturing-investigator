"""Pure, Streamlit-free views over an InvestigationResult.

Every function here derives its output ONLY from fields that already exist
on the result (report.claims, report.direct_answer*, investigation.evidence,
timeline.events, evidence_analysis.findings, report.unresolved_questions)
and from the frozen corpus on disk. Nothing is inferred by a model, and no
relationship, date, confidence or citation is invented. When data is
missing the views return empty structures so pages can show honest empty
states.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

from app import ui as U
from app.investigation.confidence import ConfidenceCalibrator
from app.investigation.models import ClaimType, EvidenceStance

# ---------------------------------------------------------------------------
# Claims
# ---------------------------------------------------------------------------

# Display groups, in reading order. Grouping uses only the backend's own
# stance / claim_type values (the same rule app.ui.render_claims uses).
CLAIM_GROUPS = OrderedDict(
    [
        ("contradicted", ("Contradicted", "red", "The cited records conflict with the statement.")),
        ("not_established", (
            "Not established and evidence gaps",
            "amber",
            "The evidence needed was not found in the retrieved records. This is not a finding that the statement is false.",
        )),
        ("supported", ("Supported", "green", "The cited records directly document the statement.")),
        ("context", ("Context only", "blue", "Background that informs the case but does not on its own support a conclusion.")),
        ("inconclusive", ("Inconclusive", "gray", "No valid citation survived verification, so no confidence is assigned.")),
        ("recommendation", ("Recommendations", "violet",
                            "Actions proposed by TRACE reasoning. A recommendation is never a finding; each one shows "
                            "what TRACE's identifier check found in its cited records, which is not proof that "
                            "the records support it.")),
    ]
)

# Stance-level caps applied by ConfidenceCalibrator._calibrate_claim. They
# are literals inside that method, so they are mirrored here for display;
# tests/unit/test_command_center.py asserts these match the calibrator's
# actual behaviour so the explanation cannot drift from the code.
STANCE_CAPS = {
    EvidenceStance.NOT_ESTABLISHED: 0.30,
    EvidenceStance.CONTRADICTS: 0.85,
    EvidenceStance.CONTEXT_ONLY: 0.65,
}


def claim_group(claim) -> str:
    ctype = getattr(claim, "claim_type", None)
    stance = getattr(claim, "stance", None)
    if ctype == ClaimType.RECOMMENDATION:
        return "recommendation"
    if ctype == ClaimType.EVIDENCE_GAP or stance == EvidenceStance.NOT_ESTABLISHED:
        return "not_established"
    if stance == EvidenceStance.SUPPORTS:
        return "supported"
    if stance == EvidenceStance.CONTRADICTS:
        return "contradicted"
    if stance == EvidenceStance.CONTEXT_ONLY:
        return "context"
    return "inconclusive"


def evidence_role(claim, evidence_ids=None) -> str:
    """How the passages a claim cites relate to it, from its stance."""
    stance = getattr(claim, "stance", None)
    if getattr(claim, "claim_type", None) == ClaimType.RECOMMENDATION:
        return RECOMMENDATION_EVIDENCE_ROLE[recommendation_check(claim, evidence_ids)]
    return {
        EvidenceStance.SUPPORTS: "Supporting evidence",
        EvidenceStance.CONTRADICTS: "Conflicting evidence",
        EvidenceStance.NOT_ESTABLISHED: "Evidence reviewed (does not establish the statement)",
        EvidenceStance.CONTEXT_ONLY: "Context evidence",
    }.get(stance, "Cited evidence")


def calibration_limits(claim) -> list[str]:
    """Deterministic confidence ceilings that applied to this claim."""
    notes = []
    ceiling = ConfidenceCalibrator._CLAIM_TYPE_CEILINGS.get(getattr(claim, "claim_type", None))
    if ceiling is not None:
        notes.append(f"{U._claim_type_label(claim)} claims are capped at {ceiling:.0%}.")
    cap = STANCE_CAPS.get(getattr(claim, "stance", None))
    if cap is not None:
        label = U._status(claim.stance)[0]
        notes.append(f"{label} claims are capped at {cap:.0%}.")
    if not list(getattr(claim, "evidence_chunk_ids", []) or []):
        notes.append("No citation resolved to retrieved evidence, so confidence is 0%.")
    return notes


@dataclass
class ClaimRow:
    idx: int
    claim: object
    group: str
    statement: str
    type_label: str
    status_label: str
    tone: str
    glyph: str
    confidence: float
    chunk_ids: list[str]
    requires_human: bool
    limits: list[str] = field(default_factory=list)


def evidence_ids(result) -> set[str]:
    """Chunk IDs of the passages actually retrieved for this result."""
    return {e.chunk_id for e in U._evidence_items(result)}


def recommendation_check(claim, evidence_ids=None) -> str:
    """What ClaimVerifier's identifier check established for a
    recommendation: "procedure_referenced", "identifier_overlap", or ""
    (nothing; shown as unverified).

    Requires ALL of: a recommendation; stance still SUPPORTS after
    verification; the verifier's recorded outcome; and the passages that
    matched being among the claim's citations AND among the retrieved
    evidence (`evidence_ids`). Without `evidence_ids` resolvability cannot
    be checked, so the answer is "" (fail closed). A non-empty list of IDs
    alone never counts. The outcome is lexical (identifiers present in the
    cited text), never semantic support."""
    if getattr(claim, "claim_type", None) != ClaimType.RECOMMENDATION:
        return ""
    if getattr(claim, "stance", None) != EvidenceStance.SUPPORTS or evidence_ids is None:
        return ""
    outcome = getattr(claim, "citation_check", "") or ""
    if outcome not in ("procedure_referenced", "identifier_overlap"):
        return ""
    matched = list(getattr(claim, "citation_check_chunk_ids", []) or [])
    cited = set(getattr(claim, "evidence_chunk_ids", []) or [])
    if not matched or not getattr(claim, "citation_check_anchors", None):
        return ""
    if not set(matched) <= cited or not set(matched) <= set(evidence_ids):
        return ""
    return outcome


RECOMMENDATION_STATUS = {
    "procedure_referenced": ("Procedure referenced · not a finding", "violet", "→"),
    "identifier_overlap": ("Related records cited · not a finding", "violet", "→"),
    "": ("Not verified · suggestion only", "gray", "?"),
}

RECOMMENDATION_EVIDENCE_ROLE = {
    "procedure_referenced": "Cited passage naming the procedure (identifier check only)",
    "identifier_overlap": "Cited records (identifier check only)",
    "": "Cited evidence (not verified)",
}


def _recommendation_why(claim, outcome: str) -> str:
    ids = ", ".join(getattr(claim, "citation_check_anchors", []) or [])
    if outcome == "procedure_referenced":
        return (f"Recommendation, not a finding. The procedure it names ({ids}) appears in a cited passage. "
                "TRACE checks identifiers only; it has not verified that the passage supports this step, so "
                "read the passage before acting. TRACE does not authorize changes.")
    if outcome == "identifier_overlap":
        return ("Recommendation, not a finding. A cited passage shares an identifier with it (equipment, "
                "document ID, temperature or year). TRACE checks identifiers only, not whether the passage "
                "supports this step.")
    return ("Recommendation, not a finding. No cited passage passed TRACE's identifier check, so it is shown "
            "as an unverified suggestion only.")


def claim_status(claim, evidence_ids=None) -> tuple[str, str, str]:
    """(label, tone, glyph) for a claim. Same vocabulary as app.ui.STATUS_META,
    except:
    - an evidence-gap claim the records SUPPORT reads "Gap confirmed": the
      records confirm something is missing, which a bare "Supported" next to
      "Evidence gap" would leave ambiguous;
    - a recommendation never reads "Supported" (a finding): it reads what
      the identifier check found (recommendation_check), never more. Pass
      `evidence_ids` (D.evidence_ids(result)); without them a
      recommendation reads "Not verified"."""
    ctype = getattr(claim, "claim_type", None)
    if ctype == ClaimType.RECOMMENDATION:
        if getattr(claim, "stance", None) == EvidenceStance.CONTRADICTS:
            return "Conflicts with the cited records", "red", "✕"
        return RECOMMENDATION_STATUS[recommendation_check(claim, evidence_ids)]
    if ctype == ClaimType.EVIDENCE_GAP and getattr(claim, "stance", None) == EvidenceStance.SUPPORTS:
        return "Gap confirmed by records", "amber", "!"
    return U._status(getattr(claim, "stance", None))


def claim_why(claim, evidence_ids=None) -> str:
    """One line saying what the claim's status means."""
    if getattr(claim, "claim_type", None) == ClaimType.RECOMMENDATION and \
            getattr(claim, "stance", None) != EvidenceStance.CONTRADICTS:
        return _recommendation_why(claim, recommendation_check(claim, evidence_ids))
    return U._WHY_IT_MATTERS.get(getattr(claim, "stance", None), "")


def status_badge(claim, evidence_ids=None) -> str:
    label, tone, glyph = claim_status(claim, evidence_ids)
    return f'<span class="badge {tone}"><span aria-hidden="true">{glyph}</span>{U._escape(label)}</span>'


def claim_rows(result) -> list[ClaimRow]:
    rows = []
    ids = evidence_ids(result)
    for idx, claim in enumerate(U._claims(result)):
        label, tone, glyph = claim_status(claim, ids)
        rows.append(
            ClaimRow(
                idx=idx,
                claim=claim,
                group=claim_group(claim),
                statement=U._display_claim_statement(claim),
                type_label=U._claim_type_label(claim),
                status_label=label,
                tone=tone,
                glyph=glyph,
                confidence=float(getattr(claim, "confidence", 0.0) or 0.0),
                chunk_ids=list(getattr(claim, "evidence_chunk_ids", []) or []),
                requires_human=U._requires_human(claim),
                limits=calibration_limits(claim),
            )
        )
    return rows


def grouped_claims(result) -> "OrderedDict[str, list[ClaimRow]]":
    groups: OrderedDict[str, list[ClaimRow]] = OrderedDict((k, []) for k in CLAIM_GROUPS)
    for row in claim_rows(result):
        groups[row.group].append(row)
    return groups


def primary_claim_index(result) -> int | None:
    claim = U._primary_claim(result)
    if claim is None:
        return None
    for idx, candidate in enumerate(U._claims(result)):
        if candidate is claim:
            return idx
    return None


# ---------------------------------------------------------------------------
# Evidence <-> claims <-> timeline cross references
# ---------------------------------------------------------------------------

def claims_by_chunk(result) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    for idx, claim in enumerate(U._claims(result)):
        for chunk_id in getattr(claim, "evidence_chunk_ids", []) or []:
            mapping.setdefault(chunk_id, []).append(idx)
    return mapping


def cited_chunk_ids(result) -> set[str]:
    return set(claims_by_chunk(result)) | U._direct_answer_ids(result)


UNCATEGORISED_LANE = "Other retrieved sources"
LANE_ORDER = [
    "Asset & equipment history",
    "Engineering record",
    "SPC & quality",
    "SOP / procedure",
    "Maintenance & calibration",
    "Informal / recollection",
    UNCATEGORISED_LANE,
]


def lane_for_path(relative_path: str) -> str:
    # Uses the explicit allow-list in app.ui; anything else (including an
    # evaluation-only folder) gets a neutral lane, never an invented type.
    return U._source_category(relative_path) or UNCATEGORISED_LANE


@dataclass
class TimelineRow:
    idx: int
    date: object
    date_label: str
    year: str
    headline: str
    description: str
    lines: list[str]
    kind: str            # "record" (record header shown as its date) | "event"
    chunk_ids: list[str]
    lane: str
    filename: str
    relative_path: str
    cited: bool
    claim_idxs: list[int]
    date_text: str = ""          # exact text the date was parsed from
    precision: str = "day"       # "day" | "month" (month-only source date)
    ambiguous: bool = False      # numeric date readable day- or month-first
    issued: bool = False         # the record's own issue/header date
    related_dates: list[tuple[str, str, str]] = field(default_factory=list)  # (ref, date label, filename)
    # Key entries are the condensed narrative events + record dates, plus at
    # least one event for every year that has any well-formed dated event,
    # so the chronology's start (e.g. the 2001 origin) and end are never
    # dropped by the keyword-ranked cap. Non-key rows are the remaining
    # well-formed extracted events (shown on demand).
    key: bool = True


_RECORD_REF = re.compile(
    r"\b(?:ECO|QUAL|ENG-NOTE|INV|QM|SPC-REV|PE-STUDY|PE-VAL|COMM|WO|REL|HPS-VAL)-[\w-]*\d\b"
)

_CACHE: "weakref.WeakKeyDictionary" = None  # set below


def _cached(result, key, build):
    """Per-result memo for derived views (results are not mutated after the
    pipeline returns them). Falls back to no caching for objects that do not
    support weak references (e.g. test doubles)."""
    global _CACHE
    if _CACHE is None:
        import weakref

        _CACHE = weakref.WeakKeyDictionary()
    try:
        bucket = _CACHE.setdefault(result, {})
    except TypeError:
        return build()
    if key not in bucket:
        bucket[key] = build()
    return bucket[key]


def date_label(value, precision: str = "day") -> str:
    return U._date_label(value, precision)


def timeline_rows(result) -> tuple[list[TimelineRow], int]:
    """Display rows built on app.ui._timeline_entries (unchanged logic):
    real extracted dates only, document header lines shown as the record's
    date, sign-offs merged, fragments hidden (but still counted). Each row
    carries the exact source date text, its precision and whether it is a
    record's own issue date."""
    return _cached(result, "timeline_rows", lambda: _timeline_rows(result))


def _timeline_rows(result) -> tuple[list[TimelineRow], int]:
    if getattr(result, "timeline", None) is None:
        return [], 0
    entries, total = U._timeline_entries(result)
    by_id = U._evidence_by_id(result)
    by_chunk = claims_by_chunk(result)
    cited = cited_chunk_ids(result)
    entries = _with_full_coverage(result, entries, by_id, cited)

    # Dates attached to the same record ID (ECO-2011-184, QUAL-2014-009 …)
    # in different documents. Shown as related dates, never merged and never
    # asserted to be a conflict: they may describe different events.
    by_ref: dict[str, set[tuple[str, str]]] = {}
    for event in list(getattr(result.timeline, "events", []) or []):
        item = by_id.get((getattr(event, "evidence_chunk_ids", []) or [None])[0])
        filename = getattr(item, "filename", "") if item else ""
        label = date_label(event.date, getattr(event, "precision", "day"))
        for ref in set(_RECORD_REF.findall(str(event.description))):
            by_ref.setdefault(ref, set()).add((label, filename))

    rows = []
    for idx, entry in enumerate(entries):
        chunk_ids = list(entry.get("chunk_ids") or [])
        item = by_id.get(chunk_ids[0]) if chunk_ids else None
        precision = entry.get("precision", "day")
        year = U._format_date(entry["date"])[0]
        readable = date_label(entry["date"], precision)
        headline = (
            f"Record dated · {entry['title']}" + (f" ({entry['ref']})" if entry.get("ref") else "")
            if entry["kind"] == "record" and entry.get("title")
            else re.sub(r"^[-–•*\s]+", "", entry["description"]) or entry["description"]
        )
        claim_idxs = sorted({i for c in chunk_ids for i in by_chunk.get(c, [])})
        filename = getattr(item, "filename", "") if item else ""
        related = []
        for ref in sorted(set(_RECORD_REF.findall(" ".join([entry["description"]] + list(entry.get("lines") or []))))):
            for label, other in sorted(by_ref.get(ref, ())):
                if (label, other) != (readable, filename):
                    related.append((ref, label, other))
        rows.append(
            TimelineRow(
                idx=idx,
                date=entry["date"],
                date_label=readable,
                year=year,
                headline=headline,
                description=entry["description"],
                lines=list(entry.get("lines") or []),
                kind=entry["kind"],
                chunk_ids=chunk_ids,
                lane=lane_for_path(getattr(item, "relative_path", "")) if item else UNCATEGORISED_LANE,
                filename=filename,
                relative_path=getattr(item, "relative_path", "") if item else "",
                cited=any(c in cited for c in chunk_ids),
                claim_idxs=claim_idxs,
                date_text=entry.get("date_text", ""),
                precision=precision,
                ambiguous=bool(entry.get("ambiguous")),
                issued=entry.get("event_kind") == "record_issued" or entry["kind"] == "record",
                related_dates=related[:6],
                key=entry.get("key", True),
            )
        )
    return rows, total


def _with_full_coverage(result, entries, by_id, cited) -> list[dict]:
    """Add every remaining well-formed extracted event (not key), then mark
    the best event of each otherwise-uncovered year as key. Uses only events
    the extractor produced; fragments stay hidden (listed in the unfiltered
    view)."""
    shown = set()
    for entry in entries:
        shown.add((U._date_sort_key(entry["date"]), entry["description"]))
        for line in entry.get("lines") or []:
            shown.add((U._date_sort_key(entry["date"]), line))
    extra = []
    for event in list(getattr(result.timeline, "events", []) or []):
        description = str(event.description).strip()
        key = (U._date_sort_key(event.date), description)
        if key in shown or U._looks_like_fragment(description):
            continue
        shown.add(key)
        chunk_ids = list(getattr(event, "evidence_chunk_ids", []) or [])
        item = by_id.get(chunk_ids[0]) if chunk_ids else None
        extra.append({
            "date": event.date, "description": description, "chunk_ids": chunk_ids, "kind": "event",
            "document": getattr(item, "document_id", None) or getattr(item, "filename", None),
            "title": U._document_title(getattr(item, "filename", "")) if item is not None else "",
            "ref": "", "lines": [],
            "date_text": getattr(event, "date_text", "") or "",
            "precision": getattr(event, "precision", "day") or "day",
            "ambiguous": bool(getattr(event, "ambiguous", False)),
            "event_kind": getattr(event, "kind", "") or "",
            "key": False,
        })
    covered = {getattr(e["date"], "year", None) for e in entries}
    by_year: dict = {}
    for entry in extra:
        by_year.setdefault(getattr(entry["date"], "year", None), []).append(entry)
    for year, candidates in by_year.items():
        if year in covered:
            continue
        best = sorted(
            candidates,
            key=lambda e: (
                not any(c in cited for c in e["chunk_ids"]),
                not equipment_mentions(e["description"]),
                e["precision"] != "day",
                U._date_sort_key(e["date"]),
            ),
        )[0]
        best["key"] = True
    merged = [dict(e, key=e.get("key", True)) for e in entries] + extra
    merged.sort(key=lambda e: U._date_sort_key(e["date"]))
    return merged


def chronology_span(rows) -> str:
    """'05 Jul 2001 → 30 Nov 2022 · 21 years with dated events' from real rows."""
    dated = [r for r in rows if hasattr(r.date, "year")]
    if not dated:
        return ""
    first, last = dated[0], dated[-1]
    years = len({r.date.year for r in dated})
    return f"{first.date_label} → {last.date_label} · {years} year{'s' if years != 1 else ''} with dated events"


def events_by_chunk(result) -> dict[str, list[int]]:
    mapping: dict[str, list[int]] = {}
    rows, _ = timeline_rows(result)
    for row in rows:
        for chunk_id in row.chunk_ids:
            mapping.setdefault(chunk_id, []).append(row.idx)
    return mapping


def undated_documents(result, chunks=None) -> list[dict]:
    """Retrieved documents none of whose RETRIEVED passages contains an
    extractable date. When the corpus chunks are supplied, the document's
    own header date (from its first passage in the corpus) is reported too,
    clearly as a date from a passage this investigation did not retrieve."""
    events = list(getattr(getattr(result, "timeline", None), "events", []) or [])
    dated_chunks = {c for e in events for c in (getattr(e, "evidence_chunk_ids", []) or [])}
    docs: "OrderedDict[str, dict]" = OrderedDict()
    for item in U._evidence_items(result):
        name = getattr(item, "filename", "")
        entry = docs.setdefault(name, {"filename": name, "relative_path": item.relative_path, "dated": False,
                                       "header_date": "", "header_chunk": ""})
        if item.chunk_id in dated_chunks:
            entry["dated"] = True
    out = [d for d in docs.values() if not d["dated"]]
    if chunks:
        from app.investigation.models import EvidenceItem
        from app.investigation.timeline import TimelineBuilder

        by_file: dict[str, list] = {}
        for chunk in chunks:
            by_file.setdefault(chunk.filename, []).append(chunk)
        for entry in out:
            for chunk in by_file.get(entry["filename"], [])[:1]:
                built = TimelineBuilder().build([EvidenceItem(chunk.chunk_id, chunk.document_id, chunk.filename,
                                                              chunk.relative_path, chunk.content, 0.0, [])])
                issued = [e for e in built.events if e.kind == "record_issued"]
                if issued:
                    entry["header_date"] = date_label(issued[0].date, issued[0].precision)
                    entry["header_chunk"] = chunk.chunk_id
    return out


# ---------------------------------------------------------------------------
# Gaps, contradictions, open questions, recommendations
# ---------------------------------------------------------------------------

@dataclass
class RegisterItem:
    kind: str          # verified_gap | claimed_gap | contradiction | open_question | recommendation
    origin: str        # who produced it, in plain words
    statement: str
    chunk_ids: list[str]
    claim_idx: int | None = None
    verified: bool = False
    # Recommendations only: what the identifier check found
    # (D.recommendation_check: "procedure_referenced" / "identifier_overlap"
    # / ""). Lexical only. A recommendation is never "verified" as a
    # finding, so `verified` stays False for it.
    citation_check: str = ""


RECOMMENDATION_ORIGIN = {
    "procedure_referenced": "procedure referenced in a cited passage (identifier check only)",
    "identifier_overlap": "related records cited (identifier check only)",
    "": "not verified",
}


REGISTER_KINDS = OrderedDict(
    [
        ("contradiction", ("Contradictions", "red")),
        ("verified_gap", ("Evidence gaps · deterministic check", "amber")),
        ("claimed_gap", ("Not established · reasoning claims", "amber")),
        ("open_question", ("Open questions", "gray")),
        ("recommendation", ("Recommended actions", "violet")),
    ]
)


def gap_register(result) -> list[RegisterItem]:
    items: list[RegisterItem] = []
    for finding in U._findings(result):
        kind = U._finding_kind(finding)
        ids = list(getattr(finding, "evidence_chunk_ids", []) or [])
        if kind == "evidence-gap":
            items.append(RegisterItem("verified_gap", "Rule-based evidence-gap analyzer", finding.statement, ids, verified=True))
        elif kind == "contradiction":
            items.append(RegisterItem("contradiction", "Rule-based evidence-gap analyzer", finding.statement, ids, verified=True))
    ids = evidence_ids(result)
    for row in claim_rows(result):
        if row.group == "contradicted":
            items.append(RegisterItem("contradiction", "Reasoning claim", row.statement, row.chunk_ids, row.idx, bool(row.chunk_ids)))
        elif row.group == "not_established":
            items.append(RegisterItem("claimed_gap", "Reasoning claim", row.statement, row.chunk_ids, row.idx, bool(row.chunk_ids)))
        elif row.group == "recommendation":
            check = recommendation_check(row.claim, ids)
            origin = "Suggested by TRACE reasoning · " + RECOMMENDATION_ORIGIN[check]
            items.append(RegisterItem("recommendation", origin, row.statement, row.chunk_ids, row.idx, False,
                                      citation_check=check))
    for question in U._unresolved(result):
        items.append(RegisterItem("open_question", "Raised by reasoning", question, []))
    return items


# ---------------------------------------------------------------------------
# Relationship graph (only relationships present in the result)
# ---------------------------------------------------------------------------

_EQUIPMENT_PATTERNS = (
    re.compile(r"\bPX-\d+\b", re.IGNORECASE),
    re.compile(r"\bPress[- ]?\d+\b", re.IGNORECASE),
)


def _equipment_key(token: str) -> str:
    text = re.sub(r"[\s\-]+", "", token.lower())
    return re.sub(r"(?<![\d.])0+(?=\d)", "", text)


def equipment_mentions(text: str) -> dict[str, str]:
    """canonical key -> first surface form, for Press-N / PX-N identifiers
    (same identifier families the analyzer compares)."""
    found: dict[str, str] = {}
    for pattern in _EQUIPMENT_PATTERNS:
        for match in pattern.finditer(text or ""):
            found.setdefault(_equipment_key(match.group(0)), match.group(0))
    return found


@dataclass
class GraphNode:
    id: str
    kind: str          # claim | document | event | equipment
    label: str
    detail: str
    chunk_ids: list[str] = field(default_factory=list)
    tone: str = "gray"
    ref: object = None  # claim idx / event idx / filename / equipment key
    x: float = 0.0
    y: float = 0.0


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    chunk_ids: list[str] = field(default_factory=list)


def relationship_graph(result, max_events: int = 14) -> tuple[list[GraphNode], list[GraphEdge]]:
    return _cached(result, ("graph", max_events), lambda: _relationship_graph(result, max_events))


def _relationship_graph(result, max_events: int = 14) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Claims, documents, timeline events and equipment, connected ONLY by
    relationships recorded in the result:

    * claim -> document : the claim cites a passage of that document
    * event -> document : the event was extracted from a passage of it
    * equipment -> document : the identifier literally appears in a retrieved
      passage of that document (only for documents already in the graph)
    """
    by_id = U._evidence_by_id(result)
    nodes: dict[str, GraphNode] = {}
    edges: dict[tuple[str, str], GraphEdge] = {}

    def doc_node(chunk_id):
        item = by_id.get(chunk_id)
        if item is None:
            return None
        node_id = f"doc:{item.filename}"
        if node_id not in nodes:
            nodes[node_id] = GraphNode(
                node_id, "document", U._document_title(item.filename),
                f"{item.relative_path}", [], "doc", item.filename,
            )
        if chunk_id not in nodes[node_id].chunk_ids:
            nodes[node_id].chunk_ids.append(chunk_id)
        return node_id

    def link(src, dst, relation, chunk_id):
        edge = edges.setdefault((src, dst), GraphEdge(src, dst, relation, []))
        if chunk_id not in edge.chunk_ids:
            edge.chunk_ids.append(chunk_id)

    for row in claim_rows(result):
        valid = [c for c in row.chunk_ids if c in by_id]
        if not valid:
            continue
        node_id = f"claim:{row.idx}"
        nodes[node_id] = GraphNode(
            node_id, "claim", f"{row.status_label} · {row.type_label}", row.statement,
            valid, row.tone, row.idx,
        )
        for chunk_id in valid:
            doc = doc_node(chunk_id)
            if doc:
                link(node_id, doc, "cites", chunk_id)

    rows, _ = timeline_rows(result)
    # Prefer events tied to cited passages, then chronology.
    chosen = sorted(rows, key=lambda r: (not r.cited, r.idx))[:max_events]
    for row in sorted(chosen, key=lambda r: r.idx):
        valid = [c for c in row.chunk_ids if c in by_id]
        if not valid:
            continue
        node_id = f"event:{row.idx}"
        nodes[node_id] = GraphNode(node_id, "event", row.date_label, row.headline, valid, "event", row.idx)
        for chunk_id in valid:
            doc = doc_node(chunk_id)
            if doc:
                link(node_id, doc, "extracted from", chunk_id)

    doc_ids = [n.id for n in nodes.values() if n.kind == "document"]
    question = str(getattr(getattr(result, "investigation", None), "question", "") or "")
    named = equipment_mentions(question)
    cited_text = " ".join(
        str(getattr(by_id[c], "content", ""))
        for c in cited_chunk_ids(result) if c in by_id
    )
    candidates = dict(equipment_mentions(cited_text))
    candidates.update(named)
    for key, surface in candidates.items():
        node_id = f"equip:{key}"
        hits = []
        for doc_id in doc_ids:
            filename = nodes[doc_id].ref
            for item in U._evidence_items(result):
                if item.filename == filename and key in equipment_mentions(item.content):
                    hits.append((doc_id, item.chunk_id))
        if not hits:
            continue
        nodes[node_id] = GraphNode(
            node_id, "equipment", surface,
            "Named in the question" if key in named else "Named in a cited passage",
            [], "equipment", key,
        )
        for doc_id, chunk_id in hits:
            link(node_id, doc_id, "mentioned in", chunk_id)
            if chunk_id not in nodes[node_id].chunk_ids:
                nodes[node_id].chunk_ids.append(chunk_id)

    _layout(nodes, list(edges.values()))
    return list(nodes.values()), list(edges.values())


def _layout(nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> None:
    """Deterministic layered layout: claims | documents | events & equipment.
    Documents are ordered by the barycentre of their neighbours to reduce
    crossings. Positions are presentation only."""
    columns = {"claim": 0.0, "document": 1.0, "event": 2.0, "equipment": 2.0}
    neighbours: dict[str, list[str]] = {}
    for e in edges:
        neighbours.setdefault(e.source, []).append(e.target)
        neighbours.setdefault(e.target, []).append(e.source)

    def spread(ids):
        n = len(ids)
        for i, node_id in enumerate(ids):
            nodes[node_id].y = 1.0 - (i + 0.5) / n if n else 0.5

    claims = [n.id for n in nodes.values() if n.kind == "claim"]
    right = [n.id for n in nodes.values() if n.kind == "event"] + [
        n.id for n in nodes.values() if n.kind == "equipment"
    ]
    docs = [n.id for n in nodes.values() if n.kind == "document"]
    spread(claims)
    spread(right)
    for _ in range(2):
        def bary(node_id, pool):
            ys = [nodes[m].y for m in neighbours.get(node_id, []) if m in pool]
            return sum(ys) / len(ys) if ys else 0.5
        docs.sort(key=lambda d: -bary(d, set(claims) | set(right)))
        spread(docs)
        claims.sort(key=lambda c: -bary(c, set(docs)))
        spread(claims)
    for node in nodes.values():
        node.x = columns[node.kind]


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

def corpus_rows(chunks, result=None) -> list[dict]:
    """One row per corpus chunk (real chunker output), annotated with how the
    active investigation used it (if any)."""
    retrieved = {e.chunk_id: e for e in U._evidence_items(result)} if result is not None else {}
    by_claim = claims_by_chunk(result) if result is not None else {}
    by_event = events_by_chunk(result) if result is not None else {}
    direct = U._direct_answer_ids(result) if result is not None else set()
    rows = []
    for chunk in chunks:
        item = retrieved.get(chunk.chunk_id)
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "document": chunk.filename,
                "title": U._document_title(chunk.filename),
                "relative_path": chunk.relative_path,
                "source_type": lane_for_path(chunk.relative_path),
                "retrieved": item is not None,
                "score": float(getattr(item, "retrieval_score", 0.0) or 0.0) if item else None,
                "channels": list(getattr(item, "retrieval_channels", []) or []) if item else [],
                "retrieved_for": list(getattr(item, "retrieved_for", []) or []) if item else [],
                "claims": by_claim.get(chunk.chunk_id, []),
                "events": by_event.get(chunk.chunk_id, []),
                "in_answer": chunk.chunk_id in direct,
                "content": chunk.content,
            }
        )
    return rows


def evidence_rows(result) -> list[dict]:
    """Rows for the passages this investigation retrieved (no corpus needed)."""
    by_claim = claims_by_chunk(result)
    by_event = events_by_chunk(result)
    direct = U._direct_answer_ids(result)
    rows = []
    for item in sorted(U._evidence_items(result), key=lambda e: -float(e.retrieval_score or 0.0)):
        rows.append(
            {
                "chunk_id": item.chunk_id,
                "document": item.filename,
                "title": U._document_title(item.filename),
                "relative_path": item.relative_path,
                "source_type": lane_for_path(item.relative_path),
                "retrieved": True,
                "score": float(item.retrieval_score or 0.0),
                "channels": list(getattr(item, "retrieval_channels", []) or []),
                "retrieved_for": list(getattr(item, "retrieved_for", []) or []),
                "claims": by_claim.get(item.chunk_id, []),
                "events": by_event.get(item.chunk_id, []),
                "in_answer": item.chunk_id in direct,
                "content": item.content,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Corpus constellation (Command Center hero, no investigation needed)
# ---------------------------------------------------------------------------

_HEADER_ID = re.compile(
    r"^(?:report\s+no|doc(?:ument)?\s+no|form\s+no|ref|eco\s+no|wo\s+no|proposal\s+no)\s*[:.]\s*"
    r"([A-Z][A-Z0-9][A-Z0-9\-./]{2,})",
    re.IGNORECASE,
)


def document_record_ids(documents) -> dict[str, str]:
    """filename -> the record ID its own header declares (first 15 lines).
    IDs declared by more than one document (e.g. every SOP revision is
    SOP-CURE-07) are dropped, because a reference to them cannot be tied to
    one document."""
    found: dict[str, str] = {}
    for doc in documents:
        for line in str(doc["content"]).splitlines()[:15]:
            match = _HEADER_ID.match(line.strip())
            if match:
                found[doc["filename"]] = match.group(1).rstrip(".,;")
                break
    counts: dict[str, int] = {}
    for rid in found.values():
        counts[rid] = counts.get(rid, 0) + 1
    return {f: rid for f, rid in found.items() if counts[rid] == 1}


def corpus_graph(documents) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Real relationships across the whole corpus, from the text alone:

    * document -> document "references": B's text contains the record ID
      that A's own header declares (e.g. 12_PX400_Qualification cites
      ECO-2011-184, the ID in 04_ECO_2011_184's header);
    * document -> equipment "mentions": a Press-N / PX-N identifier appears
      in the document text.

    Documents sit on a ring grouped by source type; equipment in the centre.
    Positions are presentation only."""
    import math

    record_ids = document_record_ids(documents)
    owner = {rid: f for f, rid in record_ids.items()}
    ordered = sorted(
        documents,
        key=lambda d: (LANE_ORDER.index(lane_for_path(d["relative_path"]))
                       if lane_for_path(d["relative_path"]) in LANE_ORDER else 99, d["filename"]),
    )
    nodes: dict[str, GraphNode] = {}
    edges: dict[tuple[str, str], GraphEdge] = {}
    n = max(1, len(ordered))
    for i, doc in enumerate(ordered):
        angle = 2 * math.pi * i / n - math.pi / 2
        lane = lane_for_path(doc["relative_path"])
        rid = record_ids.get(doc["filename"], "")
        nodes[f"doc:{doc['filename']}"] = GraphNode(
            f"doc:{doc['filename']}", "document", U._document_title(doc["filename"]),
            f"{lane} · {doc['relative_path']}" + (f" · record {rid}" if rid else ""),
            [], lane, doc["filename"], x=math.cos(angle), y=math.sin(angle),
        )
    for doc in ordered:
        src = f"doc:{doc['filename']}"
        text = str(doc["content"])
        for rid, target_file in owner.items():
            if target_file == doc["filename"] or rid not in text:
                continue
            edges[(src, f"doc:{target_file}")] = GraphEdge(src, f"doc:{target_file}", "references", [])
        for key, surface in equipment_mentions(text).items():
            node_id = f"equip:{key}"
            if node_id not in nodes:
                nodes[node_id] = GraphNode(node_id, "equipment", surface, "Equipment identifier named in the corpus",
                                           [], "equipment", key)
            edges.setdefault((src, node_id), GraphEdge(src, node_id, "mentions", []))
    equipment = sorted((nd for nd in nodes.values() if nd.kind == "equipment"),
                       key=lambda nd: -sum(1 for e in edges.values() if e.target == nd.id))
    for i, nd in enumerate(equipment):
        angle = 2 * math.pi * i / max(1, len(equipment)) + math.pi / 4
        radius = 0.0 if i == 0 else 0.38
        nd.x, nd.y = radius * math.cos(angle), radius * math.sin(angle)
        mentions = sum(1 for e in edges.values() if e.target == nd.id)
        nd.detail = f"Named in {mentions} document{'s' if mentions != 1 else ''} of the corpus"
    return list(nodes.values()), list(edges.values())


def ego_subgraph(nodes, edges, focus_id, hops: int = 1):
    """The focus node and everything within `hops` recorded links of it,
    re-laid out (copies; the cached full graph is not modified). Returns the
    full graph unchanged when the focus is unknown."""
    from dataclasses import replace as _replace

    ids = {n.id for n in nodes}
    if focus_id not in ids:
        return nodes, edges
    keep = {focus_id}
    frontier = {focus_id}
    for _ in range(max(1, hops)):
        nxt = set()
        for e in edges:
            if e.source in frontier and e.target not in keep:
                nxt.add(e.target)
            if e.target in frontier and e.source not in keep:
                nxt.add(e.source)
        keep |= nxt
        frontier = nxt
    sub_nodes = {n.id: _replace(n, chunk_ids=list(n.chunk_ids)) for n in nodes if n.id in keep}
    sub_edges = [e for e in edges if e.source in keep and e.target in keep]
    _layout(sub_nodes, sub_edges)
    return list(sub_nodes.values()), sub_edges
