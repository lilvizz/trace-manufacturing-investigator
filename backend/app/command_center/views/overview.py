"""2 · Investigation Overview: conclusion first, timeline directly beneath."""

from __future__ import annotations

import streamlit as st

from app import ui as U
from app.command_center import charts as CH
from app.command_center import components as C
from app.command_center import data as D
from app.investigation.models import ClaimType

_STRENGTH = ("Insufficient", "Moderate", "Strong")


def _verdict(result) -> None:
    label, tone = U._assessment_label(result)
    _l, _s, _t, confidence, requires_human = U._assessment(result)
    primary = U._primary_claim(result)
    status_word = U._status(primary.stance)[0] if primary is not None else "Inconclusive"
    strength = U._evidence_strength(confidence)
    level = _STRENGTH.index(strength) + 1 if strength in _STRENGTH else 0
    answer = U._clean_direct_answer(getattr(result.report, "direct_answer", "") or "")
    claim_html = ""
    if primary is not None:
        claim_html = (
            f'<div class="verdict-claim"><b>Assessed claim · {U._escape(U._claim_type_label(primary))}.</b> '
            f"{U._escape(U._display_claim_statement(primary))}</div>"
        )
    meaning = D.claim_why(primary, D.evidence_ids(result)) if primary is not None else (
        "No claim survived verification, so TRACE draws no conclusion.")
    short = _first_sentences(answer, 300)
    bars = "".join(f'<span class="{"on" if i < level else ""}"></span>' for i in range(3))
    U._md(
        f'<div class="verdict {tone}">'
        '<div class="verdict-top"><div style="flex:1;min-width:280px;">'
        f'<div class="verdict-q">Conclusion</div>'
        f'<div class="verdict-status"><span class="word">{U._escape(status_word)}</span>'
        f'{"" if status_word.upper() == label else U._badge(label, tone)}</div>'
        f'<div class="verdict-meaning">{U._escape(meaning)}</div>'
        f'<div class="verdict-answer">{U._escape(short) if short else "TRACE did not produce a direct answer supported by cited evidence for this question."}</div>'
        f"{_story_html(result)}</div>"
        f'<div class="meter"><div class="lbl">Evidence strength</div><div class="val">{U._escape(strength)}</div>'
        f'<div class="bars">{bars}</div><div class="scale"><span>Insufficient</span><span>Moderate</span><span>Strong</span></div>'
        f'<div class="note">Applies to the assessed claim only: {confidence:.0%} after deterministic calibration '
        "caps. It is not the probability that the conclusion is correct. Each claim carries its own confidence on "
        "Reasoning &amp; Claims.</div></div>"
        "</div>"
        + (
            '<div class="verify-strip"><span aria-hidden="true">⚠</span> <strong>Human verification required.</strong> '
            "TRACE does not authorize a production change.</div>"
            if requires_human
            else '<div class="verify-strip ok"><span aria-hidden="true">✓</span> Evidence is sufficient for the stated assessment.</div>'
        )
        + "</div>"
    )
    ids = list(getattr(result.report, "direct_answer_evidence_chunk_ids", []) or [])
    if ids:
        C.note("Sources cited by the answer. Select one to read the exact passage.")
        with st.container(key="cc_cites"):
            C.citations(result, ids, key=f"ov_{C.rkey()}")
    if (short and short != answer) or claim_html:
        with st.expander("Full answer and the exact assessed claim", expanded=False):
            if short != answer:
                U._md(f'<div class="verdict-answer" style="font-size:16px;">{U._escape(answer)}</div>')
            if claim_html:
                U._md(claim_html)
                if primary is not None and U._display_claim_statement(primary) != str(primary.statement):
                    U._md('<div class="vnote">The model stated: “' + U._escape(primary.statement) + '”. '
                          "Verification did not establish it, so TRACE shows the verified wording above.</div>")
                for note_text in list(getattr(primary, "verification_notes", []) or []):
                    U._md(f'<div class="vnote"><b>Deterministic verification:</b> {U._escape(note_text)}</div>')


def _first_sentences(text: str, limit: int) -> str:
    """Leading sentences of the real answer, up to `limit` characters (the
    full answer stays one click away)."""
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    import re as _re

    out = ""
    for sentence in _re.split(r"(?<=[.!?])\s+", text):
        if out and len(out) + len(sentence) + 1 > limit:
            break
        out = f"{out} {sentence}".strip()
    return out if out else text[: limit - 1].rstrip() + "…"


def _story_html(result) -> str:
    """Strongest support, most important open issue and next human action,
    each labelled by where it came from."""
    groups = D.grouped_claims(result)
    by_id = U._evidence_by_id(result)
    cells = []
    supported = sorted(groups["supported"], key=lambda r: -r.confidence)
    if supported:
        top = supported[0]
        doc = by_id.get(top.chunk_ids[0]) if top.chunk_ids else None
        source = U._document_title(doc.filename) if doc is not None else "no resolvable source"
        cells.append(("var(--trace-green)", "Strongest supporting evidence", _first_sentences(top.statement, 170),
                      f"Verified claim · {top.confidence:.0%} · {source}"))
    else:
        cells.append(("var(--trace-border-strong)", "Strongest supporting evidence",
                      "No claim was supported by its cited evidence.", ""))
    register = D.gap_register(result)
    issue = next((i for i in register if i.kind == "contradiction"), None) or \
        next((i for i in register if i.kind in ("verified_gap", "claimed_gap")), None) or \
        next((i for i in register if i.kind == "open_question"), None)
    if issue is not None:
        text = _first_sentences(issue.statement, 170)
        cells.append(("var(--trace-amber)", "Most important open issue", text, issue.origin))
    else:
        cells.append(("var(--trace-border-strong)", "Most important open issue", "No gap or contradiction was reported.", ""))
    # Prefer a recommendation that passed the identifier check; an unchecked
    # one is still shown, labelled as such. Neither is ever a finding.
    recommendations = [i for i in register if i.kind == "recommendation"]
    action = next((i for i in recommendations if i.citation_check), None) or next(iter(recommendations), None)
    if action is not None:
        check = D.RECOMMENDATION_ORIGIN[action.citation_check]
        cells.append(("var(--trace-violet)", "Next human action", _first_sentences(action.statement, 170),
                      f"Suggested by TRACE reasoning · {check} · not a verified finding · "
                      "TRACE does not authorize changes"))
    else:
        question = next((i for i in register if i.kind == "open_question"), None)
        cells.append(("var(--trace-violet)", "Next human action",
                      question.statement if question else "No action was proposed by the evidence.",
                      "Open question raised by reasoning · investigate it to get evidence" if question else ""))
    return '<div class="story">' + "".join(
        f'<div style="--s-tone:{tone}"><div class="k">{U._escape(k)}</div><div class="t">{U._escape(t)}</div>'
        f'<div class="o">{U._escape(o)}</div></div>' for tone, k, t, o in cells) + "</div>"


def event_caption(row) -> str:
    """Date provenance line shared by every timeline view."""
    parts = [f'<span class="date-prov">{U._escape(row.date_text or row.date_label)}</span>']
    parts.append("record issue date" if row.issued else "date mentioned in the text")
    if row.precision == "month":
        parts.append("month only — no day given in the source")
    if row.ambiguous:
        parts.append("ambiguous numeric date, read day-first")
    parts.append("cited in a conclusion" if row.cited else "retrieved, not cited")
    return " · ".join(parts)


def timeline_preview(result, key: str, height: int | None = None, full_link: bool = True) -> None:
    all_rows, total = D.timeline_rows(result)
    rows = [r for r in all_rows if r.key]
    if not rows:
        U._md('<div class="empty-card">No explicit dates were found in the retrieved passages, so no '
              "chronology can be shown for this question.</div>")
        return
    chart_key = f"tl_{key}_{C.rkey()}"
    focus_idx = C.chart_pick(chart_key, "event", "idx", "event")
    chart = CH.timeline_chart(rows, height=height, focus_idx=focus_idx)
    st.altair_chart(chart, width="stretch", on_select="rerun", selection_mode="event", key=chart_key, theme=None)
    by_idx = {r.idx: r for r in all_rows}
    row = by_idx.get(int(focus_idx)) if focus_idx is not None else None
    if row is None:
        C.note(
            f"{U._escape(D.chronology_span(rows))} · {len(rows)} key entries from {total} extracted dated events · "
            "square = record issued, circle = dated "
            "statement, triangle = month-only date · cyan = cited in a conclusion. Select a point to read its source."
        )
        if full_link:
            C.link_button("Open the full timeline", "timeline", key=f"tl_open_{key}")
        return
    with st.container(border=True):
        U._md(
            f'<div class="li {"blue" if row.cited else "gray"}" style="border:0;padding:0;margin:0;background:transparent;">'
            f'<div class="k"><b class="mono">{U._escape(row.date_label)}</b><span>{U._escape(row.lane)}</span></div>'
            f'<div class="t">{U._escape(row.headline)}</div>'
            f'<div class="f">{event_caption(row)}</div></div>'
        )
        C.citations(result, row.chunk_ids[:4], key=f"tlp_{key}_{row.idx}", needle=row.description)
        with st.container(horizontal=True):
            C.link_button("Open in timeline", "timeline", key=f"tl_go_{key}_{row.idx}", event=row.idx)
            if row.claim_idxs:
                C.link_button(f"Related claims ({len(row.claim_idxs)})", "claims",
                              key=f"tl_claims_{key}_{row.idx}", claim=row.claim_idxs[0])
            if st.button("Clear selection", key=f"tl_clear_{key}_{row.idx}"):
                C.focus().pop("event", None)
                st.rerun()


def _stat_strip(result, meta) -> None:
    evidence = U._evidence_items(result)
    docs = len({e.filename for e in evidence})
    run = (meta or {}).get("retrieval_run") or {}
    vector = (
        f"{run.get('vector_passages', 0)} of {run.get('passages', len(evidence))}"
        if run and not run.get("vector_failures") and (meta or {}).get("vector_enabled")
        else ("failed" if run.get("vector_failures") else "off")
    )
    cells = [
        (len(evidence), "passages retrieved"),
        (docs, "source documents"),
        (len(U._claims(result)), "claims after verification"),
        (len(getattr(getattr(result, "timeline", None), "events", []) or []), "dated events extracted"),
        (vector, "passages ranked by vector search"),
    ]
    U._md('<div class="stat-strip">' + "".join(
        f'<div><div class="n">{U._escape(n)}</div><div class="t">{U._escape(t)}</div></div>' for n, t in cells
    ) + "</div>")


def _column_items(title, items, empty, page, key):
    C.section(title)
    if not items:
        U._md(f'<div class="empty-line">{U._escape(empty)}</div>')
        return
    for n, (tone, kicker, text, focus_kw) in enumerate(items):
        U._md(f'<div class="li {tone}"><div class="k">{kicker}</div><div class="t">{U._escape(text)}</div></div>')
        if focus_kw:
            C.link_button("Inspect", page, key=f"ov_{key}_{n}", **focus_kw)


def render() -> None:
    case = C.require_case("Investigation overview", "The conclusion first, then the chronology and the evidence behind it.")
    if case is None:
        return
    _conversation, _turn, result, meta = case
    if st.session_state.pop("just_completed", False) and hasattr(st, "toast"):
        st.toast("Investigation complete")

    _verdict(result)

    C.section("Timeline", "Dates as written in the retrieved records, by source type")
    timeline_preview(result, key="ov", height=None)

    _stat_strip(result, meta)

    C.section("Decision basis", "What the evidence says about each part of the question")
    C.safe(U.render_basis_matrix, result)

    groups = D.grouped_claims(result)
    register = D.gap_register(result)
    rk = C.rkey()

    with st.expander("Supporting and contradictory evidence", expanded=False):
        rows = sorted(groups["supported"], key=lambda r: -r.confidence) + groups["contradicted"]
        if not rows:
            U._md('<div class="empty-line">No claim was supported or contradicted by its cited evidence.</div>')
        for row in rows:
            U._md(f'<div class="li {row.tone}"><div class="k">{D.status_badge(row.claim, D.evidence_ids(result))}'
                  f'<span>{U._escape(row.type_label)} · {row.confidence:.0%}</span></div>'
                  f'<div class="t">{U._escape(row.statement)}</div></div>')
            C.citations(result, row.chunk_ids, key=f"ovs_{rk}_{row.idx}", max_items=4)

    with st.expander("Historical changes and equipment transitions", expanded=False):
        changes = [r for r in D.claim_rows(result)
                   if getattr(r.claim, "claim_type", None) in (ClaimType.CURRENT_CAPABILITY, ClaimType.ADMINISTRATIVE_ACTION)]
        for row in changes:
            U._md(f'<div class="li {row.tone}"><div class="k">{D.status_badge(row.claim, D.evidence_ids(result))}'
                  f'<span>{U._escape(row.type_label)}</span></div><div class="t">{U._escape(row.statement)}</div></div>')
        issued = [r for r in D.timeline_rows(result)[0] if r.issued and D.equipment_mentions(" ".join([r.description] + r.lines))]
        if issued:
            U._md('<div class="small-muted" style="margin:8px 0 4px;"><b>Dated records naming equipment</b></div>'
                  + "".join(f'<div class="tl-row"><span class="d mono">{U._escape(r.date_label)}</span>'
                            f'<span class="t">{U._escape(r.headline)} · {U._escape(", ".join(sorted(set(D.equipment_mentions(" ".join([r.description] + r.lines)).values()))))}</span></div>'
                            for r in issued))
        if not changes and not issued:
            U._md('<div class="empty-line">No equipment or administrative change was recorded for this question.</div>')

    with st.expander("Claim-level reasoning", expanded=False):
        for key, rows in groups.items():
            for row in rows:
                U._md(f'<div class="tl-row"><span class="d">{D.status_badge(row.claim, D.evidence_ids(result))}</span>'
                      f'<span class="t"><b>C{row.idx + 1}</b> · {U._escape(row.type_label)} · {row.confidence:.0%} — '
                      f'{U._escape(row.statement)}</span></div>')
        C.link_button("Open full claim analysis", "claims", key="ov_claims_all")

    with st.expander("Critical evidence gaps", expanded=False):
        critical = [i for i in register if i.kind in ("contradiction", "verified_gap", "claimed_gap")]
        if not critical:
            U._md('<div class="empty-line">No contradictions or evidence gaps were reported.</div>')
        for n, item in enumerate(critical):
            U._md(f'<div class="li {"red" if item.kind == "contradiction" else "amber"}"><div class="k">{U._escape(item.origin)}</div>'
                  f'<div class="t">{U._escape(item.statement)}</div></div>')
            if item.claim_idx is not None:
                C.link_button("Inspect claim", "claims", key=f"ov_gap_{n}", claim=item.claim_idx)
        C.link_button("All gaps & actions", "gaps", key="ov_all_gaps")

    with st.expander("Recommended follow-up actions", expanded=False):
        nxt = [i for i in register if i.kind in ("recommendation", "open_question")]
        if not nxt:
            U._md('<div class="empty-line">No recommendation or open question was produced.</div>')
        for item in nxt:
            U._md(f'<div class="li {"violet" if item.kind == "recommendation" else "gray"}"><div class="k">{U._escape(item.origin)}'
                  f'{" · not verified" if not item.verified else ""}</div><div class="t">{U._escape(item.statement)}</div></div>')
        U._md('<div class="authority-note"><strong>Decision support only.</strong> TRACE does not approve or authorize '
              "production changes; the responsible engineer decides under change control.</div>")
        C.link_button("Plan next steps", "gaps", key="ov_all_next")

    C.section("Go deeper")
    with st.container(horizontal=True):
        for page, label in (("timeline", "Full timeline"), ("explorer", "Evidence explorer"),
                            ("claims", "Claim analysis"), ("graph", "Relationship map"), ("copilot", "Ask a follow-up")):
            st.page_link(C.PAGES[page], label=label)
