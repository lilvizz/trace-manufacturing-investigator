"""3 · Evidence Timeline: the full chronology, every event with its source."""

from __future__ import annotations

import streamlit as st

from app import ui as U
from app.command_center import charts as CH
from app.command_center import components as C
from app.command_center import data as D
from app.command_center.views.overview import event_caption

DATE_TYPES = ("Record issue dates", "Dates mentioned in text", "Month-only dates")


def _event_body(result, row, key: str) -> None:
    U._md(f'<div class="small-muted" style="margin-bottom:6px;">{event_caption(row)} · {U._escape(row.lane)} · '
          f'{U._escape(row.filename)}</div>')
    if row.lines:
        U._md("".join(f'<div class="timeline-line">{U._escape(line)}</div>' for line in row.lines[:4]))
    if row.kind == "record":
        U._md(f'<div class="small-muted">Date line in the record: <span class="mono">{U._escape(row.description)}</span></div>')
    if row.related_dates:
        U._md('<div class="small-muted" style="margin-top:6px;"><b>Other dates linked to the same record ID</b> '
              "(may describe different events; shown, never merged):</div>"
              + "".join(f'<div class="timeline-line"><span class="mono">{U._escape(ref)}</span> · '
                        f'{U._escape(label)} · {U._escape(fname)}</div>' for ref, label, fname in row.related_dates))
    by_id = U._evidence_by_id(result)
    for chunk_id in row.chunk_ids[:3]:
        item = by_id.get(chunk_id)
        if item is None:
            U._md(f'<div class="small-muted tone-red">Source passage {U._escape(chunk_id)} is not in the retrieved evidence.</div>')
            continue
        st.markdown(C.passage_html(item, needle=row.description, limit=900), unsafe_allow_html=True)
    with st.container(horizontal=True, vertical_alignment="center"):
        if row.chunk_ids:
            C.link_button("Open source passage", "explorer", key=f"tle_src_{key}", chunk=row.chunk_ids[0])
        if row.claim_idxs:
            C.link_button(f"Claims citing this ({len(row.claim_idxs)})", "claims", key=f"tle_cl_{key}", claim=row.claim_idxs[0])
        else:
            U._md('<div class="small-muted">No verified claim cites this source passage.</div>')


def render() -> None:
    case = C.require_case(
        "Evidence timeline",
        "Every date here was written in a retrieved record, shown exactly as written. TRACE does not infer or "
        "estimate dates; month-only and ambiguous dates are labelled.",
    )
    if case is None:
        return
    _c, _t, result, _m = case
    rows, total = D.timeline_rows(result)
    if not rows:
        U._md('<div class="empty-card"><b>No chronology available.</b><br>None of the retrieved passages '
              "contains an explicit date TRACE can parse.</div>")
        return
    rk = C.rkey()
    lanes = [lane for lane in D.LANE_ORDER if lane in {r.lane for r in rows}]
    years = sorted({r.date.year for r in rows if hasattr(r.date, "year")})
    f1, f2, f3 = st.columns([1.6, 1.6, 1.4])
    with f1:
        search = st.text_input("Search events, equipment or sources", key=f"tl_q_{rk}",
                               placeholder="e.g. PX-400, requalification, SOP")
    with f2:
        chosen = st.multiselect("Source type", lanes, key=f"tl_lane_{rk}", placeholder="All source types")
    with f3:
        kinds = st.multiselect("Date type", DATE_TYPES, key=f"tl_kind_{rk}", placeholder="All date types")
    g1, g2, g3 = st.columns([3, 1.1, 1.1])
    with g1:
        span = (st.slider("Years", years[0], years[-1], (years[0], years[-1]), key=f"tl_years_{rk}")
                if len(years) > 1 else (years[0], years[0]) if years else (0, 9999))
    with g2:
        cited_only = st.toggle("Supports a conclusion", key=f"tl_cited_{rk}",
                               help="Only events extracted from passages cited by the answer or a verified claim")
    with g3:
        key_only = st.toggle("Key events only", value=True, key=f"tl_keyonly_{rk}",
                             help="Key events: record dates, the condensed narrative events, and at least one "
                                  "event for every year that has a dated event. Off: every well-formed extracted event.")

    def kind_of(r):
        if r.precision == "month":
            return DATE_TYPES[2]
        return DATE_TYPES[0] if r.issued else DATE_TYPES[1]

    terms = [t for t in search.lower().split() if t]
    view = [
        r for r in rows
        if (not key_only or r.key)
        and (not chosen or r.lane in chosen)
        and (not kinds or kind_of(r) in kinds)
        and (not cited_only or r.cited)
        and (not hasattr(r.date, "year") or span[0] <= r.date.year <= span[1])
        and all(t in f"{r.headline} {r.description} {' '.join(r.lines)} {r.filename}".lower() for t in terms)
    ]
    chart_key = f"tl_full_{rk}"
    focus_idx = C.chart_pick(chart_key, "event", "idx", "event")
    chart = CH.timeline_chart(view, focus_idx=focus_idx) if view else None
    if chart is not None:
        st.altair_chart(chart, width="stretch", on_select="rerun", selection_mode="event", key=chart_key, theme=None)

    by_idx = {r.idx: r for r in rows}
    focused = by_idx.get(int(focus_idx)) if focus_idx is not None else None
    if focused is not None:
        with st.container(border=True):
            U._md(f'<div class="small-muted">Selected event</div><div class="doc-head">{U._escape(focused.date_label)} · '
                  f'{U._escape(focused.headline)}</div>')
            _event_body(result, focused, key=f"{rk}_sel_{focused.idx}")
            if st.button("Clear selection", key=f"tl_clear_{rk}"):
                C.focus().pop("event", None)
                st.rerun()
        if focused not in view:
            C.note("The selected event is hidden by the current filters.")

    try:
        _docs, chunks = C.corpus_chunks()
    except Exception:
        chunks = None
    undated = D.undated_documents(result, chunks)
    notes = [f"Chronology: {D.chronology_span([r for r in rows if r.key])}.",
             f"Showing {len(view)} of {len(rows)} entries ({sum(r.key for r in rows)} key), built from {total} extracted dated events."]
    if total > len(rows):
        notes.append("Record header lines appear as the record's issue date, sign-offs are merged into their record, "
                     "and garbled extraction fragments are hidden here (all are listed under “All extracted events”).")
    C.note(" ".join(U._escape(n) for n in notes))
    if undated:
        U._md(
            '<div class="small-muted" style="margin:2px 0 10px;"><b>Not on the timeline:</b> '
            f"{len(undated)} retrieved document{'s' if len(undated) != 1 else ''} had no extractable date in the "
            "passages retrieved for this case. "
            + "; ".join(
                U._escape(d["filename"]) + (f" (its header, in a passage not retrieved here, is dated {U._escape(d['header_date'])})"
                                             if d["header_date"] else "")
                for d in undated[:8]
            ) + ("…" if len(undated) > 8 else "") + "</div>"
        )

    if not view:
        U._md('<div class="empty-card">No events match these filters. Clear the search, source type, date type or '
              "year range.</div>")
    year = None
    for row in view:
        if row.year != year:
            year = row.year
            U._md(f'<div class="sec" style="margin:16px 0 4px;"><div class="h mono">{U._escape(year)}</div></div>')
        marker = "● " if row.cited else "○ "
        flag = " · month only" if row.precision == "month" else (" · ambiguous date" if row.ambiguous else "")
        title = f"{marker}{row.date_label}{flag}  ·  {row.headline[:115]}{'…' if len(row.headline) > 115 else ''}"
        with st.expander(title, expanded=(focused is not None and focused.idx == row.idx)):
            _event_body(result, row, key=f"{rk}_{row.idx}")
            if st.button("Show on chart", key=f"tl_pick_{rk}_{row.idx}"):
                C.focus()["event"] = row.idx
                st.rerun()

    gaps = U._gap_items(result)
    if gaps:
        statement, _ids, origin = gaps[0]
        U._md(
            '<div class="tl-open" style="margin-top:16px;"><div class="tl-year-label tone-amber">Open</div>'
            '<div class="tl-open-body"><div class="tl-open-label">Unresolved after the last dated record · '
            f'{U._escape(origin)}</div><div class="tl-open-text">{U._escape(statement)}</div></div></div>'
        )
        C.link_button("See all gaps", "gaps", key=f"tl_gaps_{rk}")

    all_events = list(getattr(result.timeline, "events", []) or [])
    with st.expander(f"All {len(all_events)} extracted dated events (unfiltered)"):
        st.markdown(
            "".join(
                '<div class="tl-row">'
                f'<span class="d mono">{U._escape(D.date_label(e.date, getattr(e, "precision", "day")))}</span>'
                f'<span class="t">{U._escape(e.description)} '
                f'<span class="date-prov">{U._escape(getattr(e, "date_text", "") or "")}</span> '
                f'<span class="chip">{U._escape(", ".join(e.evidence_chunk_ids[:2]))}</span></span>'
                "</div>"
                for e in all_events
            ),
            unsafe_allow_html=True,
        )
