"""5 · Reasoning & Claim Analysis: every verified claim, grouped by status."""

from __future__ import annotations

import streamlit as st

from app import ui as U
from app.command_center import components as C
from app.command_center import data as D


def _claim_block(result, row: D.ClaimRow, primary_idx, events_by_chunk, focused: bool, rk: str) -> None:
    card_tone = "violet" if row.group == "recommendation" else row.tone
    with st.container(border=True, key=f"claim_{card_tone}_{rk}_{row.idx}"):
        tags = [D.status_badge(row.claim, D.evidence_ids(result)), f'<span class="badge quiet">{U._escape(row.type_label)}</span>']
        if row.idx == primary_idx:
            tags.append('<span class="badge quiet" style="color:var(--trace-accent-strong);">Headline assessment</span>')
        # A recommendation's status badge already says "not a finding" /
        # "suggestion only" (D.claim_status), so no second badge is added.
        if row.requires_human:
            tags.append('<span class="badge amber">Needs human verification</span>')
        if focused:
            tags.append('<span class="badge quiet">Selected</span>')
        U._md(
            f'<div class="claim-head"><div class="claim-tags"><span class="mono small-muted">C{row.idx + 1}</span>{"".join(tags)}</div>'
            f'<span class="small-muted" title="Model confidence after deterministic caps. Not the probability that '
            f'the statement is true.">Calibrated confidence {row.confidence:.0%}</span></div>'
            f'<div class="claim-statement">{U._escape(row.statement)}</div>'
            f'<div class="claim-why">{U._escape(D.claim_why(row.claim, D.evidence_ids(result)))}</div>'
        )
        if row.statement != str(getattr(row.claim, "statement", "")):
            U._md(f'<div class="small-muted" style="margin-top:4px;">The model\'s original wording was replaced because '
                  f'verification did not establish it: “{U._escape(row.claim.statement)}”</div>')
        for note_text in list(getattr(row.claim, "verification_notes", []) or []):
            U._md(f'<div class="vnote"><b>Deterministic verification:</b> {U._escape(note_text)}</div>')
        if row.limits:
            U._md('<div class="small-muted" style="margin-top:6px;"><b>Limits applied:</b> ' + " ".join(U._escape(x) for x in row.limits) + "</div>")
        U._md(f'<div class="small-muted" style="margin:8px 0 2px;"><b>{U._escape(D.evidence_role(row.claim, D.evidence_ids(result)))}</b></div>')
        C.citations(result, row.chunk_ids, key=f"cl_{rk}_{row.idx}")
        event_idxs = sorted({e for c in row.chunk_ids for e in events_by_chunk.get(c, [])})
        with st.container(horizontal=True):
            if row.chunk_ids:
                C.link_button("Evidence", "explorer", key=f"cl_ex_{rk}_{row.idx}", chunk=row.chunk_ids[0])
            if event_idxs:
                C.link_button(f"Timeline ({len(event_idxs)})", "timeline", key=f"cl_tl_{rk}_{row.idx}", event=event_idxs[0])
            if row.chunk_ids:
                C.link_button("On the map", "graph", key=f"cl_map_{rk}_{row.idx}", node=f"claim:{row.idx}")


def render() -> None:
    case = C.require_case(
        "Reasoning & claim analysis",
        "Claims Gemini produced, after deterministic verification against the cited passages and confidence "
        "calibration. Uncited claims drop to Inconclusive with no confidence.",
    )
    if case is None:
        return
    _c, _t, result, _m = case
    rk = C.rkey()
    groups = D.grouped_claims(result)
    findings = U._findings(result)
    if not any(groups.values()) and not findings:
        U._md('<div class="empty-card">No claims or analyzer findings were produced for this investigation.</div>')
        return

    C.note("Calibrated confidence is the model's confidence after TRACE's deterministic caps (for example, current "
           "technical justification is capped at 70% and not-established claims at 30%). It is not the probability "
           "that a statement is true. <b>Not established</b> means the records needed were not found — not that the "
           "statement is false.")
    focus_idx = C.focus().get("claim")
    options = [k for k, rows in groups.items() if rows]
    labels = {k: f"{D.CLAIM_GROUPS[k][0]} · {len(groups[k])}" for k in options}
    chosen = st.pills("Status", options, selection_mode="multi", format_func=lambda k: labels[k],
                      key=f"cl_pills_{rk}", default=None) or options
    primary_idx = D.primary_claim_index(result)
    events_by_chunk = D.events_by_chunk(result)

    if focus_idx is not None and 0 <= int(focus_idx) < len(U._claims(result)):
        row = next(r for rows in groups.values() for r in rows if r.idx == int(focus_idx))
        C.section("Selected claim", f'Opened from another page · <span class="tone-{row.tone}">{U._escape(row.status_label)}</span>')
        _claim_block(result, row, primary_idx, events_by_chunk, True, rk + "_focus")
        if st.button("Clear selection", key=f"cl_clear_{rk}"):
            C.focus().pop("claim", None)
            st.rerun()

    for key in chosen:
        rows = groups.get(key) or []
        if not rows:
            continue
        title, tone, explainer = D.CLAIM_GROUPS[key]
        C.section(title, f"{len(rows)} claim{'s' if len(rows) != 1 else ''} · {U._escape(explainer)}")
        for row in sorted(rows, key=lambda r: -r.confidence):
            _claim_block(result, row, primary_idx, events_by_chunk, False, rk)

    C.section("Deterministic analyzer findings", "Rule-based checks run before reasoning; not produced by the model")
    if not findings:
        U._md('<div class="empty-line">The analyzer reported no findings.</div>')
    for n, finding in enumerate(findings):
        st.markdown(U._finding_card_html(result, finding), unsafe_allow_html=True)
