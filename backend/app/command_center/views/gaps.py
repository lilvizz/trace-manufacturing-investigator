"""8 · Gaps & Recommended Actions."""

from __future__ import annotations

import streamlit as st

from app import ui as U
from app.command_center import components as C
from app.command_center import data as D

_PROVENANCE = {
    "verified_gap": "Verified by a deterministic rule over the retrieved passages.",
    "contradiction": "Conflicts reported by the rule-based analyzer or by reasoning claims with verified citations.",
    "claimed_gap": "Reasoning claim whose citations were verified; the statement it tested was not established.",
    "open_question": "Raised by the reasoning step. Not verified; investigate it to get evidence.",
    "recommendation": "Suggested action from TRACE reasoning. A suggestion, not a verified finding.",
}


_WHY = {
    "contradiction": "Why it matters: conflicting records must be resolved before either can be relied on.",
    "verified_gap": "Why it matters: a change happened, but no record re-justifies the parameter for the changed configuration.",
    "claimed_gap": "Why it matters: the decision rests on something the records do not show. Not established is not the same as false.",
    "open_question": "Why it matters: answering it could close or confirm a gap; it has not been investigated yet.",
    "recommendation": "Why it matters: a possible next step for an engineer to decide on under change control.",
}


def _badge_for(item) -> str:
    if item.kind == "verified_gap" or (item.kind == "contradiction" and item.claim_idx is None):
        return '<span class="badge blue">deterministic</span>'
    if item.kind == "recommendation":
        label = {"procedure_referenced": "procedure referenced · not a finding",
                 "identifier_overlap": "related records cited · not a finding"}.get(item.citation_check)
        return (f'<span class="badge violet">{label}</span>' if label
                else '<span class="badge quiet">not verified</span>')
    if item.verified:
        return '<span class="badge green">citations verified</span>'
    if item.kind in ("open_question", "recommendation"):
        return '<span class="badge quiet">not verified</span>'
    return ""


def render() -> None:
    case = C.require_case(
        "Gaps & recommended actions",
        "What the evidence does not establish, where it conflicts, and what to do next — each item labelled "
        "by where it came from.",
    )
    if case is None:
        return
    conversation, _t, result, _m = case
    rk = C.rkey()
    register = D.gap_register(result)
    counts = {k: sum(1 for i in register if i.kind == k) for k in D.REGISTER_KINDS}
    U._md('<div class="stat-strip">' + "".join(
        f'<div><div class="n tone-{tone}">{counts[k]}</div><div class="t">{U._escape(title)}</div></div>'
        for k, (title, tone) in D.REGISTER_KINDS.items()
    ) + "</div>")
    U._md('<div class="authority-note" style="margin-top:12px;"><strong>Decision support only.</strong> TRACE does '
          "not approve, authorize or change production parameters; the responsible engineer decides under the "
          "plant's change-control process.</div>")
    if not register:
        U._md('<div class="empty-card"><b>Nothing to act on.</b><br>No contradictions, evidence gaps, open questions '
              "or recommendations were reported for this question.</div>")
        return

    pending = bool(st.session_state.get("pending_run"))
    first_open = True
    for kind, (title, tone) in D.REGISTER_KINDS.items():
        items = [i for i in register if i.kind == kind]
        if not items:
            continue
        with st.expander(f"{title} · {len(items)}", expanded=first_open):
            first_open = False
            C.note(U._escape(_PROVENANCE[kind]) + " " + U._escape(_WHY.get(kind, "")))
            for n, item in enumerate(items):
                _item(result, conversation, item, kind, tone, n, rk, pending)


def _item(result, conversation, item, kind, tone, n, rk, pending) -> None:
    with st.container(border=True):
        U._md(
            f'<div class="li {tone}" style="margin:0;border-top:0;border-right:0;border-bottom:0;background:transparent;">'
            f'<div class="k"><span>{U._escape(item.origin)}</span>'
            f'{_badge_for(item)}</div>'
            f'<div class="t">{U._escape(item.statement)}</div></div>'
        )
        if item.chunk_ids:
            C.citations(result, item.chunk_ids, key=f"gp_{rk}_{kind}_{n}", max_items=4)
        with st.container(horizontal=True):
            if item.claim_idx is not None:
                C.link_button("Open claim", "claims", key=f"gp_cl_{rk}_{kind}_{n}", claim=item.claim_idx)
            elif item.chunk_ids:
                C.link_button("Open evidence", "explorer", key=f"gp_ex_{rk}_{kind}_{n}", chunk=item.chunk_ids[0])
            if kind in ("open_question", "verified_gap", "claimed_gap"):
                if st.button("Investigate as follow-up", key=f"gp_fu_{rk}_{kind}_{n}", disabled=pending,
                             help="Runs the full pipeline for this question in the Copilot, with fresh retrieval"):
                    question = item.statement if kind == "open_question" else (
                        f"What evidence addresses this gap: {item.statement}"
                    )
                    if U._request_run(question, mode="follow-up", conv_id=conversation["id"]):
                        C.goto("copilot")
