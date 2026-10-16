"""7 · Investigation Copilot: contextual follow-ups through the full pipeline.

Each follow-up calls Investigator.run(question, context=ConversationContext)
via app.ui._run_pending: fresh planning and retrieval, re-verification and
calibration. Earlier answers are passed only as labelled not-evidence
context (app/investigation/conversation.py) and can never be cited.
"""

from __future__ import annotations

import streamlit as st

from app import ui as U
from app.command_center import components as C
from app.command_center import data as D


PLAN_LABELS = {
    "gemini": "plan by Gemini",
    "deterministic-narrow": "focused lookup plan (deterministic)",
    "deterministic": "deterministic plan",
    "deterministic-fallback": "deterministic fallback plan (Gemini planning failed)",
}


def _open_turn(conversation, turn, page: str, key: str, label: str, pending: bool, **focus_kw) -> None:
    if st.button(label, key=key, disabled=pending):
        U._activate_turn(conversation, turn)
        st.session_state[C.FOCUS_KEY] = {}
        C.goto(page, **focus_kw)


def _answer(conversation, turn, index: int, active_id, pending: bool) -> None:
    result = turn.get("result")
    with U._chat_message("user"):
        U._md(f'<div class="thread-turn-tag">Turn {index} · {U._escape(turn["kind"].replace("-", " "))}'
              f'{" · " + U._escape(turn.get("created_at", "")) if turn.get("created_at") else ""}</div>'
              f'<div class="cop-q">{U._escape(turn["question"])}</div>')
    with U._chat_message("assistant"):
        if result is None:
            U._md('<div class="thread-turn-tag">Clarification needed · no investigation was run</div>'
                  f'<div class="cop-a">{U._escape(turn.get("message", ""))}</div>')
            return
        label, tone = U._assessment_label(result)
        answer = U._clean_direct_answer(getattr(result.report, "direct_answer", "") or "")
        meta = turn.get("meta") or {}
        run = meta.get("retrieval_run") or {}
        resolution = getattr(result, "resolution", None)
        context = ""
        if resolution is not None and getattr(resolution, "used_context", False):
            anchors = ", ".join(getattr(resolution, "anchors", ()) or ()) or "the previous question"
            context = f" · references resolved against {U._escape(anchors)}"
        fresh = run.get("passages", len(U._evidence_items(result))) - run.get("carried_passages", 0)
        if run.get("vector_failures"):
            vector_note = f" · vector search failed in {run['vector_failures']} of {run.get('vector_calls', 0)} calls (BM25 + entity used)"
        elif run.get("vector_calls"):
            vector_note = f" · {run.get('vector_passages', 0)} ranked by vector search"
        else:
            vector_note = " · vector search off"
        plan = getattr(getattr(result, "plan", None), "source", "")
        shown = ' <span class="badge quiet">Shown on case pages</span>' if turn["id"] == active_id else ""
        U._md(
            f'{U._badge(label, tone)}{shown}'
            f'<div class="cop-a">{U._escape(answer or "The evidence retrieved for this question does not establish an answer.")}</div>'
            f'<div class="cop-meta">{fresh} passages retrieved fresh{U._escape(vector_note)}'
            f'{" · " + str(run.get("carried_passages")) + " cited passages carried forward and re-verified" if run.get("carried_passages") else ""}'
            f'{" · " + U._escape(PLAN_LABELS.get(plan, plan)) if plan else ""}'
            f'{" · " + str(meta.get("total_seconds")) + " s" if meta.get("total_seconds") is not None else ""}{context}</div>'
        )
        ids = list(getattr(result.report, "direct_answer_evidence_chunk_ids", []) or [])
        if ids:
            C.citations(result, ids, key=f"cop_{turn['id']}")
        groups = D.grouped_claims(result)
        unresolved = U._unresolved(result)
        cells = [
            (len(groups["supported"]), "established by records"),
            (len(groups["context"]) + len(groups["contradicted"]), "context or conflicting"),
            (len(groups["not_established"]) + len(groups["inconclusive"]) + len(unresolved), "unresolved"),
            (len(groups["recommendation"]), "recommendations (not findings)"),
        ]
        U._md('<div class="turn-break">' + "".join(
            f"<div><b>{n}</b><span>{U._escape(t)}</span></div>" for n, t in cells) + "</div>")
        with st.expander("What this turn established, left open and suggests"):
            for key in ("supported", "contradicted", "context", "not_established", "inconclusive", "recommendation"):
                for row in groups[key]:
                    U._md(f'<div class="tl-row"><span class="d">{D.status_badge(row.claim, D.evidence_ids(result))}</span>'
                          f'<span class="t">{U._escape(row.type_label)} — {U._escape(row.statement)}</span></div>')
            for q in unresolved:
                U._md(f'<div class="tl-row"><span class="d"><span class="badge quiet">Open question</span></span>'
                      f'<span class="t">{U._escape(q)} <span class="small-muted">(raised by reasoning, not verified)</span></span></div>')
        with st.expander(f"Evidence retrieved for this turn ({len(U._evidence_items(result))} passages)"):
            for item in sorted(U._evidence_items(result), key=lambda e: -float(e.retrieval_score or 0))[:12]:
                channels = " + ".join(getattr(item, "retrieval_channels", []) or []) or "—"
                U._md(f'<div class="tl-row"><span class="d mono">{U._escape(item.chunk_id.rsplit("_", 1)[-1])}</span>'
                      f'<span class="t">{U._escape(U._document_title(item.filename))} · '
                      f'<span class="small-muted">{U._escape(channels)}</span></span></div>')
            _open_turn(conversation, turn, "explorer", f"cop_ev_{turn['id']}", "Explore this turn's evidence", pending)
        with st.container(horizontal=True):
            _open_turn(conversation, turn, "overview", f"cop_ov_{turn['id']}", "Overview", pending)
            _open_turn(conversation, turn, "claims", f"cop_cl_{turn['id']}", "Claims", pending)
            _open_turn(conversation, turn, "timeline", f"cop_tl_{turn['id']}", "Timeline", pending)


def render() -> None:
    C.page_header(
        "Investigation copilot",
        "Ask follow-up questions about the open case. Each one is planned, retrieved and verified from the "
        "corpus again; earlier answers only help TRACE understand what you are referring to.",
    )
    request = st.session_state.get("pending_run")
    pending = bool(request)
    conversation, turn, result, _meta = C.active_case()

    U._md(
        '<div class="guard"><b>Evidence rule.</b> Earlier TRACE answers are never treated as evidence. '
        "Only passages retrieved for the new question, or the exact corpus passages a previous turn cited "
        "(re-checked like any other passage), can be cited. A question that refers back to something with "
        "no open case is answered with a clarification request, not a guess.</div>"
    )
    try:
        _, meta = U.load_pipeline()
        status = C.health_rows(meta)
    except Exception as exc:
        status = C.health_rows(None, exc)
    blocking = [r for r in status if r[1] == "FAILED" and r[0] in ("Pipeline startup", "Gemini API key")]
    for name, _state, why in blocking:
        U._md(f'<div class="trace-error"><div class="trace-error-title">{U._escape(name)} failed</div>'
              f'<div class="trace-error-copy">{U._escape(why)} Follow-ups cannot run until this is fixed.</div></div>')
    U._render_last_error()

    if conversation is None:
        U._md('<div class="empty-card"><b>No case is open.</b><br>Ask a question below to start a new '
              "investigation; it runs on the Command Center and opens on the Overview.</div>")
        question = st.chat_input("Ask a new investigation question", key="cop_new_input", disabled=pending)
        if question and question.strip() and U._request_run(question):
            C.goto("home")
        return

    if result is not None:
        C.case_bar(conversation, turn, result)
    active_id = st.session_state.get("active_id")
    for index, item in enumerate(conversation["turns"], start=1):
        _answer(conversation, item, index, active_id, pending)

    if pending and request.get("mode") == "follow-up":
        st.chat_input("Follow-up running… input is disabled until it completes", key="followup_input", disabled=True)
        U._run_pending(request)
        return

    anchor = result if result is not None else None
    suggestions = U._unresolved(anchor)[:3] if anchor is not None else []
    if suggestions:
        C.section("Suggested follow-ups", "Open questions TRACE reported for the turn shown on case pages")
        box = U._keyed_container("suggestions")
        for n, suggestion in enumerate(suggestions):
            if box.button(suggestion, key=f"cop_sug_{C.rkey()}_{n}", width="stretch", disabled=pending):
                if U._request_run(suggestion, mode="follow-up", conv_id=conversation["id"]):
                    st.rerun()

    followup = st.chat_input("Ask a follow-up about this investigation", key="followup_input", disabled=pending)
    if followup and followup.strip():
        if U._request_run(followup, mode="follow-up", conv_id=conversation["id"]):
            st.rerun()
