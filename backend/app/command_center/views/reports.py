"""9 · Reports & Investigation History (session scope, honest about it)."""

from __future__ import annotations

import re

import streamlit as st

from app import ui as U
from app.command_center import components as C
from app.command_center import export as X
from app.investigation.investigator import PIPELINE_STAGES


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:48] or "case"


def _audit(result, meta) -> None:
    plan = getattr(result, "plan", None)
    source = getattr(plan, "source", "") or "not recorded"
    U._md(f'<div class="kv"><span class="k">Plan source</span><span class="v">{U._escape(source)}'
          + (f' — Gemini planning failed: {U._escape(plan.fallback_reason)}' if getattr(plan, "fallback_reason", "") else "")
          + "</span></div>")
    timings = (meta or {}).get("stage_seconds") or {}
    if timings:
        labels = dict(PIPELINE_STAGES)
        U._md('<div class="kv">' + "".join(
            f'<span class="k">{U._escape(labels.get(k, k))}</span><span class="v mono">{v:.1f}s</span>'
            for k, v in timings.items()
        ) + f'<span class="k">Total</span><span class="v mono">{(meta or {}).get("total_seconds", 0):.1f}s</span></div>')
    U.render_retrieval_diagnostics(meta)
    U.render_conversation_context(result)
    U.render_investigation_path(result)


def render() -> None:
    C.page_header(
        "Reports & investigation history",
        "Investigations from this session, with downloadable reports built from their real results.",
    )
    U._md(
        '<div class="guard"><b>Kept for this session only.</b> TRACE holds investigations in memory while this '
        "browser session and server are running. Closing the tab or restarting the app clears them; saving "
        "cases across sessions is not available. Download the JSON case file or Markdown report to keep a record.</div>"
    )
    conversations = U._conversations()
    if not conversations:
        U._md('<div class="empty-card"><b>No investigations in this session yet.</b><br>Completed investigations '
              "and their follow-ups appear here with report downloads.</div>")
        C.link_button("Open Command Center", "home", key="rep_home", type="primary")
        return

    pending = bool(st.session_state.get("pending_run"))
    active_conv = st.session_state.get("active_conv")
    for conversation in reversed(conversations):
        turns = conversation["turns"]
        answered = [t for t in turns if t.get("result") is not None]
        latest = answered[-1] if answered else turns[-1]
        with st.container(border=True):
            badge = U._badge(*U._assessment_label(latest["result"])) if latest.get("result") is not None else U._badge("CLARIFICATION", "gray")
            U._md(
                f'<div class="claim-head"><div class="claim-tags">{badge}<span class="mono small-muted">{U._escape(conversation["id"])}</span>'
                f'{"<span class=" + chr(34) + "badge quiet" + chr(34) + ">open case</span>" if conversation["id"] == active_conv else ""}</div>'
                f'<span class="small-muted">{U._escape(turns[0].get("created_at", ""))}</span></div>'
                f'<div class="claim-statement">{U._escape(conversation["title"])}</div>'
                + "".join(
                    f'<div class="small-muted" style="margin-top:4px;">Turn {i}: {U._escape(t["question"])}'
                    f'{" · " + str((t.get("meta") or {}).get("total_seconds")) + "s" if (t.get("meta") or {}).get("total_seconds") else ""}</div>'
                    for i, t in enumerate(turns, start=1)
                )
            )
            row = st.container(horizontal=True)
            c1 = c2 = c3 = row
            with c1:
                if answered and st.button("Open case", key=f"rep_open_{conversation['id']}", disabled=pending):
                    U._activate_turn(conversation, latest)
                    C.goto("overview")
            if answered:
                with c2:
                    st.download_button(
                        "Report (.md)", data=X.case_to_markdown(conversation),
                        file_name=f"trace-{conversation['id']}-{_slug(conversation['title'])}.md",
                        mime="text/markdown", key=f"rep_md_{conversation['id']}", width="content",
                    )
                with c3:
                    st.download_button(
                        "Case file (.json)", data=X.case_to_json(conversation),
                        file_name=f"trace-{conversation['id']}-{_slug(conversation['title'])}.json",
                        mime="application/json", key=f"rep_js_{conversation['id']}", width="content",
                    )

    conversation, turn, result, meta = C.active_case()
    if result is not None:
        C.section("Open case report", "Exactly what the Markdown download contains")
        with st.expander("Preview report", expanded=False):
            st.markdown(X.case_to_markdown(conversation))
        C.section("Audit trail", "How TRACE produced the result shown on the case pages")
        C.safe(_audit, result, meta)

    C.section("System health", "Each state comes from a real check; nothing here is inferred from configuration alone")
    C.render_health(compact=False)

    st.write("")
    if st.button("Clear this session's investigations", key="rep_clear", disabled=pending):
        st.session_state["conversations"] = []
        U._reset_workspace(prefill="")
        st.rerun()
