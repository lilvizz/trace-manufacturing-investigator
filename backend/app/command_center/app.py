"""Command Center shell: page registry, persistent navigation, run routing."""

from __future__ import annotations

import streamlit as st

from app import ui as U
from app.command_center import components as C
from app.command_center import data as D
from app.command_center.theme import CC_CSS, CC_CSS_2
from app.command_center.views import (
    claims,
    copilot,
    explorer,
    gaps,
    graph,
    home,
    overview,
    reports,
    timeline,
)

# key, title, material icon, url path, render function, nav group
PAGE_SPECS = (
    ("home", "Command Center", ":material/radar:", "command-center", home.render, "Command"),
    ("overview", "Investigation Overview", ":material/fact_check:", "overview", overview.render, "Case"),
    ("timeline", "Evidence Timeline", ":material/timeline:", "timeline", timeline.render, "Case"),
    ("explorer", "Evidence Explorer", ":material/plagiarism:", "evidence", explorer.render, "Evidence"),
    ("claims", "Reasoning & Claims", ":material/account_tree:", "claims", claims.render, "Evidence"),
    ("graph", "Relationship Map", ":material/hub:", "relationships", graph.render, "Evidence"),
    ("copilot", "Investigation Copilot", ":material/forum:", "copilot", copilot.render, "Act"),
    ("gaps", "Gaps & Actions", ":material/report:", "gaps", gaps.render, "Act"),
    ("reports", "Reports & History", ":material/inventory_2:", "reports", reports.render, "Act"),
)


def _page_fn(key, fn):
    def _run():
        st.session_state["cc_page"] = key
        fn()
    _run.__name__ = f"page_{key}"
    return _run


def _count_suffix(key: str, result) -> str:
    if result is None:
        return ""
    try:
        if key == "timeline":
            return f"  {sum(r.key for r in D.timeline_rows(result)[0])}"
        if key == "explorer":
            return f"  {len(U._evidence_items(result))}"
        if key == "claims":
            return f"  {len(U._claims(result))}"
        if key == "gaps":
            n = sum(1 for i in D.gap_register(result) if i.kind != "recommendation")
            return f"  {n}" if n else ""
    except Exception:
        return ""
    return ""


def _sidebar(current: str) -> None:
    pending = bool(st.session_state.get("pending_run"))
    conversation, turn, result, _meta = C.active_case()
    with st.sidebar:
        U._md(
            '<div class="trace-brand"><div class="trace-mark">T</div><div>'
            '<div class="trace-wordmark">TRACE</div>'
            '<div class="trace-caption">Investigation Command Center</div></div></div>'
        )
        if st.button("New investigation", key="sb_new_cc", width="stretch",
                     type="primary", disabled=pending, icon=":material/add:"):
            U._reset_workspace(prefill="")
            st.session_state[C.FOCUS_KEY] = {}
            C.goto("home")

        if result is not None:
            label, tone = U._assessment_label(result)
            n_turns = len(conversation["turns"])
            U._md(
                f'<div class="case-pin">{U._badge(label, tone)}'
                f'<div class="q">{U._escape(conversation["title"])}</div>'
                f'<div class="m"><span class="mono">{U._escape(conversation["id"])}</span> · '
                f'{n_turns} turn{"s" if n_turns != 1 else ""} · this session</div></div>'
            )
        else:
            U._md('<div class="case-pin empty">No open case. Ask a question in the Command Center.</div>')

        group = None
        for key, title, icon, _url, _fn, nav_group in PAGE_SPECS:
            if nav_group != group:
                group = nav_group
                U._md(f'<div class="nav-group">{U._escape(group)}</div>')
            st.page_link(C.PAGES[key], label=title + _count_suffix(key, result), icon=icon)

        U._md('<div class="nav-group" style="margin-top:18px;">System health</div>')
        C.render_health(compact=True)
        U._md(
            '<div class="side-note" style="margin-top:12px;">Decision support only. TRACE reports what '
            "the evidence supports; it does not approve production changes.</div>"
        )


def run() -> None:
    st.set_page_config(
        page_title="TRACE · Investigation Command Center",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="auto",
    )
    U.inject_css()
    st.markdown(CC_CSS, unsafe_allow_html=True)
    st.markdown(CC_CSS_2, unsafe_allow_html=True)

    pages = {
        key: st.Page(_page_fn(key, fn), title=title, icon=icon, url_path=url, default=(key == "home"))
        for key, title, icon, url, fn, _g in PAGE_SPECS
    }
    C.PAGES.clear()
    C.PAGES.update(pages)
    nav = st.navigation(list(pages.values()), position="hidden")
    current = next((k for k, p in pages.items() if p.url_path == nav.url_path), "home")

    # Route queued runs to the page that shows their progress: a new
    # investigation runs on the Command Center, a follow-up in Copilot.
    request = st.session_state.get("pending_run")
    if request:
        target = "copilot" if request.get("mode") == "follow-up" else "home"
        if current != target:
            st.switch_page(pages[target])

    after = st.session_state.pop("cc_after_run", None)
    if after == "new" and current == "home" and C.active_case()[2] is not None:
        st.session_state[C.FOCUS_KEY] = {}
        st.switch_page(pages["overview"])

    _sidebar(current)
    nav.run()
