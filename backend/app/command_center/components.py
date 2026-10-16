"""Shared Streamlit building blocks for every Command Center page."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from app import ui as U
from app.command_center import data as D

# key -> st.Page; filled by app.run() on every script run.
PAGES: dict = {}

FOCUS_KEY = "cc_focus"


# ---------------------------------------------------------------------------
# Case context
# ---------------------------------------------------------------------------

def active_case():
    """(conversation, turn, result, meta) for the selected investigation turn,
    or (conversation|None, turn|None, None, None) when there is no result."""
    conversation = U._active_conversation()
    turn = U._find_turn(conversation, st.session_state.get("active_id"))
    if turn is None or turn.get("result") is None:
        return conversation, turn, None, None
    meta = turn.get("meta") or {}
    return conversation, turn, turn["result"], meta


def rkey() -> str:
    return U._result_key()


def focus() -> dict:
    return st.session_state.setdefault(FOCUS_KEY, {})


def focus_nonce() -> int:
    return int(st.session_state.get("cc_focus_nonce", 0))


def goto(page_key: str, **new_focus) -> None:
    """Navigate to another page, carrying a focus target (chunk / claim /
    event / node) so the destination opens on the related item."""
    if new_focus:
        current = focus()
        current.update(new_focus)
        st.session_state["cc_focus_nonce"] = focus_nonce() + 1
    page = PAGES.get(page_key)
    if page is not None:
        st.switch_page(page)
    st.rerun()


def link_button(label: str, page_key: str, key: str, help: str | None = None,
                type: str = "secondary", disabled: bool = False, **new_focus) -> None:
    if st.button(label, key=key, help=help, type=type, disabled=disabled):
        goto(page_key, **new_focus)


# ---------------------------------------------------------------------------
# Headers and empty states
# ---------------------------------------------------------------------------

def page_header(title: str, sub: str = "") -> None:
    U._md(
        f'<div class="pg-head"><div><div class="pg-title">{U._escape(title)}</div>'
        + (f'<div class="pg-sub">{sub}</div>' if sub else "")
        + "</div></div>"
    )


def case_bar(conversation, turn, result) -> None:
    """Persistent context line: which investigation (and which turn) every
    case page is showing, with a turn switcher when there are follow-ups."""
    label, tone = U._assessment_label(result)
    turns = [t for t in conversation["turns"] if t.get("result") is not None]
    position = next((i for i, t in enumerate(conversation["turns"], start=1) if t["id"] == turn["id"]), 1)
    U._md(
        '<div class="case-bar">'
        f'<div class="q">Case <b class="mono">{U._escape(conversation["id"])}</b> · '
        f'{U._escape(turn["question"])}</div>'
        f'<div class="r">{U._badge(label, tone)}<span>Turn {position} of {len(conversation["turns"])}</span></div>'
        "</div>"
    )
    if len(turns) > 1:
        options = [t["id"] for t in turns]
        labels = {t["id"]: f"Turn {conversation['turns'].index(t) + 1} · {t['question'][:90]}" for t in turns}
        chosen = st.selectbox(
            "Showing turn",
            options,
            index=options.index(turn["id"]) if turn["id"] in options else len(options) - 1,
            format_func=lambda i: labels[i],
            # Keyed by the active turn so the widget always opens on the turn
            # that is actually active (a completed follow-up is never undone
            # by a stale widget value).
            key=f"turn_pick_{conversation['id']}_{st.session_state.get('cc_page', '')}_{turn['id']}",
        )
        if chosen != turn["id"]:
            U._activate_turn(conversation, next(t for t in turns if t["id"] == chosen))
            st.session_state[FOCUS_KEY] = {}
            st.rerun()


def require_case(title: str, sub: str = ""):
    """Render the page header and return the active case, or an empty state
    with a way forward when no investigation has been run yet."""
    page_header(title, sub)
    conversation, turn, result, meta = active_case()
    if result is None:
        if turn is not None and turn.get("kind") == "clarification":
            U._md(
                '<div class="empty-card"><b>The last question needs clarification before it can be '
                f"investigated.</b><br>{U._escape(turn.get('message', ''))}</div>"
            )
        else:
            U._md(
                '<div class="empty-card"><b>No investigation is open.</b><br>Run an investigation from the '
                "Command Center, or reopen one from Reports &amp; history. This page then shows that "
                "case's real evidence.</div>"
            )
        st.write("")
        c1, c2, _ = st.columns([1, 1, 3])
        with c1:
            link_button("Open Command Center", "home", key=f"empty_home_{title}", type="primary")
        with c2:
            link_button("Reports & history", "reports", key=f"empty_rep_{title}")
        return None
    case_bar(conversation, turn, result)
    return conversation, turn, result, meta


def section(title: str, sub: str = "") -> None:
    U._md(f'<div class="sec"><div class="h">{U._escape(title)}</div><div class="s">{sub}</div></div>')


def note(text: str) -> None:
    U._md(f'<div class="note-line">{text}</div>')


# ---------------------------------------------------------------------------
# Citations: clickable, inspectable, navigable
# ---------------------------------------------------------------------------

def passage_html(item, needle: str = "", limit: int = 1600) -> str:
    text = U._passage_window(item.content, needle) if needle else item.content
    return (
        f'<div class="passage"><div class="passage-head">{U._escape(item.filename)} '
        f'<span class="src">· {U._escape(item.chunk_id)}</span></div>'
        f'<div class="doc-path">{U._escape(item.relative_path)}</div>'
        f'<div class="passage-text">{U._content_html(text, limit)}</div></div>'
    )


def citations(result, chunk_ids, key: str, label_prefix: str = "", max_items: int = 8,
              needle: str = "") -> None:
    """One popover per cited passage: read it in place, then open it in the
    Evidence Explorer. Unresolvable IDs are listed, never hidden."""
    by_id = U._evidence_by_id(result)
    ids = list(dict.fromkeys(chunk_ids or []))
    if not ids:
        U._md('<span class="small-muted tone-red">No valid citation</span>')
        return
    shown = ids[:max_items]
    cols = st.columns(4)
    for n, chunk_id in enumerate(shown):
        item = by_id.get(chunk_id)
        with cols[n % len(cols)]:
            if item is None:
                U._md(f'<span class="chip" title="Not in retrieved evidence">unresolved · {U._escape(chunk_id)}</span>')
                continue
            title = U._document_title(item.filename)
            short = title if len(title) <= 26 else title[:25] + "…"
            with st.popover(f"{label_prefix}[{n + 1}] {short}", width="stretch"):
                st.markdown(passage_html(item, needle), unsafe_allow_html=True)
                cited_by = D.claims_by_chunk(result).get(chunk_id, [])
                events = D.events_by_chunk(result).get(chunk_id, [])
                U._md(
                    f'<div class="small-muted" style="margin-top:6px;">Cited by {len(cited_by)} claim'
                    f'{"s" if len(cited_by) != 1 else ""} · {len(events)} timeline event'
                    f'{"s" if len(events) != 1 else ""} extracted</div>'
                )
                if st.button("Open in Evidence Explorer", key=f"cite_{key}_{n}_{chunk_id}"):
                    goto("explorer", chunk=chunk_id)
    if len(ids) > max_items:
        U._md(f'<div class="small-muted">+{len(ids) - max_items} more cited passages in the Evidence Explorer</div>')


# ---------------------------------------------------------------------------
# Corpus (local files only; no Gemini calls, no re-embedding)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def corpus_chunks():
    from app.config import CORPUS_PATH
    from app.ingestion.chunker import chunk_corpus
    from app.ingestion.loader import load_corpus

    documents = load_corpus(str(CORPUS_PATH))
    return documents, chunk_corpus(documents)


def document_text(relative_path: str) -> str | None:
    documents, _ = corpus_chunks()
    for doc in documents:
        if doc["relative_path"].replace("\\", "/") == str(relative_path).replace("\\", "/"):
            return doc["content"]
    return None


def safe(render: Callable, *args, **kwargs) -> None:
    """Contain a rendering failure to its own section."""
    try:
        render(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        if type(exc).__name__ in {"RerunException", "StopException"}:
            raise
        st.error(f"This section could not be rendered: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Chart selection <-> page focus synchronisation
# ---------------------------------------------------------------------------

def chart_pick(chart_key: str, param: str, field: str, focus_key: str):
    """Apply a NEW click selection from a chart (read from its widget state
    before the chart is drawn) to the page focus, and return the focus.

    Only a changed selection updates the focus, so a focus set elsewhere
    (an event list, a link from another page, a search box) is not
    overwritten by a stale chart selection."""
    from app.command_center import charts as CH

    picked = CH.selected_value(st.session_state.get(chart_key), param, field)
    seen_key = f"_seen_{chart_key}"
    if picked is not None and picked != st.session_state.get(seen_key):
        focus()[focus_key] = picked
    st.session_state[seen_key] = picked
    return focus().get(focus_key)


@st.cache_resource(show_spinner=False)
def corpus_graph_cached():
    documents, _chunks = corpus_chunks()
    return D.corpus_graph(documents)


# ---------------------------------------------------------------------------
# System health (each state backed by a real check)
# ---------------------------------------------------------------------------

STATE_TONE = {"HEALTHY": "green", "DEGRADED": "amber", "FAILED": "red", "NOT VERIFIED": "gray", "AVAILABLE": "blue"}


def health_rows(meta: dict | None, error: Exception | None = None) -> list[tuple[str, str, str]]:
    """(subsystem, STATE, evidence) rows. Definitions:
    HEALTHY = the check for that subsystem ran and passed; DEGRADED = it
    works with a stated limitation; FAILED = the check ran and failed;
    NOT VERIFIED = no check has run yet in this process/session."""
    rows: list[tuple[str, str, str]] = []
    if error is not None:
        _tone, title, message = U._classify_error(error)
        rows.append(("Pipeline startup", "FAILED", f"{title}. {message}"))
        return rows
    meta = meta or {}
    status = meta.get("vector_status") or {}
    index = status.get("index") or {}
    rows.append(("Corpus", "HEALTHY" if meta.get("documents") else "FAILED",
                 f"{meta.get('documents', 0)} documents, {meta.get('chunks', 0)} chunks loaded"))
    if not index.get("present"):
        rows.append(("Embedding index", "FAILED", "Index file not found. Rebuild: python -m app.retrieval.indexer"))
    elif not index.get("loaded") or not index.get("ids_match") or not index.get("dimension"):
        rows.append(("Embedding index", "FAILED", (index.get("error") or "Index does not match the current corpus.")
                     + " Rebuild: python -m app.retrieval.indexer"))
    elif index.get("stale_chunks"):
        rows.append(("Embedding index", "DEGRADED", f"{index.get('stale_chunks')} of {index.get('records')} embeddings "
                     "predate the current text. Rebuild: python -m app.retrieval.indexer"))
    else:
        rows.append(("Embedding index", "HEALTHY", f"{index.get('records')} chunks · {index.get('dimension')}-d · "
                     f"{index.get('model')} · IDs match corpus"))
    state = status.get("state", "off")
    if state == "on":
        rows.append(("Query embedding", "HEALTHY", f"Live probe succeeded at startup ({status.get('probe_dimension')}-d)"))
    elif state == "failed":
        rows.append(("Query embedding", "FAILED", (status.get("reason") or "probe failed")[:220]))
    else:
        rows.append(("Query embedding", "NOT VERIFIED", status.get("reason") or "Probe not run"))
    rows.append(("BM25 + entity retrieval", "AVAILABLE", "Local, built from the corpus at startup"))
    last = st.session_state.get("last_pipeline_run")
    if not status.get("api_key_present"):
        rows.append(("Gemini API key", "FAILED", "GEMINI_API_KEY not found in the environment or project .env"))
    else:
        rows.append(("Gemini API key", "AVAILABLE", "Present (value never displayed)"))
    if last is None:
        rows.append(("Gemini planning & reasoning", "NOT VERIFIED", "No investigation has run in this session"))
        rows.append(("Vector contribution", "NOT VERIFIED", "Measured per investigation; none has run yet"))
    elif last.get("ok"):
        plan = last.get("plan_source") or "unknown"
        state_ = "DEGRADED" if plan == "deterministic-fallback" else "HEALTHY"
        note = f"Last run {last['at']} · {last.get('total_seconds')}s · plan: {plan}"
        if plan == "deterministic-fallback":
            note += f" (Gemini planning failed: {last.get('plan_fallback_reason', '')[:120]})"
        rows.append(("Gemini planning & reasoning", state_, note))
        if last.get("vector_failures"):
            rows.append(("Vector contribution", "DEGRADED", f"{last['vector_failures']} of {last['vector_calls']} "
                         "vector searches failed in the last run; BM25 + entity results were used"))
        elif last.get("vector_passages"):
            rows.append(("Vector contribution", "HEALTHY", f"{last['vector_passages']} passages ranked by the vector "
                         f"channel in the last run ({last['vector_calls']} searches)"))
        else:
            rows.append(("Vector contribution", "DEGRADED" if state == "on" else "NOT VERIFIED",
                         "No passage was ranked by the vector channel in the last run"))
    else:
        rows.append(("Gemini planning & reasoning", "FAILED", f"Last run {last['at']}: {last.get('title')}"))
        rows.append(("Vector contribution", "NOT VERIFIED", "Last run did not complete"))
    rows.append(("Claim verification & calibration", "AVAILABLE", "Deterministic; runs on every investigation"))
    return rows


def render_health(compact: bool = False) -> None:
    try:
        _, meta = U.load_pipeline()
        rows, checked = health_rows(meta), meta.get("checked_at", "")
    except Exception as exc:
        rows, checked = health_rows(None, exc), ""
    if compact:
        worst = next((s for s in ("FAILED", "DEGRADED", "NOT VERIFIED") if any(r[1] == s for r in rows)), "HEALTHY")
        U._md(
            '<div class="health">' + "".join(
                f'<span class="k">{U._escape(name)}</span><span class="v tone-{STATE_TONE[state]}">{state.title()}</span>'
                for name, state, _ in rows
            ) + "</div>"
        )
        return
    U._md(
        '<div class="hl-table">' + "".join(
            f'<div class="hl-row"><span class="hl-name">{U._escape(name)}</span>'
            f'<span class="badge {STATE_TONE[state]}">{state}</span>'
            f'<span class="hl-why">{U._escape(why)}</span></div>'
            for name, state, why in rows
        ) + "</div>"
        + f'<div class="note-line">Startup checks ran {U._escape(checked) or "—"} and are cached for this app '
        "process. Investigation rows reflect the last run in this browser session.</div>"
    )
    if st.button("Re-run startup checks", key=f"health_recheck_{compact}"):
        U.load_pipeline.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Interactive graph block (shared by the Command Center and the Map)
# ---------------------------------------------------------------------------

ZOOM_LEVELS = (1.0, 1.6, 2.6, 4.0)


def graph_controls(key: str, focus_key: str) -> float:
    """Zoom in / out / fit and clear-selection controls. Returns the zoom
    factor. Fit returns to the whole graph; mouse wheel and drag also zoom
    and pan inside the chart, and double-click resets that."""
    level_key = f"_zoom_{key}"
    level = int(st.session_state.get(level_key, 0))
    with st.container(horizontal=True, vertical_alignment="center"):
        if st.button("Zoom in", key=f"zin_{key}", icon=":material/zoom_in:", disabled=level >= len(ZOOM_LEVELS) - 1):
            st.session_state[level_key] = level + 1
            st.rerun()
        if st.button("Zoom out", key=f"zout_{key}", icon=":material/zoom_out:", disabled=level <= 0):
            st.session_state[level_key] = level - 1
            st.rerun()
        if st.button("Fit to view", key=f"zfit_{key}", icon=":material/fit_screen:"):
            st.session_state[level_key] = 0
            st.session_state[f"_chartnonce_{key}"] = st.session_state.get(f"_chartnonce_{key}", 0) + 1
            st.rerun()
        if focus().get(focus_key) is not None and st.button("Clear selection", key=f"zclr_{key}",
                                                             icon=":material/close:"):
            focus().pop(focus_key, None)
            st.session_state[level_key] = 0
            st.rerun()
    return ZOOM_LEVELS[min(level, len(ZOOM_LEVELS) - 1)]


def zoomed_domains(nodes, focus_id, x_domain, y_domain, zoom: float):
    if zoom <= 1.0:
        return x_domain, y_domain
    node = next((n for n in nodes if n.id == focus_id), None)
    cx = node.x if node else (x_domain[0] + x_domain[1]) / 2
    cy = node.y if node else (y_domain[0] + y_domain[1]) / 2
    hx = (x_domain[1] - x_domain[0]) / 2 / zoom
    hy = (y_domain[1] - y_domain[0]) / 2 / zoom
    return (cx - hx, cx + hx), (cy - hy, cy + hy)


def chart_key(key: str) -> str:
    """Widget key that changes on Fit, so Vega's own wheel/drag zoom resets."""
    return f"{key}_{st.session_state.get(f'_chartnonce_{key}', 0)}"


def node_actions(node, result, key: str) -> None:
    """One useful, real action per node type."""
    with st.container(horizontal=True):
        if node.kind == "claim" and node.ref is not None:
            link_button("Open claim analysis", "claims", key=f"na_cl_{key}", claim=int(node.ref))
            if node.chunk_ids:
                link_button("Read cited passage", "explorer", key=f"na_ex_{key}", chunk=node.chunk_ids[0])
        elif node.kind == "event" and node.ref is not None:
            link_button("Open in timeline", "timeline", key=f"na_tl_{key}", event=int(node.ref))
            if node.chunk_ids:
                link_button("Read source passage", "explorer", key=f"na_ex_{key}", chunk=node.chunk_ids[0])
        elif node.kind == "document":
            chunk = node.chunk_ids[0] if node.chunk_ids else next(
                (c.chunk_id for c in corpus_chunks()[1] if c.filename == node.ref), None)
            if chunk:
                link_button("Read in Evidence Explorer", "explorer", key=f"na_doc_{key}", chunk=chunk)
        elif node.kind == "equipment":
            link_button(f"Find passages naming {node.label}", "explorer", key=f"na_eq_{key}", search=node.label)
