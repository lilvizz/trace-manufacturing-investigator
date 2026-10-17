"""4 · Evidence Explorer: retrieved passages and the full corpus."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from app import ui as U
from app.command_center import components as C
from app.command_center import data as D

_CHANNELS = {"bm25": "BM25", "vector": "vector", "entity": "entity", "carried-forward": "carried forward"}


def _document_html(full_text: str, passage: str, search: str) -> str:
    """Full document with the selected passage marked (located by exact
    text match; if it cannot be located, nothing is marked)."""
    text = str(full_text or "")
    snippet = str(passage or "").strip()
    start = text.find(snippet[:200]) if snippet else -1
    if start >= 0:
        end = start + len(snippet)
        before, hit, after = text[:start], text[start:end], text[end:]
        return (U._content_html(before, highlight=search) + '<span class="hit">'
                + U._content_html(hit, highlight=search) + "</span>" + U._content_html(after, highlight=search))
    return U._content_html(text, highlight=search)


def _preview(result, row: dict, search: str, key: str) -> None:
    # The passage is the primary focus: title, then one quiet provenance line
    # (source type, passage ID, file path; all kept), then the reading pane
    # wide on the left and the supporting metadata narrow on the right.
    U._md(
        '<div class="reader-eyebrow">Selected passage</div>'
        f'<div class="doc-head reader-head">{U._escape(row["title"])}</div>'
        f'<div class="reader-prov">{U._escape(row["source_type"])} · passage '
        f'<span class="mono">{U._escape(row["chunk_id"])}</span> · '
        f'<span class="mono">{U._escape(row["relative_path"])}</span></div>'
    )
    text_col, meta_col = st.columns([2.3, 1], gap="large")
    with meta_col:
        channels = " + ".join(_CHANNELS.get(c, c) for c in row["channels"]) or "—"
        meta = [
            ("In this case", "retrieved" + (f" · fused score {row['score']:.4f}" if row["score"] is not None else "")
             if row["retrieved"] else "not retrieved for this investigation"),
            ("Found by", U._escape(channels)),
            ("Cited by", f"{len(row['claims'])} claim(s)" + (" · the direct answer" if row["in_answer"] else "")),
            ("Timeline", f"{len(row['events'])} event(s) extracted from this passage"),
        ]
        U._md('<div class="kv kv-side">' + "".join(f'<span class="k">{k}</span><span class="v">{v}</span>' for k, v in meta) + "</div>")

        if result is not None and (row["claims"] or row["events"]):
            C.section("Referenced by")
            claims = U._claims(result)
            for n, idx in enumerate(row["claims"][:6]):
                claim = claims[idx]
                U._md(f'<div class="li {U._status(claim.stance)[1]}"><div class="k">{D.status_badge(claim, D.evidence_ids(result))}'
                      f'<span>{U._escape(U._claim_type_label(claim))}</span></div>'
                      f'<div class="t">{U._escape(U._display_claim_statement(claim))}</div></div>')
                C.link_button("Open claim", "claims", key=f"ex_cl_{key}_{n}", claim=idx)
            rows, _ = D.timeline_rows(result)
            by_idx = {r.idx: r for r in rows}
            for n, idx in enumerate(row["events"][:6]):
                event = by_idx.get(idx)
                if event is None:
                    continue
                U._md(f'<div class="li blue"><div class="k mono">{U._escape(event.date_label)}</div>'
                      f'<div class="t">{U._escape(event.headline)}</div></div>')
                C.link_button("Open event", "timeline", key=f"ex_ev_{key}_{n}", event=idx)



    with text_col:
        tab_passage, tab_doc, tab_why = st.tabs(["Passage", "Full document", "Why retrieved"])
        body_class = "doc-body mono-body" if row["relative_path"].lower().endswith(".csv") else "doc-body prose-body"
        with tab_passage:
            U._md(f'<div class="{body_class}">{U._content_html(row["content"], highlight=search)}</div>')
        with tab_doc:
            full = C.document_text(row["relative_path"])
            if full is None:
                U._md('<div class="empty-line">The source file could not be read from the corpus folder.</div>')
            else:
                U._md(f'<div class="{body_class} full-doc">{_document_html(full, row["content"], search)}</div>')
                C.note("The highlighted block is the selected passage; the view scrolls to it.")
                import streamlit.components.v1 as components

                # The tab is hidden when this first renders, so a one-off
                # scroll measured nothing. Poll briefly and scroll each
                # document box once, as soon as it is actually visible.
                components.html(
                    f"<!-- {U._escape(row['chunk_id'])} --><script>"
                    "const d = window.parent.document; let n = 0;"
                    "const t = setInterval(() => {"
                    "d.querySelectorAll('.full-doc').forEach((box) => {"
                    "const hit = box.querySelector('.hit');"
                    "if (!hit || box.clientHeight === 0 || box.dataset.scrolledTo === hit.textContent.length + ':' + box.scrollHeight) return;"
                    "box.scrollTop = hit.offsetTop - box.offsetTop - 40;"
                    "box.dataset.scrolledTo = hit.textContent.length + ':' + box.scrollHeight; });"
                    "if (++n > 300) clearInterval(t); }, 200);</script>",
                    height=0,
                )
        with tab_why:
            if row["retrieved_for"]:
                U._md("".join(f'<div class="li gray"><div class="t">{U._escape(q)}</div></div>' for q in row["retrieved_for"][:6]))
            else:
                U._md('<div class="empty-line">Not retrieved for this investigation.</div>')


def render() -> None:
    C.page_header(
        "Evidence explorer",
        "Search the passages this investigation retrieved, or the whole corpus. Select a row to read it "
        "and see which claims and events depend on it.",
    )
    conversation, turn, result, _meta = C.active_case()
    if result is not None:
        C.case_bar(conversation, turn, result)

    rk = C.rkey() if result is not None else "nocase"
    scopes = (["This investigation", "Full corpus"] if result is not None else ["Full corpus"])
    f1, f2, f3, f4 = st.columns([1.5, 2.2, 1.8, 1.4])
    # A link from another page (citation, claim, event, map node) arrives
    # with a new focus nonce: show exactly that passage. Clear any filters
    # that could hide it and pick the scope that contains it.
    nonce = C.focus_nonce()
    if st.session_state.get("_ex_applied_nonce") != nonce:
        st.session_state["_ex_applied_nonce"] = nonce
        f = C.focus()
        if f.get("chunk") or f.get("search"):
            for scope_name in ("This investigation", "Full corpus"):
                st.session_state.pop(f"ex_type_{rk}_{scope_name}", None)
                st.session_state.pop(f"ex_use_{rk}_{scope_name}", None)
            st.session_state[f"ex_q_{rk}"] = f.pop("search", "") or ""
            if f.get("chunk") and result is not None:
                st.session_state[f"ex_scope_{rk}"] = (
                    "This investigation" if f["chunk"] in U._evidence_by_id(result) else "Full corpus"
                )
    with f1:
        scope = st.radio("Scope", scopes, key=f"ex_scope_{rk}", horizontal=False)
    with f2:
        search = st.text_input("Search text, file or passage ID", key=f"ex_q_{rk}", placeholder="e.g. 42.3, PX-400, requalification")

    if scope == "Full corpus":
        try:
            _docs, chunks = C.corpus_chunks()
        except Exception as exc:
            st.error(f"The corpus could not be loaded: {type(exc).__name__}: {exc}")
            return
        rows = D.corpus_rows(chunks, result)
    else:
        rows = D.evidence_rows(result)

    with f3:
        types = sorted({r["source_type"] for r in rows}, key=lambda t: D.LANE_ORDER.index(t) if t in D.LANE_ORDER else 99)
        chosen = st.multiselect("Source type", types, key=f"ex_type_{rk}_{scope}", placeholder="All source types")
    with f4:
        usage_opts = ["All", "Cited", "Not cited"] + (["Retrieved", "Not retrieved"] if scope == "Full corpus" and result is not None else [])
        usage = st.selectbox("Use in case", usage_opts, key=f"ex_use_{rk}_{scope}", disabled=result is None)

    terms = [t for t in search.lower().split() if t]

    def keep(r):
        cited = bool(r["claims"]) or r["in_answer"]
        if chosen and r["source_type"] not in chosen:
            return False
        if usage == "Cited" and not cited or usage == "Not cited" and cited:
            return False
        if usage == "Retrieved" and not r["retrieved"] or usage == "Not retrieved" and r["retrieved"]:
            return False
        hay = f"{r['content']} {r['document']} {r['chunk_id']}".lower()
        return all(t in hay for t in terms)

    view = [r for r in rows if keep(r)]
    focus_chunk = C.focus().get("chunk")
    # The reading pane is drawn ABOVE the table (filled once the table has
    # reported its selection); the table is the secondary, navigational list.
    reader = st.container()
    left = st.container()
    with left:
        C.section("All passages")
        C.note(f"{len(view)} of {len(rows)} passages · {len({r['document'] for r in view})} documents · "
               "select a row to read it above")
        if not view:
            U._md('<div class="empty-card">No passages match. Clear the search or the filters.</div>')
            return
        frame = pd.DataFrame(
            {
                "Document": [r["title"] for r in view],
                "Passage": [r["chunk_id"].rsplit("_", 1)[-1] for r in view],
                "Source type": [r["source_type"] for r in view],
                "Claims": [len(r["claims"]) for r in view],
                "Answer": ["✓" if r["in_answer"] else "" for r in view],
                "Events": [len(r["events"]) for r in view],
                "Score": [r["score"] for r in view],
            }
        )
        default_rows = [i for i, r in enumerate(view) if r["chunk_id"] == focus_chunk]
        df_key = f"ex_df_{rk}_{scope}_{C.focus_nonce()}_{hash((search, tuple(chosen), usage)) & 0xffff}"
        try:
            has_pick = bool(default_rows) or bool(st.session_state[df_key]["selection"]["rows"])
        except Exception:
            has_pick = bool(default_rows)
        state = st.dataframe(
            frame,
            hide_index=True,
            width="stretch",
            # Shorter while a passage is open, so the passage stays the focus.
            height=min(260 if has_pick else 400, 38 + 35 * len(view)),
            on_select="rerun",
            selection_mode="single-row",
            selection_default={"selection": {"rows": default_rows[:1]}} if default_rows else None,
            key=df_key,
            column_config={
                "Document": st.column_config.TextColumn("Document", width="large"),
                "Source type": st.column_config.TextColumn("Source type", width="medium"),
                "Answer": st.column_config.TextColumn("Ans.", width=48, help="Cited by the direct answer"),
                "Score": st.column_config.NumberColumn("Score", format="%.3f", width=64, help="Fused retrieval score (blank = not retrieved)"),
                "Claims": st.column_config.NumberColumn("Claims", width=56, help="Verified claims citing this passage"),
                "Events": st.column_config.NumberColumn("Events", width=56, help="Timeline events extracted from this passage"),
                "Passage": st.column_config.TextColumn("#", width=36, help="Passage number within the document"),
            },
        )
    selected = None
    try:
        picked_rows = state.selection.rows
    except Exception:
        picked_rows = []
    if picked_rows:
        selected = view[picked_rows[0]]
    elif default_rows:
        selected = view[default_rows[0]]
    with reader:
        if selected is None:
            U._md('<div class="empty-card">Select a passage in the table below to read it, open its full source '
                  "document, and follow it to the claims and timeline events that use it.</div>")
        else:
            with st.container(border=True, key="ex_reader"):
                _preview(result, selected, search, key=re.sub(r"\W", "_", selected["chunk_id"]))
