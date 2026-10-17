"""1 · Home / Command Center: ask, see the evidence network, resume a case."""

from __future__ import annotations

import streamlit as st

from app import ui as U
from app.command_center import charts as CH
from app.command_center import components as C
from app.command_center import data as D
from app.investigation.investigator import PIPELINE_STAGES
from app.investigation.scenarios import get_scenarios


def _composer(pending: bool) -> None:
    if "pending_question" in st.session_state:
        st.session_state["question_input"] = st.session_state.pop("pending_question")
    st.session_state.setdefault("question_input", "")
    question = st.text_area(
        "Investigation question",
        key="question_input",
        height=104,
        label_visibility="collapsed",
        placeholder="e.g. Why is Press-07 limited to 42.3°C, and is its current technical basis still supported?",
        disabled=pending,
    )
    run = st.button("Investigating…" if pending else "Investigate", type="primary",
                    disabled=pending, key="cc_investigate", icon=":material/search:")
    U._md('<div class="stage-rail">' + "".join(f"<span>{U._escape(label)}</span>" for _k, label in PIPELINE_STAGES)
          + "</div>")
    if run:
        if not question.strip():
            st.error("Enter an investigation question.")
        elif U._request_run(question):
            st.rerun()


def _examples(pending: bool) -> None:
    scenarios = {s.id: s for s in get_scenarios()}
    choice = st.pills(
        "Reference questions from the corpus",
        list(scenarios), format_func=lambda k: scenarios[k].title, selection_mode="single",
        key="cc_example_pills", disabled=pending, wrap=True,
    )
    if choice and st.session_state.get("_cc_example_seen") != choice:
        st.session_state["_cc_example_seen"] = choice
        st.session_state["pending_question"] = scenarios[choice].question
        st.rerun()
    C.note("Selecting one fills the question box. Nothing runs until you choose Investigate.")


def _hero_graph(result) -> None:
    if result is not None:
        nodes, edges = D.relationship_graph(result)
        title = "Open case · how the conclusion connects to its evidence"
        sub = (f"{sum(n.kind == 'claim' for n in nodes)} verified claims · "
               f"{sum(n.kind == 'document' for n in nodes)} documents · {len(edges)} recorded links")
        x_domain, y_domain, label_kinds, base = (-1.0, 3.0), (-0.03, 1.03), ("claim", "event", "equipment"), "home_case"
        focus_key = "node"  # shared with the Relationship Map
        primary = D.primary_claim_index(result)
        anchored = f"_home_anchored_{C.rkey()}"
        if (not st.session_state.get(anchored) and C.focus().get(focus_key) is None and primary is not None
                and any(n.id == f"claim:{primary}" for n in nodes)):
            C.focus()[focus_key] = f"claim:{primary}"  # the assessed claim anchors the story (once per case)
        st.session_state[anchored] = True
    else:
        try:
            nodes, edges = C.corpus_graph_cached()
        except Exception as exc:
            st.error(f"The corpus could not be loaded: {type(exc).__name__}: {exc}")
            return
        refs = sum(e.relation == "references" for e in edges)
        title = "The plant record · documents that cite each other"
        sub = (f"{sum(n.kind == 'document' for n in nodes)} documents · {refs} record cross-references · "
               f"{sum(n.kind == 'equipment' for n in nodes)} equipment identifiers")
        x_domain, y_domain, label_kinds, base = (-1.55, 1.55), (-1.15, 1.15), ("equipment",), "home_corpus"
        focus_key = "home_corpus_node"
    widget = C.chart_key(f"{base}_{C.rkey() if result is not None else 'corpus'}")
    focus_id = C.chart_pick(widget, "node", "id", focus_key)
    U._md(f'<div class="graph-cap"><span class="t">{U._escape(title)}</span><span class="s">{U._escape(sub)}</span></div>')
    zoom = C.graph_controls(base, focus_key)
    xd, yd = C.zoomed_domains(nodes, focus_id, x_domain, y_domain, zoom)
    chart = CH.graph_chart(nodes, edges, height=440, focus_id=focus_id, label_kinds=label_kinds,
                           x_domain=xd, y_domain=yd, inward_labels=result is None)
    if chart is None:
        U._md('<div class="empty-card">No relationships to draw yet.</div>')
        return
    st.altair_chart(chart, width="stretch", on_select="rerun", selection_mode="node", key=widget, theme=None)
    if result is None:
        U._md('<div class="legend">' + "".join(
            f'<span><i style="background:{CH.LANE_COLORS[l]};border-radius:2px"></i>{U._escape(l)}</span>'
            for l in D.LANE_ORDER) + '<span><i style="background:#e7c9a0"></i>Equipment</span>'
            '<span style="color:#7aa7ff">── cites record ID</span><span style="color:#b39b7a">── names equipment</span></div>')
    else:
        U._md('<div class="legend"><span><i style="background:#4fc48b"></i>Claim (colour = status)</span>'
              '<span><i style="background:#d5dde8;border-radius:2px"></i>Document</span>'
              '<span><i style="background:#3cc4d4"></i>Dated event</span><span><i style="background:#e7c9a0"></i>Equipment</span></div>')
    node = next((n for n in nodes if n.id == focus_id), None)
    if node is None:
        C.note("Click a node to anchor the view on it: its direct links light up and everything else fades. "
               "Wheel or the buttons zoom, drag pans, Fit to view resets.")
        return
    near = CH.neighbours(edges, node.id)
    by_id = {n.id: n for n in nodes}
    U._md(
        f'<div class="li {"blue" if node.kind != "equipment" else "amber"}" style="margin-top:4px;">'
        f'<div class="k">{U._escape(node.kind.capitalize())} · {len(near)} direct connection{"s" if len(near) != 1 else ""}</div>'
        f'<div class="t"><b>{U._escape(node.label)}</b> — {U._escape(node.detail[:260])}</div>'
        f'<div class="f">{U._escape(", ".join(sorted(by_id[m].label for m in near if m in by_id))[:360])}</div></div>'
    )
    C.node_actions(node, result, key=f"home_{node.id}")
    if result is not None:
        C.link_button("Explore in the Relationship Map", "graph", key=f"hg_map_{node.id}", node=node.id)


def _corpus_composition() -> None:
    try:
        documents, chunks = C.corpus_chunks()
    except Exception:
        return
    counts: dict[str, int] = {}
    for doc in documents:
        lane = D.lane_for_path(doc["relative_path"])
        counts[lane] = counts.get(lane, 0) + 1
    total = max(1, len(documents))
    lanes = [lane for lane in D.LANE_ORDER if lane in counts]
    U._md(
        '<div class="sec" style="margin-top:6px;"><div class="h">What TRACE searches</div>'
        f'<div class="s">{len(documents)} documents · {len(chunks)} passages, by source type</div></div>'
        '<div class="statusbar">' + "".join(
            f'<span style="width:{100 * counts[l] / total:.1f}%;background:{CH.LANE_COLORS[l]}" title="{U._escape(l)}: {counts[l]}"></span>'
            for l in lanes) + "</div>"
        '<div class="status-legend">' + "".join(
            f'<span><i style="background:{CH.LANE_COLORS[l]}"></i>{U._escape(l)} · {counts[l]}</span>' for l in lanes) + "</div>"
    )


def _claim_status(result) -> None:
    groups = D.grouped_claims(result)
    total = max(1, sum(len(v) for v in groups.values()))
    colours = {"supported": "#4fc48b", "contradicted": "#f07272", "not_established": "#e9a94f",
               "context": "#7aa7ff", "inconclusive": "#9aa7b8", "recommendation": "#a58beb"}
    present = [k for k, v in groups.items() if v]
    U._md(
        '<div class="sec" style="margin-top:6px;"><div class="h">Verified claims</div>'
        f'<div class="s">{sum(len(v) for v in groups.values())} claims after deterministic verification</div></div>'
        '<div class="statusbar">' + "".join(
            f'<span style="width:{100 * len(groups[k]) / total:.1f}%;background:{colours[k]}"></span>' for k in present)
        + '</div><div class="status-legend">' + "".join(
            f'<span><i style="background:{colours[k]}"></i>{U._escape(D.CLAIM_GROUPS[k][0])} · {len(groups[k])}</span>'
            for k in present) + "</div>"
    )
    C.link_button("Open claim analysis", "claims", key="home_claims")


def _recent(pending: bool) -> None:
    C.section("Recent investigations", "This browser session")
    conversations = U._conversations()
    if not conversations:
        U._md('<div class="empty-line">None yet. Investigations stay available on every page until this session '
              "ends; Reports &amp; History lets you download them.</div>")
        return
    for conversation in list(reversed(conversations))[:4]:
        answered = [t for t in conversation["turns"] if t.get("result") is not None]
        latest = answered[-1] if answered else conversation["turns"][-1]
        badge = (U._badge(*U._assessment_label(latest["result"])) if latest.get("result") is not None
                 else U._badge("CLARIFICATION", "gray"))
        U._md(
            f'<div class="recent">{badge}<div class="q" style="margin-top:6px;">{U._escape(conversation["title"])}</div>'
            f'<div class="m"><span class="mono">{U._escape(conversation["id"])}</span> · {len(conversation["turns"])} turn(s)'
            f'{" · " + U._escape(latest.get("created_at", "")) if latest.get("created_at") else ""}</div></div>'
        )
        if latest.get("result") is not None and st.button("Open case", key=f"cc_open_{conversation['id']}", disabled=pending):
            U._activate_turn(conversation, latest)
            C.goto("overview")


def render() -> None:
    request = st.session_state.get("pending_run")
    pending = bool(request)
    conversation, turn, result, _ = C.active_case()

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        U._md(
            '<div class="hero-grid"><div class="hero-kicker">TRACE · manufacturing evidence investigator</div>'
            "<h1>Ask why a manufacturing decision exists.</h1>"
            '<div class="hero-copy">TRACE searches the plant record, rebuilds what changed and when, and separates '
            "what the documents establish from what they do not — every claim checked against its sources.</div></div>"
        )
        st.write("")
        _composer(pending)
        _examples(pending)
    with right:
        C.safe(_hero_graph, result)

    U._render_last_error()
    if pending and request.get("mode") != "follow-up":
        U._run_pending(request)
        return

    if turn is not None and turn.get("kind") == "clarification":
        U.render_clarification(conversation, turn)

    if result is not None:
        label, tone = U._assessment_label(result)
        U._md(
            f'<div class="case-bar" style="margin-top:18px;"><div class="q">Open case · <b>{U._escape(turn["question"])}</b></div>'
            f'<div class="r">{U._badge(label, tone)}</div></div>'
        )
        a, b = st.columns([1.6, 1], gap="large")
        with a:
            from app.command_center.views.overview import timeline_preview

            C.section("Chronology", "Dates as written in the retrieved records")
            C.safe(timeline_preview, result, key="home", height=None, full_link=True)
        with b:
            C.safe(_claim_status, result)
            C.link_button("Open the investigation overview", "overview", key="cc_home_to_case", type="primary")
    else:
        C.safe(_corpus_composition)

    a, b = st.columns([1, 1.2], gap="large")
    with a:
        _recent(pending)
    with b:
        C.section("System health", "Each state comes from a real check")
        try:
            _, meta = U.load_pipeline()
            rows = C.health_rows(meta)
        except Exception as exc:
            rows = C.health_rows(None, exc)
        problems = [r for r in rows if r[1] in ("FAILED", "DEGRADED")]
        unverified = [r for r in rows if r[1] == "NOT VERIFIED"]
        if problems:
            U._md('<div class="trace-error warn"><div class="trace-error-title">'
                  f'{len(problems)} subsystem{"s" if len(problems) != 1 else ""} need attention</div>'
                  '<div class="trace-error-copy">' + "<br>".join(
                      f"<b>{U._escape(n)}</b> · {s.title()} — {U._escape(w)}" for n, s, w in problems) + "</div></div>")
        else:
            U._md(f'<div class="note-line">No failures detected. {len(unverified)} check'
                  f'{"s" if len(unverified) != 1 else ""} not yet verified (they run with the first investigation).</div>')
        with st.expander("All diagnostics", expanded=False):
            C.render_health(compact=False)
