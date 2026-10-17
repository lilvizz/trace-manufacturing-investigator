"""6 · Evidence Relationship Map: only relationships recorded in the result."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app import ui as U
from app.command_center import charts as CH
from app.command_center import components as C
from app.command_center import data as D

KINDS = {"claim": "Claims", "document": "Documents", "event": "Timeline events", "equipment": "Equipment"}


def _node_panel(result, node: D.GraphNode, nodes, edges, rk: str) -> None:
    by_id = {n.id: n for n in nodes}
    kind = KINDS[node.kind][:-1] if node.kind != "equipment" else "Equipment"
    info, conn = st.columns([1.2, 1], gap="large")
    with info:
        U._md(f'<div class="small-muted">{U._escape(kind)}</div><div class="doc-head" style="margin-top:2px;">{U._escape(node.label)}</div>'
              f'<div class="pg-sub" style="margin-top:6px;">{U._escape(node.detail)}</div>')
        if node.chunk_ids:
            U._md('<div class="small-muted" style="margin:10px 0 4px;"><b>Source passages</b></div>')
            C.citations(result, node.chunk_ids, key=f"gm_{rk}_{node.id}", max_items=8)
    with conn:
        links = [(e, e.target if e.source == node.id else e.source) for e in edges if node.id in (e.source, e.target)]
        U._md(f'<div class="small-muted" style="margin:0 0 4px;"><b>{len(links)} connection{"s" if len(links) != 1 else ""}</b></div>')
        for edge, other_id in links[:16]:
            other = by_id.get(other_id)
            if other is None:
                continue
            direction = edge.relation if edge.source == node.id else f"← {edge.relation}"
            U._md(f'<div class="li gray" style="padding:7px 11px;margin-bottom:6px;"><div class="k">{U._escape(direction)} · '
                  f'{U._escape(KINDS[other.kind])} · via <span class="mono">{U._escape(", ".join(edge.chunk_ids[:2]))}</span></div>'
                  f'<div class="t" style="font-size:13px;">{U._escape(other.label)}</div></div>')
        if len(links) > 16:
            U._md(f'<div class="small-muted">+{len(links) - 16} more in the relationship list below</div>')


def render() -> None:
    case = C.require_case(
        "Evidence relationship map",
        "Claims, source documents, dated events and equipment, connected only where the investigation "
        "recorded the link: a claim cites a passage, an event was extracted from it, or an identifier "
        "appears in its text.",
    )
    if case is None:
        return
    _c, _t, result, _m = case
    rk = C.rkey()
    nodes, edges = D.relationship_graph(result)
    present = [k for k in KINDS if any(n.kind == k for n in nodes)]
    if len(edges) < 2:
        U._md('<div class="empty-card"><b>Not enough recorded relationships for a map.</b><br>This result has '
              f"{len(edges)} link(s) between claims, documents and events. The relationship list below shows "
              "what exists; nothing is drawn that the evidence does not record.</div>")
    else:
        primary = D.primary_claim_index(result)
        anchored = f"_map_anchored_{rk}"
        if (not st.session_state.get(anchored) and C.focus().get("node") is None and primary is not None
                and any(n.id == f"claim:{primary}" for n in nodes)):
            C.focus()["node"] = f"claim:{primary}"  # default anchor (once per case): the assessed claim
        st.session_state[anchored] = True
        c1, c2, c3 = st.columns([1.2, 1.3, 1.2])
        with c1:
            mode = st.radio("Show", ["Selected node and direct links", "Two steps out", "Everything"], index=1,
                            key=f"gm_mode_{rk}",
                            help="Two steps out: the selected node, what it links to, and what those link to "
                                 "(e.g. a claim → its documents → their dated events and other claims). "
                                 "Reveal more on demand.")
        with c2:
            show = st.pills("Node types", present, selection_mode="multi", default=present,
                            format_func=lambda k: f"{KINDS[k]} · {sum(1 for n in nodes if n.kind == k)}",
                            key=f"gm_kinds_{rk}") or present
        widget = C.chart_key(f"gm_chart_{rk}")
        focus_id = C.chart_pick(widget, "node", "id", "node")
        with c3:
            options = [""] + [n.id for n in sorted(nodes, key=lambda n: (list(KINDS).index(n.kind), n.label))]
            labels = {n.id: f"{KINDS[n.kind][:-1] if n.kind != 'equipment' else 'Equipment'} · {n.label}" for n in nodes}
            seen = f"_seen_gm_find_{rk}"
            chosen = st.selectbox("Find a node", options, format_func=lambda i: labels.get(i, "Choose a node…"),
                                  key=f"gm_find_{rk}")
            if chosen and chosen != st.session_state.get(seen):
                C.focus()["node"] = chosen
                focus_id = chosen
            st.session_state[seen] = chosen
        if mode == "Everything" or focus_id not in {n.id for n in nodes}:
            v_nodes, v_edges = nodes, edges
        else:
            v_nodes, v_edges = D.ego_subgraph(nodes, edges, focus_id, hops=1 if mode.startswith("Selected") else 2)
        keep = {n.id for n in v_nodes if n.kind in show or n.id == focus_id}
        v_nodes = [n for n in v_nodes if n.id in keep]
        v_edges = [e for e in v_edges if e.source in keep and e.target in keep]
        U._md(
            '<div class="legend">'
            '<span><i style="background:#4fc48b"></i>Claim (colour = status)</span>'
            '<span><i style="background:#d5dde8;border-radius:2px"></i>Document</span>'
            '<span><i style="background:#3cc4d4"></i>Dated event</span>'
            '<span><i style="background:#e7c9a0"></i>Equipment</span>'
            '<span style="color:#5f7da3">── claim cites a passage of</span><span style="color:#3cc4d4">── event extracted from</span>'
            '<span style="color:#b39b7a">── equipment named in</span>'
            "</div>"
        )
        zoom = C.graph_controls(f"gm_{rk}", "node")
        tallest = max(
            sum(1 for n in v_nodes if n.kind == "document"),
            sum(1 for n in v_nodes if n.kind == "claim"),
            sum(1 for n in v_nodes if n.kind in ("event", "equipment")),
            1,
        )
        xd, yd = C.zoomed_domains(v_nodes, focus_id, (-1.0, 3.0), (-0.03, 1.03), zoom)
        chart = CH.graph_chart(v_nodes, v_edges, height=max(300, 34 * tallest + 60), focus_id=focus_id,
                               x_domain=xd, y_domain=yd)
        st.altair_chart(chart, width="stretch", on_select="rerun", selection_mode="node", key=widget, theme=None)
        C.note(f"Showing {len(v_nodes)} of {len(nodes)} nodes and {len(v_edges)} of {len(edges)} recorded links. "
               "Left: claims · centre: source documents · right: dated events and equipment.")
        node = next((n for n in nodes if n.id == focus_id), None)
        if node is None:
            U._md('<div class="empty-card">Select a node (or use Find a node) to anchor the map on it.</div>')
        else:
            with st.container(border=True):
                _node_panel(result, node, nodes, edges, rk)
                C.node_actions(node, result, key=f"gm_{rk}_{node.id}")

    C.section("Relationship list", f"{len(edges)} recorded links · the accessible alternative to the map")
    if not edges:
        U._md('<div class="empty-line">No relationships were recorded.</div>')
        return
    by_id = {n.id: n for n in nodes}
    focus_id = C.focus().get("node")
    only = st.toggle("Only links of the selected node", key=f"gm_only_{rk}", disabled=focus_id not in by_id)
    rows = [e for e in edges if e.source in by_id and e.target in by_id
            and (not only or focus_id in (e.source, e.target))]
    frame = pd.DataFrame(
        [
            {
                "From": f"{KINDS[by_id[e.source].kind][:-1] if by_id[e.source].kind != 'equipment' else 'Equipment'} · {by_id[e.source].label}",
                "Relationship": e.relation,
                "To": by_id[e.target].label,
                "Via passages": ", ".join(e.chunk_ids[:3]) + ("…" if len(e.chunk_ids) > 3 else ""),
            }
            for e in rows
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch", height=min(420, 38 + 35 * max(1, len(frame))))
