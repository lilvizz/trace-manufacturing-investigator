"""Interactive Altair charts (Altair ships with Streamlit; no new dependency).

Selection model: a click selection is read once (components.chart_pick),
stored as the page's focus, and the focus is then drawn into the chart
data (selected / neighbour / dimmed). That keeps the chart, the detail
panel and any list on the page synchronised, including when the focus is
set from somewhere other than the chart.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from app.command_center import data as D

INK = "#070b12"
TEXT = "#b3c0d0"
MUTED = "#8593a6"
GRID = "#1d2a3b"
ACCENT = "#3cc4d4"
FONT = "IBM Plex Sans, Segoe UI, Helvetica, Arial, sans-serif"
TONES = {
    "green": "#4fc48b", "red": "#f07272", "amber": "#e9a94f", "blue": "#7aa7ff",
    "gray": "#9aa7b8", "violet": "#a58beb", "doc": "#d5dde8", "event": ACCENT, "equipment": "#e7c9a0",
}
LANE_COLORS = {
    "Asset & equipment history": "#7aa7ff",
    "Engineering record": "#3cc4d4",
    "SPC & quality": "#4fc48b",
    "SOP / procedure": "#a58beb",
    "Maintenance & calibration": "#e9a94f",
    "Informal / recollection": "#f0a3c0",
    D.UNCATEGORISED_LANE: "#6b7a8f",
}


def _style(chart):
    return (
        chart.configure(background="transparent", font=FONT)
        .configure_view(stroke=None)
        .configure_axis(labelColor=TEXT, titleColor=MUTED, gridColor=GRID, domainColor=GRID,
                        tickColor=GRID, labelFontSize=12, titleFontSize=12, titleFontWeight=500)
        .configure_legend(labelColor=TEXT, titleColor=MUTED, labelFontSize=12, titleFontSize=12,
                          orient="top", direction="horizontal", symbolStrokeWidth=0, symbolFillColor="#8593a6")
    )


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def timeline_frame(rows: list[D.TimelineRow], focus_idx=None) -> pd.DataFrame:
    records = []
    for row in rows:
        if not hasattr(row.date, "isoformat"):
            continue  # only real parsed dates are plotted
        if row.precision == "month":
            kind = "Month-only date"
        elif row.issued:
            kind = "Record issued"
        else:
            kind = "Dated statement"
        records.append(
            {
                "idx": row.idx,
                "date": pd.Timestamp(row.date.isoformat()),
                "date_label": row.date_label + (" (month only)" if row.precision == "month" else "")
                + (" (ambiguous)" if row.ambiguous else ""),
                "lane": row.lane,
                "kind": kind,
                "use": "Cited in conclusions" if row.cited else "Retrieved only",
                "headline": (row.headline[:110] + "…") if len(row.headline) > 110 else row.headline,
                "source": row.filename,
                "focus": focus_idx is not None and int(focus_idx) == row.idx,
                "size": 480 if (focus_idx is not None and int(focus_idx) == row.idx) else 150,
                "ring": "#ffffff" if (focus_idx is not None and int(focus_idx) == row.idx) else INK,
                "sw": 2.5 if (focus_idx is not None and int(focus_idx) == row.idx) else 1.0,
            }
        )
    return pd.DataFrame.from_records(records)


def timeline_chart(rows: list[D.TimelineRow], height: int | None = None, focus_idx=None):
    frame = timeline_frame(rows, focus_idx)
    if frame.empty:
        return None
    lanes = [lane for lane in D.LANE_ORDER if lane in set(frame["lane"])]
    # Snap the axis to whole years so the first and last years always get a
    # labelled tick (an axis starting at, say, 05 Sep 2001 never shows "2001"
    # and its points sit unlabelled against the edge).
    first_year, last_year = int(frame["date"].min().year), int(frame["date"].max().year)
    domain = [f"{first_year}-01-01", f"{last_year}-12-31"]
    pick = alt.selection_point(name="event", fields=["idx"], on="click", empty=False)
    base = alt.Chart(frame)
    lane_rules = base.mark_rule(color=GRID, strokeDash=[2, 4]).encode(
        y=alt.Y("lane:N", sort=lanes, title=None)
    )
    points = base.mark_point(filled=True, cursor="pointer").encode(
        x=alt.X("date:T", title=None, scale=alt.Scale(domain=domain, padding=6),
                axis=alt.Axis(format="%Y", tickCount={"interval": "year", "step": 1 if last_year - first_year <= 26 else 2},
                              grid=True, labelAngle=0, labelOverlap=True)),
        y=alt.Y("lane:N", sort=lanes, title=None, axis=alt.Axis(labelLimit=190, grid=False)),
        shape=alt.Shape("kind:N", title=None, scale=alt.Scale(
            domain=["Record issued", "Dated statement", "Month-only date"],
            range=["square", "circle", "triangle-up"])),
        color=alt.Color("use:N", title=None, scale=alt.Scale(domain=["Cited in conclusions", "Retrieved only"],
                                                             range=[ACCENT, "#5b6b80"])),
        size=alt.Size("size:Q", scale=None, legend=None),
        stroke=alt.Stroke("ring:N", scale=None, legend=None),
        strokeWidth=alt.StrokeWidth("sw:Q", scale=None, legend=None),
        tooltip=[alt.Tooltip("date_label:N", title="Date"), alt.Tooltip("headline:N", title="Event"),
                 alt.Tooltip("source:N", title="Source"), alt.Tooltip("use:N", title="Use")],
    ).add_params(pick)
    chart = (lane_rules + points).properties(height=height or max(150, 46 * len(lanes) + 40))
    return _style(chart)


# ---------------------------------------------------------------------------
# Node-link graphs (case relationship map and corpus constellation)
# ---------------------------------------------------------------------------

def neighbours(edges, node_id) -> set[str]:
    out = set()
    for e in edges:
        if e.source == node_id:
            out.add(e.target)
        elif e.target == node_id:
            out.add(e.source)
    return out


def graph_chart(nodes: list, edges: list, height: int = 560, focus_id=None,
                label_kinds=("claim", "document", "event", "equipment"), x_domain=(-1.0, 3.0),
                y_domain=(-0.03, 1.03), zoom: bool = True, legend_kinds=None, inward_labels: bool = False):
    """Generic node-link chart. Nodes need id/kind/label/detail/tone/x/y
    (GraphNode or the corpus graph's nodes). With a focus, the selected node
    is ringed, its direct neighbours stay bright with their links, and
    everything else is dimmed."""
    if not nodes:
        return None
    pos = {n.id: n for n in nodes}
    near = neighbours(edges, focus_id) if focus_id in pos else set()
    # Second ring: nodes reached through a direct neighbour. Kept legible
    # (not dimmed away) so a "two steps out" view reads as a connected story.
    near2 = set()
    for n_id in near:
        near2 |= neighbours(edges, n_id)
    near2 -= near | {focus_id}
    # Label neighbours only while there are few enough to read; a hub node
    # (e.g. Press-07, named in most documents) would otherwise stack dozens
    # of labels on top of each other. Its links stay highlighted and the
    # node panel below the chart lists them. Second-ring labels are shown in
    # the layered map only.
    label_near = len(near) <= MAX_NEIGHBOUR_LABELS
    label_near2 = not inward_labels and len(near2) <= MAX_NEIGHBOUR_LABELS

    def state(node_id):
        if focus_id not in pos:
            return "none"
        if node_id == focus_id:
            return "sel"
        if node_id in near:
            return "nbr"
        return "nbr2" if node_id in near2 else "dim"

    edge_rows = []
    for k, e in enumerate(edges):
        a, b = pos.get(e.source), pos.get(e.target)
        if a is None or b is None:
            continue
        lit = focus_id in pos and focus_id in (e.source, e.target)
        ring2 = focus_id in pos and ({e.source, e.target} & near) and ({e.source, e.target} & near2)
        if lit:
            op, sw = 0.95, 2.2
        elif ring2:
            op, sw = 0.5, 1.3
        elif focus_id in pos:
            op, sw = 0.05, 1.0
        else:
            op, sw = (0.2 if e.relation in ("mentioned in", "mentions") else 0.5), 1.1
        for order, n in enumerate((a, b)):
            edge_rows.append({"edge": k, "order": order, "x": n.x, "y": n.y, "relation": e.relation,
                              "op": op, "sw": sw})
    node_rows = []
    for n in nodes:
        label = n.label
        if n.kind == "claim" and n.ref is not None:
            parts = n.label.split(" · ")
            # A recommendation's status is long ("Related records cited · not a
            # finding"); on the chart its type is the clearer short label. The
            # full status stays in the tooltip and the node panel.
            short = "Recommendation" if parts[-1] == "Recommendation" else parts[0]
            label = f"C{int(n.ref) + 1} · {short}"
        elif len(label) > 30:
            label = label[:29] + "…"
        st_ = state(n.id)
        node_rows.append({
            "id": n.id, "x": n.x, "y": n.y, "kind": n.kind.capitalize(), "label": label,
            "detail": (n.detail[:140] + "…") if len(n.detail) > 140 else n.detail,
            "color": TONES.get(n.tone, LANE_COLORS.get(n.tone, TONES["gray"])),
            # Label placement (presentation only; node positions unchanged):
            # see _label_place.
            **dict(zip(("align", "dx", "dy"), _label_place(n, inward_labels, selected=st_ == "sel"))),
            "hl": st_,
            "size": 620 if st_ == "sel" else 180,
            "op": {"dim": 0.18, "nbr2": 0.7}.get(st_, 1.0),
            "sw": 3.0 if st_ == "sel" else 1.0,
            "text_op": {"dim": 0.25, "nbr2": 0.75}.get(st_, 1.0),
            "text_color": "#ffffff" if st_ == "sel" else TEXT,
            "show_label": (n.kind in label_kinds or st_ == "sel" or (st_ == "nbr" and label_near)
                           or (st_ == "nbr2" and label_near2)),
        })
    nf, ef = pd.DataFrame(node_rows), pd.DataFrame(edge_rows)
    pick = alt.selection_point(name="node", fields=["id"], on="click", empty=False)
    xs = alt.X("x:Q", axis=None, scale=alt.Scale(domain=list(x_domain)))
    ys = alt.Y("y:Q", axis=None, scale=alt.Scale(domain=list(y_domain)))
    layers = []
    if not ef.empty:
        layers.append(
            alt.Chart(ef).mark_line().encode(
                x=xs, y=ys, detail="edge:N", order="order:Q",
                color=alt.Color("relation:N", legend=None, scale=alt.Scale(
                    domain=["cites", "extracted from", "mentioned in", "references", "mentions"],
                    range=["#5f7da3", ACCENT, "#b39b7a", "#7aa7ff", "#b39b7a"])),
                opacity=alt.Opacity("op:Q", scale=None, legend=None),
                strokeWidth=alt.StrokeWidth("sw:Q", scale=None, legend=None),
            )
        )
    node_layer = alt.Chart(nf).mark_point(filled=True, cursor="pointer", stroke=INK).encode(
        x=xs, y=ys,
        color=alt.Color("color:N", scale=None, legend=None),
        shape=alt.Shape("kind:N", legend=None, scale=alt.Scale(
            domain=["Claim", "Document", "Event", "Equipment"],
            range=["circle", "square", "triangle-up", "diamond"])),
        size=alt.Size("size:Q", scale=None, legend=None),
        opacity=alt.Opacity("op:Q", scale=None, legend=None),
        strokeWidth=alt.StrokeWidth("sw:Q", scale=None, legend=None),
        tooltip=[alt.Tooltip("kind:N", title="Type"), alt.Tooltip("label:N", title="Node"),
                 alt.Tooltip("detail:N", title="Detail")],
    ).add_params(pick)
    if zoom:
        node_layer = node_layer.add_params(alt.selection_interval(bind="scales", name="zoom"))
    # A ring around the selected node anchors the story.
    sel_rows = nf[nf["hl"] == "sel"]
    if not sel_rows.empty:
        layers.append(alt.Chart(sel_rows).mark_point(
            shape="circle", filled=False, size=1500, stroke="#ffffff", strokeWidth=1.5, opacity=0.8, clip=False
        ).encode(x=xs, y=ys))
    layers.append(node_layer)
    shown = nf[nf["show_label"]]
    for (align, dx, dy), sub in shown.groupby(["align", "dx", "dy"], sort=False):
        limit = LABEL_LIMIT_CENTRED if align == "center" else LABEL_LIMIT
        # clip=False: the zoom/pan binding clips marks to the plot area, which
        # cut labels off at the chart edge; labels may use the padding.
        text = dict(align=align, dx=int(dx), dy=int(dy), fontSize=11.5, limit=limit, font=FONT, baseline="middle",
                    clip=False)
        enc = dict(x=xs, y=ys, text="label:N", opacity=alt.Opacity("text_op:Q", scale=None, legend=None))
        # Halo: the same text stroked in the page colour underneath, so edges
        # that cross a label never run through its glyphs.
        layers.append(alt.Chart(sub).mark_text(stroke=INK, strokeWidth=4, strokeJoin="round", fill=INK,
                                               **text).encode(**enc))
        layers.append(alt.Chart(sub).mark_text(**text).encode(
            color=alt.Color("text_color:N", scale=None, legend=None), **enc))
    chart = alt.layer(*layers).properties(height=height, width="container",
                                          padding={"left": 6, "right": 6, "top": 18, "bottom": 6})
    return _style(chart)


MAX_NEIGHBOUR_LABELS = 12  # beyond this, neighbours are highlighted but not labelled
LABEL_LIMIT = 230          # px; right-hand labels
LABEL_LIMIT_CENTRED = 170  # px; labels centred above a node


def _label_place(node, inward_labels: bool, selected: bool = False) -> tuple[str, int, int]:
    """(align, dx, dy) for a node's label.

    Layered map (claims | documents | events & equipment): claim labels sit
    centred above their node, so nothing hangs off the left edge of the
    chart; documents, events and equipment label to the right. The layered
    x domain (-1.0, 3.0) leaves room for both at narrow desktop widths.
    Ring (inward_labels): document labels face the centre; equipment labels
    point away from the centre, and the centre node labels below itself, so
    the clustered identifiers do not overwrite each other."""
    gap = 28 if selected else 12   # the selected node's label clears its white ring
    if inward_labels:
        if node.kind != "equipment":
            return ("left", gap, 0) if node.x < 0 else ("right", -gap, 0)
        if abs(node.x) < 0.05 and abs(node.y) < 0.05:
            return ("center", 0, gap + 4)
        return ("left", gap, 0) if node.x >= 0 else ("right", -gap, 0)
    if node.kind == "claim":
        return ("center", 0, -(gap + 4))
    return ("left", gap, 0)


def selected_value(state, param: str, field: str):
    """Read a point selection from st.altair_chart(on_select='rerun') state
    (the returned object or the raw session_state value)."""
    if state is None:
        return None
    try:
        selection = state.selection if hasattr(state, "selection") else state.get("selection", {})
        values = selection.get(param) if hasattr(selection, "get") else getattr(selection, param, None)
    except Exception:
        return None
    if isinstance(values, list) and values:
        first = values[0]
        if isinstance(first, dict):
            return first.get(field)
    if isinstance(values, dict):
        value = values.get(field)
        if isinstance(value, list):
            return value[0] if value else None
        return value
    return None
