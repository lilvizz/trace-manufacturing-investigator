"""Command Center design layer, applied on top of app.ui._CSS (which keeps
its tokens, so existing components such as the basis matrix, passages and
progress card stay visually consistent).

System
  Ink      #070b12 / #0e1520 / #121b28   blue-black plant-control-room base
  Signal   #3cc4d4                       the one interactive accent
  Status   green supported · red contradicted · amber not established /
           gap · blue context · violet recommendation · gray inconclusive
  Type     IBM Plex Sans for prose, IBM Plex Mono only for record data
           (dates, chunk IDs, file paths); system fallbacks if offline.
"""

CC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root {
  --trace-violet: #a58beb;
  --trace-mono: "IBM Plex Mono", "Source Code Pro", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --cc-sans: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp td, .stApp th { font-family: var(--cc-sans); }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1360px; }
.badge.violet { color: var(--trace-violet); background: rgba(165,139,235,.08); }
.tone-violet { color: var(--trace-violet); }
.mono { font-family: var(--trace-mono); }

/* ---------- Sidebar navigation ---------- */
[data-testid="stSidebar"] [data-testid="stPageLink"] a {
  border-radius: 8px; padding: 5px 10px; margin: 1px 0;
  border-left: 2px solid transparent;
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { background: var(--trace-surface-2); }
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"],
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"][aria-current="page"] {
  background: var(--trace-accent-soft); border-left-color: var(--trace-accent);
}
[data-testid="stSidebar"] [data-testid="stPageLink"] p { font-size: 14px; color: var(--trace-text-2); }
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] p { color: var(--trace-text); font-weight: 600; }
.nav-group { color: var(--trace-muted); font-size: 12px; font-weight: 600; margin: 14px 0 3px 2px; }
.case-pin {
  border: 1px solid var(--trace-border-strong); border-radius: 10px; background: var(--trace-surface);
  padding: 10px 12px; margin: 4px 0 6px;
}
.case-pin .q { color: var(--trace-text); font-size: 13px; line-height: 1.45; margin-top: 6px;
  display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.case-pin .m { color: var(--trace-muted); font-size: 12px; margin-top: 6px; }
.case-pin.empty { border-style: dashed; color: var(--trace-muted); font-size: 13px; }
.health { display: grid; grid-template-columns: auto 1fr; gap: 5px 10px; font-size: 12.5px; }
.health .k { color: var(--trace-muted); }
.health .v { color: var(--trace-text-2); }

/* ---------- Page header / case bar ---------- */
.pg-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; margin: 0 0 14px; }
.pg-title { color: var(--trace-text); font-size: 25px; font-weight: 650; letter-spacing: -.01em; line-height: 1.2; }
.pg-sub { color: var(--trace-muted); font-size: 14px; line-height: 1.5; margin-top: 4px; max-width: 760px; }
.case-bar {
  display: flex; gap: 14px; align-items: center; justify-content: space-between;
  border: 1px solid var(--trace-border); background: var(--trace-bg-2);
  border-radius: 10px; padding: 9px 14px; margin-bottom: 14px;
}
.case-bar .q { color: var(--trace-text-2); font-size: 13.5px; line-height: 1.4; min-width: 0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.case-bar .q b { color: var(--trace-text); font-weight: 600; }
.case-bar .r { display: flex; gap: 8px; align-items: center; flex: 0 0 auto; color: var(--trace-muted); font-size: 12.5px; }

/* ---------- Verdict plate (overview) ---------- */
.verdict {
  position: relative; overflow: hidden;
  border: 1px solid var(--trace-border-strong); border-radius: 14px;
  background: linear-gradient(135deg, #0f1c29 0%, #0b131e 60%);
  padding: 20px 24px 18px;
}
.verdict:before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--v-tone, var(--trace-gray)); }
.verdict.green { --v-tone: var(--trace-green); } .verdict.red { --v-tone: var(--trace-red); }
.verdict.amber { --v-tone: var(--trace-amber); } .verdict.blue { --v-tone: var(--trace-blue); }
.verdict.gray { --v-tone: var(--trace-gray); }
.verdict-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; flex-wrap: wrap; }
.verdict-q { color: var(--trace-text-2); font-size: 14px; line-height: 1.5; max-width: 860px; }
.verdict-status { display: flex; align-items: center; gap: 10px; margin-top: 10px; }
.verdict-status .word { font-size: 30px; font-weight: 700; letter-spacing: -.01em; color: var(--v-tone); line-height: 1.1; }
.verdict-answer { color: var(--trace-text); font-size: 18px; line-height: 1.6; margin-top: 12px; max-width: 900px; }
.verdict-claim { color: var(--trace-text-2); font-size: 14px; line-height: 1.55; margin-top: 12px;
  border-top: 1px solid var(--trace-border); padding-top: 10px; max-width: 900px; }
.verdict-claim b { color: var(--trace-text); font-weight: 600; }
.meter { min-width: 230px; }
.meter .lbl { color: var(--trace-muted); font-size: 12.5px; }
.meter .val { color: var(--trace-text); font-size: 18px; font-weight: 650; margin: 2px 0 6px; }
.meter .bars { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; }
.meter .bars span { height: 6px; border-radius: 3px; background: var(--trace-border-strong); }
.meter .bars span.on { background: var(--v-tone, var(--trace-accent)); }
.meter .scale { display: grid; grid-template-columns: repeat(3, 1fr); color: var(--trace-muted); font-size: 11px; margin-top: 4px; }
.meter .note { color: var(--trace-muted); font-size: 12px; line-height: 1.45; margin-top: 8px; max-width: 260px; }

.stat-strip { display: grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap: 1px;
  background: var(--trace-border); border: 1px solid var(--trace-border); border-radius: 10px; overflow: hidden; margin: 12px 0 4px; }
.stat-strip > div { background: var(--trace-bg-2); padding: 10px 14px; }
.stat-strip .n { color: var(--trace-text); font-size: 20px; font-weight: 650; font-variant-numeric: tabular-nums; }
.stat-strip .t { color: var(--trace-muted); font-size: 12.5px; }

.sec { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 4px 16px; margin: 22px 0 8px; }
.sec .h { color: var(--trace-text); font-size: 16.5px; font-weight: 650; white-space: nowrap; }
.sec .s { max-width: 760px; }
[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] p { color: var(--trace-accent-ink) !important; font-weight: 700; }
.sec .s { color: var(--trace-muted); font-size: 13px; }

/* ---------- Compact list items ---------- */
.li { border: 1px solid var(--trace-border); background: var(--trace-surface); border-radius: 10px;
  padding: 11px 14px; margin-bottom: 8px; border-left: 3px solid var(--li-tone, var(--trace-border-strong)); }
.li.green { --li-tone: var(--trace-green); } .li.red { --li-tone: var(--trace-red); }
.li.amber { --li-tone: var(--trace-amber); } .li.blue { --li-tone: var(--trace-blue); }
.li.violet { --li-tone: var(--trace-violet); } .li.gray { --li-tone: var(--trace-gray); }
.li .k { color: var(--trace-muted); font-size: 12px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.li .t { color: var(--trace-text); font-size: 14px; line-height: 1.55; margin-top: 5px; }
.li .f { color: var(--trace-muted); font-size: 12.5px; margin-top: 6px; }
.chip { display: inline-block; font-family: var(--trace-mono); font-size: 11.5px; color: var(--trace-text-2);
  border: 1px solid var(--trace-border-strong); border-radius: 5px; padding: 1px 6px; background: var(--trace-bg-2); margin: 2px 4px 0 0; }
.empty-card { border: 1px dashed var(--trace-border-strong); border-radius: 12px; padding: 22px 24px;
  color: var(--trace-text-2); font-size: 14px; line-height: 1.6; background: rgba(14,21,32,.5); }
.empty-card b { color: var(--trace-text); }
.note-line { color: var(--trace-muted); font-size: 12.5px; line-height: 1.5; margin: 4px 0 10px; }

/* ---------- Home ---------- */
.hero { padding: 22px 0 6px; }
.hero h1 { font-size: 40px !important; line-height: 1.12; font-weight: 650 !important; letter-spacing: -.02em; margin: 0 0 10px !important; padding: 0 !important; max-width: 900px; }
.hero p { color: var(--trace-text-2); font-size: 16px; line-height: 1.6; max-width: 760px; margin: 0; }
.stage-rail { display: flex; flex-wrap: wrap; gap: 0; margin: 14px 0 4px; }
.stage-rail span { color: var(--trace-muted); font-size: 12.5px; padding: 3px 10px 3px 0; }
.stage-rail span:not(:last-child):after { content: "›"; margin-left: 10px; color: var(--trace-border-strong); }
.feature { border: 1px solid var(--trace-border); border-radius: 10px; background: var(--trace-surface); padding: 12px 14px; min-height: 92px; }
.feature .h { color: var(--trace-text); font-size: 14px; font-weight: 600; }
.feature .s { color: var(--trace-muted); font-size: 12.5px; line-height: 1.5; margin-top: 4px; }
.recent { border-bottom: 1px solid var(--trace-border); padding: 9px 0; }
.recent .q { color: var(--trace-text); font-size: 14px; line-height: 1.45; }
.recent .m { color: var(--trace-muted); font-size: 12px; margin-top: 3px; }

/* ---------- Explorer preview ---------- */
.doc-head { color: var(--trace-text); font-size: 16px; font-weight: 600; }
.doc-path { color: var(--trace-muted); font-family: var(--trace-mono); font-size: 12px; margin-top: 2px; word-break: break-all; }
.doc-body { font-family: var(--trace-mono); font-size: 12.5px; line-height: 1.65; color: var(--trace-text-2);
  background: var(--trace-bg-2); border: 1px solid var(--trace-border); border-radius: 10px; padding: 14px 16px;
  max-height: 520px; overflow: auto; white-space: normal; }
.doc-body mark { background: rgba(60,196,212,.22); color: var(--trace-text); border-radius: 2px; }
.doc-body .hit { background: rgba(60,196,212,.08); border-left: 2px solid var(--trace-accent); padding-left: 8px; display: block; margin: 4px 0; }
.kv { display: grid; grid-template-columns: 130px 1fr; gap: 5px 12px; font-size: 13px; margin: 10px 0; }
.kv .k { color: var(--trace-muted); } .kv .v { color: var(--trace-text-2); word-break: break-word; }

/* ---------- Map legend ---------- */
.legend { display: flex; flex-wrap: wrap; gap: 14px; color: var(--trace-text-2); font-size: 12.5px; margin: 2px 0 6px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }

/* ---------- Copilot ---------- */
.cop-q { color: var(--trace-text); font-size: 15px; line-height: 1.5; font-weight: 500; }
.cop-a { color: var(--trace-text); font-size: 15px; line-height: 1.62; margin-top: 8px; }
.cop-meta { color: var(--trace-muted); font-size: 12.5px; margin-top: 8px; }
.guard { border: 1px solid var(--trace-border-strong); border-radius: 10px; background: var(--trace-bg-2);
  padding: 10px 13px; color: var(--trace-text-2); font-size: 13px; line-height: 1.55; margin-bottom: 12px; }
.guard b { color: var(--trace-text); }

/* popover / citation buttons */
[data-testid="stPopover"] button { min-height: 30px !important; padding: 2px 10px !important; }
[data-testid="stPopover"] button p { font-family: var(--trace-mono); font-size: 12px !important; color: var(--trace-accent-strong) !important; }
[data-testid="stPopoverBody"] { max-width: 640px; }
.st-key-cc_cites [data-testid="stHorizontalBlock"] { gap: 6px; }

@media (max-width: 1000px) {
  .stat-strip { grid-template-columns: repeat(2, minmax(0,1fr)); }
  .hero h1 { font-size: 30px !important; }
  .case-bar { flex-direction: column; align-items: flex-start; }
  .case-bar .q { white-space: normal; }
  .kv { grid-template-columns: 1fr; }
}
</style>
"""

CC_CSS_2 = """
<style>
.hl-table { border: 1px solid var(--trace-border); border-radius: 10px; overflow: hidden; }
.hl-row { display: grid; grid-template-columns: 210px 120px 1fr; gap: 10px; align-items: center;
  padding: 8px 12px; border-bottom: 1px solid var(--trace-border); background: var(--trace-bg-2); }
.hl-row:last-child { border-bottom: 0; }
.hl-name { color: var(--trace-text); font-size: 13.5px; }
.hl-why { color: var(--trace-text-2); font-size: 13px; line-height: 1.45; }
.hl-row .badge { justify-self: start; font-size: 11px; }
.hero-grid h1 { font-size: 44px !important; line-height: 1.08; font-weight: 650 !important; letter-spacing: -.02em;
  margin: 6px 0 12px !important; padding: 0 !important; }
.hero-kicker { color: var(--trace-accent-strong); font-size: 13px; font-weight: 600; }
.hero-copy { color: var(--trace-text-2); font-size: 15.5px; line-height: 1.6; max-width: 560px; }
.graph-cap { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; flex-wrap: wrap; margin-bottom: 2px; }
.graph-cap .t { color: var(--trace-text); font-size: 15px; font-weight: 600; }
.graph-cap .s { color: var(--trace-muted); font-size: 12.5px; }
.mini-stats { display: flex; flex-wrap: wrap; gap: 18px; margin: 6px 0 2px; }
.mini-stats div { color: var(--trace-muted); font-size: 12.5px; }
.mini-stats b { color: var(--trace-text); font-size: 18px; font-weight: 650; display: block; }
.statusbar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; margin: 8px 0 6px; background: var(--trace-border); }
.statusbar span { display: block; height: 100%; }
.status-legend { display: flex; flex-wrap: wrap; gap: 14px; color: var(--trace-text-2); font-size: 12.5px; }
.status-legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; }
.story { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin-top: 14px; }
.story > div { border-top: 2px solid var(--s-tone, var(--trace-border-strong)); padding-top: 8px; }
.story .k { color: var(--trace-muted); font-size: 12.5px; }
.story .t { color: var(--trace-text); font-size: 14px; line-height: 1.5; margin-top: 4px; }
.story .o { color: var(--trace-muted); font-size: 12px; margin-top: 4px; }
.date-prov { font-family: var(--trace-mono); font-size: 12px; color: var(--trace-text-2); background: var(--trace-bg-2);
  border: 1px solid var(--trace-border); border-radius: 5px; padding: 1px 6px; }
.turn-break { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 8px; margin: 10px 0 4px; }
.turn-break div { border: 1px solid var(--trace-border); border-radius: 8px; padding: 7px 10px; background: var(--trace-bg-2); }
.turn-break b { display: block; font-size: 17px; color: var(--trace-text); }
.turn-break span { color: var(--trace-muted); font-size: 12px; }
.vnote { color: #f0c486; font-size: 12.5px; line-height: 1.5; border-left: 2px solid var(--trace-amber); padding-left: 9px; margin-top: 6px; }
@media (max-width: 1000px) {
  .hl-row { grid-template-columns: 1fr; gap: 3px; }
  .story, .turn-break { grid-template-columns: 1fr 1fr; }
  .hero-grid h1 { font-size: 32px !important; }
}
@media (max-width: 600px) { .story, .turn-break { grid-template-columns: 1fr; } }
[data-testid="stVegaLiteChart"] summary, .vega-embed summary { opacity: .7; }
</style>
"""

# The Overview's story row already states the most important open issue;
# hide the duplicate box the shared basis-matrix renderer adds (classic UI
# keeps it).
CC_CSS_2 += "<style>.stApp .open-issue { display: none; }</style>"

CC_CSS_2 += """<style>
.doc-body.prose-body { font-family: var(--cc-sans); font-size: 14.5px; line-height: 1.72; color: var(--trace-text); }
.doc-body.mono-body { font-size: 12.5px; }
.doc-body .hit { background: rgba(60,196,212,.10); border-left: 3px solid var(--trace-accent); padding: 4px 10px; border-radius: 4px; }
</style>"""

CC_CSS_2 += """<style>
.verdict-meaning { color: var(--trace-text-2); font-size: 14px; line-height: 1.5; margin-top: 6px; max-width: 760px; }
.verdict-answer { font-size: 17px; }
</style>"""

CC_CSS_2 += """<style>
[class*="st-key-claim_green"] { border-left: 4px solid var(--trace-green) !important; }
[class*="st-key-claim_amber"] { border-left: 4px solid var(--trace-amber) !important; }
[class*="st-key-claim_red"] { border-left: 4px solid var(--trace-red) !important; }
[class*="st-key-claim_blue"] { border-left: 4px solid var(--trace-blue) !important; }
[class*="st-key-claim_violet"] { border-left: 4px solid var(--trace-violet) !important; }
[class*="st-key-claim_gray"] { border-left: 4px dashed var(--trace-gray) !important; }
</style>"""

CC_CSS_2 += """<style>
/* Evidence Explorer: the selected passage is the primary focus. */
.st-key-ex_reader { border-color: var(--trace-border-strong) !important; background: var(--trace-surface); }
.reader-eyebrow { color: var(--trace-accent); font-size: 11.5px; letter-spacing: .06em; text-transform: uppercase; }
.reader-head { font-size: 19px; margin-top: 2px; }
.reader-prov { color: var(--trace-muted); font-size: 12px; margin: 4px 0 10px; word-break: break-word; }
.reader-prov .mono { font-size: 11.5px; }
.st-key-ex_reader .doc-body.prose-body { font-size: 15px; }
.st-key-ex_reader .doc-body { max-height: 560px; }
.kv-side { grid-template-columns: 1fr; gap: 0; font-size: 12.5px; margin-top: 4px; }
.kv-side .k { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
.kv-side .v { margin-bottom: 9px; }
</style>"""
