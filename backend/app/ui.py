"""
TRACE — Manufacturing Evidence Investigator
Judge-facing Streamlit UI.

Presentation layer only:
- Uses the existing Investigator pipeline (load_pipeline -> Investigator.run).
- Does not change investigation logic, retrieval, verification, or
  confidence calibration.
- Keeps evidence, claims, timeline, and gaps traceable to backend objects:
  every grouping in this file is derived from fields that already exist on
  InvestigationResult; nothing is invented for presentation.
"""

from __future__ import annotations

import json
import os
import sys
import re
import time
from html import unescape
from html.parser import HTMLParser
from dataclasses import asdict
from pathlib import Path
from typing import Any

import textwrap
import streamlit as st
import streamlit.components.v1 as components

# Allow `streamlit run app/ui.py` from the backend directory.
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import CORPUS_PATH, INDEX_PATH  # noqa: E402
from app.ingestion.chunker import chunk_corpus  # noqa: E402
from app.ingestion.loader import load_corpus  # noqa: E402
from app.investigation.build import build_investigator  # noqa: E402
from app.investigation.models import ClaimType, EvidenceStance  # noqa: E402
from app.investigation.planner import InvestigationPlanner  # noqa: E402
from app.investigation.scenarios import get_scenarios  # noqa: E402
from app.retrieval.bm25_search import BM25Searcher  # noqa: E402
from app.retrieval.entity_matcher import EntityMatcher  # noqa: E402
from app.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402


# ---------------------------------------------------------------------------
# Product copy
# ---------------------------------------------------------------------------

APP_NAME = "TRACE"
APP_SUBTITLE = "Manufacturing Evidence Investigator"

HERO_QUESTION = (
    "Why is Press-07 limited to 42.3°C, what changed since that limit "
    "was established, and is its current technical basis still supported?"
)

EXAMPLE_QUESTIONS = [
    HERO_QUESTION,
    "What established the original 42.3°C boundary for Press-07?",
    "Did the PX-200 to PX-400 replacement re-establish the 42.3°C limit?",
    "Is there explicit current technical justification for keeping the 42.3°C limit?",
]

STANCE_META = {
    EvidenceStance.SUPPORTS: ("NO ESCALATION", "green"),
    EvidenceStance.CONTRADICTS: ("FLAG / ESCALATE", "red"),
    EvidenceStance.NOT_ESTABLISHED: ("NOT ESTABLISHED", "amber"),
    EvidenceStance.CONTEXT_ONLY: ("CONTEXT ONLY", "blue"),
    EvidenceStance.UNKNOWN: ("INCONCLUSIVE", "gray"),
}

FINDING_META = {
    "supported": ("Supported finding", "green"),
    "contradiction": ("Contradiction", "red"),
    "evidence-gap": ("Evidence gap", "amber"),
    "context": ("Context", "blue"),
}

# Evidence-language status used by the basis matrix and the claims view.
# Every status carries a glyph AND a word, so meaning never depends on
# color alone. This is a display vocabulary for the backend's existing
# EvidenceStance values -- it adds no new states.
STATUS_META = {
    EvidenceStance.SUPPORTS: ("Supported", "green", "✓"),
    EvidenceStance.CONTRADICTS: ("Contradicted", "red", "✕"),
    EvidenceStance.NOT_ESTABLISHED: ("Not established", "amber", "○"),
    EvidenceStance.CONTEXT_ONLY: ("Context only", "blue", "◇"),
    EvidenceStance.UNKNOWN: ("Inconclusive", "gray", "?"),
}

CLAIM_TYPE_LABELS = {
    ClaimType.ORIGINAL_BASIS: "Historical basis",
    ClaimType.CURRENT_CAPABILITY: "Equipment & capability",
    ClaimType.ADMINISTRATIVE_ACTION: "Administrative record",
    ClaimType.CURRENT_TECHNICAL_JUSTIFICATION: "Current technical justification",
    ClaimType.EVIDENCE_GAP: "Evidence gap",
    ClaimType.RECOMMENDATION: "Recommendation",
}

# The four questions the conclusion matrix answers, in reading order:
# where did it come from -> what changed -> how was it carried -> is it
# technically current. Rows are only filled from claims of that type.
BASIS_ROWS = (
    ClaimType.ORIGINAL_BASIS,
    ClaimType.CURRENT_CAPABILITY,
    ClaimType.ADMINISTRATIVE_ACTION,
    ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
)

# Source-type labels for the corpus folders. Deliberately an explicit
# allow-list: a folder that is not listed gets no category label at all
# (its real relative path is still shown as provenance). This keeps the UI
# from inventing a source type, and from turning an evaluation-only folder
# name into an on-screen judgement about a document.
SOURCE_CATEGORY_LABELS = {
    "A_asset_equipment_history": "Asset & equipment history",
    "B_historical_engineering": "Engineering record",
    "C_spc_quality": "SPC & quality",
    "D_sop_evolution": "SOP / procedure",
    "E_maintenance_calibration": "Maintenance & calibration",
    "F_human_informal": "Informal / recollection",
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_CSS = """
<style>
:root {
    --trace-bg: #070b12;
    --trace-bg-2: #0a0f18;
    --trace-surface: #0e1520;
    --trace-surface-2: #121b28;
    --trace-surface-3: #172334;
    --trace-border: #1d2a3b;
    --trace-border-strong: #2a3b52;
    --trace-text: #e7edf5;
    --trace-text-2: #b3c0d0;
    --trace-muted: #8593a6;
    --trace-accent: #3cc4d4;
    --trace-accent-strong: #67d8e5;
    --trace-accent-ink: #04232a;
    --trace-accent-soft: rgba(60,196,212,.10);
    --trace-blue: #7aa7ff;
    --trace-cyan: #67d8e5;
    --trace-green: #4fc48b;
    --trace-amber: #e9a94f;
    --trace-red: #f07272;
    --trace-gray: #9aa7b8;
    --trace-mono: "Source Code Pro", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --trace-radius: 12px;
    --trace-radius-sm: 8px;
}

html, body { background: var(--trace-bg); }

.stApp {
    background:
        radial-gradient(1200px 500px at 85% -10%, rgba(60,196,212,.06), transparent 60%),
        var(--trace-bg);
    color: var(--trace-text);
}

[data-testid="stHeader"] { background: rgba(7,11,18,.85); }
[data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div {
    background: var(--trace-bg) !important;
}

.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1320px;
    padding-top: 3.4rem;
    padding-bottom: 6rem;
}

h1, h2, h3, h4 { color: var(--trace-text) !important; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--trace-bg-2);
    border-right: 1px solid var(--trace-border);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li { color: var(--trace-text-2); }

.trace-brand { display: flex; align-items: center; gap: 12px; margin: 4px 0 22px; }
.trace-mark {
    width: 34px; height: 34px; border-radius: 9px;
    display: grid; place-items: center;
    background: linear-gradient(160deg, #123344, #0b1a26);
    border: 1px solid #2c6273;
    color: var(--trace-accent-strong);
    font-family: var(--trace-mono); font-weight: 700; font-size: 15px;
}
.trace-wordmark { font-size: 18px; font-weight: 700; letter-spacing: .04em; color: var(--trace-text); }
.trace-caption { color: var(--trace-muted); font-size: 12.5px; margin-top: 1px; }

.side-heading {
    color: var(--trace-text-2); font-size: 12px; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; margin: 18px 0 8px;
}
.side-note { color: var(--trace-muted); font-size: 12.5px; line-height: 1.5; }
.sys-table { display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; font-size: 13px; }
.sys-table .k { color: var(--trace-muted); }
.sys-table .v { color: var(--trace-text); font-weight: 600; }
.sys-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: 1px; }
.sys-dot.on { background: var(--trace-green); }
.sys-dot.off { background: var(--trace-amber); }

/* ---------- Header / composer ---------- */
.trace-topbar { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 14px; }
.trace-title { font-size: 26px; font-weight: 700; letter-spacing: -.01em; color: var(--trace-text); }
.trace-title span { color: var(--trace-muted); font-weight: 500; font-size: 16px; margin-left: 10px; letter-spacing: 0; }
.trace-tagline { color: var(--trace-text-2); font-size: 14.5px; line-height: 1.55; max-width: 820px; margin: -6px 0 18px; }

.question-label {
    color: var(--trace-text-2); font-size: 13px; font-weight: 600; margin-bottom: 6px;
}
.composer-hint { color: var(--trace-muted); font-size: 13px; line-height: 1.5; padding-top: 9px; }
.try-label { color: var(--trace-muted); font-size: 13px; padding-top: 8px; }

div[data-testid="stTextArea"] textarea {
    background: #0a111b !important;
    color: var(--trace-text) !important;
    border: 1px solid var(--trace-border-strong) !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    line-height: 1.5 !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: var(--trace-accent) !important;
    box-shadow: 0 0 0 3px rgba(60,196,212,.18) !important;
}
div[data-testid="stTextInput"] input {
    background: #0a111b !important; color: var(--trace-text) !important;
    border-radius: 9px !important;
}

/* ---------- Buttons ---------- */
.stButton > button, [data-testid="stBaseButton-secondary"] {
    border-radius: 9px !important;
    border: 1px solid var(--trace-border-strong) !important;
    background: var(--trace-surface-2) !important;
    color: var(--trace-text) !important;
    font-weight: 600 !important;
    min-height: 40px !important;
}
.stButton > button:hover { border-color: var(--trace-accent) !important; color: #fff !important; }
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {
    background: var(--trace-accent) !important;
    border-color: var(--trace-accent) !important;
    color: var(--trace-accent-ink) !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
    background: var(--trace-accent-strong) !important; color: var(--trace-accent-ink) !important;
}
.stButton > button:disabled { opacity: .55 !important; }
.stButton > button:focus-visible,
[role="tab"]:focus-visible,
[data-testid="stExpander"] summary:focus-visible,
.trace-details summary:focus-visible {
    outline: 2px solid var(--trace-accent-strong) !important;
    outline-offset: 2px !important;
}
/* Example-question chips: smaller, text-like secondary buttons. */
.st-key-examples button { min-height: 34px !important; }

/* ---------- Cards & labels ---------- */
.result-card {
    background: var(--trace-surface);
    border: 1px solid var(--trace-border);
    border-radius: var(--trace-radius);
    padding: 18px 20px;
    margin-bottom: 12px;
}
.result-card.compact { padding: 14px 16px; }
.section-label {
    color: var(--trace-text-2); font-size: 12px; font-weight: 700;
    letter-spacing: .08em; text-transform: uppercase; margin-bottom: 10px;
}
.section-title { color: var(--trace-text); font-size: 17px; font-weight: 700; margin: 4px 0 4px; }
.section-sub { color: var(--trace-muted); font-size: 13.5px; line-height: 1.5; margin-bottom: 12px; }
.small-muted { color: var(--trace-muted); font-size: 12.5px; line-height: 1.5; }

.badge {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 6px; padding: 4px 9px;
    font-size: 12px; font-weight: 700; letter-spacing: .04em;
    border: 1px solid currentColor; white-space: nowrap;
}
.badge.green { color: var(--trace-green); background: rgba(79,196,139,.08); }
.badge.amber { color: var(--trace-amber); background: rgba(233,169,79,.08); }
.badge.red { color: var(--trace-red); background: rgba(240,114,114,.08); }
.badge.blue { color: var(--trace-blue); background: rgba(122,167,255,.08); }
.badge.gray { color: var(--trace-gray); background: rgba(154,167,184,.07); }
.badge.quiet { border-color: var(--trace-border-strong); color: var(--trace-text-2); background: transparent; font-weight: 600; }
.tone-green { color: var(--trace-green); } .tone-amber { color: var(--trace-amber); }
.tone-red { color: var(--trace-red); } .tone-blue { color: var(--trace-blue); }
.tone-gray { color: var(--trace-gray); }

/* ---------- Result header ---------- */
.trace-result-header {
    border: 1px solid var(--trace-border);
    border-radius: var(--trace-radius);
    background: linear-gradient(180deg, var(--trace-surface-2), var(--trace-surface));
    padding: 16px 20px; margin-bottom: 12px;
}
.trace-result-kicker { color: var(--trace-accent-strong); font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.trace-result-question { color: var(--trace-text); font-size: 19px; line-height: 1.4; font-weight: 650; margin-top: 6px; max-width: 1100px; }
.trace-result-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
.trace-result-chip {
    display: inline-flex; align-items: center; gap: 6px;
    border: 1px solid var(--trace-border-strong); background: rgba(7,11,18,.5);
    border-radius: 999px; padding: 4px 11px; color: var(--trace-text-2); font-size: 12.5px;
}
.trace-result-chip strong { color: var(--trace-text); font-weight: 700; }
.trace-result-chip.subject { border-color: rgba(60,196,212,.35); color: var(--trace-accent-strong); }

/* ---------- Direct answer + assessment ---------- */
.direct-answer-card {
    background: linear-gradient(180deg, #0f1c29, #0d1621);
    border: 1px solid rgba(60,196,212,.28);
    border-left: 3px solid var(--trace-accent);
    border-radius: var(--trace-radius);
    padding: 18px 22px; margin-bottom: 12px; height: 100%;
}
.direct-answer-label { color: var(--trace-accent-strong); font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 8px; }
.direct-answer-text { color: var(--trace-text); font-size: 17px; line-height: 1.62; font-weight: 500; }
.direct-answer-evidence {
    color: var(--trace-muted); font-size: 12.5px; line-height: 1.6;
    margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--trace-border);
}
.direct-answer-evidence .src { font-family: var(--trace-mono); font-size: 12px; color: var(--trace-text-2); }

.assessment-card {
    background: var(--trace-surface); border: 1px solid var(--trace-border);
    border-radius: var(--trace-radius); padding: 18px 20px; margin-bottom: 12px;
}
.assessment-row { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px 14px; }
.assessment-title { color: var(--trace-text); font-size: 17px; line-height: 1.45; font-weight: 650; margin-top: 10px; }
.assessment-about { color: var(--trace-muted); font-size: 12.5px; margin-top: 6px; }
.confidence { text-align: right; }
.confidence-number { font-size: 20px; font-weight: 700; color: var(--trace-text); }
.confidence-label { color: var(--trace-muted); font-size: 12px; }
.verify-strip {
    margin-top: 14px; padding: 10px 12px;
    border: 1px solid rgba(233,169,79,.32); background: rgba(233,169,79,.07);
    border-radius: 9px; color: #f0c486; font-size: 13.5px; line-height: 1.45;
}
.verify-strip.ok { border-color: rgba(79,196,139,.3); background: rgba(79,196,139,.06); color: #8fdcb4; }

/* ---------- Basis matrix ---------- */
.basis-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin-bottom: 10px; }
.basis-tile {
    background: var(--trace-surface); border: 1px solid var(--trace-border);
    border-top: 3px solid var(--trace-border-strong);
    border-radius: 10px; padding: 13px 14px 12px; min-width: 0;
}
.basis-tile.green { border-top-color: var(--trace-green); }
.basis-tile.amber { border-top-color: var(--trace-amber); background: linear-gradient(180deg, rgba(233,169,79,.06), var(--trace-surface) 60%); }
.basis-tile.red { border-top-color: var(--trace-red); background: linear-gradient(180deg, rgba(240,114,114,.07), var(--trace-surface) 60%); }
.basis-tile.blue { border-top-color: var(--trace-blue); }
.basis-tile.gray { border-top-style: dashed; }
.basis-name { color: var(--trace-text-2); font-size: 12.5px; font-weight: 700; }
.basis-status { font-size: 15px; font-weight: 700; margin-top: 6px; }
.basis-strength { color: var(--trace-muted); font-size: 12.5px; margin-top: 2px; }
.basis-statement {
    color: var(--trace-text-2); font-size: 13px; line-height: 1.5; margin-top: 8px;
    display: -webkit-box; -webkit-line-clamp: 4; -webkit-box-orient: vertical; overflow: hidden;
}
.basis-foot { color: var(--trace-muted); font-size: 12px; margin-top: 8px; }

.signal-row { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin-bottom: 12px; }
.signal {
    border: 1px solid var(--trace-border); border-radius: 10px;
    background: var(--trace-bg-2); padding: 10px 13px;
    display: flex; align-items: baseline; gap: 10px;
}
.signal-value { font-size: 20px; font-weight: 700; color: var(--trace-text); }
.signal-label { color: var(--trace-text-2); font-size: 13px; line-height: 1.35; }

.open-issue {
    border: 1px solid rgba(233,169,79,.30); border-left: 3px solid var(--trace-amber);
    background: rgba(233,169,79,.05); border-radius: 10px; padding: 12px 15px; margin-bottom: 12px;
}
.open-issue-label { color: var(--trace-amber); font-size: 12px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
.open-issue-text { color: var(--trace-text); font-size: 14px; line-height: 1.55; margin-top: 4px; }

.confidence-context {
    display: flex; align-items: center; justify-content: space-between; gap: 18px;
    border: 1px solid var(--trace-border); border-radius: 10px;
    background: var(--trace-bg-2); padding: 11px 15px; margin: 0 0 16px;
}
.confidence-context-title { color: var(--trace-text-2); font-size: 13px; font-weight: 700; }
.confidence-context-copy { color: var(--trace-muted); font-size: 13px; line-height: 1.55; margin-top: 3px; }
.confidence-context-copy strong { color: var(--trace-text-2); }
.confidence-context-stat { flex: 0 0 auto; text-align: right; border-left: 1px solid var(--trace-border); padding-left: 16px; }
.confidence-context-stat strong { display: block; color: var(--trace-text); font-size: 18px; line-height: 1; }
.confidence-context-stat span { display: block; color: var(--trace-muted); font-size: 12px; margin-top: 4px; }

/* ---------- Tabs ---------- */
/* Covers both the BaseWeb tab DOM (Streamlit <= 1.4x) and the
   role="tab" DOM used by newer releases. */
[data-baseweb="tab-list"], [data-testid="stTabs"] [role="tablist"] {
    gap: 4px; border-bottom: 1px solid var(--trace-border);
}
[role="tab"] { padding: 10px 14px !important; }
[role="tab"] p { font-size: 14.5px !important; font-weight: 600 !important; color: var(--trace-text-2); }
[role="tab"][aria-selected="true"] p { color: var(--trace-text) !important; }
[data-baseweb="tab-highlight"] { background-color: var(--trace-accent) !important; }

/* ---------- Expanders ---------- */
div[data-testid="stExpander"] details {
    border: 1px solid var(--trace-border) !important;
    background: var(--trace-surface) !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] summary p { font-size: 14px; color: var(--trace-text); }

/* ---------- Timeline ---------- */
.tl { position: relative; }
.tl-year { display: grid; grid-template-columns: 74px 1fr; gap: 0 14px; }
.tl-year-label {
    color: var(--trace-text); font-size: 18px; font-weight: 700;
    font-variant-numeric: tabular-nums; padding-top: 1px; text-align: right;
}
.tl-year-events { border-left: 2px solid var(--trace-border-strong); padding: 0 0 6px 20px; }
.timeline-event { position: relative; padding: 0 0 13px 0; }
.timeline-event:before {
    content: ""; position: absolute; left: -27px; top: 5px;
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--trace-bg); border: 2px solid var(--trace-accent);
}
.timeline-date { color: var(--trace-accent-strong); font-size: 12.5px; font-weight: 700; font-variant-numeric: tabular-nums; }
.timeline-description { color: var(--trace-text); font-size: 14px; line-height: 1.5; margin-top: 2px; }
.timeline-source { color: var(--trace-muted); font-size: 12.5px; margin-top: 4px; }
.timeline-line { color: var(--trace-text-2); font-size: 13px; line-height: 1.45; margin-top: 3px; padding-left: 10px; border-left: 2px solid var(--trace-border); }
.timeline-source .src, .src-chip {
    display: inline-block; font-family: var(--trace-mono); font-size: 11.5px;
    color: var(--trace-text-2); border: 1px solid var(--trace-border-strong);
    border-radius: 5px; padding: 1px 6px; margin: 2px 4px 0 0; background: var(--trace-bg-2);
}
.tl-open { display: grid; grid-template-columns: 74px 1fr; gap: 0 14px; }
.tl-open-body {
    border-left: 2px dashed rgba(233,169,79,.55); padding: 0 0 4px 20px; position: relative;
}
.tl-open-body:before {
    content: "?"; position: absolute; left: -12px; top: 0;
    width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center;
    background: var(--trace-bg); border: 2px solid var(--trace-amber); color: var(--trace-amber);
    font-size: 11px; font-weight: 800;
}
.tl-open-label { color: var(--trace-amber); font-size: 12.5px; font-weight: 700; }
.tl-open-text { color: var(--trace-text); font-size: 14px; line-height: 1.5; margin-top: 2px; }
.tl-row {
    display: grid; grid-template-columns: 112px 1fr; gap: 12px;
    padding: 7px 0; border-bottom: 1px solid var(--trace-border); font-size: 13px;
}
.tl-row:last-child { border-bottom: 0; }
.tl-row .d { color: var(--trace-text-2); font-variant-numeric: tabular-nums; }
.tl-row .t { color: var(--trace-text-2); }

/* ---------- details/summary (native, keyboard accessible) ---------- */
.trace-details { margin-top: 6px; }
.trace-details summary {
    cursor: pointer; color: var(--trace-accent-strong); font-size: 12.5px; font-weight: 600;
    list-style: none; display: inline-block; border-radius: 4px;
}
.trace-details summary::-webkit-details-marker { display: none; }
.trace-details summary:before { content: "▸ "; }
.trace-details[open] summary:before { content: "▾ "; }
.passage {
    margin-top: 8px; border: 1px solid var(--trace-border); border-left: 2px solid var(--trace-accent);
    background: var(--trace-bg-2); border-radius: 8px; padding: 10px 12px;
}
.passage-head { color: var(--trace-text-2); font-size: 12.5px; font-weight: 600; }
.passage-head .src { font-family: var(--trace-mono); font-weight: 400; color: var(--trace-muted); }
.passage-text {
    max-height: 280px; overflow: auto;
    color: var(--trace-text-2); font-size: 13px; line-height: 1.6; margin-top: 6px;
    font-family: var(--trace-mono); font-size: 12.5px;
}
.passage-text mark { background: rgba(60,196,212,.22); color: var(--trace-text); border-radius: 2px; padding: 0 1px; }

/* ---------- Evidence & claims ---------- */
.evidence-card {
    border: 1px solid var(--trace-border); background: var(--trace-bg-2);
    border-radius: 10px; padding: 13px 15px; margin-bottom: 9px;
}
.evidence-top { display: flex; justify-content: space-between; gap: 12px; align-items: center; flex-wrap: wrap; }
.evidence-id { color: var(--trace-accent-strong); font-size: 12.5px; font-family: var(--trace-mono); }
.evidence-file { color: var(--trace-text); font-size: 14px; font-weight: 600; margin-top: 4px; line-height: 1.45; }
.evidence-path { color: var(--trace-muted); font-size: 12.5px; margin-top: 3px; font-family: var(--trace-mono); }
.evidence-content {
    color: var(--trace-text-2); font-size: 12.5px; line-height: 1.6; margin-top: 10px;
    font-family: var(--trace-mono); max-height: 320px; overflow: auto;
    border-top: 1px solid var(--trace-border); padding-top: 9px;
}
.evidence-content mark { background: rgba(60,196,212,.22); color: var(--trace-text); border-radius: 2px; }
.evidence-reason { color: var(--trace-muted); font-size: 12.5px; margin-top: 8px; line-height: 1.5; }
.evidence-reason b { color: var(--trace-text-2); font-weight: 600; }
.cited-by { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
.cited-by-item { font-size: 12.5px; color: var(--trace-text-2); line-height: 1.45; }

.claim-card {
    border: 1px solid var(--trace-border); background: var(--trace-surface);
    border-radius: 10px; padding: 13px 15px; margin-bottom: 9px;
}
.claim-head { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; justify-content: space-between; }
.claim-tags { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.claim-statement { color: var(--trace-text); font-size: 14.5px; line-height: 1.55; margin-top: 8px; font-weight: 500; }
.claim-meta { color: var(--trace-muted); font-size: 12.5px; margin-top: 6px; line-height: 1.5; }
.claim-why { color: var(--trace-muted); font-size: 12.5px; margin-top: 6px; font-style: italic; }
.group-head { display: flex; align-items: baseline; gap: 10px; margin: 16px 0 8px; }
.group-title { color: var(--trace-text); font-size: 15.5px; font-weight: 700; }
.group-count { color: var(--trace-muted); font-size: 13px; }
.group-explainer {
    color: var(--trace-text-2); font-size: 13px; line-height: 1.55;
    border: 1px dashed var(--trace-border-strong); border-radius: 8px; padding: 9px 12px; margin-bottom: 9px;
}
.empty-line { color: var(--trace-muted); font-size: 13px; padding: 4px 0 6px; }

/* ---------- Analyzer findings / path / metrics ---------- */
.finding-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 9px; }
.finding { border: 1px solid var(--trace-border); background: var(--trace-bg-2); border-radius: 9px; padding: 12px; }
.finding-name { font-size: 13.5px; color: var(--trace-text); line-height: 1.5; margin-top: 7px; }
.finding-meta { margin-top: 6px; color: var(--trace-muted); font-size: 12.5px; }
.path-item {
    display: grid; grid-template-columns: 30px 1fr auto; gap: 12px; align-items: center;
    padding: 10px 0; border-bottom: 1px solid var(--trace-border);
}
.path-item:last-child { border-bottom: 0; }
.path-index {
    width: 26px; height: 26px; border: 1px solid var(--trace-border-strong); border-radius: 7px;
    display: grid; place-items: center; color: var(--trace-text-2); font-size: 12px; font-variant-numeric: tabular-nums;
}
.path-question { color: var(--trace-text); font-size: 13.5px; line-height: 1.45; }
.path-status { font-size: 12px; font-weight: 700; color: var(--trace-green); white-space: nowrap; }
.path-status.warn { color: var(--trace-amber); }
.metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.metric { border: 1px solid var(--trace-border); background: var(--trace-bg-2); border-radius: 9px; padding: 12px; }
.metric-value { color: var(--trace-text); font-size: 20px; font-weight: 700; }
.metric-label { color: var(--trace-muted); font-size: 12.5px; margin-top: 2px; }

/* ---------- Next steps ---------- */
.gap-card {
    border: 1px solid rgba(233,169,79,.30); border-left: 3px solid var(--trace-amber);
    background: rgba(233,169,79,.05); border-radius: 10px; padding: 13px 15px; margin-bottom: 9px;
}
.gap-title { color: var(--trace-amber); font-size: 13px; font-weight: 700; }
.gap-copy { color: var(--trace-text); font-size: 14px; line-height: 1.55; margin-top: 4px; }
.human-action {
    border: 1px solid rgba(122,167,255,.28); border-left: 3px solid var(--trace-blue);
    background: rgba(122,167,255,.05); border-radius: 10px; padding: 13px 15px; margin-bottom: 9px;
}
.human-title { color: var(--trace-blue); font-size: 13px; font-weight: 700; }
.human-copy { color: var(--trace-text); font-size: 14px; line-height: 1.55; margin-top: 4px; }
.authority-note {
    border: 1px solid var(--trace-border-strong); border-radius: 10px; background: var(--trace-bg-2);
    padding: 11px 14px; color: var(--trace-text-2); font-size: 13.5px; line-height: 1.55; margin-bottom: 14px;
}
.authority-note strong { color: var(--trace-text); }
.oq { color: var(--trace-text); font-size: 14px; line-height: 1.5; padding-top: 8px; }

/* ---------- Home ---------- */
.landing-shell {
    border: 1px solid var(--trace-border); border-radius: var(--trace-radius);
    background: var(--trace-surface); padding: 16px 18px; margin: 6px 0 4px;
}
.how-grid { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; }
.how-step { border-left: 2px solid var(--trace-border-strong); padding: 2px 0 2px 12px; }
.how-num { color: var(--trace-accent-strong); font-family: var(--trace-mono); font-size: 12px; font-weight: 700; }
.how-title { color: var(--trace-text); font-size: 14px; font-weight: 650; margin-top: 2px; }
.how-copy { color: var(--trace-muted); font-size: 12.5px; line-height: 1.5; margin-top: 3px; }
.case-intro { margin: 22px 0 10px; }
.case-intro-title { color: var(--trace-text); font-size: 16px; font-weight: 700; }
.case-intro-copy { color: var(--trace-muted); font-size: 13.5px; line-height: 1.5; margin-top: 3px; }
.case-card {
    border: 1px solid var(--trace-border); background: var(--trace-surface);
    border-radius: 10px; padding: 14px 16px; min-height: 128px; margin-bottom: 8px;
}
.case-num { color: var(--trace-accent-strong); font-family: var(--trace-mono); font-size: 12px; font-weight: 700; }
.case-title { color: var(--trace-text); font-size: 15.5px; font-weight: 650; margin-top: 5px; }
.case-copy { color: var(--trace-muted); font-size: 13px; line-height: 1.5; margin-top: 5px; }
.case-outcome { margin-top: 8px; }

/* ---------- List-style buttons (thread, suggestions) ---------- */
.st-key-history button, .st-key-suggestions button, .st-key-examples button,
.st-key-history button > div, .st-key-suggestions button > div, .st-key-examples button > div {
    justify-content: flex-start !important; text-align: left !important;
}
.st-key-history button p, .st-key-suggestions button p, .st-key-examples button p {
    text-align: left !important; font-size: 13.5px !important; line-height: 1.4 !important;
}
.st-key-history button, .st-key-examples button {
    font-weight: 500 !important; background: transparent !important; color: var(--trace-text-2) !important;
}
.st-key-suggestions button { font-weight: 500 !important; }

/* ---------- Thread ---------- */
.thread-title { color: var(--trace-text); font-size: 17px; font-weight: 700; margin: 26px 0 2px; }
.thread-note { color: var(--trace-muted); font-size: 13px; line-height: 1.5; margin-bottom: 10px; }
.thread-answer { color: var(--trace-text-2); font-size: 14px; line-height: 1.55; margin-top: 6px; }
[data-testid="stChatMessage"] { background: var(--trace-surface) !important; border: 1px solid var(--trace-border); border-radius: 12px; }
[data-testid="stChatInput"] textarea { font-size: 15px !important; }
[data-testid="stChatInput"], [data-testid="stChatInput"] > div {
    background: var(--trace-surface-2) !important;
    border-color: var(--trace-border-strong) !important;
}
[data-testid="stChatInput"] textarea { color: var(--trace-text) !important; background: transparent !important; }
[data-testid="stChatInput"]:focus-within { box-shadow: 0 0 0 2px rgba(60,196,212,.35) !important; border-radius: 12px; }
[data-testid="stChatMessage"] p { color: var(--trace-text); }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] {
    background: var(--trace-surface-3) !important; color: var(--trace-accent-strong) !important;
    border: 1px solid var(--trace-border-strong);
}
[data-testid="stToast"] { background: var(--trace-surface-3) !important; color: var(--trace-text) !important; border: 1px solid var(--trace-border-strong); }

/* ---------- Errors / progress ---------- */
.trace-error {
    border: 1px solid rgba(240,114,114,.35); border-left: 3px solid var(--trace-red);
    background: rgba(240,114,114,.06); border-radius: 10px; padding: 13px 15px; margin: 6px 0 10px;
}
.trace-error.warn { border-color: rgba(233,169,79,.35); border-left-color: var(--trace-amber); background: rgba(233,169,79,.06); }
.trace-error-title { color: var(--trace-text); font-size: 15px; font-weight: 700; }
.trace-error-copy { color: var(--trace-text-2); font-size: 13.5px; line-height: 1.55; margin-top: 4px; }

/* ---------- Status additions ---------- */
.sys-dot.fail { background: var(--trace-red); }
.side-warn {
    margin-top: 10px; border: 1px solid rgba(233,169,79,.32); background: rgba(233,169,79,.06);
    border-radius: 8px; padding: 8px 10px; color: #f0c486; font-size: 12.5px; line-height: 1.5;
}
.side-warn code { font-size: 11.5px; color: var(--trace-text); background: transparent; }
.trace-result-chip.warn { border-color: rgba(233,169,79,.45); color: var(--trace-amber); }
.trace-result-chip.context { border-color: rgba(122,167,255,.45); color: var(--trace-blue); }

/* ---------- Citations ---------- */
.cite-label { color: var(--trace-muted); font-size: 12.5px; margin-bottom: 4px; }
details.cite { margin: 3px 0; }
details.cite > summary {
    cursor: pointer; list-style: none; display: inline-block; border-radius: 5px;
}
details.cite > summary::-webkit-details-marker { display: none; }
details.cite > summary .src {
    border: 1px solid var(--trace-border-strong); border-radius: 5px; padding: 1px 7px;
    background: var(--trace-bg-2); color: var(--trace-accent-strong);
}
details.cite > summary:hover .src { border-color: var(--trace-accent); }
details.cite > summary:focus-visible { outline: 2px solid var(--trace-accent-strong); outline-offset: 2px; }
details.cite[open] > summary .src { background: var(--trace-accent-soft); }

/* ---------- Conversation ---------- */
.conv-strip { border: 1px solid var(--trace-border); border-radius: 10px; background: var(--trace-bg-2); padding: 4px 12px; margin-bottom: 8px; }
.strip-row { display: grid; grid-template-columns: 56px auto 1fr auto; gap: 10px; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--trace-border); }
.strip-row:last-child { border-bottom: 0; }
.strip-row .badge { font-size: 11px; padding: 2px 7px; }
.strip-n { color: var(--trace-muted); font-size: 12px; font-weight: 700; font-variant-numeric: tabular-nums; }
.strip-q { color: var(--trace-text-2); font-size: 13.5px; line-height: 1.4; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.strip-tag { color: var(--trace-muted); font-size: 12px; white-space: nowrap; }
.strip-row.current .strip-q { color: var(--trace-text); font-weight: 600; }
.strip-row.current .strip-tag { color: var(--trace-accent-strong); font-weight: 600; }
.thread-turn-tag { color: var(--trace-muted); font-size: 12px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; margin-bottom: 4px; }
.thread-question { color: var(--trace-text); font-size: 14.5px; line-height: 1.5; }
.thread-current {
    color: var(--trace-accent-strong); font-size: 12.5px; font-weight: 600;
    border-left: 2px solid var(--trace-accent); padding: 4px 10px; margin: 6px 0 10px;
}

/* ---------- Live progress (backend-reported stages) ---------- */
.pt {
    border: 1px solid var(--trace-border-strong); border-left: 3px solid var(--trace-accent);
    background: var(--trace-surface); border-radius: 12px; padding: 15px 18px; margin: 6px 0 12px;
}
.pt-top { display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }
.pt-k { color: var(--trace-accent-strong); font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
.pt-q { color: var(--trace-text); font-size: 15px; font-weight: 600; line-height: 1.45; margin-top: 3px; }
.pt-clock { text-align: right; font-variant-numeric: tabular-nums; flex: 0 0 auto; }
.pt-clock span { display: block; font-size: 20px; font-weight: 700; color: var(--trace-text); }
.pt-clock small { color: var(--trace-muted); font-size: 11.5px; }
.pt-steps { list-style: none; padding: 0; margin: 12px 0 8px; display: flex; flex-wrap: wrap; gap: 6px; }
.pt-steps li {
    font-size: 13px; color: var(--trace-muted); border: 1px solid var(--trace-border);
    border-radius: 999px; padding: 4px 11px 4px 9px; display: inline-flex; align-items: center; gap: 7px;
}
.pt-steps li .mk { font-size: 11px; }
.pt-steps li .t { color: var(--trace-muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.pt-steps li.done { color: var(--trace-text-2); }
.pt-steps li.done .mk { color: var(--trace-green); }
.pt-steps li.on { color: var(--trace-text); border-color: var(--trace-accent); }
.pt-steps li.on .mk { color: var(--trace-accent); animation: trace-pulse 1.3s ease-in-out infinite; }
.pt-note { color: var(--trace-muted); font-size: 12.5px; line-height: 1.5; margin-top: 4px; }
@keyframes trace-pulse { 0%,100% { opacity: 1; } 50% { opacity: .3; } }

/* ---------- Diagnostics ---------- */
.diag-row { display: grid; grid-template-columns: 220px 1fr; gap: 12px; padding: 6px 0; border-bottom: 1px solid var(--trace-border); font-size: 13px; }
.diag-row:last-child { border-bottom: 0; }
.diag-row .k { color: var(--trace-muted); }
.diag-row .v { color: var(--trace-text-2); word-break: break-word; }

@media (max-width: 1100px) {
    .basis-grid, .signal-row { grid-template-columns: repeat(2, minmax(0,1fr)); }
    .how-grid { grid-template-columns: repeat(2, minmax(0,1fr)); }
}
@media (max-width: 700px) {
    .basis-grid, .signal-row, .finding-grid, .metric-row, .how-grid { grid-template-columns: 1fr; }
    .assessment-row, .confidence-context { flex-direction: column; align-items: flex-start; }
    .confidence, .confidence-context-stat { text-align: left; border-left: 0; padding-left: 0; }
    .tl-year, .tl-open { grid-template-columns: 1fr; }
    .tl-year-label { text-align: left; margin-bottom: 6px; }
    .tl-row { grid-template-columns: 1fr; gap: 2px; }
    .diag-row { grid-template-columns: 1fr; gap: 2px; }
    .strip-row { grid-template-columns: auto 1fr; }
    .strip-row .strip-q { grid-column: 1 / -1; white-space: normal; }
    .strip-row .strip-tag { display: none; }
    .pt-top { flex-direction: column; }
    .pt-clock { text-align: left; }
    .trace-result-question { font-size: 17px; }
    .direct-answer-text { font-size: 16px; }
    .block-container, [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
}
@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Construct the existing TRACE pipeline once per Streamlit process.

    The UI deliberately delegates investigation behavior to
    app.investigation.build.build_investigator().

    Vector retrieval is enabled only when ALL of these hold:
      1. the persisted index loads and its chunk IDs/dimensions match the
         current corpus (app.retrieval.vector_health.load_index);
      2. a real query embedding succeeds with the live provider and has
         the index's dimensionality (probe_query_embedding).
    An index file or an API key on its own is never reported as
    "vector on". Any failure leaves lexical + entity retrieval running
    and is surfaced in meta["vector_status"] for the UI.
    """
    from app.retrieval.vector_health import (
        LEGACY_INDEX_MODEL,
        load_index,
        probe_query_embedding,
    )

    documents = load_corpus(str(CORPUS_PATH))
    chunks = chunk_corpus(documents)

    bm25 = BM25Searcher(chunks)
    entities = EntityMatcher(chunks)

    vector_searcher = None
    index_path = Path(INDEX_PATH)
    api_key_present = bool(os.getenv("GEMINI_API_KEY"))

    vector_status = {
        "state": "off",
        "reason": "",
        "api_key_present": api_key_present,
        "index": {},
        "stale_chunk_ids": [],
        "probe_dimension": None,
    }

    print(
        f"[TRACE VECTOR DIAG] "
        f"api_key_present={api_key_present} "
        f"index_path={index_path} "
        f"index_exists={index_path.exists()} "
        f"documents={len(documents)} chunks={len(chunks)}",
        flush=True,
    )

    health, _index_ids, embeddings = load_index(index_path, chunks)
    vector_status["index"] = health.summary()
    vector_status["stale_chunk_ids"] = list(health.stale_chunk_ids)

    print(
        f"[TRACE VECTOR DIAG] "
        f"index_format={health.index_format or 'n/a'} "
        f"index_chunks={health.records} "
        f"embeddings={len(embeddings)} "
        f"dimension={health.dimension} "
        f"chunk_ids_match={health.ids_match} "
        f"stale_chunks={len(health.stale_chunk_ids)}",
        flush=True,
    )

    if not health.present:
        vector_status["reason"] = "Embedding index file not found."
        print("[TRACE VECTOR DIAG] INDEX_MISSING", flush=True)
    elif not health.usable:
        vector_status["reason"] = (
            health.error
            or ("Index chunk IDs do not match the current corpus."
                if not health.ids_match
                else "Index embeddings have inconsistent dimensions.")
        )
        print("[TRACE VECTOR DIAG] CHUNK_ID_MISMATCH" if not health.ids_match
              else f"[TRACE VECTOR DIAG] INDEX_UNUSABLE {health.error}", flush=True)
    elif not api_key_present:
        vector_status["reason"] = "GEMINI_API_KEY is not set, so queries cannot be embedded."
        print("[TRACE VECTOR DIAG] API_KEY_MISSING", flush=True)
    else:
        try:
            from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
            from app.retrieval.vector_search import VectorSearcher

            # Bounded for the UI: one retry and a 20 s timeout per query
            # embedding, so a slow or rate-limited embedding call degrades
            # to a recorded vector failure instead of stalling the demo.
            provider = GeminiEmbeddingProvider(
                model=health.model or LEGACY_INDEX_MODEL,
                max_retries=1,
                base_delay_seconds=0.5,
                timeout_seconds=20,
            )
            probe = probe_query_embedding(provider, health.dimension)
            vector_status["probe_dimension"] = probe.dimension

            if probe.ok:
                # Pair index embeddings with the CURRENT corpus chunks
                # (IDs verified identical above), so passage text handed
                # to the reasoner always matches the corpus on disk.
                vector_searcher = VectorSearcher(
                    chunks=chunks,
                    embeddings=embeddings,
                    embedding_provider=provider,
                )
                vector_status["state"] = "on"
                print(
                    f"[TRACE VECTOR DIAG] VECTOR_INITIALIZED query_probe=ok dim={probe.dimension}",
                    flush=True,
                )
            else:
                vector_status["state"] = "failed"
                vector_status["reason"] = f"Live query embedding failed: {probe.error}"
                print(f"[TRACE VECTOR DIAG] QUERY_PROBE_FAILED {probe.error}", flush=True)
        except Exception as exc:
            vector_status["state"] = "failed"
            vector_status["reason"] = f"{type(exc).__name__}: {exc}"[:400]
            print(
                f"[TRACE VECTOR INIT ERROR] "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            vector_searcher = None

    retriever = HybridRetriever(
        bm25_searcher=bm25,
        vector_searcher=vector_searcher,
        entity_matcher=entities,
    )

    # Keep UI retries short: a judge should see a clear transient Gemini
    # failure quickly instead of waiting through several exponential retries.
    from app.investigation.gemini_client import GeminiReasoningClient

    ui_gemini = GeminiReasoningClient(
        max_retries=1,
        base_delay_seconds=0.5,
        timeout_seconds=120,
    )

    return build_investigator(retriever, gemini_client=ui_gemini), {
        "documents": len(documents),
        "chunks": len(chunks),
        "vector_enabled": vector_searcher is not None,
        "vector_status": vector_status,
        "retriever": retriever,
        # When these startup checks ran (they are cached for the process and
        # re-run only via the explicit "Re-check" action).
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _badge(label: str, tone: str) -> str:
    return f'<span class="badge {tone}">{_escape(label)}</span>'


def _confidence_level(confidence: float) -> str:
    """
    Map a calibrated confidence score to a qualitative label, for the
    top-level hero display ONLY.

    This performs no recalculation of confidence and does not alter the
    `confidence` value itself -- it only chooses which of three fixed
    words to display in place of the raw percentage next to the headline
    assessment. The raw percentage is still shown everywhere else in the
    UI (each individual claim's confidence, and the "What the confidence
    applies to" explanation), unchanged. This exists purely to avoid the
    optical contradiction of a high top-level number sitting next to a
    low individual-claim percentage on first skim -- it changes nothing
    about what was actually calibrated.
    """
    if confidence >= 0.85:
        return "High confidence"
    if confidence >= 0.60:
        return "Moderate confidence"
    return "Low confidence"


def _evidence_strength(confidence: float) -> str:
    """
    Map a calibrated confidence score to a plain evidence-sufficiency
    word, for the top-level hero display ONLY. Reuses `_confidence_level`'s
    thresholds (no new calibration, no recalculation) but deliberately
    uses different wording for a different question.

    `_confidence_level` (still used/tested on its own, unchanged above)
    answers "how confident is TRACE in this specific stated assessment"
    -- a percentage-flavored framing that a judge skimming several
    different claims' percentages can misread as "there is a 30% chance
    this conclusion is correct", when NOT_ESTABLISHED actually means
    "the evidence needed to establish this was not found", a yes/no
    finding, not a probability. "Evidence strength: Insufficient" asks
    the second question directly and cannot be misread as the first.

    The raw percentage remains fully visible elsewhere -- every
    individual claim's confidence (render_claims) and the "What the
    confidence applies to" explanation (render_confidence_context) are
    both untouched by this function or its use in the hero.
    """
    if confidence >= 0.85:
        return "Strong"
    if confidence >= 0.60:
        return "Moderate"
    return "Insufficient"


# Explicit phrasings that mean the question is specifically asking whether
# the CURRENT technical basis for a parameter/limit is established or
# re-justified -- not merely a question that happens to mention a
# technical change or event in passing (e.g. "...after the PX-400
# replacement..." is an equipment/administrative reference, not this).
# Phrasings that unambiguously mean the question is about the CURRENT
# (post-replacement) technical basis, regardless of whether the question
# also mentions "original"/"established" in passing (the hero question
# does exactly this: "...what changed since that limit was established,
# and is its current technical basis still supported?"). These always
# win, because there is no legitimate "original basis" reading of them.
_CURRENT_JUSTIFICATION_STRONG_TRIGGERS = (
    "current technical basis",
    "currently justified",
    "currently supported",
    "still supported",
    "still justified",
    "still valid",
    "re-justification",
    "re justification",
    "rejustification",
    "re-justified",
    "rejustified",
    "revalidation",
    "re-validation",
    "revalidated",
    "re-validated",
    "re-derivation",
    "re derivation",
    "rederivation",
    "re-derived",
    "rederived",
    "after the px-400",
    "after px-400",
    "px-400 replacement",
    "after replacement",
    "technically supported",
    "technically valid",
)

# Phrasings that are ambiguous on their own: "technical basis" or
# "technical justification" can equally describe the ORIGINAL 2001
# derivation ("what was the original technical basis for setting the
# UCL...") or the CURRENT re-justification question. These only count
# as a current-justification signal when no original-basis marker (see
# _ORIGINAL_BASIS_MARKERS) is also present in the question.
_TECHNICAL_JUSTIFICATION_WEAK_TRIGGERS = (
    "technical justification",
    "technically justified",
    "technical rationale",
    "technical basis",
)

# "original"/"originally" already doubles as a scoring term for
# ORIGINAL_BASIS below; reusing it here (rather than inventing a new,
# separate marker set) keeps the "is this question about the original
# derivation" check consistent with the rest of this function.
_ORIGINAL_BASIS_MARKERS = ("original", "originally")


def _question_targets_original_basis(question: str) -> bool:
    """True when the question explicitly asks about the ORIGINAL basis/
    derivation/establishment of a limit (e.g. "...in 2001"), as opposed
    to its current re-justification."""
    return any(marker in question for marker in _ORIGINAL_BASIS_MARKERS)


def _question_targets_technical_justification(question: str) -> bool:
    """
    True when the question explicitly targets the CURRENT technical
    basis/re-justification -- a narrow, explicit phrase match, not a
    scoring signal. It exists so that an unambiguous "is the current
    technical basis established?" question always wins the
    primary-claim slot, instead of being out-scored by an unrelated but
    higher-confidence claim whose keyword terms happen to also appear in
    the question's wording (the original bug: "replacement" matching
    ADMINISTRATIVE_ACTION's terms even though the question was really
    asking about current technical justification).

    A strong trigger (see _CURRENT_JUSTIFICATION_STRONG_TRIGGERS) always
    wins. A weak/ambiguous trigger (bare "technical basis" or "technical
    justification") only counts when the question does NOT also contain
    an original-basis marker -- otherwise "what was the ORIGINAL
    technical basis ... in 2001?" would incorrectly be forced onto the
    CURRENT_TECHNICAL_JUSTIFICATION claim (the bug this fixes).
    """
    if any(trigger in question for trigger in _CURRENT_JUSTIFICATION_STRONG_TRIGGERS):
        return True

    if any(trigger in question for trigger in _TECHNICAL_JUSTIFICATION_WEAK_TRIGGERS):
        return not _question_targets_original_basis(question)

    return False


def _primary_claim(result):
    """Choose the assessment claim that best matches the user's question."""
    claims = list(getattr(result.report, "claims", []) or [])
    if not claims:
        return None

    question = getattr(result.investigation, "question", "").lower()

    # An explicit technical-justification question must win outright,
    # before any scoring: the keyword-overlap scoring below can otherwise
    # be tipped toward an unrelated but higher-confidence claim. This
    # only fires on an explicit phrase match (see
    # _question_targets_technical_justification); any other question
    # (asking about original basis, equipment/capability, or an
    # administrative decision) falls through to the unchanged scoring
    # below exactly as before.
    if _question_targets_technical_justification(question):
        technical_claim = next(
            (
                claim for claim in claims
                if claim.claim_type == ClaimType.CURRENT_TECHNICAL_JUSTIFICATION
            ),
            None,
        )
        if technical_claim is not None:
            return technical_claim

    # Symmetrically, an explicit original-basis question (e.g. "what was
    # the original technical basis ... in 2001?") must win outright onto
    # ORIGINAL_BASIS, for the same reason: without this, "basis" and
    # "original" scoring terms can still be out-weighed by an unrelated
    # higher-confidence claim below.
    elif _question_targets_original_basis(question):
        original_claim = next(
            (
                claim for claim in claims
                if claim.claim_type == ClaimType.ORIGINAL_BASIS
            ),
            None,
        )
        if original_claim is not None:
            return original_claim

    # Prefer the claim type that directly answers the question rather than
    # always prioritizing current technical justification.
    type_terms = {
        ClaimType.ORIGINAL_BASIS: (
            "why", "original", "established", "basis", "rationale", "history",
        ),
        ClaimType.CURRENT_CAPABILITY: (
            "what equipment", "which equipment", "what changed", "controller",
            "sensor", "capability", "performance", "installed",
        ),
        ClaimType.ADMINISTRATIVE_ACTION: (
            "approved", "approval", "rejected", "proposal", "decision",
            "replacement", "change order", "eco",
        ),
        ClaimType.CURRENT_TECHNICAL_JUSTIFICATION: (
            "technically justified", "technical justification", "revalidated",
            "revalidation", "revalidated", "re-validated", "re-justified",
            "re-justification", "re-derived", "re-derived", "still supported",
            "current technical basis", "prove the limit", "does that prove",
        ),
        ClaimType.EVIDENCE_GAP: (
            # Deliberately excludes the bare word "evidence": this is an
            # evidence-investigation tool, so "evidence" appears in nearly
            # every question (e.g. "does the evidence establish..."). As a
            # bare keyword it was a near-universal false-positive match that
            # could outscore every real-answer claim's confidence and hijack
            # the hero onto an unrelated, low-confidence gap finding. Only
            # phrases that actually signal the question is about a gap or
            # missing evidence should count.
            "evidence gap", "missing evidence", "what evidence",
        ),
        ClaimType.RECOMMENDATION: (
            "recommend", "should we", "what should", "action",
        ),
    }

    scored = []
    for claim in claims:
        score = 0
        terms = type_terms.get(claim.claim_type, ())
        score += sum(3 for term in terms if term in question)

        # Direct factual questions should not be hijacked by a
        # NOT_ESTABLISHED technical-justification claim.
        if (
            claim.claim_type == ClaimType.CURRENT_TECHNICAL_JUSTIFICATION
            and any(question.startswith(prefix) for prefix in ("what ", "which ", "who ", "when "))
            and not any(term in question for term in type_terms[ClaimType.CURRENT_TECHNICAL_JUSTIFICATION])
        ):
            score -= 8

        score += float(getattr(claim, "confidence", 0.0) or 0.0)
        scored.append((score, claim))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def _claim_for_type(result, claim_type):
    matches = [
        c for c in result.report.claims
        if c.claim_type == claim_type
    ]
    return max(matches, key=lambda c: c.confidence) if matches else None


def _assessment(result) -> tuple[str, str, str, float, bool]:
    claim = _primary_claim(result)

    if claim is None:
        return (
            "INVESTIGATION INCONCLUSIVE",
            "TRACE did not produce a claim strong enough to summarize the investigation.",
            "gray",
            0.0,
            True,
        )

    label, tone = STANCE_META.get(
        claim.stance,
        ("UNKNOWN", "gray"),
    )

    requires_human = (
        claim.stance in {
            EvidenceStance.NOT_ESTABLISHED,
            EvidenceStance.CONTRADICTS,
            EvidenceStance.UNKNOWN,
        }
        or claim.claim_type in {
            ClaimType.EVIDENCE_GAP,
            ClaimType.RECOMMENDATION,
        }
    )

    return (
        label,
        claim.statement,
        tone,
        claim.confidence,
        requires_human,
    )


def _human_action(result) -> str:
    gap_claim = _claim_for_type(result, ClaimType.EVIDENCE_GAP)
    recommendation = _claim_for_type(result, ClaimType.RECOMMENDATION)

    if recommendation:
        return recommendation.statement

    if gap_claim:
        return (
            "Verify the missing evidence identified by TRACE before "
            "treating the current technical basis as established."
        )

    if result.report.unresolved_questions:
        return result.report.unresolved_questions[0]

    return (
        "No additional human verification step was generated by the "
        "available evidence."
    )


# ---------------------------------------------------------------------------
# Safe accessors and small formatting helpers
# ---------------------------------------------------------------------------
#
# Every renderer below reads result fields through these helpers so that a
# partially-populated result (an older report object, a test double, or a
# report missing an optional field) degrades to an empty section instead of
# raising inside the UI.

def _claims(result) -> list:
    return list(getattr(getattr(result, "report", None), "claims", []) or [])


def _evidence_items(result) -> list:
    return list(getattr(getattr(result, "investigation", None), "evidence", []) or [])


def _findings(result) -> list:
    return list(getattr(getattr(result, "evidence_analysis", None), "findings", []) or [])


def _unresolved(result) -> list[str]:
    return [
        str(q).strip()
        for q in (getattr(getattr(result, "report", None), "unresolved_questions", []) or [])
        if str(q).strip()
    ]


def _finding_kind(finding) -> str:
    kind = getattr(finding, "finding_type", "")
    return str(getattr(kind, "value", kind) or "")


def _evidence_by_id(result) -> dict:
    return {
        getattr(item, "chunk_id", ""): item
        for item in _evidence_items(result)
    }


def _one_line(html: str) -> str:
    """Collapse authored multi-line HTML into a single physical line.

    Streamlit's markdown renderer ends a raw-HTML block at the first blank
    line and renders everything after it as literal text (the root cause
    of the earlier HTML-leak bug). Collapsing authored markup to one line
    removes that failure mode by construction.
    """
    return " ".join(
        line.strip() for line in str(html).splitlines() if line.strip()
    )


def _md(html: str) -> None:
    st.markdown(_one_line(html), unsafe_allow_html=True)


def _content_html(text: str, limit: int | None = None, highlight: str = "") -> str:
    """Escape document text for display inside an HTML block.

    Newlines become <br> so that paragraph breaks inside a document excerpt
    can never terminate the surrounding raw-HTML block.
    """
    value = str(text or "").strip()
    if limit is not None and len(value) > limit:
        value = value[:limit].rstrip() + "…"
    value = re.sub(r"\n{3,}", "\n\n", value)
    escaped = _escape(value)
    if highlight:
        terms = [t for t in re.split(r"\s+", highlight.strip()) if len(t) >= 2]
        for term in terms[:6]:
            escaped = re.sub(
                "(" + re.escape(_escape(term)) + ")",
                r"<mark>\1</mark>",
                escaped,
                flags=re.IGNORECASE,
            )
    return escaped.replace("\r", "").replace("\n", "<br>")


def _passage_window(content: str, needle: str, radius: int = 320) -> str:
    """Return the part of a chunk around the line a timeline event came from.

    Falls back to the start of the chunk when the line cannot be located.
    Purely a display window over the real chunk text.
    """
    text = str(content or "")
    probe = " ".join(str(needle or "").split())[:60]
    index = -1
    if probe:
        flat = " ".join(text.split())
        flat_index = flat.find(probe)
        if flat_index >= 0:
            text = flat
            index = flat_index
    if index < 0:
        return text[: radius * 2].strip() + ("…" if len(text) > radius * 2 else "")
    start = max(0, index - radius // 2)
    end = min(len(text), index + len(probe) + radius)
    return ("…" if start > 0 else "") + text[start:end].strip() + ("…" if end < len(text) else "")


def _source_category(relative_path: str) -> str | None:
    top = str(relative_path or "").replace("\\", "/").split("/", 1)[0]
    return SOURCE_CATEGORY_LABELS.get(top)


def _status(stance) -> tuple[str, str, str]:
    return STATUS_META.get(stance, ("Inconclusive", "gray", "?"))


def _status_badge(stance) -> str:
    label, tone, glyph = _status(stance)
    return f'<span class="badge {tone}"><span aria-hidden="true">{glyph}</span>{_escape(label)}</span>'


def _claim_type_label(claim) -> str:
    ctype = getattr(claim, "claim_type", None)
    return CLAIM_TYPE_LABELS.get(ctype, "Untyped claim")


def _requires_human(claim) -> bool:
    """Same rule _assessment() uses for the headline verification flag."""
    return (
        getattr(claim, "stance", None) in {
            EvidenceStance.NOT_ESTABLISHED,
            EvidenceStance.CONTRADICTS,
            EvidenceStance.UNKNOWN,
        }
        or getattr(claim, "claim_type", None) in {
            ClaimType.EVIDENCE_GAP,
            ClaimType.RECOMMENDATION,
        }
    )


def _format_date(value) -> tuple[str, str]:
    """Return (year, readable date) for a date object or ISO-like string."""
    if hasattr(value, "strftime"):
        return str(value.year), value.strftime("%d %b %Y")
    text = str(value or "")
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        from datetime import date as _date
        try:
            parsed = _date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
            return str(parsed.year), parsed.strftime("%d %b %Y")
        except ValueError:
            pass
    year = re.match(r"^(\d{4})", text)
    return (year.group(1) if year else "—"), text


def _date_label(value, precision: str = "day") -> str:
    """Readable date that never shows a placeholder day: month-precision
    dates (source wrote only "Sept 2001") render as "Sep 2001"."""
    if precision == "month" and hasattr(value, "strftime"):
        return value.strftime("%b %Y")
    return _format_date(value)[1]


def _source_chips(result, chunk_ids, limit: int = 3) -> str:
    by_id = _evidence_by_id(result)
    # One chip per source document; the chunk IDs stay in the tooltip.
    labels: dict[str, list[str]] = {}
    for chunk_id in list(chunk_ids or []):
        item = by_id.get(chunk_id)
        label = getattr(item, "filename", None) or chunk_id
        labels.setdefault(label, []).append(chunk_id)
    chips = [
        f'<span class="src" title="{_escape(", ".join(ids))}">{_escape(label)}</span>'
        for label, ids in list(labels.items())[:limit]
    ]
    extra = len(labels) - limit
    if extra > 0:
        chips.append(f'<span class="small-muted">+{extra} more</span>')
    return "".join(chips)


def _passages_details(result, chunk_ids, summary: str, needle: str = "", limit_chars: int = 900) -> str:
    """A native <details> drawer listing the real cited passages.

    Only chunk IDs that exist in the retrieved evidence are shown; an ID
    that does not resolve is listed as unresolved rather than dropped
    silently, so provenance problems stay visible.
    """
    by_id = _evidence_by_id(result)
    blocks = []
    for chunk_id in list(chunk_ids or []):
        item = by_id.get(chunk_id)
        if item is None:
            blocks.append(
                f'<div class="passage"><div class="passage-head">Unresolved citation '
                f'<span class="src">{_escape(chunk_id)}</span></div></div>'
            )
            continue
        text = (
            _passage_window(getattr(item, "content", ""), needle)
            if needle
            else getattr(item, "content", "")
        )
        blocks.append(
            '<div class="passage">'
            f'<div class="passage-head">{_escape(getattr(item, "filename", ""))} '
            f'<span class="src">· {_escape(chunk_id)}</span></div>'
            f'<div class="passage-text">{_content_html(text, limit_chars)}</div>'
            "</div>"
        )
    if not blocks:
        return ""
    return (
        '<details class="trace-details">'
        f"<summary>{_escape(summary)}</summary>"
        + "".join(blocks)
        + "</details>"
    )


# ---------------------------------------------------------------------------
# Derived (presentation-only) views over the result
# ---------------------------------------------------------------------------

_EQUIPMENT_TOKEN = re.compile(r"\b[A-Z][A-Za-z]{1,12}-\d[\w.\-]*\b")
_TEMPERATURE_TOKEN = re.compile(r"(\d{2,3}(?:\.\d+)?)\s?°\s?C")


def _question_subjects(result) -> list[tuple[str, int]]:
    """Equipment IDs / temperatures named in the question that also appear
    in the retrieved evidence. A subject is only shown when at least one
    retrieved passage actually contains it, so the chip is a fact about the
    evidence, not a guess about the question."""
    question = str(getattr(getattr(result, "investigation", None), "question", "") or "")
    evidence = _evidence_items(result)
    subjects: list[tuple[str, int]] = []
    seen = set()

    for token in _EQUIPMENT_TOKEN.findall(question):
        key = token.lower().rstrip(".,;:?")
        if key in seen:
            continue
        seen.add(key)
        count = sum(1 for e in evidence if key in str(getattr(e, "content", "")).lower())
        if count:
            subjects.append((token.rstrip(".,;:?"), count))

    for number in _TEMPERATURE_TOKEN.findall(question):
        if number in seen:
            continue
        seen.add(number)
        pattern = re.compile(r"(?<![\d.])" + re.escape(number) + r"(?![\d])")
        count = sum(1 for e in evidence if pattern.search(str(getattr(e, "content", ""))))
        if count:
            subjects.append((f"{number}°C", count))

    return subjects


def _basis_claim(result, claim_type):
    """Pick the claim shown in one basis-matrix tile.

    For the current-technical-justification row the most conservative
    stance wins (not established / contradicted / inconclusive before
    supported), so a single optimistic claim can never visually outrank a
    verified gap. Other rows show the highest-confidence claim of the type.
    Returns (claim, number_of_other_claims_with_a_different_stance).
    """
    matches = [c for c in _claims(result) if getattr(c, "claim_type", None) == claim_type]
    if not matches:
        return None, 0

    if claim_type == ClaimType.CURRENT_TECHNICAL_JUSTIFICATION:
        order = {
            EvidenceStance.NOT_ESTABLISHED: 0,
            EvidenceStance.CONTRADICTS: 1,
            EvidenceStance.UNKNOWN: 2,
            EvidenceStance.CONTEXT_ONLY: 3,
            EvidenceStance.SUPPORTS: 4,
        }
        chosen = sorted(
            matches,
            key=lambda c: (order.get(c.stance, 2), -float(c.confidence or 0.0)),
        )[0]
    else:
        chosen = max(matches, key=lambda c: float(getattr(c, "confidence", 0.0) or 0.0))

    differing = sum(1 for c in matches if c is not chosen and c.stance != chosen.stance)
    return chosen, differing


def _gap_items(result) -> list[tuple[str, list[str], str]]:
    """(statement, chunk_ids, origin) for every evidence gap the backend
    reported: deterministic analyzer gaps first, then reasoning claims of
    type EVIDENCE_GAP."""
    items = []
    for finding in _findings(result):
        if _finding_kind(finding) == "evidence-gap":
            items.append((finding.statement, list(finding.evidence_chunk_ids or []), "Analyzer finding"))
    for claim in _claims(result):
        if getattr(claim, "claim_type", None) == ClaimType.EVIDENCE_GAP:
            items.append((_display_claim_statement(claim), list(claim.evidence_chunk_ids or []), "Reasoning claim"))
    return items


def _contradiction_count(result) -> int:
    return sum(
        1 for c in _claims(result) if getattr(c, "stance", None) == EvidenceStance.CONTRADICTS
    ) + sum(1 for f in _findings(result) if _finding_kind(f) == "contradiction")


def _key_open_issue(result) -> str:
    gaps = _gap_items(result)
    if gaps:
        return gaps[0][0]
    unresolved = _unresolved(result)
    return unresolved[0] if unresolved else ""


def _citations_by_chunk(result) -> dict[str, list]:
    """chunk_id -> claims citing it (the evidence -> claim direction)."""
    cited: dict[str, list] = {}
    for claim in _claims(result):
        for chunk_id in getattr(claim, "evidence_chunk_ids", []) or []:
            cited.setdefault(chunk_id, []).append(claim)
    return cited


def _direct_answer_ids(result) -> set[str]:
    return set(getattr(getattr(result, "report", None), "direct_answer_evidence_chunk_ids", []) or [])


def _result_key() -> str:
    """Stable per-investigation suffix for widget keys inside the result."""
    return str(st.session_state.get("active_id", "r"))


def _keyed_container(key: str):
    """st.container(key=...) where supported (adds a .st-key-* CSS hook);
    a plain container on older Streamlit versions."""
    try:
        return st.container(key=key)
    except TypeError:
        return st.container()


# ---------------------------------------------------------------------------
# Session model: conversations, turns, run requests, reset
# ---------------------------------------------------------------------------
#
# st.session_state["conversations"] : list of conversations, oldest first
#   {"id": "inv-3", "title": <first question>, "turns": [turn, ...]}
# turn : {"id": int, "kind": "investigation" | "follow-up" | "clarification",
#         "question": str, "result": InvestigationResult | None,
#         "meta": dict | None, "message": str (clarification text)}
# st.session_state["active_conv"] / ["active_id"] : what the workspace shows
# st.session_state["pending_run"] : {"question", "mode": "new"|"follow-up",
#                                     "conv_id"} -- at most one at a time
#
# A follow-up only ever sees the turns of its own conversation, so context
# cannot leak between investigations; a new investigation always starts a
# new conversation with no context.

def _conversations() -> list[dict]:
    return st.session_state.setdefault("conversations", [])


def _find_conversation(conv_id) -> dict | None:
    return next((c for c in _conversations() if c["id"] == conv_id), None)


def _active_conversation() -> dict | None:
    return _find_conversation(st.session_state.get("active_conv"))


def _find_turn(conversation: dict | None, turn_id) -> dict | None:
    if not conversation:
        return None
    return next((t for t in conversation["turns"] if t["id"] == turn_id), None)


def _next_id(key: str) -> int:
    value = st.session_state.get(key, 0) + 1
    st.session_state[key] = value
    return value


def _request_run(question: str, mode: str = "new", conv_id=None) -> bool:
    """Queue an investigation for the next script run.

    Two-phase on purpose: the next run renders the composer and chat input
    disabled and only then calls the pipeline, so a second click cannot
    start a duplicate investigation while one is in flight. Returns False
    (and queues nothing) for empty input or while a run is pending.
    """
    question = str(question or "").strip()
    if not question or st.session_state.get("pending_run"):
        return False
    st.session_state["pending_run"] = {"question": question, "mode": mode, "conv_id": conv_id}
    st.session_state["run_question"] = question
    if mode == "new":
        # Show the question in the composer (applied before the widget is
        # instantiated on the next run).
        st.session_state["pending_question"] = question
    st.session_state.pop("last_error", None)
    return True


def _activate_turn(conversation: dict, turn: dict) -> None:
    st.session_state["active_conv"] = conversation["id"]
    st.session_state["active_id"] = turn["id"]
    if turn.get("result") is not None:
        st.session_state["result"] = turn["result"]
        st.session_state["pipeline_meta"] = turn["meta"]
        st.session_state["executed_question"] = turn["question"]
    st.session_state["scroll_top"] = True
    st.session_state.pop("last_error", None)


def _reset_workspace(prefill: str = HERO_QUESTION) -> None:
    """Start a genuinely fresh investigation: no active conversation, no
    result, no queued run. Earlier conversations stay listed (read-only)."""
    for key in (
        "result",
        "executed_question",
        "active_question",
        "investigating",
        "pipeline_meta",
        "run_question",
        "pending_run",
        "last_error",
        "active_id",
        "active_conv",
    ):
        st.session_state.pop(key, None)
    # Routed through pending_question: the composer widget may already be
    # instantiated in this run, so writing its key directly would raise
    # StreamlitAPIException.
    st.session_state["pending_question"] = prefill


def _assessment_label(result) -> tuple[str, str]:
    claim = _primary_claim(result)
    if claim is None:
        return "INCONCLUSIVE", "gray"
    return STANCE_META.get(claim.stance, ("UNKNOWN", "gray"))


def _vector_state(meta: dict | None) -> tuple[str, str, str]:
    """(dot class, short label, explanation) for the system panel. "ON"
    requires a successful live query-embedding probe (see load_pipeline)."""
    status = (meta or {}).get("vector_status") or {}
    state = status.get("state", "on" if (meta or {}).get("vector_enabled") else "off")
    reason = status.get("reason", "")
    if state == "on":
        return "on", "ON · query verified", "A live query embedding succeeded at startup."
    if state == "failed":
        return "fail", "FAILED · fallback", reason or "Vector initialization failed."
    return "off", "OFF", reason or "Vector retrieval is not configured."


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(result_state=None) -> None:
    pending = bool(st.session_state.get("pending_run"))
    with st.sidebar:
        _md(
            """
            <div class="trace-brand">
                <div class="trace-mark">T</div>
                <div>
                    <div class="trace-wordmark">TRACE</div>
                    <div class="trace-caption">Manufacturing evidence investigator</div>
                </div>
            </div>
            """
        )

        if st.button("＋  Start new investigation", use_container_width=True, key="sb_new",
                     disabled=pending):
            _reset_workspace()
            st.rerun()

        _md('<div class="side-heading">Investigations</div>')

        conversations = _conversations()
        if not conversations:
            _md(
                '<div class="side-note">No investigations yet in this session. Each '
                "investigation and its follow-up questions are kept here.</div>"
            )
        else:
            active_conv = st.session_state.get("active_conv")
            box = _keyed_container("history")
            for conversation in reversed(conversations):
                title = conversation["title"]
                short = title if len(title) <= 70 else title[:69].rstrip() + "…"
                answered = [t for t in conversation["turns"] if t.get("result") is not None]
                follow_ups = max(0, len(conversation["turns"]) - 1)
                marker = "● " if conversation["id"] == active_conv else ""
                suffix = f"  ·  +{follow_ups} follow-up{'s' if follow_ups != 1 else ''}" if follow_ups else ""
                if box.button(
                    f"{marker}{short}{suffix}",
                    key=f"hist_{conversation['id']}",
                    use_container_width=True,
                    help=title,
                    disabled=pending,
                ):
                    latest = answered[-1] if answered else conversation["turns"][-1]
                    _activate_turn(conversation, latest)
                    st.rerun()
            if st.button("Clear list", key="sb_clear", use_container_width=True, disabled=pending):
                st.session_state["conversations"] = []
                _reset_workspace()
                st.rerun()

        _md('<div class="side-heading">System</div>')

        try:
            _, meta = load_pipeline()
            dot, label, why = _vector_state(meta)
            status = meta.get("vector_status") or {}
            index = status.get("index") or {}
            stale = len(status.get("stale_chunk_ids") or [])
            index_line = (
                f"{index.get('records', 0)} chunks · {index.get('dimension') or '?'}-d"
                if index.get("loaded")
                else "not loaded"
            )
            _md(
                f"""
                <div class="sys-table">
                    <span class="k">Corpus</span><span class="v">{meta.get("documents", 0)} documents · {meta.get("chunks", 0)} chunks</span>
                    <span class="k">Index</span><span class="v">{_escape(index_line)}</span>
                    <span class="k">Vector</span><span class="v"><span class="sys-dot {dot}"></span>{_escape(label)}</span>
                    <span class="k">Retrieval</span><span class="v">{"BM25 + vector + entity" if dot == "on" else "BM25 + entity"}</span>
                    <span class="k">Reasoning</span><span class="v">Gemini + deterministic verification</span>
                </div>
                <div class="side-note" style="margin-top:8px;">{_escape(why)}</div>
                """
            )
            if stale:
                _md(
                    f'<div class="side-warn">{stale}/{index.get("records", 0)} embeddings predate the '
                    "current chunk text (passages shown are always current). Rebuild: "
                    "<code>python -m app.retrieval.indexer</code></div>"
                )
            if dot != "on":
                if st.button("Re-check vector retrieval", key="sb_vector_recheck",
                             use_container_width=True, disabled=pending):
                    load_pipeline.clear()
                    st.rerun()
        except Exception as exc:
            message = str(exc)
            hint = (
                "GEMINI_API_KEY is not set. Add it to the project .env file and restart."
                if "GEMINI_API_KEY" in message
                else f"{type(exc).__name__}: {message}"[:240]
            )
            _md(
                '<div class="side-warn"><strong>Pipeline not initialized.</strong> '
                f"{_escape(hint)}</div>"
            )

        _md(
            '<div class="side-note" style="margin-top:14px;">TRACE reports what the '
            "evidence supports. It does not approve or authorize production changes.</div>"
        )


# ---------------------------------------------------------------------------
# Header, composer, home state
# ---------------------------------------------------------------------------

def render_header() -> None:
    _md(
        """
        <div class="trace-topbar">
            <div class="trace-title">TRACE<span>Manufacturing Evidence Investigator</span></div>
        </div>
        """
    )


def scroll_to_question(nonce: int) -> None:
    """Scroll the parent Streamlit page to the investigation question."""
    components.html(
        f"""
        <!-- scroll-nonce:{nonce} -->
        <script>
        (() => {{
            // Embedding the nonce (as both a comment above and a variable
            // here) makes this component's HTML content unique per call.
            // components.html() renders an iframe whose embedded <script>
            // only (re)executes when the iframe's srcdoc content actually
            // changes between Streamlit reruns; without this, every call
            // emitted byte-identical markup and only the very first call
            // (the first time this iframe element existed at all) ever
            // ran the scroll/focus script.
            const scrollNonce = {nonce};
            const scroll = () => {{
                const parentDoc = window.parent.document;
                const target = parentDoc.querySelector(
                    'textarea[aria-label="Investigation question"]'
                );
                if (target) {{
                    target.scrollIntoView({{
                        behavior: 'smooth',
                        block: 'center'
                    }});
                    target.focus({{preventScroll: true}});
                }}
            }};
            setTimeout(scroll, 80);
            setTimeout(scroll, 350);
            setTimeout(scroll, 700);
        }})();
        </script>
        """,
        height=0,
    )


def scroll_to_top(nonce: int) -> None:
    """Bring the conclusion into view after an investigation completes.

    Streamlit keeps a chat-style page pinned to the bottom when a chat
    input is present; for TRACE the conclusion at the top must lead.
    The nonce makes the iframe content unique so the script re-executes.
    """
    components.html(
        f"""
        <!-- top-nonce:{nonce} -->
        <script>
        (() => {{
            const toTop = () => {{
                const doc = window.parent.document;
                // Covers current (section.stMain) and older Streamlit DOMs.
                doc.querySelectorAll(
                    'section.stMain, [data-testid="stAppScrollToBottomContainer"], '
                    + '[data-testid="stMain"], section.main'
                ).forEach((el) => el.scrollTo({{top: 0}}));
                window.parent.scrollTo(0, 0);
            }};
            [60, 300, 700, 1300, 2200].forEach((t) => setTimeout(toTop, t));
        }})();
        </script>
        """,
        height=0,
    )


def render_question_input(disabled: bool = False, compact: bool = False) -> str:
    if not compact:
        _md(
            '<div class="trace-tagline">Ask why a manufacturing decision exists. TRACE '
            "plans an evidence search, reconstructs what changed, separates what the "
            "records establish from what they do not, and verifies every claim against "
            "its sources.</div>"
        )
    _md('<div class="question-label">Investigation question</div>')

    # Apply example selections BEFORE instantiating the text widget.
    # This avoids Streamlit's StreamlitWidgetAlreadyInstantiatedError.
    case_selected = "pending_question" in st.session_state
    if case_selected:
        st.session_state["question_input"] = st.session_state.pop("pending_question")

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = HERO_QUESTION

    question = st.text_area(
        "Investigation question",
        height=80 if compact else 96,
        label_visibility="collapsed",
        key="question_input",
        placeholder="e.g. Why is this parameter set where it is, and is its current basis documented?",
        disabled=disabled,
    )

    if case_selected and st.session_state.get("scroll_nonce") and not compact:
        scroll_to_question(st.session_state.get("scroll_nonce", 0))

    cols = st.columns([1.15, 4])
    with cols[0]:
        run = st.button(
            "Investigating…" if disabled else "Investigate",
            use_container_width=True,
            type="primary",
            disabled=disabled,
            key="investigate_btn",
        )
    with cols[1]:
        _md(
            '<div class="composer-hint">Plans evidence-seeking questions → retrieves '
            "across the corpus → reconstructs the timeline → analyzes gaps → Gemini "
            "reasoning → claim verification → confidence calibration.</div>"
        )

    if not compact and not disabled:
        with _keyed_container("examples"):
            _md('<div class="try-label">Try a focused question:</div>')
            for index, example in enumerate(EXAMPLE_QUESTIONS[1:], start=1):
                if st.button(example, key=f"example_{index}", use_container_width=True):
                    st.session_state["pending_question"] = example
                    st.session_state.pop("run_question", None)
                    st.rerun()

    if run:
        if not question.strip():
            st.error("Enter an investigation question.")
            return ""
        if question.strip() == st.session_state.get("executed_question"):
            if hasattr(st, "toast"):
                st.toast("This question is already investigated below.")
        else:
            _request_run(question)
            st.rerun()

    return st.session_state.get("run_question", "")


def render_empty_state() -> None:
    _md(
        """
        <div class="landing-shell">
            <div class="how-grid">
                <div class="how-step"><div class="how-num">01 · Plan</div><div class="how-title">Evidence-seeking questions</div><div class="how-copy">The question is decomposed into what would have to be documented for an answer.</div></div>
                <div class="how-step"><div class="how-num">02 · Retrieve</div><div class="how-title">Hybrid search across the corpus</div><div class="how-copy">BM25, vector and entity matching over the plant record, with provenance kept per passage.</div></div>
                <div class="how-step"><div class="how-num">03 · Reconstruct</div><div class="how-title">Timeline and evidence gaps</div><div class="how-copy">Dated events are rebuilt; missing re-justification is reported as a gap, not filled in.</div></div>
                <div class="how-step"><div class="how-num">04 · Verify</div><div class="how-title">Claims checked against sources</div><div class="how-copy">Uncited claims are discarded and confidence is deterministically capped.</div></div>
            </div>
        </div>
        """
    )


def _render_reference_cases() -> None:
    _md(
        '<div class="case-intro">'
        '<div class="case-intro-title">Reference investigations</div>'
        '<div class="case-intro-copy">Representative cases from the corpus. Selecting one '
        "fills the question above; nothing runs until you choose Investigate.</div></div>"
    )

    scenarios = get_scenarios()
    cols = st.columns(2)
    for index, scenario in enumerate(scenarios):
        with cols[index % 2]:
            _md(
                f"""
                <div class="case-card">
                    <div class="case-num">CASE {index + 1:02d}</div>
                    <div class="case-title">{_escape(scenario.title)}</div>
                    <div class="case-copy">{_escape(scenario.why_it_matters)}</div>
                </div>
                """
            )
            if st.button(
                "Use this question",
                key=f"scenario_{scenario.id}",
                use_container_width=True,
            ):
                st.session_state["pending_question"] = scenario.question
                st.session_state["selected_scenario"] = scenario.id
                st.session_state["scroll_nonce"] = (
                    st.session_state.get("scroll_nonce", 0) + 1
                )
                st.session_state.pop("run_question", None)
                st.session_state.pop("executed_question", None)
                st.session_state.pop("result", None)
                st.session_state.pop("pipeline_meta", None)
                st.session_state.pop("active_id", None)
                st.rerun()


# ---------------------------------------------------------------------------
# Conclusion: header, direct answer, assessment, basis matrix
# ---------------------------------------------------------------------------

def _retrieval_chip(meta) -> str:
    """Retrieval mode chip, from what actually happened in this run when
    the run statistics are available (never from configuration alone)."""
    meta = meta or {}
    run = meta.get("retrieval_run")
    vector_on = bool(meta.get("vector_enabled"))
    if not run:
        text = "Hybrid retrieval · BM25 + vector + entity" if vector_on else "Lexical + entity retrieval (vector off)"
        return f'<span class="trace-result-chip">{_escape(text)}</span>'
    if run.get("vector_failures"):
        return (
            '<span class="trace-result-chip warn" title="'
            f'{_escape(run.get("vector_last_error") or "")}">Vector failed in '
            f'{run["vector_failures"]} of {run.get("vector_calls", 0)} searches · BM25 + entity fallback</span>'
        )
    if not vector_on:
        return '<span class="trace-result-chip">Lexical + entity retrieval · vector off</span>'
    return (
        '<span class="trace-result-chip" title="Passages ranked by the vector channel in this run">'
        f'Hybrid retrieval · vector ranked <strong>{run.get("vector_passages", 0)}</strong> of '
        f'{run.get("passages", 0)} passages</span>'
    )


def _turn_position(result) -> tuple[int, int, str] | None:
    """(turn number, total turns, kind) of `result` in its conversation."""
    for conversation in _conversations():
        for index, turn in enumerate(conversation["turns"], start=1):
            if turn.get("result") is result:
                return index, len(conversation["turns"]), turn["kind"]
    return None


def render_result_header(result, meta) -> None:
    question = getattr(getattr(result, "investigation", None), "question", "")
    position = _turn_position(result)
    if position and position[0] > 1:
        kicker = f"Follow-up · turn {position[0]} of {position[1]}"
    elif position and position[1] > 1:
        kicker = f"Investigation · turn 1 of {position[1]}"
    else:
        kicker = "Investigation"

    resolution = getattr(result, "resolution", None)
    context_chip = ""
    if resolution is not None and getattr(resolution, "used_context", False):
        anchors = ", ".join(getattr(resolution, "anchors", ()) or ()) or "the previous question"
        context_chip = (
            '<span class="trace-result-chip context" title="References in this follow-up were resolved '
            'against earlier questions in this investigation. Evidence was retrieved fresh.">'
            f"Context · {_escape(anchors)}</span>"
        )
    run = (meta or {}).get("retrieval_run") or {}
    carried_chip = (
        f'<span class="trace-result-chip">{run["carried_passages"]} passage'
        f'{"s" if run["carried_passages"] != 1 else ""} carried forward from the previous turn</span>'
        if run.get("carried_passages")
        else ""
    )

    evidence = _evidence_items(result)
    documents = len({getattr(e, "document_id", getattr(e, "filename", "")) for e in evidence})

    subject_chips = "".join(
        f'<span class="trace-result-chip subject" title="Named in the question and found in {count} retrieved passage(s)">'
        f"<strong>{_escape(token)}</strong> · {count} passages</span>"
        for token, count in _question_subjects(result)
    )

    _md(
        f"""
        <div class="trace-result-header">
            <div class="trace-result-kicker">{_escape(kicker)}</div>
            <div class="trace-result-question">{_escape(question)}</div>
            <div class="trace-result-meta">
                {context_chip}
                {subject_chips}
                <span class="trace-result-chip"><strong>{len(evidence)}</strong> passages from <strong>{documents}</strong> documents</span>
                {_retrieval_chip(meta)}
                {carried_chip}
                <span class="trace-result-chip">Claims verified against sources</span>
            </div>
        </div>
        """
    )


def render_assessment(result) -> None:
    label, statement, tone, confidence, requires_human = _assessment(result)
    evidence_strength = _evidence_strength(confidence)

    # The headline must never show a stale positive model statement under a
    # NOT ESTABLISHED badge. _display_claim_statement already guards this
    # for the claims list; apply the same guard to the headline.
    claim = _primary_claim(result)
    if claim is not None:
        statement = _display_claim_statement(claim)
        about = _claim_type_label(claim)
    else:
        about = "Overall investigation"

    verify = (
        '<div class="verify-strip"><span aria-hidden="true">⚠</span> '
        "<strong>Human verification required.</strong> TRACE does not authorize "
        "a production change.</div>"
        if requires_human
        else '<div class="verify-strip ok"><span aria-hidden="true">✓</span> '
        "Evidence is sufficient for the stated assessment.</div>"
    )

    st.markdown(
        _one_line(
            f"""
            <div class="assessment-card">
                <div class="section-label">Current assessment</div>
                <div class="assessment-row">
                    <div>{_badge(label, tone)}</div>
                    <div class="confidence">
                        <div class="confidence-number">{_escape(evidence_strength)}</div>
                        <div class="confidence-label">evidence strength</div>
                    </div>
                </div>
                <div class="assessment-title">{_escape(statement)}</div>
                <div class="assessment-about">Assessed claim · {_escape(about)}</div>
                {verify}
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_basis_matrix(result) -> None:
    """Four-row conclusion matrix: historical basis, equipment change,
    administrative carry-forward, current technical justification.

    A row with no claim of that type reads "Not assessed" -- distinct from
    "Not established", which is a verified finding that evidence is absent.
    """
    claims = _claims(result)
    if not claims:
        return

    tiles = []
    for claim_type in BASIS_ROWS:
        claim, differing = _basis_claim(result, claim_type)
        name = CLAIM_TYPE_LABELS[claim_type]
        if claim is None:
            tiles.append(
                f'<div class="basis-tile gray"><div class="basis-name">{_escape(name)}</div>'
                '<div class="basis-status tone-gray">— Not assessed</div>'
                '<div class="basis-statement">No claim of this type was produced for this question.</div></div>'
            )
            continue
        status, tone, glyph = _status(claim.stance)
        statement = _display_claim_statement(claim)
        n_sources = len(getattr(claim, "evidence_chunk_ids", []) or [])
        foot = f"{n_sources} cited source{'s' if n_sources != 1 else ''}"
        if differing:
            foot += (
                f" · {differing} other claim{'s differ' if differing != 1 else ' differs'}"
                " — see Claims & gaps"
            )
        tiles.append(
            f'<div class="basis-tile {tone}">'
            f'<div class="basis-name">{_escape(name)}</div>'
            f'<div class="basis-status tone-{tone}"><span aria-hidden="true">{glyph}</span> {_escape(status)}</div>'
            f'<div class="basis-strength">Evidence strength: {_escape(_evidence_strength(claim.confidence))}</div>'
            f'<div class="basis-statement" title="{_escape(statement)}">{_escape(statement)}</div>'
            f'<div class="basis-foot">{_escape(foot)}</div>'
            "</div>"
        )

    gaps = _gap_items(result)
    contradictions = _contradiction_count(result)
    unresolved = _unresolved(result)
    needs_human = any(_requires_human(c) for c in claims)

    signals = (
        '<div class="signal-row">'
        f'<div class="signal"><span class="signal-value {"tone-red" if contradictions else ""}">{contradictions}</span>'
        f'<span class="signal-label">material contradiction{"s" if contradictions != 1 else ""} found</span></div>'
        f'<div class="signal"><span class="signal-value {"tone-amber" if gaps else ""}">{len(gaps)}</span>'
        f'<span class="signal-label">evidence gap{"s" if len(gaps) != 1 else ""} reported</span></div>'
        f'<div class="signal"><span class="signal-value">{len(unresolved)}</span>'
        f'<span class="signal-label">open question{"s" if len(unresolved) != 1 else ""}</span></div>'
        f'<div class="signal"><span class="signal-value {"tone-amber" if needs_human else "tone-green"}">'
        f'{"Yes" if needs_human else "No"}</span><span class="signal-label">human verification required</span></div>'
        "</div>"
    )

    issue = _key_open_issue(result)
    issue_html = (
        '<div class="open-issue"><div class="open-issue-label">Most important open issue</div>'
        f'<div class="open-issue-text">{_escape(issue)}</div></div>'
        if issue
        else ""
    )

    _md('<div class="basis-grid">' + "".join(tiles) + "</div>" + signals + issue_html)


def render_confidence_context(result) -> None:
    label, statement, tone, confidence, requires_human = _assessment(result)
    evidence_strength = _evidence_strength(confidence)
    claims = list(getattr(result.report, "claims", []) or [])
    high_conf = sum(
        1 for claim in claims
        if float(getattr(claim, "confidence", 0.0) or 0.0) >= 0.80
    )
    st.markdown(
        _one_line(
            f"""
            <div class="confidence-context">
                <div class="confidence-context-main">
                    <div class="confidence-context-title">What the evidence strength applies to</div>
                    <div class="confidence-context-copy">
                        The <strong>{_escape(evidence_strength)}</strong> evidence-strength rating
                        applies to the <strong>stated assessment</strong>, not to every fact TRACE
                        found. Historical facts can be strongly documented while the current
                        technical justification remains not established. Each claim's own
                        calibrated confidence is listed under Claims &amp; gaps.
                    </div>
                </div>
                <div class="confidence-context-stat">
                    <strong>{high_conf}</strong>
                    <span>high-confidence claims</span>
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


_FENCE_OPEN = re.compile(r"```(?:html|text|markdown)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"```")
_TAG = re.compile(r"<\/?[a-zA-Z][^<>\n]{0,200}>")


def _clean_direct_answer(answer: str) -> str:
    """Return plain answer text; UI/HTML markup can never reach the renderer.

    This intentionally does NOT use html.parser.HTMLParser. HTMLParser is
    built to parse well-formed HTML, and when fed a malformed tag (for
    example a quoted attribute whose closing quote is missing), it
    greedily treats everything that follows as still being inside that
    one tag and never calls handle_data for it again — so the entire
    real answer text is silently discarded, not just the markup. A
    bounded, iteratively-applied regex has no such failure mode: it only
    ever removes text that matches `<tag ...>` and never consumes
    anything past the next `>` or newline.
    """
    text = unescape(str(answer or "").strip())

    # Strip markdown code fences wherever they appear (not just at the
    # very start of the string) so a model response with preamble text
    # before a fenced block doesn't leak the fence markers.
    text = _FENCE_OPEN.sub("", text)
    text = _FENCE_CLOSE.sub("", text)

    # Repeatedly strip anything that looks like an HTML/XML tag. This is
    # bounded ([^<>\n]{0,200}) and line-scoped, so it cannot run away or
    # swallow real content the way an unterminated-attribute tag can
    # confuse a stateful HTML parser.
    prev = None
    while prev != text:
        prev = text
        text = _TAG.sub("", text)

    # Entity decoding can expose another literal tag layer; strip once more.
    text = unescape(text)
    text = _TAG.sub("", text)

    return " ".join(text.split()).strip()


_HEADER_LINE = re.compile(
    r"^(ref|report\s+no|date|prepared\s+by|reviewed\s+by|approved\s+by|"
    r"distribution|title|document\s+no|revision|owner|effective\s+date|"
    r"scope|page\s+\d)\s*[:.]",
    re.IGNORECASE,
)

_DANGLING_WORDS = (
    "and", "or", "to", "of", "the", "see", "ref", "a", "an", "see,",
)


def _looks_like_fragment(description: str) -> bool:
    """Generic structural check for a timeline event description that is
    an ill-formed mid-sentence/document-header extraction artifact rather
    than a clean, self-contained statement.

    This is intentionally content-agnostic: it never inspects which words
    (dates, IDs, topics) appear, only the shape of the text. That keeps it
    equally applicable to any investigation question, not just the
    Press-07 flagship case.
    """
    text = description.strip()
    if not text:
        return True

    # Looks like a document-header field ("Ref: ...", "Report No: ...").
    if _HEADER_LINE.match(text):
        return True

    # A description that starts mid-sentence (lowercase first letter)
    # is a wrapped continuation fragment, not a standalone statement.
    first_alpha = next((c for c in text if c.isalpha()), None)
    if first_alpha is not None and first_alpha.islower():
        return True

    # Unbalanced parentheses/brackets indicate a sentence was cut
    # mid-clause (common with "(see Section 4, ...)"-style asides).
    if text.count(")") != text.count("("):
        return True
    if text.count("]") != text.count("["):
        return True

    # Two or more "Label: value" fields on one line is a form/record
    # excerpt (e.g. "WO No: WO-18-0322 ... Date Completed: 12-SEP-2018"),
    # not a narrative sentence. Generic on field count, not field names.
    if len(re.findall(r"[A-Za-z][A-Za-z \-]{1,30}:\s", text)) >= 2:
        return True

    # Looks like a raw delimited data record (a CSV/export row pulled in
    # verbatim): several short, comma-separated tokens rather than prose.
    # Generic on shape (token count/length), not on any column name.
    segments = [s.strip() for s in text.split(",")]
    if len(segments) >= 5:
        short_segments = sum(1 for s in segments if len(s) <= 20)
        if short_segments >= len(segments) - 1:
            return True

    # A dangling trailing connective/preposition, or trailing punctuation
    # that implies continuation, means the extraction stopped mid-clause.
    last_word = re.split(r"\s+", text.strip())[-1].lower().strip(".,;:()")
    if text[-1] in ",:;-" or last_word in _DANGLING_WORDS:
        return True

    return False


def _condense_timeline_events(result):
    """Keep the displayed timeline focused on clean, decision-relevant
    events, for any investigation question (not only the flagship case).

    This does not alter or discard the underlying extracted timeline
    data returned by the backend (`result.timeline.events`); it only
    selects which already-extracted events are shown, in the same way
    for every question.
    """
    events = list(getattr(result.timeline, "events", []) or [])
    if not events:
        return [], 0

    # Remove exact duplicate date/description pairs first.
    unique = []
    seen = set()
    for event in events:
        key = (str(event.date), str(event.description).strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)

    total_unique = len(unique)

    # Drop ill-formed extraction fragments (document-header lines,
    # mid-sentence continuations, unbalanced clauses). This is a
    # structural filter, not a topic/keyword filter, so it applies
    # identically to every question.
    well_formed = [
        event for event in unique
        if not _looks_like_fragment(str(event.description))
    ]

    # If filtering would remove every event (e.g. a corpus whose dated
    # lines are all header-style), fall back to the deduped set rather
    # than showing an empty timeline.
    candidates = well_formed if well_formed else unique

    priority_terms = (
        "42.3", "limit", "upper control", "boundary", "px-200", "px-400",
        "replacement", "change", "eco-", "qualification", "requalification",
        "validation", "revalidation", "characterization", "44.0", "43.0",
        "proposal", "rejected", "current", "sop", "spc",
    )

    scored = []
    for index, event in enumerate(candidates):
        text = f"{event.date} {event.description}".lower()
        score = sum(2 for term in priority_terms if term in text)
        # Preserve chronology as the tie-breaker.
        scored.append((score, index, event))

    max_events = 12
    if len(scored) <= max_events:
        selected = sorted(scored, key=lambda item: item[1])
        return [item[2] for item in selected], total_unique

    selected = sorted(
        sorted(scored, key=lambda item: (-item[0], item[1]))[:max_events],
        key=lambda item: item[1],
    )
    return [item[2] for item in selected], total_unique


def _display_claim_statement(claim) -> str:
    """Prevent a verified NOT_ESTABLISHED technical claim from displaying
    a stale/contradictory positive statement returned by Gemini.
    """
    if (
        claim.claim_type == ClaimType.CURRENT_TECHNICAL_JUSTIFICATION
        and claim.stance == EvidenceStance.NOT_ESTABLISHED
    ):
        return (
            "The available evidence does not establish the current technical "
            "basis for the parameter or limit on the current equipment configuration."
        )
    return str(claim.statement or "")


def _is_simple_factual_question(question: str) -> bool:
    """Identify narrow factual lookups that do not need the full case UI."""
    q = " ".join(str(question or "").lower().split())

    simple_starts = (
        "what ", "which ", "who ", "when ", "where ", "how many ",
    )
    complex_terms = (
        "why", "evidence", "prove", "justif", "revalid", "re-",
        "supported", "current basis", "still", "changed as a result",
        "what changed", "history", "rationale", "approve", "reject",
        "gap", "contradict", "should",
    )

    return q.startswith(simple_starts) and not any(term in q for term in complex_terms)


def render_direct_answer(result) -> None:
    answer = getattr(result.report, "direct_answer", "").strip()
    evidence_ids = getattr(
        result.report,
        "direct_answer_evidence_chunk_ids",
        [],
    )

    if not answer:
        return

    answer = _clean_direct_answer(answer)

    evidence_by_id = {
        item.chunk_id: item
        for item in result.investigation.evidence
    }

    # Each citation is a native <details> element: click (or Enter) to read
    # the cited passage in place. Only IDs present in this investigation's
    # evidence are shown.
    citations = []

    for chunk_id in evidence_ids:
        item = evidence_by_id.get(chunk_id)
        if item is None:
            continue

        label = f"{item.filename} · {chunk_id}"
        path = getattr(item, "relative_path", "") or ""
        citations.append(
            '<details class="cite">'
            f'<summary><span class="src">{_escape(label)}</span></summary>'
            '<div class="passage">'
            f'<div class="passage-head"><span class="src">{_escape(path)}</span></div>'
            f'<div class="passage-text">{_content_html(getattr(item, "content", ""), 1400)}</div>'
            "</div></details>"
        )

    evidence_html = ""

    if citations:
        evidence_html = (
            '<div class="direct-answer-evidence">'
            '<div class="cite-label">Cited evidence · select a source to read the passage</div>'
            + "".join(citations)
            + "</div>"
        )

    # Built as a single-line string deliberately: a multi-line f-string
    # with a blank line between the inner <div>s causes Streamlit's
    # markdown renderer to treat the blank line as the end of the raw-HTML
    # block. Everything after it then gets parsed as an indented code block
    # and shown as literal "<div>...</div>" text instead of being rendered.
    card_html = (
        '<div class="direct-answer-card">'
        '<div class="direct-answer-label">DIRECT ANSWER</div>'
        f'<div class="direct-answer-text">{_escape(answer)}</div>'
        f"{evidence_html}"
        "</div>"
    )

    st.markdown(card_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

_DOC_REF = re.compile(
    r"(?:report\s+no|doc(?:ument)?\s+no|form\s+no|ref)\s*[:.]\s*([A-Z0-9][A-Z0-9\-./]{3,})",
    re.IGNORECASE,
)


# Record-header fields not covered by _HEADER_LINE (which is shared with
# the tested fragment filter and is left unchanged).
_RECORD_HEADER = re.compile(r"^(doc(?:ument)?\s+no|form\s+no)\s*[:.]", re.IGNORECASE)


def _document_title(filename: str) -> str:
    """Readable title from the real corpus filename
    ("07_Thermal_Validation_2001.txt" -> "Thermal Validation 2001")."""
    stem = Path(str(filename or "")).stem
    stem = re.sub(r"^\d+[_\-\s]+", "", stem)
    return stem.replace("_", " ").strip() or str(filename or "")


def _date_sort_key(value) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value or "")


def _timeline_entries(result) -> tuple[list[dict], int]:
    """Display model for the timeline, built on _condense_timeline_events.

    _condense_timeline_events (unchanged) hides document-header lines such
    as "Report No: PE-STUDY-2001-07  Date: 05-JUL-2001" because they are
    not narrative sentences. Those lines are, however, the date each
    record was issued -- often the most important milestones. When the
    retrieved evidence is available, each such header line is shown as a
    "record dated" milestone titled from the real source filename, and
    entries from the same document within a few days (e.g. a report and
    its sign-off lines) are merged into one. Nothing is invented: titles,
    references and dates all come from the extracted event and its chunk.
    """
    events, total = _condense_timeline_events(result)
    by_id = _evidence_by_id(result)

    def make(event, kind):
        chunk_ids = list(getattr(event, "evidence_chunk_ids", []) or [])
        item = by_id.get(chunk_ids[0]) if chunk_ids else None
        description = str(event.description).strip()
        ref = _DOC_REF.search(description) if kind == "record" else None
        return {
            "date": event.date,
            "description": description,
            "chunk_ids": chunk_ids,
            "kind": kind,
            "document": getattr(item, "document_id", None) or getattr(item, "filename", None),
            "title": _document_title(getattr(item, "filename", "")) if item is not None else "",
            "ref": ref.group(1) if ref else "",
            "lines": [],
            # Date provenance from the extractor (defaults for older events).
            "date_text": getattr(event, "date_text", "") or "",
            "precision": getattr(event, "precision", "day") or "day",
            "ambiguous": bool(getattr(event, "ambiguous", False)),
            "event_kind": getattr(event, "kind", "") or "",
        }

    entries = [make(event, "event") for event in events]

    if by_id:
        shown = {(_date_sort_key(e["date"]), e["description"]) for e in entries}
        for event in list(getattr(getattr(result, "timeline", None), "events", []) or []):
            description = str(event.description).strip()
            key = (_date_sort_key(event.date), description)
            if key in shown or not (
                _HEADER_LINE.match(description) or _RECORD_HEADER.match(description)
            ):
                continue
            shown.add(key)
            entries.append(make(event, "record"))

    entries.sort(key=lambda e: _date_sort_key(e["date"]))

    merged: list[dict] = []
    for entry in entries:
        previous = merged[-1] if merged else None
        close = False
        if previous and entry["document"] and entry["document"] == previous["document"]:
            a, b = previous["date"], entry["date"]
            if hasattr(a, "toordinal") and hasattr(b, "toordinal"):
                close = abs(b.toordinal() - a.toordinal()) <= 7
            else:
                close = str(a) == str(b)
        if close:
            if entry["kind"] == "record" and previous["kind"] != "record":
                # The record's own date line becomes the headline.
                entry["lines"] = [previous["description"]] + previous["lines"]
                entry["chunk_ids"] = list(dict.fromkeys(entry["chunk_ids"] + previous["chunk_ids"]))
                merged[-1] = entry
            else:
                previous["lines"].append(entry["description"])
                previous["chunk_ids"] = list(dict.fromkeys(previous["chunk_ids"] + entry["chunk_ids"]))
            continue
        merged.append(entry)

    return merged, total


def render_timeline(result) -> None:
    entries, total_events = _timeline_entries(result)
    events = entries

    _md(
        '<div class="section-title">Reconstructed timeline</div>'
        '<div class="section-sub">Dated events extracted from the retrieved records, in '
        "chronological order. Expand an event to read the passage it was extracted from.</div>"
    )

    if not events:
        _md(
            '<div class="empty-line">No explicit dated events were extracted from the '
            "retrieved evidence.</div>"
        )
        return

    if total_events > len(events):
        _md(
            f'<div class="small-muted" style="margin-bottom:12px;">Showing {len(events)} key '
            f"entries built from {total_events} extracted dated events. Header lines are shown "
            "as the record's date; sign-offs are merged into their record; garbled fragments are "
            "hidden here, not removed from the result.</div>"
        )

    has_evidence = bool(_evidence_items(result))

    # Group consecutive events by year so a decade of records reads as a
    # handful of eras instead of an undifferentiated list.
    groups: list[tuple[str, list]] = []
    for event in events:
        year, _ = _format_date(event["date"])
        if groups and groups[-1][0] == year:
            groups[-1][1].append(event)
        else:
            groups.append((year, [event]))

    # Built as single-line fragments joined with no separators: a blank line
    # anywhere in this markup would end Streamlit's raw-HTML block.
    html = ['<div class="tl">']
    for year, year_events in groups:
        html.append(
            f'<div class="tl-year"><div class="tl-year-label">{_escape(year)}</div>'
            '<div class="tl-year-events">'
        )
        for event in year_events:
            readable = _date_label(event["date"], event.get("precision", "day"))
            chunk_ids = event["chunk_ids"]
            if has_evidence:
                sources = _source_chips(result, chunk_ids)
                drawer = _passages_details(
                    result,
                    chunk_ids[:2],
                    "View source passage",
                    needle=event["description"],
                    limit_chars=700,
                )
            else:
                sources = _escape(", ".join(chunk_ids[:3]))
                drawer = ""
            if event["kind"] == "record" and event["title"]:
                ref = f' <span class="src-chip">{_escape(event["ref"])}</span>' if event["ref"] else ""
                headline = (
                    f'<span class="badge quiet" style="margin-right:6px;">Record dated</span>'
                    f"<strong>{_escape(event['title'])}</strong>{ref}"
                )
            else:
                headline = _escape(event["description"])
            lines = "".join(
                f'<div class="timeline-line">{_escape(line)}</div>' for line in event["lines"][:3]
            )
            html.append(
                '<div class="timeline-event">'
                f'<div class="timeline-date">{_escape(readable)}</div>'
                f'<div class="timeline-description">{headline}</div>'
                f"{lines}"
                f'<div class="timeline-source">Evidence · {sources}</div>'
                f"{drawer}"
                "</div>"
            )
        html.append("</div></div>")

    # The unresolved question is part of the history: if the backend
    # reported an evidence gap, end the timeline on it rather than on the
    # last dated record, so "nothing after this re-justified the limit"
    # is visible in the chronology itself.
    gaps = _gap_items(result)
    if gaps:
        statement, chunk_ids, _origin = gaps[0]
        html.append(
            '<div class="tl-open"><div class="tl-year-label tone-amber">Open</div>'
            '<div class="tl-open-body">'
            '<div class="tl-open-label">Unresolved · reported evidence gap</div>'
            f'<div class="tl-open-text">{_escape(statement)}</div>'
            "</div></div>"
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    if total_events > len(events):
        all_events = list(getattr(result.timeline, "events", []) or [])
        with st.expander(f"All {len(all_events)} extracted dated events (unfiltered)"):
            rows = []
            for event in all_events:
                readable = _date_label(event.date, getattr(event, "precision", "day"))
                rows.append(
                    '<div class="tl-row">'
                    f'<span class="d">{_escape(readable)}</span>'
                    f'<span class="t">{_escape(event.description)} '
                    f'{_source_chips(result, getattr(event, "evidence_chunk_ids", []), limit=2)}</span>'
                    "</div>"
                )
            st.markdown("".join(rows), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Evidence explorer
# ---------------------------------------------------------------------------

_CHANNEL_LABELS = {
    "bm25": "BM25",
    "vector": "vector",
    "entity": "entity",
    "carried-forward": "carried forward from previous turn",
}


def _timeline_by_chunk(result) -> dict[str, list[str]]:
    """chunk_id -> readable timeline entries that cite it."""
    mapping: dict[str, list[str]] = {}
    if getattr(result, "timeline", None) is None:
        return mapping
    entries, _total = _timeline_entries(result)
    for entry in entries:
        readable = _date_label(entry["date"], entry.get("precision", "day"))
        text = entry["title"] if entry["kind"] == "record" and entry["title"] else entry["description"]
        for chunk_id in entry["chunk_ids"]:
            mapping.setdefault(chunk_id, []).append(f"{readable} · {_one_line(text)[:90]}")
    return mapping


def _evidence_card_html(result, item, cited_claims, in_direct_answer: bool, highlight: str = "",
                        timeline_refs=None) -> str:
    channels = [
        _CHANNEL_LABELS.get(c, c) for c in (getattr(item, "retrieval_channels", []) or [])
    ]
    channel_html = (
        f'<span class="small-muted">found by {_escape(" + ".join(channels))}</span>'
        if channels
        else ""
    )
    timeline_html = (
        '<div class="evidence-reason"><b>Timeline</b>'
        + "".join(f"<div>· {_escape(ref)}</div>" for ref in (timeline_refs or [])[:3])
        + "</div>"
        if timeline_refs
        else ""
    )
    retrieved_for = list(getattr(item, "retrieved_for", []) or [])
    reason = (
        "".join(f"<div>· {_escape(q)}</div>" for q in retrieved_for[:3])
        if retrieved_for
        else "<div>· Retrieved as supporting investigation evidence.</div>"
    )
    more = f'<div class="small-muted">+{len(retrieved_for) - 3} more</div>' if len(retrieved_for) > 3 else ""

    cited_rows = []
    if in_direct_answer:
        cited_rows.append('<div class="cited-by-item"><span class="badge quiet">Direct answer</span></div>')
    for claim in cited_claims:
        cited_rows.append(
            '<div class="cited-by-item">'
            f"{_status_badge(claim.stance)} "
            f"<b>{_escape(_claim_type_label(claim))}</b> · "
            f"{_escape(_display_claim_statement(claim))}</div>"
        )
    cited_html = (
        '<div class="evidence-reason"><b>Cited by</b></div><div class="cited-by">' + "".join(cited_rows) + "</div>"
        if cited_rows
        else '<div class="evidence-reason">Retrieved, but not cited by any verified claim.</div>'
    )

    return (
        '<div class="evidence-card">'
        '<div class="evidence-top">'
        f'<span class="evidence-id">{_escape(item.chunk_id)}</span>'
        f'<span class="small-muted">{channel_html}{" · " if channel_html else ""}fused score {float(getattr(item, "retrieval_score", 0.0) or 0.0):.4f}</span>'
        "</div>"
        f'<div class="evidence-content">{_content_html(getattr(item, "content", ""), 2400, highlight)}</div>'
        f'<div class="evidence-reason"><b>Retrieved for</b>{reason}{more}</div>'
        f"{timeline_html}"
        f"{cited_html}"
        "</div>"
    )


def render_evidence(result) -> None:
    evidence = _evidence_items(result)

    _md(
        '<div class="section-title">Evidence explorer</div>'
        '<div class="section-sub">Every passage TRACE retrieved, grouped by source document, '
        "with its real file path, the investigation question it was retrieved for, and the "
        "claims that cite it.</div>"
    )

    if not evidence:
        _md('<div class="empty-line">No evidence was retrieved.</div>')
        return

    cited = _citations_by_chunk(result)
    direct_ids = _direct_answer_ids(result)
    timeline_refs = _timeline_by_chunk(result)
    key = _result_key()

    c1, c2, c3 = st.columns([2.2, 1.4, 1.6])
    with c1:
        search = st.text_input(
            "Search passages",
            key=f"ev_search_{key}",
            placeholder="Filter by text, e.g. 42.3, PX-400, requalification",
        )
    with c2:
        scope = st.selectbox(
            "Show",
            ("All passages", "Cited in conclusions", "Not cited"),
            key=f"ev_scope_{key}",
        )
    categories = sorted(
        {(_source_category(getattr(e, "relative_path", "")) or "Uncategorised") for e in evidence}
    )
    with c3:
        selected = st.multiselect(
            "Source type",
            categories,
            key=f"ev_cat_{key}",
            placeholder="All source types",
        )

    def keep(item) -> bool:
        is_cited = item.chunk_id in cited or item.chunk_id in direct_ids
        if scope == "Cited in conclusions" and not is_cited:
            return False
        if scope == "Not cited" and is_cited:
            return False
        category = _source_category(getattr(item, "relative_path", "")) or "Uncategorised"
        if selected and category not in selected:
            return False
        if search.strip():
            haystack = " ".join(
                [str(item.content), str(item.filename), str(item.chunk_id)]
            ).lower()
            if not all(term in haystack for term in search.lower().split()):
                return False
        return True

    filtered = [item for item in evidence if keep(item)]

    # Group by document, documents ordered by their best-ranked passage
    # (preserves the backend's retrieval ranking).
    documents: dict[str, list] = {}
    for item in sorted(filtered, key=lambda e: e.retrieval_score, reverse=True):
        documents.setdefault(getattr(item, "relative_path", item.filename), []).append(item)

    n_cited = sum(1 for e in evidence if e.chunk_id in cited or e.chunk_id in direct_ids)
    _md(
        f'<div class="small-muted" style="margin:2px 0 10px;">{len(filtered)} of {len(evidence)} '
        f"passages · {len(documents)} documents shown · {n_cited} passages are cited by the "
        "direct answer or a verified claim</div>"
    )

    if not filtered:
        _md('<div class="empty-line">No passages match these filters.</div>')
        return

    for path, items in documents.items():
        first = items[0]
        category = _source_category(path)
        cite_count = sum(len(cited.get(i.chunk_id, [])) + (1 if i.chunk_id in direct_ids else 0) for i in items)
        parts = [first.filename]
        if category:
            parts.append(category)
        parts.append(f"{len(items)} passage{'s' if len(items) != 1 else ''}")
        parts.append(f"cited {cite_count}×" if cite_count else "not cited")
        with st.expander("  ·  ".join(parts), expanded=False):
            _md(
                f'<div class="evidence-path">{_escape(path)}</div>'
                f'<div class="small-muted" style="margin:2px 0 8px;">Document ID '
                f'<span class="evidence-id">{_escape(getattr(first, "document_id", ""))}</span></div>'
            )
            st.markdown(
                "".join(
                    _evidence_card_html(
                        result,
                        item,
                        cited.get(item.chunk_id, []),
                        item.chunk_id in direct_ids,
                        highlight=search,
                        timeline_refs=timeline_refs.get(item.chunk_id),
                    )
                    for item in items
                ),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Claims, contradictions, gaps
# ---------------------------------------------------------------------------

_WHY_IT_MATTERS = {
    EvidenceStance.SUPPORTS: "Supported: the cited records directly document this.",
    EvidenceStance.CONTRADICTS: "Contradicted: the cited records conflict with this; resolve before relying on it.",
    EvidenceStance.NOT_ESTABLISHED: (
        "Not established: the evidence needed for this was not found in the available "
        "records. This is not a finding that the statement is false."
    ),
    EvidenceStance.CONTEXT_ONLY: "Context only: background that informs the investigation but does not by itself support a conclusion.",
    EvidenceStance.UNKNOWN: "Inconclusive: no valid citation survived verification, so TRACE assigns no confidence.",
}


def _claim_card_html(result, claim) -> str:
    human = (
        '<span class="badge amber" style="font-weight:600;">Needs human verification</span>'
        if _requires_human(claim)
        else ""
    )
    chunk_ids = list(getattr(claim, "evidence_chunk_ids", []) or [])
    sources = _source_chips(result, chunk_ids, limit=4) if chunk_ids else '<span class="tone-red">No valid citation</span>'
    drawer = _passages_details(
        result,
        chunk_ids,
        f"View {len(chunk_ids)} cited passage{'s' if len(chunk_ids) != 1 else ''}",
    ) if chunk_ids else ""
    return (
        '<div class="claim-card">'
        '<div class="claim-head"><div class="claim-tags">'
        f"{_status_badge(claim.stance)}"
        f'<span class="badge quiet">{_escape(_claim_type_label(claim))}</span>'
        f"{human}</div>"
        f'<span class="small-muted">{float(claim.confidence or 0.0):.0%} calibrated confidence</span>'
        "</div>"
        f'<div class="claim-statement">{_escape(_display_claim_statement(claim))}</div>'
        f'<div class="claim-why">{_escape(_WHY_IT_MATTERS.get(claim.stance, ""))}</div>'
        f'<div class="claim-meta timeline-source">Reasoning claim · verified after Gemini · {sources}</div>'
        f"{drawer}"
        "</div>"
    )


def _finding_card_html(result, finding) -> str:
    label, tone = FINDING_META.get(_finding_kind(finding), ("Finding", "gray"))
    chunk_ids = list(getattr(finding, "evidence_chunk_ids", []) or [])
    drawer = _passages_details(
        result,
        chunk_ids,
        f"View {len(chunk_ids)} related passage{'s' if len(chunk_ids) != 1 else ''}",
    ) if chunk_ids else ""
    return (
        '<div class="claim-card">'
        '<div class="claim-head"><div class="claim-tags">'
        f"{_badge(label, tone)}"
        '<span class="badge quiet">Deterministic analyzer finding</span>'
        "</div></div>"
        f'<div class="claim-statement">{_escape(finding.statement)}</div>'
        f'<div class="claim-meta timeline-source">{_source_chips(result, chunk_ids, limit=4)}</div>'
        f"{drawer}"
        "</div>"
    )


def _group(title: str, cards: list[str], empty: str, explainer: str = "") -> str:
    head = (
        f'<div class="group-head"><span class="group-title">{_escape(title)}</span>'
        f'<span class="group-count">{len(cards)}</span></div>'
    )
    body = "".join(cards) if cards else f'<div class="empty-line">{_escape(empty)}</div>'
    note = f'<div class="group-explainer">{explainer}</div>' if explainer else ""
    return head + note + body


def render_claims(result) -> None:
    """Claims & gaps view.

    Groups reasoning claims (post-hoc verified) and deterministic analyzer
    findings by what they mean for the decision. Groupings use only the
    backend's own stance / claim_type / finding_type fields.
    """
    claims = _claims(result)
    findings = _findings(result)

    _md(
        '<div class="section-title">Reasoning claims · verified after Gemini</div>'
        '<div class="section-sub">Each claim is checked against the passages it cites; '
        "uncited claims are dropped to Inconclusive and confidence is capped "
        "deterministically. Analyzer findings come from rule-based evidence-gap analysis.</div>"
    )

    supported, contradicted, gaps, context, inconclusive, recommendations = [], [], [], [], [], []
    for claim in claims:
        card = _claim_card_html(result, claim)
        ctype = getattr(claim, "claim_type", None)
        if ctype == ClaimType.RECOMMENDATION:
            recommendations.append(card)
        elif ctype == ClaimType.EVIDENCE_GAP or claim.stance == EvidenceStance.NOT_ESTABLISHED:
            gaps.append(card)
        elif claim.stance == EvidenceStance.SUPPORTS:
            supported.append(card)
        elif claim.stance == EvidenceStance.CONTRADICTS:
            contradicted.append(card)
        elif claim.stance == EvidenceStance.CONTEXT_ONLY:
            context.append(card)
        else:
            inconclusive.append(card)

    for finding in findings:
        card = _finding_card_html(result, finding)
        kind = _finding_kind(finding)
        if kind == "supported":
            supported.append(card)
        elif kind == "contradiction":
            contradicted.append(card)
        elif kind == "evidence-gap":
            gaps.append(card)
        else:
            context.append(card)

    if not claims and not findings:
        _md('<div class="empty-line">No claims or findings were generated for this investigation.</div>')
        return

    html = [
        _group(
            "Not established · evidence gaps",
            gaps,
            "No evidence gaps were reported.",
            "<strong>Not established is not the same as false.</strong> It means the records "
            "needed to establish the statement were not found in the available evidence. "
            "A contradicted statement, by contrast, is one the records actively conflict with.",
        ),
        _group(
            "Contradictions",
            contradicted,
            "No contradictions were identified in the retrieved evidence.",
        ),
        _group("Supported", supported, "No supported claims."),
        _group("Context", context, "No context-only items."),
    ]
    if inconclusive:
        html.append(_group("Inconclusive", inconclusive, ""))
    st.markdown("".join(html), unsafe_allow_html=True)

    if recommendations:
        _md(
            f'<div class="small-muted" style="margin-top:8px;">{len(recommendations)} '
            "recommendation claim(s) are shown under Next steps.</div>"
        )


def render_finding_summary(result) -> None:
    findings = _findings(result)

    _md('<div class="section-label" style="margin-top:6px;">Deterministic analyzer output</div>')

    if not findings:
        _md('<div class="small-muted">No structured evidence findings were generated.</div>')
        return

    cards = []
    for finding in findings[:6]:
        label, tone = FINDING_META.get(_finding_kind(finding), ("Finding", "gray"))
        cards.append(
            '<div class="finding">'
            f"{_badge(label, tone)}"
            f'<div class="finding-name">{_escape(finding.statement)}</div>'
            f'<div class="finding-meta">{len(finding.evidence_chunk_ids)} cited evidence item(s)</div>'
            "</div>"
        )
    st.markdown('<div class="finding-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Next steps (decision support)
# ---------------------------------------------------------------------------

def render_gap_and_action(result) -> None:
    claims = _claims(result)
    recommendations = [c for c in claims if getattr(c, "claim_type", None) == ClaimType.RECOMMENDATION]
    gaps = _gap_items(result)
    unresolved = _unresolved(result)

    _md(
        '<div class="section-title">Next steps</div>'
        '<div class="authority-note"><strong>Decision support only.</strong> These steps '
        "come from the gaps and recommendations TRACE found in the evidence. TRACE does not "
        "approve, authorize, or change production parameters; the responsible engineer decides, "
        "under the plant's change-control process.</div>"
    )

    html = []
    if recommendations:
        for claim in recommendations:
            chunk_ids = list(claim.evidence_chunk_ids or [])
            html.append(
                '<div class="human-action">'
                '<div class="human-title">Recommended by TRACE</div>'
                f'<div class="human-copy">{_escape(_display_claim_statement(claim))}</div>'
                f'<div class="timeline-source">{_status_badge(claim.stance)} '
                f"{_source_chips(result, chunk_ids, limit=4)}</div>"
                f"{_passages_details(result, chunk_ids, 'View supporting passages')}"
                "</div>"
            )
    else:
        html.append(
            '<div class="human-action">'
            '<div class="human-title">Human verification</div>'
            f'<div class="human-copy">{_escape(_human_action(result))}</div>'
            "</div>"
        )

    if gaps:
        html.append('<div class="group-head"><span class="group-title">Gaps to close</span>'
                    f'<span class="group-count">{len(gaps)}</span></div>')
        for statement, chunk_ids, origin in gaps:
            html.append(
                '<div class="gap-card">'
                f'<div class="gap-title">Evidence gap · {_escape(origin)}</div>'
                f'<div class="gap-copy">{_escape(statement)}</div>'
                f'<div class="timeline-source">{_source_chips(result, chunk_ids, limit=4)}</div>'
                "</div>"
            )
    st.markdown("".join(html), unsafe_allow_html=True)

    if unresolved:
        _md(
            '<div class="group-head"><span class="group-title">Open questions</span>'
            f'<span class="group-count">{len(unresolved)}</span></div>'
            '<div class="small-muted" style="margin-bottom:6px;">Investigating one runs a new, '
            "independent evidence investigation.</div>"
        )
        key = _result_key()
        for index, question in enumerate(unresolved):
            c1, c2 = st.columns([5, 1.3])
            with c1:
                _md(f'<div class="oq">{_escape(question)}</div>')
            with c2:
                if st.button("Investigate", key=f"oq_{key}_{index}", use_container_width=True):
                    _request_run(question)
                    st.rerun()


# ---------------------------------------------------------------------------
# Method (how TRACE investigated)
# ---------------------------------------------------------------------------

def render_investigation_path(result) -> None:
    questions = list(getattr(getattr(result, "plan", None), "subquestions", []) or [])
    _md(
        '<div class="section-label" style="margin-top:14px;">Investigation path · '
        "evidence-seeking questions TRACE generated</div>"
    )
    if not questions:
        _md('<div class="small-muted">No sub-questions were recorded for this investigation.</div>')
        return

    rows = []
    evidence = _evidence_items(result)
    for index, question in enumerate(questions, start=1):
        associated = [item for item in evidence if question in (item.retrieved_for or [])]
        status = f"✓ {len(associated)} passages" if associated else "○ No evidence"
        tone = "" if associated else " warn"
        rows.append(
            '<div class="path-item">'
            f'<div class="path-index">{index:02d}</div>'
            f'<div class="path-question">{_escape(question)}</div>'
            f'<div class="path-status{tone}">{status}</div>'
            "</div>"
        )
    st.markdown('<div class="result-card compact">' + "".join(rows) + "</div>", unsafe_allow_html=True)


def _render_investigation_coverage(result) -> None:
    """
    "Investigation coverage" metrics card (questions / evidence items /
    timeline events) plus its deterministic-calibration note. Shared by
    both render_result (the full case UI) and render_simple_result (the
    compact factual-lookup UI), so both paths show the same coverage
    metrics -- this card must not disappear just because a question was
    classified as a "simple" factual lookup.
    """
    plan_count = len(getattr(getattr(result, "plan", None), "subquestions", []) or [])
    evidence_count = len(_evidence_items(result))
    event_count = len(getattr(getattr(result, "timeline", None), "events", []) or [])
    st.markdown(
        _one_line(
            f"""
            <div class="result-card compact">
                <div class="section-label">Investigation coverage</div>
                <div class="metric-row">
                    <div class="metric"><div class="metric-value">{plan_count}</div><div class="metric-label">questions</div></div>
                    <div class="metric"><div class="metric-value">{evidence_count}</div><div class="metric-label">evidence items</div></div>
                    <div class="metric"><div class="metric-value">{event_count}</div><div class="metric-label">timeline events</div></div>
                </div>
                <div class="small-muted" style="margin-top:10px;">Claims are post-hoc verified against retrieved evidence. Confidence is deterministically calibrated.</div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


def render_retrieval_diagnostics(meta) -> None:
    """Measured retrieval facts for this run plus the vector status found
    at startup. Counts here are real counters, not model estimates."""
    meta = meta or {}
    run = meta.get("retrieval_run") or {}
    status = meta.get("vector_status") or {}
    index = status.get("index") or {}
    dot, label, why = _vector_state(meta)

    rows = [
        ("Vector status at startup", f"{label} — {why}"),
        ("Index", (
            f"{index.get('records', 0)} chunks · {index.get('dimension') or '?'}-dimensional · "
            f"{index.get('format') or 'unknown format'} · model "
            f"{index.get('model') or '?'}{'' if index.get('model_declared') else ' (assumed; legacy index does not record it)'}"
        ) if index else "Not reported"),
        ("Index ↔ corpus", (
            f"chunk IDs {'match' if index.get('ids_match') else 'DO NOT match'}; "
            f"{index.get('stale_chunks', 0)} chunk(s) embedded from older text"
        ) if index else "Not reported"),
    ]
    if run:
        rows += [
            ("Vector searches this run", f"{run.get('vector_calls', 0)} calls, {run.get('vector_failures', 0)} failed"),
            ("Passages ranked by vector", f"{run.get('vector_passages', 0)} of {run.get('passages', 0)}"),
            ("Carried forward", f"{run.get('carried_passages', 0)} passage(s) from the previous turn"),
        ]
        if run.get("vector_last_error"):
            rows.append(("Last vector error", run["vector_last_error"]))
    body = "".join(
        f'<div class="diag-row"><span class="k">{_escape(k)}</span><span class="v">{_escape(v)}</span></div>'
        for k, v in rows
    )
    _md(
        '<div class="section-label" style="margin-top:14px;">Retrieval diagnostics</div>'
        f'<div class="result-card compact diag">{body}</div>'
    )


def render_conversation_context(result) -> None:
    context = getattr(result, "conversation", None)
    resolution = getattr(result, "resolution", None)
    if context is None:
        _md(
            '<div class="section-label" style="margin-top:14px;">Conversation context</div>'
            '<div class="small-muted">None. This turn started a new investigation, so the reasoner '
            "received no earlier conversation.</div>"
        )
        return
    summary = context.summary()
    _md(
        '<div class="section-label" style="margin-top:14px;">Conversation context</div>'
        f'<div class="small-muted">Follow-up in <span class="evidence-id">{_escape(summary["conversation_id"])}</span> '
        f'using {summary["turns"]} earlier turn(s). Retrieval query used for reference resolution: '
        f'<em>{_escape(getattr(resolution, "retrieval_question", ""))}</em></div>'
    )
    with st.expander("Exact context sent to the reasoner (labelled as not evidence)"):
        try:
            st.code(context.to_prompt(), language="text", wrap_lines=True)
        except TypeError:  # wrap_lines needs Streamlit >= 1.38
            st.code(context.to_prompt(), language="text")


def render_method(result, meta=None) -> None:
    _md(
        '<div class="section-title">How TRACE investigated</div>'
        '<div class="section-sub">The plan, coverage, retrieval diagnostics and deterministic '
        "analysis behind this result, shown for auditability.</div>"
    )
    _render_investigation_coverage(result)
    render_retrieval_diagnostics(meta)
    render_conversation_context(result)
    render_investigation_path(result)
    render_finding_summary(result)


# ---------------------------------------------------------------------------
# Result layouts
# ---------------------------------------------------------------------------

def render_simple_result(result, meta) -> None:
    """Compact result view for narrow factual questions."""
    render_direct_answer(result)
    render_assessment(result)

    # Investigation coverage and the claim breakdown stay visible
    # regardless of assessment type or question classification.
    _render_investigation_coverage(result)
    render_claims(result)

    # Keep only a small amount of supporting provenance for a factual lookup.
    evidence = sorted(
        _evidence_items(result),
        key=lambda item: item.retrieval_score,
        reverse=True,
    )

    if not evidence:
        _md(
            '<div class="section-title" style="margin-top:14px;">Supporting evidence</div>'
            '<div class="empty-line">No supporting evidence was retrieved.</div>'
        )
        return

    cited = _citations_by_chunk(result)
    direct_ids = _direct_answer_ids(result)
    st.markdown(
        '<div class="section-title" style="margin-top:14px;">Supporting evidence</div>'
        + "".join(
            '<div class="evidence-card">'
            f'<div class="evidence-file">{_escape(item.filename)}</div>'
            f'<div class="evidence-path">{_escape(getattr(item, "relative_path", ""))} · {_escape(item.chunk_id)}</div>'
            f'<div class="evidence-content">{_content_html(item.content, 600)}</div>'
            + (
                '<div class="evidence-reason">Cited by the answer or a verified claim.</div>'
                if item.chunk_id in cited or item.chunk_id in direct_ids
                else ""
            )
            + "</div>"
            for item in evidence[:4]
        ),
        unsafe_allow_html=True,
    )


def render_result(result, meta) -> None:
    question = getattr(result.investigation, "question", "")
    if _is_simple_factual_question(question):
        render_result_header(result, meta)
        render_simple_result(result, meta)
        with st.expander("How TRACE investigated · retrieval diagnostics and context"):
            render_method(result, meta)
        return

    render_result_header(result, meta)

    # Conclusion first: what TRACE answers, and what the evidence does and
    # does not establish. Direct answer must render before the assessment.
    c1, c2 = st.columns([1.55, 1], gap="medium")
    with c1:
        render_direct_answer(result)
    with c2:
        render_assessment(result)

    render_basis_matrix(result)
    render_confidence_context(result)

    claims = _claims(result)
    evidence = _evidence_items(result)
    events, _total = _timeline_entries(result) if getattr(result, "timeline", None) else ([], 0)
    n_gaps = len(_gap_items(result))

    tabs = st.tabs(
        [
            f"Timeline · {len(events)}",
            f"Claims & gaps · {len(claims) + len(_findings(result))}",
            f"Evidence · {len(evidence)}",
            f"Next steps{' · ' + str(n_gaps) + ' gap' + ('s' if n_gaps != 1 else '') if n_gaps else ''}",
            "Method",
        ]
    )
    with tabs[0]:
        render_timeline(result)
    with tabs[1]:
        render_claims(result)
    with tabs[2]:
        render_evidence(result)
    with tabs[3]:
        render_gap_and_action(result)
    with tabs[4]:
        render_method(result, meta)


# ---------------------------------------------------------------------------
# Investigation thread (follow-up questions)
# ---------------------------------------------------------------------------
#
# Investigator.run(question) is stateless: it takes one question string and
# carries no conversation history. The thread therefore records a sequence
# of independent investigations; each follow-up runs the full pipeline from
# scratch and the UI says so. It does not pretend earlier turns are passed
# to the reasoner.

def _chat_message(role: str):
    """Chat bubble with restrained, on-brand avatars (Material icons are
    supported from Streamlit 1.37; older versions fall back to defaults)."""
    icon = ":material/person:" if role == "user" else ":material/troubleshoot:"
    try:
        return st.chat_message(role, avatar=icon)
    except Exception:
        return st.chat_message(role)


# ---------------------------------------------------------------------------
# Investigation conversation
# ---------------------------------------------------------------------------
#
# Follow-ups are contextual (see app/investigation/conversation.py): each
# one runs the full pipeline with fresh retrieval and verification, and the
# reasoner additionally receives a bounded, not-evidence summary of the
# earlier turns of the SAME conversation to resolve references.

def _turn_answer_html(turn: dict, limit: int = 420) -> str:
    result = turn.get("result")
    if result is None:
        return f'<div class="thread-answer">{_escape(turn.get("message", ""))}</div>'
    label, tone = _assessment_label(result)
    answer = _clean_direct_answer(getattr(result.report, "direct_answer", "") or "")
    if len(answer) > limit:
        answer = answer[: limit - 1].rstrip() + "…"
    ids = list(getattr(result.report, "direct_answer_evidence_chunk_ids", []) or [])
    sources = _passages_details(
        result, ids, f"{len(ids)} cited passage{'s' if len(ids) != 1 else ''} · inspect"
    ) if ids else ""
    return (
        f"{_badge(label, tone)}"
        f'<div class="thread-answer">{_escape(answer or "No direct answer was established.")}</div>'
        f"{sources}"
    )


def render_conversation_history(conversation: dict, active_turn_id) -> None:
    """Earlier turns of the active investigation, shown above the current
    answer so earlier answers are visibly distinct from the current one."""
    turns = conversation["turns"]
    earlier = [t for t in turns if t["id"] != active_turn_id]
    if not earlier:
        return
    position = next((i for i, t in enumerate(turns) if t["id"] == active_turn_id), len(turns) - 1)

    # Always-visible compact strip: one line per turn, current one marked.
    rows = []
    for index, turn in enumerate(turns, start=1):
        if turn.get("result") is not None:
            label, tone = _assessment_label(turn["result"])
            badge = _badge(label, tone)
        else:
            badge = _badge("CLARIFICATION", "gray")
        current = turn["id"] == active_turn_id
        rows.append(
            f'<div class="strip-row{" current" if current else ""}">'
            f'<span class="strip-n">Turn {index}</span>{badge}'
            f'<span class="strip-q">{_escape(turn["question"])}</span>'
            f'<span class="strip-tag">{"current answer" if current else ("earlier" if index - 1 < position else "later")}</span>'
            "</div>"
        )
    _md('<div class="conv-strip" aria-label="Conversation turns">' + "".join(rows) + "</div>")

    label = f"Read the other answer{'s' if len(earlier) != 1 else ''} in this conversation ({len(earlier)})"
    with st.expander(label, expanded=False):
        for index, turn in enumerate(turns, start=1):
            if turn["id"] == active_turn_id:
                _md(
                    f'<div class="thread-current">Turn {index} · shown below as the current answer</div>'
                )
                continue
            tag = "Earlier answer" if index - 1 < position else "Later answer"
            with _chat_message("user"):
                _md(
                    f'<div class="thread-turn-tag">Turn {index} · '
                    f'{_escape(turn["kind"].replace("-", " "))}</div>'
                    f'<div class="thread-question">{_escape(turn["question"])}</div>'
                )
            with _chat_message("assistant"):
                _md(f'<div class="thread-turn-tag">{tag}</div>' + _turn_answer_html(turn))
                if turn.get("result") is not None:
                    if st.button("Open this answer", key=f"open_turn_{turn['id']}"):
                        _activate_turn(conversation, turn)
                        st.rerun()


def render_followup_panel(result, conversation: dict | None, pending: bool) -> None:
    """Follow-up guidance and suggestions directly above the pinned chat
    input, so the user never has to leave the investigation to ask."""
    _md(
        '<div class="thread-title">Continue this investigation</div>'
        '<div class="thread-note">Ask a follow-up in the box at the bottom (Enter to send). '
        "TRACE uses this investigation's earlier questions and verified claims only to "
        "understand what you are referring to, then plans, retrieves and verifies "
        "<strong>fresh evidence</strong> for the new question. Earlier answers are never "
        "treated as evidence. Use <em>Start new investigation</em> for an unrelated question.</div>"
    )
    suggestions = _unresolved(result)[:3]
    if suggestions and conversation is not None:
        _md('<div class="small-muted" style="margin:4px 0 6px;">Suggested follow-ups (open questions TRACE reported):</div>')
        box = _keyed_container("suggestions")
        for index, suggestion in enumerate(suggestions):
            if box.button(suggestion, key=f"suggest_{_result_key()}_{index}",
                          use_container_width=True, disabled=pending):
                if _request_run(suggestion, mode="follow-up", conv_id=conversation["id"]):
                    st.rerun()


def render_thread(result) -> None:
    """Backward-compatible entry point: render the active conversation's
    follow-up panel for `result`."""
    render_followup_panel(result, _active_conversation(), bool(st.session_state.get("pending_run")))


# ---------------------------------------------------------------------------
# Live investigation progress
# ---------------------------------------------------------------------------

# Stage sentences in the pipeline's real order, kept for reference and for
# the narrow/deep route label. Live progress is now driven by
# Investigator.run(progress_callback=...), i.e. by the backend itself.
_PROGRESS_SENTENCES_DEEP = (
    "Planning investigation…",
    "Searching & fusing evidence…",
    "Reconstructing timeline…",
    "Analyzing evidence gaps…",
    "Reasoning over evidence…",
    "Verifying claims…",
    "Calibrating confidence…",
)

_PROGRESS_SENTENCES_NARROW = (
    "Routing question…",
    "Searching relevant evidence…",
    "Reconstructing timeline…",
    "Analyzing evidence gaps…",
    "Reasoning over evidence…",
    "Verifying claims…",
    "Calibrating confidence…",
)


class _StageProgress:
    """Renders backend-reported stage transitions into one placeholder.

    Each update is a single small markdown write (milliseconds), made only
    when the pipeline reports a new stage, so it adds no meaningful time to
    the investigation. Times shown are real wall-clock times per stage.
    """

    def __init__(self, placeholder, question: str, mode: str, resolution=None):
        from app.investigation.investigator import PIPELINE_STAGES

        self.placeholder = placeholder
        self.question = question
        self.mode = mode
        self.resolution = resolution
        self.stages = list(PIPELINE_STAGES)
        self.started = time.monotonic()
        self.stage_started: dict[str, float] = {}
        self.durations: dict[str, float] = {}
        self.current: str | None = None
        self.render()

    def __call__(self, stage: str, label: str) -> None:
        now = time.monotonic()
        if self.current is not None:
            self.durations[self.current] = now - self.stage_started[self.current]
        self.current = None if stage == "done" else stage
        if self.current:
            self.stage_started[self.current] = now
        self.render()

    def render(self) -> None:
        elapsed = time.monotonic() - self.started
        rows = []
        for key, label in self.stages:
            if key in self.durations:
                cls, mark, note = "done", "✓", f"{self.durations[key]:.1f}s"
            elif key == self.current:
                cls, mark, note = "on", "●", "running"
            else:
                cls, mark, note = "", "○", ""
            rows.append(
                f'<li class="{cls}"><span class="mk" aria-hidden="true">{mark}</span>'
                f'{_escape(label)}<span class="t">{note}</span></li>'
            )
        is_narrow = InvestigationPlanner._is_narrow_question(self.question)
        route = "Focused lookup · deterministic plan" if is_narrow else "Full investigation · Gemini-planned"
        context_line = ""
        if self.mode == "follow-up" and self.resolution is not None:
            anchors = ", ".join(self.resolution.anchors) or "the previous question"
            context_line = (
                f'<div class="pt-note">Follow-up · references resolved against '
                f"{_escape(anchors)}. Retrieval and verification are fresh for this question.</div>"
            )
        self.placeholder.markdown(
            _one_line(
                f"""
                <div class="pt" role="status" aria-live="polite">
                    <div class="pt-top">
                        <div><div class="pt-k">{"Follow-up" if self.mode == "follow-up" else "Investigating"}</div>
                        <div class="pt-q">{_escape(self.question)}</div></div>
                        <div class="pt-clock"><span>{elapsed:.0f}s</span><small>elapsed at last stage change</small></div>
                    </div>
                    <ol class="pt-steps">{"".join(rows)}</ol>
                    {context_line}
                    <div class="pt-note">{_escape(route)}. Stages are reported by the pipeline as they start;
                    retrieval and Gemini reasoning usually take longest.</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )


def _render_progress_animation(placeholder, question: str) -> None:
    """Backward-compatible helper: show the stage card before any stage has
    been reported (all stages pending)."""
    _StageProgress(placeholder, question, "new")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

def _classify_error(exc: Exception) -> tuple[str, str, str]:
    """(tone, title, message) for a pipeline failure. Preserves the
    existing 503 / 429 handling and wording."""
    error_text = str(exc)
    if "503" in error_text and "UNAVAILABLE" in error_text:
        return (
            "warn",
            "Gemini is temporarily unavailable",
            "TRACE reached the Gemini reasoning stage, but Gemini is "
            "temporarily unavailable. Your evidence pipeline is intact. "
            "Please retry the investigation in a moment.",
        )
    if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
        return (
            "warn",
            "Gemini API quota exhausted",
            "TRACE reached the Gemini reasoning stage, but the Gemini "
            "API quota is currently exhausted. No investigation result "
            "was generated; retry when quota is available.",
        )
    if "GEMINI_API_KEY" in error_text:
        return (
            "warn",
            "Gemini is not configured",
            "GEMINI_API_KEY is not set, so TRACE cannot run Gemini planning or "
            "reasoning. Add the key to the project .env file and restart the app.",
        )
    if "API key not valid" in error_text or "API_KEY_INVALID" in error_text:
        return (
            "warn",
            "Gemini rejected the API key",
            "The configured GEMINI_API_KEY was rejected by Gemini. Check the key in "
            "the project .env file and restart the app.",
        )
    timeout_markers = ("ReadTimeout", "WriteTimeout", "PoolTimeout", "TimeoutException",
                       "timed out", "DEADLINE_EXCEEDED")
    if any(marker in f"{type(exc).__name__}: {error_text}" for marker in timeout_markers):
        return (
            "warn",
            "Gemini timed out",
            "A Gemini request exceeded the time limit TRACE allows during an investigation. "
            "Retrieval is local and unaffected; retry when the connection is stable.",
        )
    network_markers = ("ProxyError", "ConnectError", "ConnectionError", "ConnectTimeout",
                       "getaddrinfo", "Name or service not known", "ReadTimeout")
    if any(marker in f"{type(exc).__name__}: {error_text}" for marker in network_markers):
        return (
            "warn",
            "Gemini could not be reached",
            "TRACE could not connect to the Gemini API (network, proxy or firewall). "
            "Evidence retrieval is local, but planning and reasoning need Gemini. "
            "Check the connection and retry.",
        )
    if "PERMISSION_DENIED" in error_text:
        return (
            "warn",
            "Gemini refused the request",
            "Gemini returned permission denied (403) for this key or model. Check that the "
            "key is enabled for the Gemini API and the configured model, then retry.",
        )
    if ("404" in error_text and "NOT_FOUND" in error_text) or "is not found for API version" in error_text:
        return (
            "warn",
            "Gemini model not available",
            "The configured Gemini model was not found for this API key. Check the model "
            "name in app/investigation/gemini_client.py, then restart.",
        )
    if "invalid JSON" in error_text or "empty response" in error_text:
        return (
            "warn",
            "Gemini returned an unusable response",
            "Gemini answered, but not with the structured JSON TRACE requires, so no "
            "claims were produced. Retry; nothing from that response is shown as evidence.",
        )
    return (
        "",
        "Investigation could not be completed",
        "TRACE could not complete this investigation. "
        "The backend pipeline returned an error.",
    )


def _render_last_error() -> None:
    error = st.session_state.get("last_error")
    if not error:
        return
    tone, title, message = _classify_error(error["exc"])
    kind = "Follow-up" if error.get("mode") == "follow-up" else "Question"
    _md(
        f'<div class="trace-error {tone}" role="alert">'
        f'<div class="trace-error-title">{_escape(title)}</div>'
        f'<div class="trace-error-copy">{_escape(message)}</div>'
        f'<div class="small-muted" style="margin-top:6px;">{kind}: {_escape(error["question"])}</div>'
        "</div>"
    )
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("Retry", key="retry_btn", type="primary", use_container_width=True):
            st.session_state.pop("last_error", None)
            if _request_run(error["question"], mode=error.get("mode", "new"), conv_id=error.get("conv_id")):
                st.rerun()
    with c2:
        if st.button("Dismiss", key="dismiss_err", use_container_width=True):
            st.session_state.pop("last_error", None)
            st.rerun()
    with st.expander("Technical details", expanded=False):
        st.exception(error["exc"])


# ---------------------------------------------------------------------------
# Running a queued request
# ---------------------------------------------------------------------------

def _retrieval_run_stats(meta: dict, before: tuple[int, int], result) -> dict:
    """Per-investigation retrieval facts from the real retriever counters
    and the channels recorded on each evidence item."""
    retriever = meta.get("retriever")
    calls = failures = 0
    last_error = None
    if retriever is not None:
        calls = getattr(retriever, "vector_calls", 0) - before[0]
        failures = getattr(retriever, "vector_failures", 0) - before[1]
        if failures:
            last_error = getattr(retriever, "last_vector_error", None)
    evidence = _evidence_items(result)
    return {
        "vector_calls": max(0, calls),
        "vector_failures": max(0, failures),
        "vector_last_error": last_error,
        "passages": len(evidence),
        "vector_passages": sum(1 for e in evidence if "vector" in (getattr(e, "retrieval_channels", []) or [])),
        "carried_passages": sum(
            1 for e in evidence if "carried-forward" in (getattr(e, "retrieval_channels", []) or [])
        ),
    }


def _run_pending(request: dict) -> None:
    """Execute the queued request through the real pipeline.

    Failed runs never leave a stale result on screen and never auto-retry:
    the request is cleared and the user retries explicitly.
    """
    from app.investigation.conversation import ConversationContext, resolve_question

    question = request["question"]
    mode = request.get("mode", "new")
    conversation = _find_conversation(request.get("conv_id")) if mode == "follow-up" else None
    if mode == "follow-up" and conversation is None:
        mode = "new"

    context = None
    if mode == "follow-up":
        prior = [t["result"] for t in conversation["turns"] if t.get("result") is not None]
        context = ConversationContext.from_results(conversation["id"], prior)
        resolution = context.resolve(question)
    else:
        resolution = resolve_question(question)

    # A question that points back at something ("why isn't that enough?")
    # with no conversation to resolve it against gets a clarification, not
    # a guessed investigation.
    if resolution.needs_clarification:
        conv = {"id": f"inv-{_next_id('conv_seq')}", "title": question, "turns": []}
        turn = {"id": _next_id("turn_seq"), "kind": "clarification", "question": question,
                "result": None, "meta": None, "message": resolution.clarification,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        conv["turns"].append(turn)
        _conversations().append(conv)
        st.session_state.pop("pending_run", None)
        st.session_state.pop("run_question", None)
        for key in ("result", "pipeline_meta", "executed_question"):
            st.session_state.pop(key, None)
        st.session_state["active_conv"] = conv["id"]
        st.session_state["active_id"] = turn["id"]
        st.rerun()
        return

    if mode == "new":
        # A new investigation never shows the previous one while running.
        for key in ("result", "pipeline_meta", "active_id", "active_conv"):
            st.session_state.pop(key, None)

    st.session_state["investigating"] = True
    st.session_state["active_question"] = question

    placeholder = st.empty()
    progress = _StageProgress(placeholder, question, mode, resolution if mode == "follow-up" else None)

    try:
        investigator, meta = load_pipeline()
        retriever = meta.get("retriever")
        before = (getattr(retriever, "vector_calls", 0), getattr(retriever, "vector_failures", 0))
        result = investigator.run(question, context=context, progress_callback=progress)
    except Exception as exc:
        placeholder.empty()
        st.session_state["investigating"] = False
        # Last real pipeline invocation, for the health panel. Only the
        # classified title is kept (never request details or secrets).
        st.session_state["last_pipeline_run"] = {
            "ok": False,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "title": _classify_error(exc)[1],
            "stage": getattr(progress, "current", None),
        }
        for key in ("active_question", "pending_run", "run_question"):
            st.session_state.pop(key, None)
        if mode == "new":
            for key in ("executed_question", "result", "pipeline_meta", "active_id", "active_conv"):
                st.session_state.pop(key, None)
        st.session_state["last_error"] = {
            "question": question,
            "exc": exc,
            "mode": mode,
            "conv_id": conversation["id"] if conversation else None,
        }
        st.rerun()
        return

    placeholder.empty()

    run_meta = {k: v for k, v in meta.items() if k != "retriever"}
    run_meta["retrieval_run"] = _retrieval_run_stats(meta, before, result)
    # Measured wall-clock time per pipeline stage (from the progress
    # callback) and in total, for the Reports & history page.
    run_meta["stage_seconds"] = dict(getattr(progress, "durations", {}) or {})
    run_meta["total_seconds"] = round(time.monotonic() - getattr(progress, "started", time.monotonic()), 1)

    turn = {"id": _next_id("turn_seq"), "kind": "follow-up" if mode == "follow-up" else "investigation",
            "question": question, "result": result, "meta": run_meta, "message": "",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if conversation is None:
        conversation = {"id": f"inv-{_next_id('conv_seq')}", "title": question, "turns": []}
        _conversations().append(conversation)
    conversation["turns"].append(turn)

    st.session_state.pop("pending_run", None)
    st.session_state["investigating"] = False
    st.session_state["just_completed"] = True
    plan = getattr(result, "plan", None)
    st.session_state["last_pipeline_run"] = {
        "ok": True,
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "plan_source": getattr(plan, "source", ""),
        "plan_fallback_reason": getattr(plan, "fallback_reason", ""),
        "vector_calls": run_meta["retrieval_run"].get("vector_calls", 0),
        "vector_failures": run_meta["retrieval_run"].get("vector_failures", 0),
        "vector_passages": run_meta["retrieval_run"].get("vector_passages", 0),
        "total_seconds": run_meta.get("total_seconds"),
    }
    # Tells the Command Center shell which page to show next (a new
    # investigation opens on the Overview; a follow-up stays in Copilot).
    st.session_state["cc_after_run"] = mode
    _activate_turn(conversation, turn)
    st.session_state["scroll_top"] = False
    st.rerun()


def render_clarification(conversation: dict, turn: dict) -> None:
    with _chat_message("user"):
        _md(f'<div class="thread-question">{_escape(turn["question"])}</div>')
    with _chat_message("assistant"):
        _md(
            '<div class="thread-turn-tag">Clarification needed · no investigation was run</div>'
            f'<div class="thread-answer">{_escape(turn["message"])}</div>'
        )
    _md(
        '<div class="small-muted" style="margin-top:6px;">Edit the question in the composer '
        "above to name the equipment, parameter, or document, then choose Investigate.</div>"
    )


def main() -> None:
    """Launch TRACE.

    The default UI is the multi-page Investigation Command Center
    (app/command_center). Set TRACE_CLASSIC_UI=1 to run the previous
    single-page workspace instead (main_classic below, unchanged).
    """
    if os.getenv("TRACE_CLASSIC_UI", "").strip() in {"1", "true", "yes"}:
        main_classic()
        return
    from app.command_center.app import run

    run()


def main_classic() -> None:
    st.set_page_config(
        page_title="TRACE — Manufacturing Evidence Investigator",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="auto",
    )

    inject_css()

    request = st.session_state.get("pending_run")
    pending = bool(request)

    conversation = _active_conversation()
    active_turn = _find_turn(conversation, st.session_state.get("active_id"))
    result = active_turn.get("result") if active_turn else None
    show_result = result is not None and not (pending and request.get("mode") == "new")

    # render_header()/render_question_input() render the primary,
    # above-the-fold content and have no dependency on _render_sidebar().
    # _render_sidebar is called after them so the cold-start load_pipeline()
    # call it makes for the system-status panel does not block the header
    # and composer from streaming to the browser first.
    render_header()

    if show_result:
        # Results view: the composer stays available (same widget, same
        # state) but collapses so the conclusion leads the page.
        with st.expander("Start a different investigation (new question, no carried context)", expanded=False):
            render_question_input(disabled=pending, compact=True)
    else:
        render_question_input(disabled=pending)

    _render_sidebar(result)

    _render_last_error()

    if pending and request.get("mode") == "new":
        _run_pending(request)
        return

    if active_turn is not None and active_turn.get("kind") == "clarification":
        render_clarification(conversation, active_turn)
        return

    if not show_result:
        if not st.session_state.get("last_error"):
            render_empty_state()
        _render_reference_cases()
        return

    if st.session_state.pop("just_completed", False):
        if hasattr(st, "toast"):
            st.toast("Investigation complete")
        st.session_state["top_nonce"] = st.session_state.get("top_nonce", 0) + 1
        scroll_to_top(st.session_state["top_nonce"])
    elif st.session_state.pop("scroll_top", False):
        st.session_state["top_nonce"] = st.session_state.get("top_nonce", 0) + 1
        scroll_to_top(st.session_state["top_nonce"])

    meta = active_turn.get("meta") or st.session_state.get(
        "pipeline_meta",
        {"documents": 0, "chunks": 0, "vector_enabled": False},
    )

    render_conversation_history(conversation, active_turn["id"])

    latest = conversation["turns"][-1]
    if active_turn["id"] != latest["id"] and latest.get("result") is not None:
        if st.button("↩  Back to the latest answer", key="back_latest"):
            _activate_turn(conversation, latest)
            st.rerun()

    render_result(result, meta)

    if pending and request.get("mode") == "follow-up":
        # Re-render the chat input disabled BEFORE the blocking run, so the
        # previous (enabled) input cannot queue a second follow-up.
        st.chat_input(
            "Follow-up running… the input is disabled until it completes",
            key="followup_input",
            disabled=True,
        )
        _md('<div class="thread-title">Running follow-up</div>')
        _run_pending(request)
        return

    render_followup_panel(result, conversation, pending)

    followup = st.chat_input(
        "Ask a follow-up about this investigation (Enter to send)",
        key="followup_input",
        disabled=pending,
    )
    if followup and followup.strip():
        if _request_run(followup, mode="follow-up", conv_id=conversation["id"]):
            st.rerun()


if __name__ == "__main__":
    main()
