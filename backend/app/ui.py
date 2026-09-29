"""
TRACE — Manufacturing Evidence Investigator
Judge-facing Streamlit UI.

Presentation layer only:
- Uses the existing Investigator pipeline.
- Does not change investigation logic.
- Keeps evidence, claims, timeline, and gaps traceable to backend objects.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import streamlit as st

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


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --trace-bg: #080b10;
            --trace-surface: #0f141b;
            --trace-surface-2: #121923;
            --trace-border: #202a36;
            --trace-border-strong: #2b3948;
            --trace-text: #f2f5f8;
            --trace-muted: #8e9aaa;
            --trace-blue: #5da9ff;
            --trace-cyan: #69d6e7;
            --trace-green: #58c58a;
            --trace-amber: #e7ad58;
            --trace-red: #e36d6d;
            --trace-purple: #a78bfa;
        }

        .stApp {
            background:
                radial-gradient(circle at 82% 0%, rgba(93,169,255,.08), transparent 28rem),
                linear-gradient(180deg, #080b10 0%, #090d12 100%);
            color: var(--trace-text);
        }

        .block-container {
            max-width: 1450px;
            padding-top: 2.1rem;
            padding-bottom: 4rem;
        }

        [data-testid="stHeader"] {
            background: rgba(8,11,16,.78);
        }

        [data-testid="stSidebar"] {
            background: #090d12;
            border-right: 1px solid var(--trace-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.4rem;
        }

        .trace-brand {
            display: flex;
            align-items: center;
            gap: 11px;
            margin-bottom: 2.1rem;
        }

        .trace-mark {
            width: 31px;
            height: 31px;
            border: 1px solid #3d89d8;
            border-radius: 8px;
            display: grid;
            place-items: center;
            color: var(--trace-cyan);
            font-weight: 800;
            font-size: 13px;
            letter-spacing: -.04em;
            box-shadow: 0 0 22px rgba(93,169,255,.10);
        }

        .trace-wordmark {
            font-size: 18px;
            font-weight: 760;
            letter-spacing: .02em;
        }

        .trace-caption {
            color: var(--trace-muted);
            font-size: 11px;
            letter-spacing: .07em;
            text-transform: uppercase;
        }

        .topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: .7rem;
        }

        .eyebrow {
            color: var(--trace-cyan);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .14em;
            text-transform: uppercase;
        }

        .system-chip {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            color: #b9c5d1;
            border: 1px solid var(--trace-border);
            background: rgba(15,20,27,.72);
            border-radius: 999px;
            padding: 6px 10px;
            font-size: 11px;
        }

        .system-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--trace-green);
            box-shadow: 0 0 9px rgba(88,197,138,.7);
        }

        h1, h2, h3 {
            color: var(--trace-text) !important;
        }

        .page-title {
            font-size: clamp(27px, 3vw, 42px);
            line-height: 1.08;
            letter-spacing: -.035em;
            font-weight: 720;
            margin: 0;
        }

        .page-subtitle {
            color: var(--trace-muted);
            margin-top: 8px;
            margin-bottom: 28px;
            font-size: 14px;
        }

        .question-box {
            border: 1px solid var(--trace-border-strong);
            background: linear-gradient(145deg, rgba(18,25,35,.96), rgba(11,15,21,.96));
            border-radius: 14px;
            padding: 18px 20px 14px;
            box-shadow: 0 18px 50px rgba(0,0,0,.20);
        }

        .question-label {
            color: var(--trace-muted);
            font-size: 10px;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .question-text {
            color: #e9eef4;
            font-size: 18px;
            line-height: 1.45;
            font-weight: 530;
        }

        div[data-testid="stTextArea"] textarea {
            background: #0b1016 !important;
            color: #e9eef4 !important;
            border: 1px solid var(--trace-border-strong) !important;
            border-radius: 10px !important;
            font-size: 15px !important;
        }

        div[data-testid="stTextInput"] input {
            background: #0b1016 !important;
            color: #e9eef4 !important;
            border: 1px solid var(--trace-border-strong) !important;
            border-radius: 9px !important;
        }

        .result-card {
            background: linear-gradient(145deg, #111821, #0d131a);
            border: 1px solid var(--trace-border);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 12px;
        }

        .result-card.compact {
            padding: 15px 17px;
        }

        .section-label {
            color: #7f8b9a;
            font-size: 10px;
            font-weight: 750;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .assessment-row {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 20px;
        }

        .assessment-title {
            color: #e8eef4;
            font-size: 20px;
            line-height: 1.38;
            font-weight: 720;
            letter-spacing: -.018em;
            max-width: 900px;
        }

        .confidence {
            min-width: 106px;
            text-align: right;
        }

        .confidence-number {
            font-size: 25px;
            font-weight: 730;
            color: #dce7f2;
        }

        .confidence-label {
            color: var(--trace-muted);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: .1em;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 6px;
            padding: 5px 8px;
            font-size: 10px;
            font-weight: 760;
            letter-spacing: .08em;
            text-transform: uppercase;
            border: 1px solid currentColor;
        }

        .badge.green { color: var(--trace-green); background: rgba(88,197,138,.08); }
        .badge.amber { color: var(--trace-amber); background: rgba(231,173,88,.08); }
        .badge.red { color: var(--trace-red); background: rgba(227,109,109,.08); }
        .badge.blue { color: var(--trace-blue); background: rgba(93,169,255,.08); }
        .badge.gray { color: #9ba7b5; background: rgba(155,167,181,.07); }

        .verify-strip {
            margin-top: 16px;
            padding: 11px 13px;
            border: 1px solid rgba(231,173,88,.28);
            background: rgba(231,173,88,.06);
            border-radius: 9px;
            color: #d9b77e;
            font-size: 12px;
        }

        .finding-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0,1fr));
            gap: 9px;
        }

        .finding {
            border: 1px solid var(--trace-border);
            background: #0b1117;
            border-radius: 9px;
            padding: 12px;
        }

        .finding-name {
            font-size: 12px;
            font-weight: 650;
            color: #e2e8ee;
        }

        .finding-meta {
            margin-top: 5px;
            color: var(--trace-muted);
            font-size: 10px;
        }

        .path-item {
            display: grid;
            grid-template-columns: 27px 1fr auto;
            gap: 10px;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #19222c;
        }

        .path-item:last-child { border-bottom: 0; }

        .path-index {
            width: 24px;
            height: 24px;
            border: 1px solid var(--trace-border-strong);
            border-radius: 7px;
            display: grid;
            place-items: center;
            color: #8390a0;
            font-size: 10px;
        }

        .path-question {
            color: #dce4ec;
            font-size: 12px;
            line-height: 1.42;
        }

        .path-status {
            font-size: 10px;
            color: var(--trace-green);
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        .path-status.warn { color: var(--trace-amber); }

        .timeline {
            position: relative;
            margin: 3px 0 4px 5px;
            padding-left: 25px;
            border-left: 1px solid #293441;
        }

        .timeline-event {
            position: relative;
            padding: 0 0 18px 0;
        }

        .timeline-event:before {
            content: "";
            position: absolute;
            left: -30px;
            top: 4px;
            width: 9px;
            height: 9px;
            border: 2px solid #65b4e8;
            background: #0b1016;
            border-radius: 50%;
            box-shadow: 0 0 0 4px rgba(101,180,232,.06);
        }

        .timeline-date {
            color: #65b4e8;
            font-size: 10px;
            font-weight: 760;
            letter-spacing: .1em;
            text-transform: uppercase;
        }

        .timeline-description {
            color: #d5dde5;
            font-size: 12px;
            line-height: 1.5;
            margin-top: 4px;
        }

        .timeline-source {
            color: #697786;
            font-size: 10px;
            margin-top: 4px;
        }

        .evidence-card {
            border: 1px solid var(--trace-border);
            background: #0b1016;
            border-radius: 10px;
            padding: 13px 14px;
            margin-bottom: 8px;
        }

        .evidence-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: center;
        }

        .evidence-id {
            color: #7bc5ee;
            font-size: 10px;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: .02em;
        }

        .evidence-file {
            color: #dfe7ee;
            font-size: 12px;
            font-weight: 650;
            margin-top: 5px;
        }

        .evidence-path {
            color: #697786;
            font-size: 10px;
            margin-top: 3px;
        }

        .evidence-content {
            color: #aab5c1;
            font-size: 11px;
            line-height: 1.55;
            margin-top: 10px;
            white-space: pre-wrap;
        }

        .evidence-reason {
            color: #7f8e9e;
            font-size: 10px;
            margin-top: 9px;
        }

        .gap-card {
            border: 1px solid rgba(231,173,88,.25);
            background: linear-gradient(145deg, rgba(55,40,20,.24), rgba(16,19,23,.9));
            border-radius: 12px;
            padding: 15px;
        }

        .gap-title {
            color: #e9bd72;
            font-size: 13px;
            font-weight: 700;
        }

        .gap-copy {
            color: #b8b4aa;
            font-size: 12px;
            line-height: 1.55;
            margin-top: 5px;
        }

        .human-action {
            border: 1px solid rgba(93,169,255,.25);
            background: rgba(93,169,255,.045);
            border-radius: 12px;
            padding: 15px;
        }

        .human-title {
            color: #8fc7ff;
            font-size: 12px;
            font-weight: 750;
            letter-spacing: .08em;
            text-transform: uppercase;
        }

        .human-copy {
            color: #b6c1cd;
            font-size: 12px;
            line-height: 1.55;
            margin-top: 5px;
        }

        .empty-state {
            border: 1px dashed #2a3643;
            border-radius: 14px;
            padding: 48px 30px;
            text-align: center;
            background: rgba(15,20,27,.52);
        }

        .empty-title {
            color: #e2e9ef;
            font-size: 18px;
            font-weight: 680;
        }

        .empty-copy {
            color: #7f8b99;
            font-size: 12px;
            max-width: 570px;
            margin: 7px auto 0;
            line-height: 1.55;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
        }

        .metric {
            border: 1px solid var(--trace-border);
            background: #0b1016;
            border-radius: 9px;
            padding: 12px;
        }

        .metric-value {
            color: #e6edf3;
            font-size: 18px;
            font-weight: 720;
        }

        .metric-label {
            color: #707d8b;
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: .1em;
            margin-top: 3px;
        }

        .stButton > button {
            border-radius: 9px !important;
            border: 1px solid #2d6fa9 !important;
            background: linear-gradient(180deg, #175a8f, #124b76) !important;
            color: #f4f9fd !important;
            font-weight: 700 !important;
            min-height: 42px !important;
        }

        .stButton > button:hover {
            border-color: #65b4e8 !important;
            background: linear-gradient(180deg, #1a679f, #155580) !important;
        }

        div[data-testid="stExpander"] {
            border-color: var(--trace-border) !important;
            background: rgba(11,16,22,.68) !important;
            border-radius: 10px !important;
        }

        .small-muted {
            color: #697786;
            font-size: 10px;
            line-height: 1.45;
        }

        .trace-result-header {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 92% 0%, rgba(93,169,255,.10), transparent 20rem),
                linear-gradient(145deg, #111a23, #0b1016);
            border: 1px solid #263544;
            border-radius: 16px;
            padding: 18px 20px 17px;
            margin-bottom: 12px;
        }

        .trace-result-header:before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 3px;
            background: linear-gradient(180deg, #69d6e7, #5da9ff);
        }

        .trace-result-kicker {
            color: #69d6e7;
            font-size: 9px;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .trace-result-question {
            color: #edf3f8;
            font-size: 19px;
            line-height: 1.38;
            font-weight: 680;
            letter-spacing: -.012em;
            max-width: 1050px;
            margin-top: 6px;
        }

        .trace-result-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin-top: 12px;
        }

        .trace-result-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid #263544;
            background: rgba(8,12,17,.62);
            border-radius: 7px;
            padding: 5px 8px;
            color: #8997a6;
            font-size: 9px;
            letter-spacing: .06em;
            text-transform: uppercase;
        }

        .trace-result-chip strong {
            color: #cbd6df;
            font-weight: 750;
        }

        @media (max-width: 850px) {
            .finding-grid, .metric-row {
                grid-template-columns: 1fr;
            }
            .assessment-row {
                flex-direction: column;
            }
            .confidence {
                text-align: left;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Construct the existing TRACE pipeline once per Streamlit process.

    The UI deliberately delegates investigation behavior to
    app.investigation.build.build_investigator().
    """
    documents = load_corpus(str(CORPUS_PATH))
    chunks = chunk_corpus(documents)

    bm25 = BM25Searcher(chunks)
    entities = EntityMatcher(chunks)

    # The persisted index has existed in two formats over TRACE's
    # development. We keep this UI adapter local rather than changing
    # the backend contract during UI work.
    vector_searcher = None

    if os.getenv("GEMINI_API_KEY"):
        try:
            import json
            from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
            from app.retrieval.vector_search import VectorSearcher
            from app.ingestion.chunker import DocumentChunk

            index_path = Path(INDEX_PATH)

            if index_path.exists():
                payload = json.loads(index_path.read_text(encoding="utf-8"))

                if isinstance(payload, list):
                    records = payload
                    index_chunks = [
                        DocumentChunk(
                            document_id=r["document_id"],
                            filename=r["filename"],
                            relative_path=r["relative_path"],
                            chunk_id=r["chunk_id"],
                            content=r["content"],
                        )
                        for r in records
                    ]
                    embeddings = [r["embedding"] for r in records]

                    if [c.chunk_id for c in index_chunks] == [c.chunk_id for c in chunks]:
                        provider = GeminiEmbeddingProvider()
                        vector_searcher = VectorSearcher(
                            chunks=index_chunks,
                            embeddings=embeddings,
                            embedding_provider=provider,
                        )
                elif isinstance(payload, dict):
                    from app.retrieval.embedding_store import EmbeddingStore

                    store = EmbeddingStore(index_path)
                    index_chunks, embeddings = store.load(
                        expected_model="gemini-embedding-001"
                    )

                    if [c.chunk_id for c in index_chunks] == [c.chunk_id for c in chunks]:
                        provider = GeminiEmbeddingProvider()
                        vector_searcher = VectorSearcher(
                            chunks=index_chunks,
                            embeddings=embeddings,
                            embedding_provider=provider,
                        )
        except Exception:
            # The investigation remains usable with lexical + entity
            # retrieval if the optional vector layer cannot initialize.
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
    )

    return build_investigator(retriever, gemini_client=ui_gemini), {
        "documents": len(documents),
        "chunks": len(chunks),
        "vector_enabled": vector_searcher is not None,
    }


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _badge(label: str, tone: str) -> str:
    return f'<span class="badge {tone}">{_escape(label)}</span>'


def _primary_claim(result):
    claims = result.report.claims

    priority = [
        ClaimType.CURRENT_TECHNICAL_JUSTIFICATION,
        ClaimType.EVIDENCE_GAP,
        ClaimType.ORIGINAL_BASIS,
        ClaimType.CURRENT_CAPABILITY,
        ClaimType.ADMINISTRATIVE_ACTION,
        ClaimType.RECOMMENDATION,
    ]

    for claim_type in priority:
        matches = [c for c in claims if c.claim_type == claim_type]
        if matches:
            return max(matches, key=lambda c: c.confidence)

    return claims[0] if claims else None


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


def _render_sidebar(result_state: dict | None) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="trace-brand">
                <div class="trace-mark">T</div>
                <div>
                    <div class="trace-wordmark">TRACE</div>
                    <div class="trace-caption">Evidence Investigator</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**NEW INVESTIGATION**")
        st.caption("Ask TRACE to investigate a manufacturing decision, limit, change, or historical rationale.")

        if st.button("＋  Start new investigation", use_container_width=True):
            for key in (
                "result",
                "executed_question",
                "active_question",
                "investigating",
                "pipeline_meta",
                "pending_question",
                "run_question",
            ):
                st.session_state.pop(key, None)
            st.session_state["question_input"] = HERO_QUESTION
            st.rerun()

        st.markdown("---")
        st.markdown("**RECENT**")

        if result_state:
            q = result_state.get("question", HERO_QUESTION)
            st.markdown(
                f'<div class="small-muted">Current investigation</div>'
                f'<div style="color:#dce5ed;font-size:12px;margin-top:5px;">'
                f'{_escape(q[:105])}{"…" if len(q) > 105 else ""}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="small-muted">No investigation run yet.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**SYSTEM**")

        try:
            _, meta = load_pipeline()
            vector_label = "ON" if meta["vector_enabled"] else "OFF"
            st.markdown(
                f"""
                <div class="small-muted">
                    CORPUS&nbsp;&nbsp; {meta["documents"]} documents<br>
                    CHUNKS&nbsp;&nbsp;&nbsp; {meta["chunks"]}<br>
                    VECTOR&nbsp;&nbsp;&nbsp; {vector_label}<br>
                    REASONING&nbsp; Gemini + deterministic verification
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                '<div class="small-muted">Pipeline not initialized.</div>',
                unsafe_allow_html=True,
            )


def render_header() -> None:
    st.markdown(
        """
        <div class="topline">
            <div class="eyebrow">Manufacturing evidence system</div>
            <div class="system-chip">
                <span class="system-dot"></span>
                Evidence system ready
            </div>
        </div>
        <div class="page-title">TRACE</div>
        <div class="page-subtitle">
            Investigate why a manufacturing decision exists, what changed,
            and whether its current technical basis is still supported by evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_question_input() -> str:
    st.markdown(
        '<div class="question-label">Investigation question</div>',
        unsafe_allow_html=True,
    )

    # Apply example selections BEFORE instantiating the text widget.
    # This avoids Streamlit's StreamlitWidgetAlreadyInstantiatedError.
    if "pending_question" in st.session_state:
        st.session_state["question_input"] = st.session_state.pop("pending_question")

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = HERO_QUESTION

    question = st.text_area(
        "Investigation question",
        height=94,
        label_visibility="collapsed",
        key="question_input",
        placeholder="Ask a manufacturing evidence question…",
    )

    cols = st.columns([1, 5, 1.6])
    with cols[0]:
        run = st.button(
            "Investigate",
            use_container_width=True,
            type="primary",
        )
    with cols[1]:
        st.markdown(
            '<div class="small-muted" style="padding:11px 0 0 4px;">'
            "TRACE will plan evidence-seeking questions, retrieve across the corpus, "
            "reconstruct the timeline, analyze evidence gaps, and verify the final claims."
            "</div>",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            '<div style="padding-top:10px;text-align:right;">'
            '<span class="small-muted">Gemini reasoning + evidence verification</span></div>',
            unsafe_allow_html=True,
        )

    if run:
        if not question.strip():
            st.error("Enter an investigation question.")
            return ""
        st.session_state["run_question"] = question.strip()

    return st.session_state.get("run_question", "")


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-title">Ready to investigate.</div>
            <div class="empty-copy">
                Start with a specific manufacturing question. TRACE will show
                the investigation path, evidence trail, historical changes,
                evidence gaps, and the human verification step.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assessment(result) -> None:
    label, statement, tone, confidence, requires_human = _assessment(result)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="section-label">Current assessment</div>
            <div class="assessment-row">
                <div style="flex:1;">
                    {_badge(label, tone)}
                    <div class="assessment-title" style="margin-top:9px;">
                        {_escape(statement)}
                    </div>
                </div>
                <div class="confidence">
                    <div class="confidence-number">{confidence:.0%}</div>
                    <div class="confidence-label">calibrated confidence</div>
                </div>
            </div>
            {
                '<div class="verify-strip">⚠ &nbsp; HUMAN VERIFICATION REQUIRED &nbsp; '
                'TRACE does not authorize a production change.</div>'
                if requires_human else
                '<div class="verify-strip" style="border-color:rgba(88,197,138,.25);'
                'background:rgba(88,197,138,.045);color:#86d3a8;">'
                '✓ &nbsp; Evidence is sufficient for the stated assessment.</div>'
            }
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_finding_summary(result) -> None:
    st.markdown(
        '<div class="result-card"><div class="section-label">What TRACE found</div>',
        unsafe_allow_html=True,
    )

    findings = result.evidence_analysis.findings

    if not findings:
        st.markdown(
            '<div class="small-muted">No structured evidence findings were generated.</div>',
            unsafe_allow_html=True,
        )
    else:
        cards = []
        for finding in findings[:6]:
            label, tone = FINDING_META.get(
                finding.finding_type.value,
                ("Finding", "gray"),
            )
            cards.append(
                f"""
                <div class="finding">
                    {_badge(label, tone)}
                    <div class="finding-name">{_escape(finding.statement[:190])}
                    {"…" if len(finding.statement) > 190 else ""}</div>
                    <div class="finding-meta">
                        {len(finding.evidence_chunk_ids)} cited evidence item(s)
                    </div>
                </div>
                """
            )

        st.markdown(
            '<div class="finding-grid">' + "".join(cards) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_investigation_path(result) -> None:
    st.markdown(
        '<div class="result-card"><div class="section-label">'
        'Investigation path · TRACE generated evidence-seeking questions'
        '</div>',
        unsafe_allow_html=True,
    )

    questions = result.plan.subquestions

    for index, question in enumerate(questions, start=1):
        associated = [
            item for item in result.investigation.evidence
            if question in item.retrieved_for
        ]
        status = "EVIDENCE FOUND" if associated else "GAP"
        tone = "" if associated else " warn"

        st.markdown(
            f"""
            <div class="path-item">
                <div class="path-index">{index:02d}</div>
                <div class="path-question">{_escape(question)}</div>
                <div class="path-status{tone}">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_timeline(result) -> None:
    st.markdown(
        '<div class="result-card"><div class="section-label">Evidence timeline</div>',
        unsafe_allow_html=True,
    )

    if not result.timeline.events:
        st.markdown(
            '<div class="small-muted">No explicit dated events were extracted from the retrieved evidence.</div>',
            unsafe_allow_html=True,
        )
    else:
        html = ['<div class="timeline">']
        for event in result.timeline.events:
            sources = ", ".join(event.evidence_chunk_ids[:4])
            html.append(
                f"""
                <div class="timeline-event">
                    <div class="timeline-date">{_escape(event.date)}</div>
                    <div class="timeline-description">{_escape(event.description)}</div>
                    <div class="timeline-source">Evidence · {_escape(sources)}</div>
                </div>
                """
            )
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_evidence(result) -> None:
    st.markdown(
        '<div class="result-card"><div class="section-label">'
        'Evidence trail · source-level provenance'
        '</div>',
        unsafe_allow_html=True,
    )

    if not result.investigation.evidence:
        st.markdown(
            '<div class="small-muted">No evidence was retrieved.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Show the most useful evidence first while preserving backend ranking.
    evidence = sorted(
        result.investigation.evidence,
        key=lambda item: item.retrieval_score,
        reverse=True,
    )

    for item in evidence[:12]:
        retrieved_for = item.retrieved_for
        reason = (
            retrieved_for[0]
            if retrieved_for
            else "Retrieved as supporting investigation evidence."
        )

        preview = item.content.strip()
        if len(preview) > 700:
            preview = preview[:700].rstrip() + "…"

        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="evidence-top">
                    <span class="evidence-id">{_escape(item.chunk_id)}</span>
                    <span class="small-muted">retrieval {item.retrieval_score:.4f}</span>
                </div>
                <div class="evidence-file">{_escape(item.filename)}</div>
                <div class="evidence-path">{_escape(item.relative_path)}</div>
                <div class="evidence-content">{_escape(preview)}</div>
                <div class="evidence-reason">Retrieved for · {_escape(reason)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if len(evidence) > 12:
        st.caption(f"{len(evidence) - 12} additional evidence items available in the investigation result.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_claims(result) -> None:
    st.markdown(
        '<div class="result-card"><div class="section-label">'
        'Reasoning claims · verified after Gemini'
        '</div>',
        unsafe_allow_html=True,
    )

    for claim in result.report.claims:
        label, tone = STANCE_META.get(claim.stance, ("UNKNOWN", "gray"))
        evidence_ids = ", ".join(claim.evidence_chunk_ids) or "No valid citation"

        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="evidence-top">
                    {_badge(label, tone)}
                    <span class="small-muted">{claim.confidence:.0%} confidence</span>
                </div>
                <div class="evidence-file">{_escape(claim.statement)}</div>
                <div class="evidence-path">
                    {_escape(claim.claim_type.value if claim.claim_type else "untyped claim")}
                    · Evidence { _escape(evidence_ids) }
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_gap_and_action(result) -> None:
    gaps = [
        f for f in result.evidence_analysis.findings
        if f.finding_type.value == "evidence-gap"
    ]

    if gaps:
        st.markdown(
            '<div class="gap-card">'
            '<div class="gap-title">Evidence gap</div>'
            f'<div class="gap-copy">{_escape(gaps[0].statement)}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="human-action">
            <div class="human-title">Human verification</div>
            <div class="human-copy">{_escape(_human_action(result))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_header(result, meta) -> None:
    question = getattr(result.investigation, "question", "")
    retrieval_mode = (
        "HYBRID RETRIEVAL"
        if meta.get("vector_enabled")
        else "LEXICAL + ENTITY RETRIEVAL"
    )

    st.markdown(
        f"""
        <div class="trace-result-header">
            <div class="trace-result-kicker">Investigation complete · evidence review</div>
            <div class="trace-result-question">{_escape(question)}</div>
            <div class="trace-result-meta">
                <span class="trace-result-chip"><strong>{meta.get("documents", 0)}</strong> documents</span>
                <span class="trace-result-chip"><strong>{meta.get("chunks", 0)}</strong> evidence chunks</span>
                <span class="trace-result-chip">{_escape(retrieval_mode)}</span>
                <span class="trace-result-chip">post-hoc claim verification</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(result, meta) -> None:
    render_result_header(result, meta)
    render_assessment(result)

    c1, c2 = st.columns([1.35, 1], gap="medium")
    with c1:
        render_finding_summary(result)
    with c2:
        st.markdown(
            '<div class="result-card"><div class="section-label">Investigation coverage</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="metric-row">
                <div class="metric">
                    <div class="metric-value">{len(result.plan.subquestions)}</div>
                    <div class="metric-label">questions</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(result.investigation.evidence)}</div>
                    <div class="metric-label">evidence items</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(result.timeline.events)}</div>
                    <div class="metric-label">timeline events</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="small-muted" style="margin-top:10px;">'
            "Claims are post-hoc verified against retrieved evidence. "
            "Confidence is deterministically calibrated."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    render_investigation_path(result)

    c1, c2 = st.columns([1.1, 1], gap="medium")
    with c1:
        render_timeline(result)
    with c2:
        render_gap_and_action(result)

    render_evidence(result)
    render_claims(result)

    with st.expander("Investigation data", expanded=False):
        st.json(
            {
                "question": result.investigation.question,
                "subquestions": result.plan.subquestions,
                "unresolved_questions": result.report.unresolved_questions,
                "evidence_count": len(result.investigation.evidence),
                "timeline_events": len(result.timeline.events),
            }
        )

def main() -> None:
    st.set_page_config(
        page_title="TRACE — Manufacturing Evidence Investigator",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    previous = st.session_state.get("result")
    _render_sidebar(previous)

    render_header()

    question = render_question_input()

    if question and question != st.session_state.get("executed_question"):
        try:
            # A new question must never display the previous investigation
            # while the new one is running or if it fails.
            st.session_state.pop("result", None)
            st.session_state.pop("pipeline_meta", None)
            st.session_state["investigating"] = True
            st.session_state["active_question"] = question

            status = st.status(
                "TRACE is investigating…",
                expanded=True,
            )
            status.write(
                "Planning the investigation, retrieving evidence, reconstructing "
                "the timeline, and generating the verified conclusion."
            )
            investigator, meta = load_pipeline()

            result = investigator.run(question)

            status.update(
                label="Investigation complete",
                state="complete",
                expanded=False,
            )

            st.session_state["result"] = result
            st.session_state["executed_question"] = question
            st.session_state["pipeline_meta"] = meta
            st.session_state["investigating"] = False
            st.rerun()

        except Exception as exc:
            st.session_state["investigating"] = False
            st.session_state.pop("active_question", None)
            error_text = str(exc)
            if "503" in error_text and "UNAVAILABLE" in error_text:
                st.warning(
                    "TRACE reached the Gemini reasoning stage, but Gemini is "
                    "temporarily unavailable. Your evidence pipeline is intact. "
                    "Please retry the investigation in a moment."
                )
            elif "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                st.warning(
                    "TRACE reached the Gemini reasoning stage, but the Gemini "
                    "API quota is currently exhausted. No investigation result "
                    "was generated; retry when quota is available."
                )
            else:
                st.error(
                    "TRACE could not complete this investigation. "
                    "The backend pipeline returned an error."
                )
            with st.expander("Technical details", expanded=False):
                st.exception(exc)
            # Failed runs must not block retries and must not leave stale
            # results from a previous question.
            st.session_state.pop("executed_question", None)
            st.session_state.pop("result", None)
            st.session_state.pop("pipeline_meta", None)

    result = st.session_state.get("result")

    if result is None:
        render_empty_state()
        st.markdown(
            '<div style="height:16px"></div>'
            '<div class="section-label">Investigation cases</div>'
            '<div class="small-muted" style="margin:5px 0 12px;">'
            'Use representative manufacturing cases to see how TRACE handles failures, supported conditions, and evidence gaps.'
            '</div>',
            unsafe_allow_html=True,
        )

        scenarios = get_scenarios()
        cols = st.columns(2)
        for index, scenario in enumerate(scenarios):
            with cols[index % 2]:
                st.markdown(
                    f"""
                    <div class="result-card compact" style="min-height:145px;">
                        <div style="color:#69d6e7;font-size:10px;letter-spacing:.12em;text-transform:uppercase;font-weight:700;">CASE {index + 1:02d}</div>
                        <div style="color:#eef3f7;font-size:16px;font-weight:650;margin-top:7px;">{_escape(scenario.title)}</div>
                        <div class="small-muted" style="margin-top:7px;line-height:1.45;">{_escape(scenario.why_it_matters)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Investigate this case",
                    key=f"scenario_{scenario.id}",
                    use_container_width=True,
                ):
                    st.session_state["pending_question"] = scenario.question
                    st.session_state["selected_scenario"] = scenario.id
                    st.session_state.pop("run_question", None)
                    st.session_state.pop("executed_question", None)
                    st.session_state.pop("result", None)
                    st.session_state.pop("pipeline_meta", None)
                    st.rerun()

        st.markdown(
            '<div style="height:12px"></div>'
            '<div class="section-label">Or ask your own question</div>',
            unsafe_allow_html=True,
        )
        return

    meta = st.session_state.get(
        "pipeline_meta",
        {"documents": 0, "chunks": 0, "vector_enabled": False},
    )

    st.markdown(
        '<div style="height:4px"></div>',
        unsafe_allow_html=True,
    )
    render_result(result, meta)


if __name__ == "__main__":
    main()
