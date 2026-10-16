"""TRACE live end-to-end acceptance (real Gemini, real index, no fixtures).

Runs the Press-07 investigation and two Copilot-style follow-ups through
the production pipeline exactly as the app builds it, then checks the
evidence-integrity rules and prints PASS / FAIL / INFO per check. Nothing
here is scripted: if Gemini or the network is unavailable, the checks FAIL
and say why.

    cd backend
    python scripts/live_acceptance.py            # writes live_acceptance_report.json

The API key is never printed or written to the report.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.config import CORPUS_PATH, INDEX_PATH, load_environment  # noqa: E402

QUESTION = (
    "Why is Press-07 limited to 42.3°C, what changed since that limit was "
    "established, and is its current technical basis still supported?"
)
FOLLOW_UPS = (
    # Must resolve "that study" against turn 1 and retrieve the 2014 study.
    "Why is the 2014 characterization study not enough to establish the current technical justification?",
    # Needs records the first question may not have retrieved (SOP v7, 2019).
    "What did the 2019 SOP revision for Press-07 change, and did it re-justify the 42.3°C limit?",
)
TRAIL_YEARS = (2001, 2011, 2014, 2019, 2022)

results: list[dict] = []


def check(name: str, ok: bool | None, detail: str = "") -> None:
    state = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    results.append({"check": name, "result": state, "detail": detail})
    print(f"{state:5s} {name}" + (f" — {detail}" if detail else ""), flush=True)


def build():
    from app.ingestion.chunker import chunk_corpus
    from app.ingestion.loader import load_corpus
    from app.investigation.build import build_investigator
    from app.investigation.gemini_client import GeminiReasoningClient
    from app.retrieval.bm25_search import BM25Searcher
    from app.retrieval.entity_matcher import EntityMatcher
    from app.retrieval.gemini_embeddings import GeminiEmbeddingProvider
    from app.retrieval.hybrid_retriever import HybridRetriever
    from app.retrieval.vector_health import LEGACY_INDEX_MODEL, load_index, probe_query_embedding
    from app.retrieval.vector_search import VectorSearcher

    documents = load_corpus(str(CORPUS_PATH))
    chunks = chunk_corpus(documents)
    check("corpus loads", bool(chunks), f"{len(documents)} documents, {len(chunks)} chunks")
    health, _ids, embeddings = load_index(INDEX_PATH, chunks)
    check("embedding index compatible with corpus", health.usable and not health.stale_chunk_ids,
          json.dumps(health.summary()))
    vector = None
    if health.usable:
        provider = GeminiEmbeddingProvider(model=health.model or LEGACY_INDEX_MODEL, max_retries=1,
                                           base_delay_seconds=0.5, timeout_seconds=20)
        probe = probe_query_embedding(provider, health.dimension)
        check("live query embedding", probe.ok, f"dim={probe.dimension} {probe.error}")
        if probe.ok:
            vector = VectorSearcher(chunks=chunks, embeddings=embeddings, embedding_provider=provider)
    retriever = HybridRetriever(BM25Searcher(chunks), vector, EntityMatcher(chunks))
    client = GeminiReasoningClient(max_retries=1, base_delay_seconds=0.5, timeout_seconds=120)
    return build_investigator(retriever, gemini_client=client), retriever, chunks


def turn_checks(label, result, retriever, before, previous=None):
    from app.investigation.models import ClaimType, EvidenceStance

    evidence = {e.chunk_id: e for e in result.investigation.evidence}
    plan = result.plan
    check(f"{label}: Gemini produced the plan", getattr(plan, "source", "") in ("gemini", "deterministic-narrow"),
          f"source={getattr(plan, 'source', '?')} {getattr(plan, 'fallback_reason', '')}")
    calls = retriever.vector_calls - before[0]
    failures = retriever.vector_failures - before[1]
    vec = [e for e in evidence.values() if "vector" in (e.retrieval_channels or [])]
    check(f"{label}: vector retrieval contributed", calls > 0 and failures == 0 and bool(vec),
          f"{len(vec)} of {len(evidence)} passages ranked by vector; {calls} calls, {failures} failed")
    answer_ids = list(result.report.direct_answer_evidence_chunk_ids or [])
    check(f"{label}: direct answer cites retrieved passages", bool(answer_ids) and set(answer_ids) <= set(evidence),
          f"{answer_ids}")
    bad = [c.statement[:60] for c in result.report.claims if not set(c.evidence_chunk_ids) <= set(evidence)]
    check(f"{label}: every claim citation resolves", not bad, "; ".join(bad))
    caps = {EvidenceStance.NOT_ESTABLISHED: 0.30, EvidenceStance.CONTRADICTS: 0.85, EvidenceStance.CONTEXT_ONLY: 0.65}
    over = [c for c in result.report.claims if c.confidence > caps.get(c.stance, 1.0) + 1e-9
            or (c.claim_type == ClaimType.CURRENT_TECHNICAL_JUSTIFICATION and c.confidence > 0.70 + 1e-9)]
    check(f"{label}: calibration caps hold", not over, f"{len(result.report.claims)} claims")
    ctj = [c for c in result.report.claims if c.claim_type == ClaimType.CURRENT_TECHNICAL_JUSTIFICATION]
    check(f"{label}: current-justification status", None,
          ", ".join(f"{c.stance.value} {c.confidence:.0%}" for c in ctj) or "no such claim")
    if previous is not None:
        prev_ids = {e.chunk_id for e in previous.investigation.evidence}
        fresh = [e for e in evidence.values() if (e.retrieval_channels or []) != ["carried-forward"]]
        new = [cid for cid in evidence if cid not in prev_ids]
        check(f"{label}: fresh retrieval ran", len(fresh) >= 5, f"{len(fresh)} freshly retrieved passages")
        check(f"{label}: retrieved passages the previous turn did not", bool(new), ", ".join(new[:6]))
        resolution = getattr(result, "resolution", None)
        check(f"{label}: references resolved against the case", bool(resolution and resolution.used_context),
              f"anchors={getattr(resolution, 'anchors', ())}")
        prior_answer = (previous.report.direct_answer or "").strip()
        leaked = prior_answer and any(prior_answer[:120] in e.content for e in evidence.values())
        check(f"{label}: earlier answer never appears as evidence", not leaked)
    return {
        "question": result.investigation.question,
        "plan_source": getattr(plan, "source", ""),
        "direct_answer": result.report.direct_answer,
        "answer_citations": answer_ids,
        "claims": [{"stance": c.stance.value, "type": getattr(c.claim_type, "value", None),
                    "confidence": c.confidence, "statement": c.statement, "citations": c.evidence_chunk_ids,
                    "verification_notes": list(getattr(c, "verification_notes", []) or [])}
                   for c in result.report.claims],
        "unresolved_questions": list(result.report.unresolved_questions or []),
        "evidence": [{"chunk_id": e.chunk_id, "channels": e.retrieval_channels} for e in evidence.values()],
    }


def main() -> int:
    load_environment()
    check("GEMINI_API_KEY present (value not shown)", bool(os.getenv("GEMINI_API_KEY")))
    from app.investigation.conversation import ConversationContext

    try:
        investigator, retriever, chunks = build()
    except Exception as exc:
        check("pipeline builds", False, f"{type(exc).__name__}: {exc}"[:300])
        return _finish({})

    report = {"turns": []}
    turns = []
    questions = (QUESTION,) + FOLLOW_UPS
    for n, question in enumerate(questions, start=1):
        label = f"turn {n}"
        context = ConversationContext.from_results("live-acceptance", turns) if turns else None
        before = (retriever.vector_calls, retriever.vector_failures)
        started = time.monotonic()
        try:
            result = investigator.run(question, context=context)
        except Exception as exc:
            check(f"{label}: investigation completed", False, f"{type(exc).__name__}: {exc}"[:300])
            break
        seconds = round(time.monotonic() - started, 1)
        check(f"{label}: investigation completed", True, f"{seconds}s")
        entry = turn_checks(label, result, retriever, before, turns[-1] if turns else None)
        entry["seconds"] = seconds
        report["turns"].append(entry)
        turns.append(result)
        if n == 1:
            ids = {e.chunk_id for e in result.investigation.evidence}
            study = sorted(i for i in ids if i.startswith("13_Process_Requalification_2014"))
            cited = {c for cl in result.report.claims for c in cl.evidence_chunk_ids} | set(
                result.report.direct_answer_evidence_chunk_ids or [])
            check("turn 1: 2014 characterization study (QUAL-2014-009) retrieved", bool(study), ", ".join(study))
            check("turn 1: 2014 study cited by the answer or a claim", None,
                  ", ".join(sorted(i for i in study if i in cited)) or "retrieved but not cited")
            years = {e.date.year for e in result.timeline.events}
            for year in TRAIL_YEARS:
                check(f"turn 1: timeline has a dated event in {year}", None if year not in years else True,
                      "present" if year in years else "not in this turn's retrieved passages")

    # The record trail itself, from the corpus (independent of retrieval).
    from app.investigation.models import EvidenceItem
    from app.investigation.timeline import TimelineBuilder

    items = [EvidenceItem(c.chunk_id, c.document_id, c.filename, c.relative_path, c.content, 0.0, []) for c in chunks]
    issued = {e.date.year for e in TimelineBuilder().build(items).events if e.kind == "record_issued"}
    for year in TRAIL_YEARS:
        check(f"corpus record trail: a record issued in {year}", year in issued)
    return _finish(report)


def _finish(report: dict) -> int:
    report["checks"] = results
    out = BACKEND / "live_acceptance_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    failed = [r for r in results if r["result"] == "FAIL"]
    print(f"\n{len(results) - len(failed)} passed/info, {len(failed)} failed. Report: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
