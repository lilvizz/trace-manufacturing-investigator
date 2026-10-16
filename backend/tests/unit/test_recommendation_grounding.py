"""Recommendation grounding: regression tests for the C8 failure.

Live run, Press-07: the recommendation
  "Process Engineering should conduct a formal change control review and
   technical evaluation under QP-002 ..."
was marked Inconclusive at 0% with "none of the specific identifiers ...
(press7) appears in its cited passages". The verifier's grounding anchors
(app.investigation.claim_verifier._anchors) did not include quality-system
procedure / policy identifiers, so the claim's only checkable anchor was
"Press-07", which the cited procedural passages do not repeat.

What the verifier can and cannot establish (and what these tests pin):
- It checks identifiers lexically. For a recommendation that names a
  procedure (QP-n / CQ-POL-n), a cited passage must contain THAT
  identifier; the equipment name alone does not count.
- It cannot tell a passage that supports the proposed step from one that
  merely mentions the procedure. So the UI says "Procedure referenced ·
  not a finding", never that a basis was verified, and the two cases
  below (relevant vs unrelated QP-002 passage) get the same label.
- The status helper requires the verifier's recorded outcome AND matching
  citations that resolve to retrieved evidence. A non-empty ID list alone,
  or a missing evidence set, reads "Not verified".

Real ClaimVerifier and ConfidenceCalibrator throughout; real corpus
passages where named. Behaviour is checked, not source text.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.command_center import data as D
from app.command_center import export as X
from app.investigation.claim_verifier import ClaimVerifier
from app.investigation.confidence import ConfidenceCalibrator
from app.investigation.models import (
    ClaimType,
    EvidenceItem,
    EvidenceStance,
    Investigation,
    InvestigationReport,
    ReasoningClaim,
)
from app.investigation.timeline import InvestigationTimeline

C8 = (
    "Process Engineering should conduct a formal change control review and technical evaluation under "
    "QP-002 to establish a documented technical justification for Press-07's operating limits."
)
PROCEDURE_REFERENCED = "Procedure referenced · not a finding"
NOT_VERIFIED = "Not verified · suggestion only"

# Synthetic: names QP-002 for a reason unrelated to change control.
UNRELATED_QP002 = ("Canteen notice: the QP-002 document binder has moved to the second-floor shelf. "
                   "Lunch hours are unchanged.")


def _corpus():
    from app.config import CORPUS_PATH
    from app.ingestion.chunker import chunk_corpus
    from app.ingestion.loader import load_corpus
    return {c.chunk_id: c for c in chunk_corpus(load_corpus(str(CORPUS_PATH)))}


CORPUS = _corpus()


def _real(chunk_id):
    if chunk_id not in CORPUS:
        pytest.skip("corpus not available")
    c = CORPUS[chunk_id]
    return EvidenceItem(c.chunk_id, c.document_id, c.filename, c.relative_path, c.content, 0.5, ["q1"],
                        retrieval_channels=["vector"])


def _synthetic(chunk_id, content):
    return EvidenceItem(chunk_id, chunk_id.rsplit("_", 1)[0], f"{chunk_id}.txt", f"X/{chunk_id}.txt",
                        content, 0.5, ["q1"], retrieval_channels=["vector"])


def _pipeline(claims, evidence):
    """Verifier then calibrator, in the order Investigator.run applies them."""
    investigation = Investigation(question="q", evidence=list(evidence))
    report = ClaimVerifier().verify(InvestigationReport("q", list(claims)), investigation)
    return ConfidenceCalibrator().calibrate(report, investigation).claims, investigation


def _rec(ids, statement=C8, stance=EvidenceStance.SUPPORTS):
    return ReasoningClaim(statement, stance, list(ids), 0.9, ClaimType.RECOMMENDATION)


def _ids(evidence):
    return {e.chunk_id for e in evidence}


def _result(claims, investigation):
    return SimpleNamespace(
        investigation=investigation,
        report=InvestigationReport(question="Why is Press-07 limited to 42.3°C?", claims=list(claims),
                                   direct_answer="", direct_answer_evidence_chunk_ids=[]),
        timeline=InvestigationTimeline(events=[]),
        evidence_analysis=SimpleNamespace(findings=[]),
        plan=SimpleNamespace(subquestions=["q1"]),
    )


# 1. A relevant QP-002 passage (real corpus) ---------------------------------

@pytest.mark.parametrize("chunk_id", [
    "13_Process_Requalification_2014_2",            # "... full justification memo per QP-002 Section 7"
    "40_Plant_Wide_Temperature_Control_Policy_0",   # "... per site-level procedures (e.g., QP-002 ...)"
])
def test_relevant_qp002_passage_reads_procedure_referenced_not_basis_verified(chunk_id):
    evidence = [_real(chunk_id)]
    assert "QP-002" in evidence[0].content and "Press-07" not in evidence[0].content   # the C8 situation
    (claim,), _ = _pipeline([_rec([chunk_id])], evidence)

    assert claim.stance == EvidenceStance.SUPPORTS
    assert claim.confidence == pytest.approx(0.80)                  # RECOMMENDATION cap, unchanged
    assert claim.citation_check == "procedure_referenced"
    assert claim.citation_check_anchors == ["QP-002"]
    assert claim.citation_check_chunk_ids == [chunk_id]
    assert D.recommendation_check(claim, _ids(evidence)) == "procedure_referenced"
    assert D.claim_status(claim, _ids(evidence))[0] == PROCEDURE_REFERENCED
    why = D.claim_why(claim, _ids(evidence))
    assert "identifiers only" in why and "not verified that the passage supports" in why
    assert D.claim_group(claim) == "recommendation"                 # never "supported"


# 2. An unrelated passage that mentions QP-002 --------------------------------

def test_unrelated_qp002_mention_gets_the_same_honest_label_not_support():
    # The verifier is lexical: it cannot tell this apart from a relevant
    # QP-002 passage. The test pins that it does NOT claim more than that.
    evidence = [_synthetic("99_Canteen_Notice_0", UNRELATED_QP002)]
    (claim,), _ = _pipeline([_rec(["99_Canteen_Notice_0"])], evidence)
    assert claim.citation_check == "procedure_referenced"
    label = D.claim_status(claim, _ids(evidence))[0]
    assert label == PROCEDURE_REFERENCED
    assert "basis" not in label.lower() and "support" not in label.lower() and "verified" not in label.lower()
    assert D.claim_group(claim) == "recommendation"


# 3. Press-07-only meeting minutes (real corpus) ------------------------------

def test_press07_only_meeting_minutes_do_not_count_for_a_qp002_recommendation():
    evidence = [_real("34_Production_Meeting_Minutes_2018_0")]
    assert "Press-07" in evidence[0].content and "QP-002" not in evidence[0].content
    (claim,), _ = _pipeline([_rec(["34_Production_Meeting_Minutes_2018_0"])], evidence)
    assert claim.stance == EvidenceStance.UNKNOWN and claim.confidence == 0.0
    assert claim.citation_check == "failed"
    assert any("Grounding check" in n and "QP-002" in n for n in claim.verification_notes)
    assert D.claim_status(claim, _ids(evidence))[0] == NOT_VERIFIED


# 4. Missing and invalid citations; non-empty IDs alone -------------------------

def test_recommendation_without_citations_is_not_verified():
    evidence = [_real("13_Process_Requalification_2014_2")]
    (claim,), _ = _pipeline([_rec([])], evidence)
    assert claim.stance == EvidenceStance.UNKNOWN and claim.confidence == 0.0
    assert D.claim_status(claim, _ids(evidence))[0] == NOT_VERIFIED


def test_recommendation_citing_nonexistent_ids_is_not_verified():
    evidence = [_real("13_Process_Requalification_2014_2")]
    (claim,), _ = _pipeline([_rec(["99_Invented_Record_0", "QP-002"])], evidence)
    assert claim.evidence_chunk_ids == [] and claim.stance == EvidenceStance.UNKNOWN
    assert any("Removed 2 citation" in n for n in claim.verification_notes)
    assert D.claim_status(claim, _ids(evidence))[0] == NOT_VERIFIED


def test_supports_stance_with_ids_but_no_recorded_check_is_not_verified():
    # e.g. a claim built without going through ClaimVerifier
    raw = _rec(["13_Process_Requalification_2014_2"])
    assert D.recommendation_check(raw, {"13_Process_Requalification_2014_2"}) == ""
    assert D.claim_status(raw, {"13_Process_Requalification_2014_2"})[0] == NOT_VERIFIED


def test_check_result_requires_citations_that_resolve_to_retrieved_evidence():
    evidence = [_real("13_Process_Requalification_2014_2")]
    (claim,), _ = _pipeline([_rec(["13_Process_Requalification_2014_2"])], evidence)
    assert D.claim_status(claim, _ids(evidence))[0] == PROCEDURE_REFERENCED
    # Same claim shown against a result whose evidence lacks that passage
    assert D.claim_status(claim, {"07_Thermal_Validation_2001_0"})[0] == NOT_VERIFIED
    # No evidence set supplied: resolvability cannot be checked -> fail closed
    assert D.claim_status(claim)[0] == NOT_VERIFIED
    # Recorded match not among the claim's own citations
    tampered = replace(claim, evidence_chunk_ids=["07_Thermal_Validation_2001_0"])
    assert D.recommendation_check(tampered, _ids(evidence) | {"07_Thermal_Validation_2001_0"}) == ""


# 5. Supported factual claim, and no loosening for findings --------------------

def test_supported_factual_claim_still_passes():
    evidence = [_real("07_Thermal_Validation_2001_2")]
    fact = ReasoningClaim("The 2001 thermal validation derived the 42.3°C limit for the PX-200.",
                          EvidenceStance.SUPPORTS, [evidence[0].chunk_id], 0.95, ClaimType.ORIGINAL_BASIS)
    (claim,), _ = _pipeline([fact], evidence)
    assert claim.stance == EvidenceStance.SUPPORTS and claim.confidence == pytest.approx(0.95)
    assert claim.citation_check == "identifier_overlap"
    assert D.claim_group(claim) == "supported"
    assert D.claim_status(claim, _ids(evidence))[0] == "Supported"


def test_factual_claim_naming_a_procedure_is_checked_exactly_as_before():
    # Procedure identifiers count for recommendations only.
    evidence = [_real("13_Process_Requalification_2014_2")]
    fact = ReasoningClaim("Press-07's limit was reviewed under QP-002.", EvidenceStance.SUPPORTS,
                          [evidence[0].chunk_id], 0.9, ClaimType.ADMINISTRATIVE_ACTION)
    (claim,), _ = _pipeline([fact], evidence)
    assert claim.stance == EvidenceStance.UNKNOWN and claim.confidence == 0.0


def test_evidence_gap_still_reads_not_established_not_false():
    evidence = [_real("13_Process_Requalification_2014_2")]
    gap = ReasoningClaim("No record re-justifies the 42.3°C limit for the PX-400.", EvidenceStance.NOT_ESTABLISHED,
                         [evidence[0].chunk_id], 0.9, ClaimType.EVIDENCE_GAP)
    (claim,), _ = _pipeline([gap], evidence)
    assert claim.stance == EvidenceStance.NOT_ESTABLISHED and claim.confidence == pytest.approx(0.30)
    assert D.claim_status(claim, _ids(evidence))[0] == "Not established"
    assert "not a finding that the statement is false" in D.claim_why(claim, _ids(evidence))


def test_recommendation_without_a_named_procedure_keeps_the_general_rule():
    text = "Commission a PX-400 thermal characterization before any limit change."
    unrelated = _synthetic("98_Shift_Roster_0", "Shift roster for week 32. No equipment changes noted.")
    evidence = [_real("12_PX400_Qualification_2011_0"), unrelated]
    (claim,), _ = _pipeline([_rec(["12_PX400_Qualification_2011_0"], statement=text)], evidence)
    assert claim.citation_check == "identifier_overlap"
    assert D.claim_status(claim, _ids(evidence))[0] == "Related records cited · not a finding"
    (claim,), _ = _pipeline([_rec(["98_Shift_Roster_0"], statement=text)], evidence)
    assert claim.stance == EvidenceStance.UNKNOWN and claim.citation_check == "failed"
    assert D.claim_status(claim, _ids(evidence))[0] == NOT_VERIFIED


# 6. Same label and confidence on every page ------------------------------------

@pytest.mark.parametrize("chunk_id,label,confidence,origin", [
    ("13_Process_Requalification_2014_2", PROCEDURE_REFERENCED, 0.80, "procedure referenced in a cited passage"),
    ("34_Production_Meeting_Minutes_2018_0", NOT_VERIFIED, 0.0, "not verified"),
])
def test_recommendation_label_and_confidence_agree_across_pages(chunk_id, label, confidence, origin):
    evidence = [_real("13_Process_Requalification_2014_2"), _real("34_Production_Meeting_Minutes_2018_0")]
    claims, investigation = _pipeline([_rec([chunk_id])], evidence)
    result = _result(claims, investigation)

    # Reasoning & Claims and Copilot render these rows
    (row,) = D.grouped_claims(result)["recommendation"]
    assert (row.status_label, row.confidence) == (label, pytest.approx(confidence))
    assert not D.grouped_claims(result)["supported"]
    assert D.status_badge(row.claim, D.evidence_ids(result)).endswith(f"{label}</span>")

    # Gaps & Actions and the Overview "Next human action" card read the register
    (item,) = [i for i in D.gap_register(result) if i.kind == "recommendation"]
    assert item.verified is False                                    # never a verified finding
    assert origin in item.origin
    from app.command_center.views.overview import _story_html
    card = _story_html(result)
    assert "Next human action" in card and "not a verified finding" in card and origin in card

    # Reports & History: Markdown and JSON exports
    assert f"**{label}** · Recommendation · {confidence:.0%}" in X.result_to_markdown(result)
    exported = json.loads(json.dumps(X.result_to_dict(result), default=str))["claims"][0]
    assert exported["status_label"] == label
    assert exported["calibrated_confidence"] == pytest.approx(confidence)
