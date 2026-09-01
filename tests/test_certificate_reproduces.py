"""A committed certificate must still hold at the verifier now in the tree.

A certificate is an artifact of three things — a document, a receipt set, and a verifier — and it
records all three so that drift is *detectable*. Nothing checked the third. The verifier moves every
cycle, and a certificate committed at an older one goes on asserting its old verdict.

`PROSPECTUS_knowsay_2026_07_27.certificate.json` asserts OATH-HELD with zero UNGROUNDED tokens. The
same document against the same receipts, re-run at HEAD, is OATH-FAILED with four — four markdown
table row ordinals that later recall work began obligating. The failure is real, and it was
invisible for as long as nobody re-ran the certificate.

A document that carries a passing certificate it no longer passes is precisely the defect class this
repository exists to document: an absent measurement surfaced as a result.

Only VERDICT drift is gated. Counts drift is expected and benign — a recall improvement moves tokens
between ABSTAIN and VERIFIED without changing whether the oath holds — and gating it would demand
regenerating 139 certificates on every verifier change, which converts a real signal into noise.
`papers/closed-model-frontier/oath_certificate_drift_census.py` reports the full surface.
"""
import json
from pathlib import Path

import pytest

from styxx.certify import certify_doc
from styxx.corpus_audit import _resolve_receipts

ROOT = Path(__file__).resolve().parents[1]

# Documents whose committed certificate no longer reproduces at HEAD.
#
# This set is asserted EXACTLY, in both directions. A new drift fails the test, and a drift that has
# been repaired ALSO fails it until the entry is removed — so the list can only shrink, and it can
# never become a quiet place to park a failing document.
KNOWN_VERDICT_DRIFT = {
    # Committed OATH-HELD (34/27/0) in 2026-06; OATH-FAILED (34/27/1) at HEAD with its full
    # receipt set resolved. Newly VISIBLE rather than newly drifted: this document cites two
    # receipts from `papers/grounded-honesty-axis/`, and until the resolver above was given a
    # search_root it was one of 36 certificates this guard skipped entirely.
    #
    # The accusation is on line 13, and it looks correct. The line is TRUNCATED in the source —
    # it ends mid-sentence at "n=48 -> 43 scored (27 HELD, 16 CAVED, 4" — so the dangling `4`
    # has lost the vocabulary that would bind it. Its value is in the receipts
    # (behavioral_sycophancy_result.json:n_nogate = 4) but nothing on the surviving text names
    # that quantity, while `27` and `16` still bind through HELD/CAVED. The verifier is
    # reporting a real defect in a published FINDING.
    #
    # Parked, not hidden: repairing a published document and re-certifying it is its own cycle
    # with its own prereg, and this entry exists so that cycle cannot be forgotten. It leaves
    # this set only by repair — the list can still only shrink.
    "papers/closed-model-frontier/FINDING_behavioral_sycophancy_blackbox_2026_06_09.md",
}
# The v0.11 entry is gone from this set, and it left by repair, not by deletion.
#
# That entry was PROSPECTUS_knowsay_2026_07_27.md: four table row ordinals
# ("| 3 |", "| 4 |", "| 5 |", "| 8 |") extracted as numeric claims and obligated by trigger
# vocabulary in their own row text. Under value-only matching they false-verified against
# `per_item[3].i`, a leaf equal to its own subscript; under count-binding they were accused. Both
# regimes were wrong, because a row ordinal has no truth condition at all —
# `V11_ORDINAL_LABEL` (PREREG_oath_v11_row_ordinal_retraction_2026_08_25) demotes it to ABSTAIN
# with the reason `row_ordinal_label`. The document's live verdict returned to OATH-HELD, the
# `repaired` assertion below went red, and the entry was deleted IN THE SHIP COMMIT — the only way
# an entry is allowed to leave this set. The committed certificate was never hand-edited; it
# reproduces.
#
# For a few hours on 2026-08-26 this set was empty and the direction test below was vacuous.
# Widening the resolver ended that: an empty set turned out to mean the guard could not see 36 of
# 178 certificates, not that the corpus had none left to find. An empty set is still what this
# repository is trying to be — but only once it is empty over the whole corpus.


def _resolvable():
    for cp in sorted(ROOT.glob("papers/**/*.certificate.json")):
        if "anc" in cp.parts:
            continue
        doc = cp.with_name(cp.name.replace(".certificate.json", ".md"))
        if not doc.exists():
            continue
        try:
            rec = json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            continue
        # search_root is REQUIRED, not optional. Without it `_resolve_receipts` looks only next
        # to the certificate, so any document citing a receipt from another folder resolves as
        # `missing` and is skipped by the guard below. That silently excluded 36 of 178
        # certificates — 20% of the corpus, and disproportionately the cross-arc syntheses most
        # likely to drift. The resolver's own docstring already records this defect being fixed
        # once, for `corpus_audit`; this caller never got the fix.
        receipts, missing, _ = _resolve_receipts(cp, rec, ROOT / "papers")
        if receipts and not missing:
            yield doc, receipts, rec


def test_no_committed_certificate_silently_stops_holding():
    drifted = set()
    examined = 0
    for doc, receipts, rec in _resolvable():
        try:
            live = certify_doc(doc, receipts)
        except Exception:
            continue
        examined += 1
        if rec.get("verdict") != live["verdict"]:
            drifted.add(doc.relative_to(ROOT).as_posix())

    if examined == 0:
        pytest.skip("no document with fully-resolvable receipts — nothing to reproduce")

    newly_drifted = drifted - KNOWN_VERDICT_DRIFT
    assert not newly_drifted, (
        f"committed certificate(s) no longer reproduce at this verifier: {sorted(newly_drifted)}. "
        "Either the verifier change is wrong, or the document needs repair and re-certification — "
        "do not add it to KNOWN_VERDICT_DRIFT without a reason recorded there."
    )

    repaired = KNOWN_VERDICT_DRIFT - drifted
    assert not repaired, (
        f"these documents now reproduce and must be removed from KNOWN_VERDICT_DRIFT: "
        f"{sorted(repaired)}. The list is asserted exactly so it can only shrink."
    )


def test_the_known_drift_is_a_held_certificate_that_now_fails():
    """Pin the DIRECTION of the known drift: a passing certificate that no longer passes.

    Drift toward OATH-FAILED is the dangerous direction — the document advertises a verdict it does
    not earn. If this ever inverts (the certificate says FAILED and the live run says HELD) that is
    a different situation and should be looked at rather than silently tolerated.
    """
    by_doc = {d.relative_to(ROOT).as_posix(): (d, r, rec) for d, r, rec in _resolvable()}
    for rel in sorted(KNOWN_VERDICT_DRIFT):
        if rel not in by_doc:
            pytest.skip(f"{rel} is not resolvable in this checkout")
        doc, receipts, rec = by_doc[rel]
        live = certify_doc(doc, receipts)
        assert rec.get("verdict") == "OATH-HELD", f"{rel}: expected a HELD certificate"
        assert live["verdict"] == "OATH-FAILED", f"{rel}: expected the live run to FAIL"
        assert live["counts"]["UNGROUNDED"] > 0
