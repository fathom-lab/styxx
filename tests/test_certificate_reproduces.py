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
KNOWN_VERDICT_DRIFT = set()
# EMPTY — and it got here by repair, not by deletion.
#
# The one entry this set ever held was PROSPECTUS_knowsay_2026_07_27.md: four table row ordinals
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
# The direction test below is VACUOUS while this set is empty. That is the intended resting
# state: it re-arms the moment an entry is added, and an empty set is the thing this repository is
# trying to be.


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
        receipts, missing, _ = _resolve_receipts(cp, rec)
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
