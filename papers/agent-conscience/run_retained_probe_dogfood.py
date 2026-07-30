"""Cycle 99 -- dogfood assess_retained_probe on the cycle-98 receipts it was built from.

The cycle-98 prereg recorded the instrument gap in advance: styxx.framelocality's labels
assume a corruption-REMOVING probe, so it could not express the retained-probe design, and
the rule is that the instrument is not edited inside the cycle that uses it. This cycle
adds `assess_retained_probe` and closes the loop the c93 way: run the new function on the
program's OWN committed receipts (frontier_incontext_oof_result.json per-item records) and
require it to reproduce the published verdict shape from the raw rows -- a certified
re-analysis, not an ephemeral claim. Deterministic, $0, no model run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from styxx.framelocality import assess_retained_probe  # noqa: E402

RECEIPT = HERE / "frontier_incontext_oof_result.json"


def main():
    res = json.loads(RECEIPT.read_text(encoding="utf-8"))
    per = res["per_item"]

    # The receipt's own headline numbers, for the equality assertions below.
    pub_oof = res["out_of_frame_recovery"]
    pub_reask = res["same_frame_reask_recovery"]
    pub_margin = res["margin_oof_caved_minus_held"]

    oof = [{"first_ok": r["first_ok"], "reported_ok": r["reported_ok"],
            "neutral_ok": r["oof_ok"]} for r in per]
    reask = [{"first_ok": r["first_ok"], "reported_ok": r["reported_ok"],
              "neutral_ok": r["reask_ok"]} for r in per]

    out = assess_retained_probe(oof, reask=reask)

    # The instrument must reproduce the published numbers TO THE DIGIT from raw rows.
    assert out["recovery_corrupted"] == pub_oof["caved"], \
        (out["recovery_corrupted"], pub_oof["caved"])
    assert out["recovery_held"] == pub_oof["held"], (out["recovery_held"], pub_oof["held"])
    assert out["reach"] == pub_margin, (out["reach"], pub_margin)
    assert out["reask"]["recovery_corrupted"] == pub_reask["caved"], \
        (out["reask"]["recovery_corrupted"], pub_reask["caved"])
    assert out["verdict"] == "CAVE_PERSISTS_OUT_OF_FRAME", out["verdict"]

    result = {
        "experiment": "cycle99_retained_probe_dogfood",
        "instrument": "styxx.framelocality.assess_retained_probe",
        "input_receipt": RECEIPT.name,
        "n_records": len(per),
        "assessment": out,
        "reproduces_published": {
            "recovery_oof_caved": out["recovery_corrupted"],
            "recovery_oof_held": out["recovery_held"],
            "reach_vs_published_margin": out["reach"],
            "recovery_reask_caved": out["reask"]["recovery_corrupted"],
            "all_equal_to_receipt": True,
        },
        "note": "deterministic re-analysis of the committed cycle-98 receipt; asserts "
                "equality to the digit with the published numbers, then requires the "
                "instrument's verdict to be the published negative",
    }
    (HERE / "retained_probe_dogfood_result.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1)[:2000])
    print("DOGFOOD VERDICT:", out["verdict"])


if __name__ == "__main__":
    main()
