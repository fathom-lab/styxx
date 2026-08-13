"""Does context-resolution pick the RIGHT path, or merely a different one?

Ambiguity disclosure was the previous commit. Resolution is this one — and resolution is a
stronger claim, so it needs a stronger test. "14 claims resolved by context" is worthless if the
resolutions are wrong; it would be the same defect as the headline grounding rate, one level up.

So: build documents where the intended path is KNOWN by construction, and measure accuracy
against a dict-order baseline. A resolver that cannot beat dict order has bought nothing.

PRE-STATED, before running:
  - dict-order baseline accuracy on these items will be near 1/n_candidates (chance)
  - context resolution must beat that baseline substantially, or the feature does not ship
  - I expect SOME wrong resolutions; a resolver claiming 100% would mean the fixture is too easy
"""
from __future__ import annotations
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from styxx.claim_audit import audit_grounding, _load_sources, _candidates  # noqa: E402

# A receipt with deliberately colliding values across semantically distinct cells.
RECEIPT = {
    "null": {"rho_low": {"mean_licensed": 0.250, "cp_upper": 0.410},
             "rho_high": {"mean_licensed": 0.250, "cp_upper": 0.620}},
    "knee": {"rho_low": {"mean_licensed": 0.830, "cp_upper": 0.250},
             "rho_high": {"mean_licensed": 0.410, "cp_upper": 0.910}},
    "blockconf": {"rho_low": {"mean_licensed": 0.620, "cp_upper": 0.830},
                  "rho_high": {"mean_licensed": 0.910, "cp_upper": 0.250}},
}

# EASY fixture: sentences reuse the receipt's exact key names. This is the friendly case and a
# 100% score here means the fixture is too easy, exactly as pre-stated. Kept as the floor.
# (sentence, the path the sentence is ABOUT)
CASES = [
    ("the null cohort at rho_low had mean_licensed of 0.250", "null.rho_low.mean_licensed"),
    ("the null cohort at rho_high had mean_licensed of 0.250", "null.rho_high.mean_licensed"),
    ("at the knee, rho_low, cp_upper reached 0.250", "knee.rho_low.cp_upper"),
    ("under blockconf at rho_high the cp_upper was 0.250", "blockconf.rho_high.cp_upper"),
    ("knee rho_low mean_licensed came out at 0.830", "knee.rho_low.mean_licensed"),
    ("blockconf rho_low cp_upper was 0.830", "blockconf.rho_low.cp_upper"),
    ("the null rho_high cp_upper was 0.620", "null.rho_high.cp_upper"),
    ("blockconf rho_low mean_licensed was 0.620", "blockconf.rho_low.mean_licensed"),
    ("knee rho_high mean_licensed was 0.410", "knee.rho_high.mean_licensed"),
    ("null rho_low cp_upper was 0.410", "null.rho_low.cp_upper"),
    ("blockconf rho_high mean_licensed hit 0.910", "blockconf.rho_high.mean_licensed"),
    ("knee rho_high cp_upper hit 0.910", "knee.rho_high.cp_upper"),
]


# HARD fixture: prose a human would actually write. Key names are paraphrased or absent, so the
# resolver must work from partial overlap ("licensing rate" vs mean_licensed) or fail honestly.
# This is where the real number lives.
HARD_CASES = [
    ("under the null at low autocorrelation the licensing rate was 0.250",
     "null.rho_low.mean_licensed"),
    ("the confound cohort's upper bound at high rho came to 0.250",
     "blockconf.rho_high.cp_upper"),
    ("near the knee, licensing at low rho reached 0.830", "knee.rho_low.mean_licensed"),
    ("the block-confound arm's bound at low rho was 0.830", "blockconf.rho_low.cp_upper"),
    ("licensing under the confound arm at low rho sat at 0.620",
     "blockconf.rho_low.mean_licensed"),
    ("the null's bound at high rho was 0.620", "null.rho_high.cp_upper"),
    ("we saw 0.910 licensing in the confound arm at high rho",
     "blockconf.rho_high.mean_licensed"),
    ("the knee's bound at high rho was 0.910", "knee.rho_high.cp_upper"),
]


def run(cases, label):
    vals = _load_sources([RECEIPT])
    n_ctx_right = n_base_right = n_amb = 0
    rows = []
    for sentence, truth in cases:
        rep = audit_grounding(sentence, RECEIPT)
        items = [i for i in rep.items if i.status == "grounded"]
        if not items:
            rows.append((sentence, truth, "NO-MATCH", "", 0))
            continue
        it = items[0]
        cands = _candidates(it, vals)
        if len(cands) > 1:
            n_amb += 1
        ok = it.source == truth
        n_ctx_right += ok
        # dict-order baseline: what first-match would have returned
        base_ok = cands[0] == truth
        n_base_right += base_ok
        rows.append((sentence, truth, it.source, it.resolved_by, len(cands), ok, base_ok))

    n = len(cases)
    print("=" * 78)
    print(f"CONTEXT RESOLUTION vs DICT-ORDER BASELINE — {label}")
    print("=" * 78)
    for r in rows:
        if len(r) == 5:
            print(f"  [no match] {r[0]}")
            continue
        s, truth, got, how, k, ok, base_ok = r
        mark = "OK " if ok else "XX "
        print(f"  {mark} ({k} cands, {how:>9}) {s}")
        if not ok:
            print(f"        wanted {truth}")
            print(f"        got    {got}")
    print(f"\n  ambiguous items          : {n_amb}/{n}")
    print(f"  dict-order baseline right: {n_base_right}/{n} = {n_base_right/n:.3f}")
    print(f"  context resolution right : {n_ctx_right}/{n} = {n_ctx_right/n:.3f}")
    lift = (n_ctx_right - n_base_right) / n
    print(f"  LIFT OVER BASELINE       : {lift:+.3f}")
    verdict = ("RESOLVER_EARNS_ITS_KEEP" if n_ctx_right > n_base_right
               else "RESOLVER_IS_DECORATION__no_better_than_dict_order")
    print(f"\n  VERDICT: {verdict}")
    return {"n_cases": n, "n_ambiguous": n_amb,
            "dict_order_accuracy": round(n_base_right / n, 4),
            "context_accuracy": round(n_ctx_right / n, 4),
            "lift": round(lift, 4), "verdict": verdict}


def main():
    easy = run(CASES, "EASY (sentences reuse exact key names)")
    print()
    hard = run(HARD_CASES, "HARD (paraphrased prose — the honest test)")
    out = {"easy": easy, "hard": hard,
           "note": "The EASY fixture reuses the receipt's own key names and scores 1.000; that "
                   "was pre-stated as a sign the fixture is too easy, and it is kept only as a "
                   "floor. The HARD fixture is the number to quote."}
    (HERE / "context_resolution_validation.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\nwrote context_resolution_validation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
