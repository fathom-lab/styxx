"""One float, one item, and the published verdict changes -- with nothing to catch it.

Run:  python verdict_flip.py

The calibration next door reports the miss list as rates.  This file spends the largest miss on
the lab's own published result, because a rate is easier to discount than a flipped verdict.

THE TARGET.  `first_verdict_2026_09_09` compares bf16 against fp16 on three channels and reports:

    exact  distance=0.062500000  floor=0.046875000  ratio=1.333  exceeds_floor
    seqlp  distance=0.058564664  floor=0.036070694  ratio=1.624  exceeds_floor
    topk   distance=1.407087824  floor=2.140233900  ratio=0.657  same

THE MOVE.  Change `seq_logprob` on ONE item of the bf16 canonical fingerprint.  Nothing else: not
a token id, not an output digest, not a distribution, not a floor number.

WHY IT IS PERMITTED.  The chosen item generated 14 tokens and carries 8 recorded distributions, so
the first-claim battery has no equality to apply -- only P8b, which says the score may not EXCEED
the recorded prefix sum.  The honest score sits 1.089 nats below that ceiling and the value being
written sits 3.378 nats below it, so the bound is satisfied with room to spare.

WHY NO CHALLENGE SEES IT.  A second party re-running that prompt reproduces the same tokens and
the same output digest.  The forged byte is a float the digest does not cover.

Every distance below is computed by `styxx.v8.distances`, the shipped Appendix B implementation,
not by a reimplementation here; the script first reproduces all three published floor and verdict
numbers from the certificates as a check that it is measuring the same thing the verdict did.
"""
from __future__ import annotations

import copy
import itertools
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))

import artifact as A                                            # noqa: E402
import forgeries as F                                           # noqa: E402
from styxx.v8 import distances as D                             # noqa: E402

PUBLISHED = {
    "exact": {"distance": 0.0625, "floor": 0.046875, "verdict": "exceeds_floor"},
    "seqlp": {"distance": 0.058564664, "floor": 0.036070694, "verdict": "exceeds_floor"},
    "topk": {"distance": 1.407087824, "floor": 2.1402339, "verdict": "same"},
}


def channels(a, b, roles):
    return {
        "exact": D.exact(a, b, roles)[0],
        "seqlp": D.seqlp(a, b),
        "topk": D.topk(a, b),
    }


def floors(runs, roles):
    out = {}
    for ch in ("exact", "seqlp", "topk"):
        ds = []
        for i, j in itertools.combinations(range(5), 2):
            ds.append(channels(runs[i], runs[j], roles)[ch])
        out[ch] = max(ds)
    return out


def verdict(dist, floor):
    """styxx/v8/floor.py: both sides are rounded to ROUND_PLACES (9) and `d <= f` reads `same`.

    The `<=` is why scenario C below only has to REACH the floor, not pass it, and the rounding is
    applied here so this script decides the way the shipped verdict decides.
    """
    return "exceeds_floor" if D.rounded(dist) > D.rounded(floor) else "same"


def main() -> int:
    roles = A.roles_of()
    certs = {i: A.load_cert(p) for i, p in A.BF16.items()}
    runs = {i: A.items_of(c) for i, c in certs.items()}
    fp16 = A.items_of(A.load_cert(A.FP16))
    check_item = A.load_battery_module().check_item

    print("--- reproducing the published verdict from the certificates")
    d0 = channels(runs[0], fp16, roles)
    f0 = floors(runs, roles)
    for ch in ("exact", "seqlp", "topk"):
        p = PUBLISHED[ch]
        ok_d = abs(d0[ch] - p["distance"]) < 5e-9
        ok_f = abs(f0[ch] - p["floor"]) < 5e-9
        print(f"  {ch:<6} distance {d0[ch]:.9f} (published {p['distance']}) {'ok' if ok_d else 'MISMATCH'}"
              f"   floor {f0[ch]:.9f} (published {p['floor']}) {'ok' if ok_f else 'MISMATCH'}"
              f"   -> {verdict(d0[ch], f0[ch])}")
        assert ok_d and ok_f, f"{ch}: this script is not measuring what the verdict measured"

    # --- choose the item: the largest honest gap that sits in the unrecorded-tail region
    gaps = sorted(((abs(runs[0][i]["seq_logprob"] - fp16[i]["seq_logprob"]), i) for i in runs[0]),
                  reverse=True)
    target = None
    for gap, iid in gaps:
        it = runs[0][iid]
        prefix, covered = F.coverage(it)
        if covered < it["n_generated"] and fp16[iid]["seq_logprob"] <= prefix:
            target = (iid, gap, prefix, covered)
            break
    iid, gap, prefix, covered = target
    it = runs[0][iid]
    print(f"\n--- the item")
    print(f"  item_id            {iid}")
    print(f"  output_text        {it['output_text']!r}")
    print(f"  n_generated        {it['n_generated']}   recorded distributions {covered}"
          f"   emitted tokens with none {it['n_generated'] - covered}")
    print(f"  recorded prefix sum {prefix:.9f}   <- P8b's ceiling")
    print(f"  honest seq_logprob  {it['seq_logprob']:.9f}   ({prefix - it['seq_logprob']:.6f} below it)")
    print(f"  fp16 seq_logprob    {fp16[iid]['seq_logprob']:.9f}   "
          f"({prefix - fp16[iid]['seq_logprob']:.6f} below it)")
    print(f"  the write: understate this item's score by {gap:.9f} nats, to the fp16 value")

    # --- two scenarios: edit the reference alone, or the whole batch-1 trio
    def apply(scen_runs, which):
        out = {i: copy.deepcopy(r) for i, r in scen_runs.items()}
        for i in which:
            out[i][iid]["seq_logprob"] = fp16[iid]["seq_logprob"]
        return out

    for label, which, why in (
        ("A: the reference run alone", (0,),
         "cheapest, but it breaks the log's own record that the three batch-1 runs agree exactly"),
        ("B: all three batch-1 runs", (0, 3, 4),
         "keeps run0 = run3 = run4 byte-identical, which entry 6 of the log already asserts"),
    ):
        g = apply(runs, which)
        dg = channels(g[0], fp16, roles)
        fg = floors(g, roles)
        print(f"\n--- scenario {label}")
        print(f"    ({why})")
        for ch in ("exact", "seqlp", "topk"):
            was = verdict(d0[ch], f0[ch])
            now = verdict(dg[ch], fg[ch])
            flag = "  <== FLIPPED" if was != now else ""
            print(f"  {ch:<6} distance {dg[ch]:.9f}  floor {fg[ch]:.9f}  "
                  f"ratio {(dg[ch] / fg[ch] if fg[ch] else float('nan')):.3f}  {was} -> {now}{flag}")
        overall_was = "exceeds_floor" if any(
            verdict(d0[c], f0[c]) == "exceeds_floor" for c in d0) else "same"
        overall_now = "exceeds_floor" if any(
            verdict(dg[c], fg[c]) == "exceeds_floor" for c in dg) else "same"
        print(f"  overall {overall_was} -> {overall_now}")

        # --- what catches it
        forged_cert = copy.deepcopy(certs[0])
        for item in forged_cert["body"]["items"]:
            if item["item_id"] == iid:
                item["seq_logprob"] = fp16[iid]["seq_logprob"]
        caught, fails = A.internal_verdict(forged_cert, check_item)
        dig = A.differing(A.items_of(forged_cert), runs[0], A.DIGEST_FIELDS)
        ful = A.differing(A.items_of(forged_cert), runs[0], A.FULL_FIELDS)
        print(f"  internal battery on the forged cert : "
              f"{'CAUGHT ' + str(fails) if caught else 'all predicates hold -- MISSED'}")
        print(f"  items a digest challenge would flag  : {len(dig)} of 64 -> "
              f"P(catch) = {A.catch_prob(len(dig), 64):.3f} even at k=64")
        print(f"  items a full-record challenge flags  : {len(ful)} of 64 -> "
              f"P(catch) = {A.catch_prob(len(ful), 1):.3f} at k=1, "
              f"{A.catch_prob(len(ful), 8):.3f} at k=8")

    # --- scenario C: take the overall verdict, which needs the exact channel too
    print("\n--- scenario C: the overall verdict, which the exact channel is still holding up")
    print("    Scenarios A and B flip seqlp and stop, because exact stays at 4 disagreeing items")
    print("    against a floor of 3.  Removing one disagreement means writing a token id, and a")
    print("    token id is exactly what a challenger reproduces.  This measures that price.")
    exact_diff = [i for i in runs[0]
                  if runs[0][i]["token_ids"] != fp16[i]["token_ids"]]
    print(f"    items where bf16 and fp16 disagree on tokens: {len(exact_diff)} of 64")
    g = {i: copy.deepcopy(r) for i, r in runs.items()}
    for i in (0, 3, 4):
        g[i][iid]["seq_logprob"] = fp16[iid]["seq_logprob"]
    victim = exact_diff[0]
    for i in (0, 3, 4):
        for k in F.BODY_FIELDS:
            g[i][victim][k] = copy.deepcopy(fp16[victim][k])
    dg = channels(g[0], fp16, roles)
    fg = floors(g, roles)
    any_exceed = False
    for ch in ("exact", "seqlp", "topk"):
        was = verdict(d0[ch], f0[ch])
        now = verdict(dg[ch], fg[ch])
        any_exceed = any_exceed or now == "exceeds_floor"
        flag = "  <== FLIPPED" if was != now else ""
        print(f"  {ch:<6} distance {dg[ch]:.9f}  floor {fg[ch]:.9f}  {was} -> {now}{flag}")
    print(f"  overall exceeds_floor -> {'exceeds_floor' if any_exceed else 'same'}")
    forged_cert = copy.deepcopy(certs[0])
    for item in forged_cert["body"]["items"]:
        if item["item_id"] == iid:
            item["seq_logprob"] = fp16[iid]["seq_logprob"]
        if item["item_id"] == victim:
            for k in F.BODY_FIELDS:
                item[k] = copy.deepcopy(fp16[victim][k])
    caught, fails = A.internal_verdict(forged_cert, check_item)
    dig = A.differing(A.items_of(forged_cert), runs[0], A.DIGEST_FIELDS)
    print(f"  internal battery : "
          f"{'CAUGHT ' + str(fails) if caught else 'all predicates hold -- MISSED'}")
    print(f"  items a digest challenge would flag: {len(dig)} of 64")
    for k in (1, 2, 4, 8, 16, 32, 64):
        print(f"      k={k:<3} P(catch) = {A.catch_prob(len(dig), k):.4f}")
    print("  THE PRICE: the seqlp half of this forgery is free at every k; the exact half is")
    print(f"  exposed on {len(dig)} item(s), which is what the challenge is for.")

    # --- and the whole-region version, for scale
    g = {i: copy.deepcopy(r) for i, r in runs.items()}
    n = 0
    for i, item in g[0].items():
        p, c = F.coverage(item)
        if c < item["n_generated"] and fp16[i]["seq_logprob"] <= p:
            item["seq_logprob"] = fp16[i]["seq_logprob"]
            n += 1
    print(f"\n--- for scale: the same move on all {n} eligible partial items")
    print(f"  seqlp distance {D.seqlp(g[0], fp16):.9f} against the untouched published floor "
          f"{f0['seqlp']:.9f}  ratio {D.seqlp(g[0], fp16) / f0['seqlp']:.3f}")
    print("  (the floor is set by the run1/run2 pair, which this move does not touch)")

    print("""
READING
  The seqlp channel of the published verdict is decided by a quantity that one party writes and
  that, on 27 of 64 items, no predicate over the certificate constrains from below and no output
  digest covers.  A k-item challenge is the right instrument for the classes that move tokens.
  It is the wrong instrument for this one, at every k, including k = 64.

  What reaches it is a challenger who publishes their own seq_logprob -- and the published log
  says that comparison is only meaningful at a fixed batch size, where three honest pairs agreed
  on all 64 items and all seven cross-batch pairs disagreed on all 64.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
