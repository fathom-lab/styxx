"""Measure what a k-item challenge catches, and what it does not.

Run:  python calibrate.py            (writes results.json beside this file and prints the tables)

For every mutation class in `forgeries.py`, at every f in its sweep, this builds N independent
forgeries of the published fingerprint certificate and reports, per forgery:

  1  the internal-consistency battery's verdict -- the published
     `class_two_empty_2026_09_09/first_claim_battery.py`, imported and called, costing nothing
  2  the exact detection probability of a k-item output-digest challenge
  3  the exact detection probability of a k-item full-record challenge (the proposed extension)

A forgery is a RESIDUAL MISS when the battery does not catch it AND its output digests are
identical to the honest ones -- in which case no k, including k=64, can catch it either.  The
per-cell residual-miss rate is the deliverable, and the miss list is every cell where it is above
zero.  A calibration with an empty miss list has failed to generate hard enough mutations.

NO-OPS.  Some mutations are vacuous on some draws: copying item A's result onto item B when the
two honestly produced the same answer changes no byte.  A forgery that changes nothing is not an
undetected forgery, so those are counted separately and excluded from the rates.  Three of the ten
published output texts are short enough to recur; ten groups of the 64 items share an output
digest, which is where the no-ops come from.

FALSE ALARM.  A detection number without its false-alarm rate is not a number.  This file measures
the challenge's false-alarm rate from the artifact itself: the five published bf16 floor runs are
five honest executions of the same battery, and each of the ten pairs stands in for an honest
challenger's re-run.  The result decides the protocol.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import artifact as A                                            # noqa: E402
import forgeries as F                                           # noqa: E402

KS = [1, 2, 4, 8, 16, 32, 64]
N_FORGERIES = 200
SEED = 20260909
OUT = pathlib.Path(__file__).resolve().parent / "results.json"


def measure_cell(name, fn, f, honest_cert, honest, check_item, rng):
    dig, ful, internal, residual = [], [], [], 0
    predicates: dict = {}
    noops = 0
    for _ in range(N_FORGERIES):
        forged = fn(honest_cert, f, rng)
        if forged["body"]["items"] == honest_cert["body"]["items"]:
            noops += 1
            continue
        fi = A.items_of(forged)
        d = len(A.differing(fi, honest, A.DIGEST_FIELDS))
        u = len(A.differing(fi, honest, A.FULL_FIELDS))
        caught, fl = A.internal_verdict(forged, check_item)
        for k, v in fl.items():
            predicates[k] = predicates.get(k, 0) + v
        dig.append(d)
        ful.append(u)
        internal.append(caught)
        if not caught and d == 0:
            residual += 1
    n = len(dig)
    return {
        "class": name,
        "f": f,
        "n_effective": n,
        "n_noop": noops,
        "internal_caught_rate": (sum(internal) / n) if n else None,
        "internal_predicates": predicates,
        "digest_changed_mean": (sum(dig) / n) if n else None,
        "digest_changed_min": min(dig) if n else None,
        "digest_changed_max": max(dig) if n else None,
        "full_changed_mean": (sum(ful) / n) if n else None,
        "residual_miss_rate": (residual / n) if n else None,
        "challenge_digest": {str(k): (sum(A.catch_prob(b, k) for b in dig) / n) if n else 0.0
                             for k in KS},
        "challenge_full": {str(k): (sum(A.catch_prob(b, k) for b in ful) / n) if n else 0.0
                           for k in KS},
    }


def main() -> int:
    honest_cert = A.load_cert(A.BF16[0])
    honest = A.items_of(honest_cert)
    check_item = A.load_battery_module().check_item

    caught, fails = A.internal_verdict(honest_cert, check_item)
    print(f"target: {A.BF16[0].name}")
    print(f"  {len(honest)} items; internal battery on the HONEST certificate: "
          f"{'FAILS ' + str(fails) if caught else 'all predicates hold'}")
    partial = [i for i, it in honest.items() if F.is_partial(it)]
    free = sum(it["n_generated"] - F.coverage(it)[1] for it in honest.values())
    print(f"  {len(honest) - len(partial)} items fully tied by P8, {len(partial)} partial, "
          f"{free} emitted tokens with no recorded distribution\n")

    print("--- closed form vs 20000 draws (b = marked items, k = challenge size)")
    for b, k in ((1, 8), (3, 16), (8, 4)):
        print(f"  b={b:<3} k={k:<3} exact {A.catch_prob(b, k):.6f}   "
              f"monte-carlo {A.montecarlo_catch(b, k):.6f}")

    fa = A.same_batch_reproducibility()["pairs"]
    same_b = [p for p in fa if p["batch_size"][0] == p["batch_size"][1]]
    diff_b = [p for p in fa if p["batch_size"][0] != p["batch_size"][1]]
    worst = max(p["digest_disagreements"] for p in diff_b)
    print("\n--- FALSE ALARM, from the five published bf16 floor runs (10 honest pairs)")
    print(f"  same batch size ({len(same_b)} pairs): digest disagreements "
          f"{[p['digest_disagreements'] for p in same_b]}, full-record "
          f"{[p['full_record_disagreements'] for p in same_b]}")
    print(f"  different batch size ({len(diff_b)} pairs): digest disagreements "
          f"{[p['digest_disagreements'] for p in diff_b]}, full-record "
          f"{[p['full_record_disagreements'] for p in diff_b]}")
    print(f"  => at the SAME batch size the false-alarm rate is 0 at every k on both comparisons,")
    print(f"     over {len(same_b) * len(honest)} item comparisons.")
    print(f"     OFF-batch a digest challenge false-alarms with probability "
          f"{A.catch_prob(worst, 16):.4f} at k=16, and the full-record challenge with "
          f"probability 1.000 at k=1.")

    rows = []
    for name, fn, sweep, blurb in F.CLASSES:
        for f in sweep:
            rng = random.Random(f"{SEED}:{name}:{f}")
            row = measure_cell(name, fn, f, honest_cert, honest, check_item, rng)
            row["description"] = blurb
            rows.append(row)

    print("\n--- DETECTION.  internal = fraction of forgeries the free battery catches;")
    print("    k columns = P(a k-item output-digest challenge catches it); resid = fraction that")
    print("    neither reaches at any k.  noop = vacuous draws, excluded from the rates.")
    hdr = (f"{'class':<26}{'f':>3} {'n':>4} {'noop':>5} {'internal':>9} {'resid':>7}  "
           + "".join(f"k={k:<6}" for k in KS))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = "".join(f"{r['challenge_digest'][str(k)]:<8.3f}" for k in KS)
        print(f"{r['class']:<26}{r['f']:>3} {r['n_effective']:>4} {r['n_noop']:>5} "
              f"{r['internal_caught_rate']:>9.3f} {r['residual_miss_rate']:>7.3f}  {cells}")

    misses = [r for r in rows if r["residual_miss_rate"] > 0]
    print("\n=== THE MISS LIST ===")
    print("Every cell below contains forgeries that neither the free internal battery nor an")
    print("output-digest challenge of ANY size catches -- k=64 is re-running the whole battery.")
    if not misses:
        print("  EMPTY -- which would mean the mutations were too easy, not that the system is")
        print("  sound.  Generate harder ones before reporting a clean sheet.")
    else:
        print(f"\n{'class':<26}{'f':>3} {'resid':>7} {'internal':>9} {'full k=1':>9} "
              f"{'full k=64':>10}  what is being claimed")
        for r in misses:
            print(f"{r['class']:<26}{r['f']:>3} {r['residual_miss_rate']:>7.3f} "
                  f"{r['internal_caught_rate']:>9.3f} {r['challenge_full']['1']:>9.3f} "
                  f"{r['challenge_full']['64']:>10.3f}  {r['description']}")
        total_resid = [r for r in misses if r["residual_miss_rate"] == 1.0]
        print(f"\n  {len(total_resid)} of {len(misses)} cells miss on EVERY forgery generated:")
        for r in total_resid:
            print(f"    {r['class']} f={r['f']}")

    print("\n--- WHERE THE CHALLENGE BUDGET SHOULD NOT GO.  Caught by the free internal battery on")
    print("    every forgery at every f in the sweep, so a re-run buys nothing against them.")
    by_class: dict = {}
    for r in rows:
        by_class.setdefault(r["class"], []).append(r["internal_caught_rate"])
    for c, v in sorted(by_class.items()):
        if min(v) == 1.0:
            print(f"  {c}")

    print("\n--- WHERE IT SHOULD GO.  Invisible to the internal battery on every forgery, and")
    print("    reached only by a second party's bytes.")
    for c, v in sorted(by_class.items()):
        if max(v) == 0.0:
            best = max(r["challenge_digest"]["64"] for r in rows if r["class"] == c)
            print(f"  {c:<26} digest challenge at k=64 catches {best:.3f}")

    results = {
        "target": A.BF16[0].name,
        "n_items": len(honest),
        "partial_items": len(partial),
        "free_tokens": free,
        "n_forgeries_per_cell": N_FORGERIES,
        "ks": KS,
        "false_alarm": fa,
        "classes": rows,
        "miss_list": [{"class": r["class"], "f": r["f"],
                       "residual_miss_rate": r["residual_miss_rate"],
                       "internal_caught_rate": r["internal_caught_rate"],
                       "challenge_digest_k64": r["challenge_digest"]["64"],
                       "challenge_full_k64": r["challenge_full"]["64"],
                       "description": r["description"]} for r in misses],
    }
    OUT.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
