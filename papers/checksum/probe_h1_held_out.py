#!/usr/bin/env python3
"""probe_h1_held_out.py — can the beacon-draw PREREG's held-out H1 read SAME when the loads are not bit-identical?

    python papers/checksum/probe_h1_held_out.py      # writes probe_h1_held_out_result.json

PREREG_checksum_beacon_draw_2026_09_14 froze H1's pair clause in the held-out form that
CORRECTION_prereg_deploy_quant_H1 rule 5 promised: A vs A′ is graded against the larger of the A–A″ and
A′–A″ distances, so the graded pair is no longer inside the floor that grades it. The red team of
2026-09-14 argued that this removes only the case where SAME was impossible, not the problem: `distance()`
grades SAME only when the bootstrap UPPER bound of the graded pair's mean lies below the floor, and the
held-out floor is a POINT estimate of a quantity distributed exactly like the graded pair's mean. Three
loads of the same weights give three exchangeable pair distances, so the graded pair's upper bound sits
below the larger of the other two only rarely.

This probe measures it with the same synthetic construction as probe_h1_by_construction.py (three
same-weights fingerprints with per-item jitter at fixed seeds), under the runner's exact held-out rule
(run_deploy_quant.py: held_out_floor = max(mean|A−A″|, mean|A′−A″|); distance(A, A′, floor_nats=held_out_floor)),
at a jitter of zero (bit-identical loads, what the 0.5B instrument checks measured on this box) and at
four scales inside and around the PREREG's ≤ 1e-3 floor band. It also counts, for the same trials, the
worst-pairwise form the 2026-09-13 PREREG froze, so the two forms are compared on identical draws.
No model is involved; the arithmetic is the runner's own. Seeds are fixed; the result is deterministic.
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from styxx import checksum as ck  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
V = 2000
N_TRIALS = 60
N_BOOT = 2000                      # the runner's N_BOOT
SEED = 20260913                    # the runner's SEED
JITTERS = (0.0, 1e-4, 3e-4, 1e-3, 5e-3)

_rng = np.random.default_rng(0)
BASE_LP = _rng.normal(-3, 1.5, len(ck.CANARIES))
BASE_B = _rng.normal(size=(len(ck.CANARIES), V))
_INDEX = {p: i for i, (_, p, _) in enumerate(ck.CANARIES)}


def make(jitter: float, seed: int) -> ck.Fingerprint:
    r = np.random.default_rng(seed)

    def probe(prompt, continuation):
        i = _INDEX[prompt]
        return ck.Probe(cont_logprobs=[BASE_LP[i] + r.normal(0, jitter)], next_logprobs=BASE_B[i] + r.normal(0, jitter, V))
    return ck.fingerprint(probe, "synthetic same weights", tokenizer_id="synthetic")


def pair(a, b) -> float:
    return float(np.abs(a.mean_lp - b.mean_lp).mean())


def main():
    out = {"schema": "styxx.checksum/probe-h1-held-out/v0", "n_trials_per_jitter": N_TRIALS, "n_boot": N_BOOT, "seed": SEED,
           "vocab": V, "n_items": len(ck.CANARIES),
           "rule_held_out": "floor = max(mean|A-A''|, mean|A'-A''|); graded pair A vs A'; SAME iff bootstrap upper bound < max(floor, 1e-4)",
           "rule_worst_pairwise": "floor = max of all three pair means; graded pair A vs A'; SAME iff bootstrap upper bound < max(floor, 1e-4)",
           "jitters": {}}
    for jitter in JITTERS:
        held = {"SAME": 0, "INCONCLUSIVE": 0, "DRIFT": 0}
        worst = {"SAME": 0, "INCONCLUSIVE": 0, "DRIFT": 0}
        floors = []
        graded_below_held_out_point = 0
        for trial in range(N_TRIALS):
            A, A2, A3 = (make(jitter, 7000 * trial + k + int(jitter * 1e7)) for k in range(3))
            held_floor = max(pair(A, A3), pair(A2, A3))
            worst_floor = ck.null_floor([A, A2, A3])
            held[ck.distance(A, A2, n_boot=N_BOOT, seed=SEED, floor_nats=held_floor).verdict] += 1
            worst[ck.distance(A, A2, n_boot=N_BOOT, seed=SEED, floor_nats=worst_floor).verdict] += 1
            graded_below_held_out_point += int(pair(A, A2) <= held_floor)
            floors.append(worst_floor)
        key = f"{jitter:g}"
        out["jitters"][key] = {"held_out": held, "worst_pairwise": worst,
                               "trials_where_the_graded_pair_mean_is_at_most_the_held_out_floor": graded_below_held_out_point,
                               "floor_min": min(floors), "floor_max": max(floors),
                               "floor_inside_H1_band": bool(max(floors) <= 1e-3),
                               "floor_above_resolution": bool(min(floors) > ck.RESOLUTION_NATS)}
        print(key, "held-out", held, "worst-pairwise", worst, "point<=floor", graded_below_held_out_point,
              "floor", f"{min(floors):.2e}..{max(floors):.2e}", flush=True)
    with open(os.path.join(HERE, "probe_h1_held_out_result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print("wrote probe_h1_held_out_result.json")


if __name__ == "__main__":
    main()
