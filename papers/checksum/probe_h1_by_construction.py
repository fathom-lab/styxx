#!/usr/bin/env python3
"""probe_h1_by_construction.py — can the deploy-scale PREREG's H1 read SAME at all?

    python papers/checksum/probe_h1_by_construction.py      # writes probe_h1_by_construction_result.json

PREREG_checksum_deploy_quant_2026_09_13 freezes: the null floor is the WORST pairwise distance among
three loads A, A′, A″ of the same weights, and H1 says "the floor is > 0 and ≤ 1e-3 nats/token; A vs
A′ reads SAME". But `distance()` grades SAME only when the bootstrap upper bound is BELOW the floor,
and A–A′ is one of the pairs the floor is the maximum of — so whenever A–A′ is the worst pair, SAME is
impossible, and when it is not, SAME needs its whole interval under the worst pair's mean. This probe
draws three synthetic same-weights fingerprints with GPU-like jitter at three scales, many trials each,
and counts the verdicts. No model is involved; the arithmetic is the PREREG's own.

Written 2026-09-13 by the red team (deploy-2, checksum-1), committed so the CORRECTION beside the
PREREG can swear to its counts. Seeds are fixed; the result is deterministic.
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
N_TRIALS = 30
JITTERS = (3e-4, 1e-3, 5e-3)

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


def main():
    out = {"schema": "styxx.checksum/probe-h1/v0", "n_trials_per_jitter": N_TRIALS, "vocab": V, "n_items": len(ck.CANARIES),
           "rule": "floor = worst pairwise among A, A', A''; graded pair A vs A'; SAME iff bootstrap upper bound < max(floor, 1e-4)",
           "jitters": {}}
    for jitter in JITTERS:
        counts = {"SAME": 0, "INCONCLUSIVE": 0, "DRIFT": 0}
        worst_is_graded_pair = 0
        floors = []
        for trial in range(N_TRIALS):
            A, A2, A3 = (make(jitter, 1000 * trial + k + int(jitter * 1e6)) for k in range(3))
            floor = ck.null_floor([A, A2, A3])
            pairs = {"A-A2": float(np.abs(A.mean_lp - A2.mean_lp).mean()),
                     "A-A3": float(np.abs(A.mean_lp - A3.mean_lp).mean()),
                     "A2-A3": float(np.abs(A2.mean_lp - A3.mean_lp).mean())}
            d = ck.distance(A, A2, n_boot=500, floor_nats=floor)
            counts[d.verdict] += 1
            worst_is_graded_pair += int(max(pairs, key=pairs.get) == "A-A2")
            floors.append(floor)
        out["jitters"][f"{jitter:g}"] = {"verdicts_A_vs_Aprime": counts, "trials_where_the_graded_pair_is_the_worst": worst_is_graded_pair,
                                         "floor_min": min(floors), "floor_max": max(floors),
                                         "floor_inside_H1_band": bool(1e-4 < min(floors) and max(floors) <= 1e-3)}
        print(jitter, counts, "worst=graded in", worst_is_graded_pair, "floor", f"{min(floors):.2e}..{max(floors):.2e}")
    with open(os.path.join(HERE, "probe_h1_by_construction_result.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print("wrote probe_h1_by_construction_result.json")


if __name__ == "__main__":
    main()
