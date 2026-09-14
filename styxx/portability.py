# -*- coding: utf-8 -*-
"""styxx.portability — what a recipe's numbers mean on another machine.

The null floor answers one question: same weights, same machine, how far apart do two fingerprints
land? The portability floor answers the next one: same recipe, another machine, how far apart do
the NUMBERS land — and do the VERDICTS survive the move?

    python -m styxx.portability certs_a.json certs_b.json [more...]
        [--fingerprints fp_a.json fp_b.json [more...]] [--label-a build] [--label-b alienware] --out portability.json

Inputs are certs files written by the same recipe — the shape `run_smollm_quant.py` and
`run_deploy_quant.py` write: named arms, each holding a `styxx.checksum/compare/*` cert — and,
optionally, the fingerprints files those runs wrote. Output is a portability cert:

- per arm: the verdict on each machine and whether they agree; for every numeric leaf of the
  distance (mean_abs_nats, ci_mean_abs, rdm_r, corr_dist), the values and the largest pairwise
  absolute difference across machines — that difference is the tolerance a RESULT can state for
  that number, measured, with the machines named;
- per fingerprint arm, when fingerprints are given: the largest per-item |Δ mean log-prob| across
  machines — for the base arm this is the CROSS-MACHINE NULL FLOOR, the floor the null floor
  does not see because it never leaves the machine;
- an overall reading: `verdicts` is AGREE when every arm's verdict is the same on every machine,
  else FLIP (which arms); `magnitudes` is WITHIN-FLOOR when every numeric difference is at or
  below the cross-machine null floor (or the resolution, when no fingerprints are given), else
  MOVE (which arms, by how much);
- a digest over the comparison; the file names and labels ride outside it.

What this is not: a verdict on any model. It grades the recipe's portability between the
machines it was given, and only those. Two machines give one difference; the tolerance it prints
is a floor on the spread, not a ceiling. Written 2026-09-13, the day the lab's own recipe
reproduced every verdict and no magnitude on a second machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

RESOLUTION_NATS = 1e-4          # styxx.checksum.RESOLUTION_NATS, restated so this module imports nothing heavy
NUMERIC_LEAVES = ("mean_abs_nats", "rdm_r", "corr_dist")
INTERVAL_LEAVES = ("ci_mean_abs", "ci_rdm_r")


def _load(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))


def arms_of(certs: dict) -> dict:
    """The compare certs inside a runner's output: keys whose value carries a distance and a compare schema."""
    out = {}
    for k, v in certs.items():
        if isinstance(v, dict) and isinstance(v.get("distance"), dict) and str(v.get("schema", "")).startswith("styxx.checksum/compare"):
            out[k] = v
    return out


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _spread(values: list) -> float | None:
    vals = [v for v in values if _finite(v)]
    if len(vals) < 2:
        return None
    return float(max(vals) - min(vals))


def cross_machine_floor(fingerprints: list[dict]) -> dict:
    """Per arm present in every fingerprints file: the largest per-item |Δ mean log-prob| between any two
    machines, and how many items differ at all."""
    common = set(fingerprints[0].keys())
    for f in fingerprints[1:]:
        common &= set(f.keys())
    out = {}
    for arm in sorted(common):
        worst, n_diff, n = 0.0, 0, None
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                a, b = fingerprints[i][arm].get("mean_lp"), fingerprints[j][arm].get("mean_lp")
                if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)):
                    continue
                d = [abs(float(p) - float(q)) for p, q in zip(a, b)]
                worst = max(worst, max(d) if d else 0.0)
                n_diff = max(n_diff, sum(1 for v in d if v > 0))
                n = len(d)
        out[arm] = {"max_abs_diff_mean_lp": worst, "n_items_differing": n_diff, "n_items": n}
    return out


def compare(certs: list[dict], labels: list[str], fingerprints: list[dict] | None = None, base_arm: str = "A") -> dict:
    if len(certs) < 2:
        raise ValueError("portability needs certs from at least two machines")
    if len(labels) != len(certs):
        raise ValueError("one label per certs file")
    arm_sets = [arms_of(c) for c in certs]
    common = set(arm_sets[0])
    for s in arm_sets[1:]:
        common &= set(s)
    if not common:
        raise ValueError("the certs files share no compare arms; are they from the same recipe?")
    canary = {s[a]["canary_sha256"] for s in arm_sets for a in common}
    if len(canary) != 1:
        raise ValueError(f"the runs used different canary sets ({len(canary)}); not the same recipe")
    floors = cross_machine_floor(fingerprints) if fingerprints else {}
    null_floor = floors.get(base_arm, {}).get("max_abs_diff_mean_lp")
    grading_floor = max(null_floor, RESOLUTION_NATS) if null_floor is not None else RESOLUTION_NATS
    arms, flips, moves = {}, [], []
    for arm in sorted(common):
        ds = [s[arm]["distance"] for s in arm_sets]
        verdicts = [d.get("verdict") for d in ds]
        entry = {"verdicts": dict(zip(labels, verdicts)), "verdicts_agree": len(set(verdicts)) == 1, "numbers": {}}
        if not entry["verdicts_agree"]:
            flips.append(arm)
        for leaf in NUMERIC_LEAVES:
            vals = [d.get(leaf) for d in ds]
            sp = _spread(vals)
            entry["numbers"][leaf] = {"values": dict(zip(labels, vals)), "max_abs_diff": sp}
            if sp is not None and sp > grading_floor:
                moves.append(f"{arm}.{leaf}")
        for leaf in INTERVAL_LEAVES:
            ivs = [d.get(leaf) for d in ds]
            if all(isinstance(iv, list) and len(iv) == 2 for iv in ivs):
                lo, hi = _spread([iv[0] for iv in ivs]), _spread([iv[1] for iv in ivs])
                entry["numbers"][leaf] = {"values": dict(zip(labels, ivs)), "max_abs_diff": [lo, hi]}
                if (lo is not None and lo > grading_floor) or (hi is not None and hi > grading_floor):
                    moves.append(f"{arm}.{leaf}")
        arms[arm] = entry
    body = {
        "schema": "styxx.portability/v0",
        "recipe_canary_sha256": next(iter(canary)),
        "n_machines": len(certs),
        "labels": labels,
        "arms": arms,
        "cross_machine_floor": floors,
        "base_arm": base_arm,
        "grading_floor_nats": grading_floor,
        "grading_floor_source": "cross-machine null floor of the base arm" if null_floor is not None else "resolution (no fingerprints given)",
        "verdicts": "AGREE" if not flips else "FLIP",
        "verdict_flips": flips,
        "magnitudes": "WITHIN-FLOOR" if not moves else "MOVE",
        "magnitudes_moved": moves,
        "reading": ("every verdict survives the move between these machines; the numbers do not, and the per-number "
                    "max_abs_diff is the tolerance a RESULT may state for them, measured on these machines only"
                    if not flips and moves else
                    "every verdict and every number survive the move between these machines" if not flips else
                    "a verdict does not survive the move: the recipe is not portable between these machines at the verdict level"),
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body["digest"] = hashlib.sha256(blob).hexdigest()
    body["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.portability", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("certs", nargs="+", help="two or more certs files from the same recipe on different machines")
    ap.add_argument("--fingerprints", nargs="*", default=[], help="the matching fingerprints files, same order")
    ap.add_argument("--labels", nargs="*", default=[], help="one label per certs file (default: the file names)")
    ap.add_argument("--base-arm", default="A")
    ap.add_argument("--out", default="portability.json")
    a = ap.parse_args(argv)
    labels = a.labels or a.certs
    fps = [_load(p) for p in a.fingerprints] if a.fingerprints else None
    if fps is not None and len(fps) != len(a.certs):
        raise SystemExit("one fingerprints file per certs file, in the same order")
    rec = compare([_load(p) for p in a.certs], labels, fps, base_arm=a.base_arm)
    rec["inputs"] = {"certs": a.certs, "fingerprints": a.fingerprints}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1, allow_nan=False) + "\n")
    print(f"verdicts {rec['verdicts']}  magnitudes {rec['magnitudes']}  grading floor {rec['grading_floor_nats']:.3g} nats/token "
          f"({rec['grading_floor_source']})  digest {rec['digest'][:12]} -> {a.out}")
    for arm, e in rec["arms"].items():
        m = e["numbers"]["mean_abs_nats"]
        print(f"  {arm:10s} verdicts {list(e['verdicts'].values())}  mean_abs {m['values']}  max|Δ| {m['max_abs_diff']}")
    if rec["verdict_flips"]:
        print("  FLIP:", rec["verdict_flips"])
    if rec["magnitudes_moved"]:
        print("  MOVE:", ", ".join(rec["magnitudes_moved"]))
    return 0 if rec["verdicts"] == "AGREE" else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
