# -*- coding: utf-8 -*-
"""styxx.portability — what a recipe's numbers mean on another machine.

The null floor answers one question: same weights, same machine, how far apart do two fingerprints
land? The portability reading answers the next one: same recipe, another machine, how far apart do
the NUMBERS land — and do the VERDICTS survive the move?

    python -m styxx.portability certs_a.json certs_b.json [more...]
        [--fingerprints fp_a.json fp_b.json [more...]] [--labels build alienware ...] [--base-arm A]
        --out portability.json

Inputs are certs files written by the same recipe — the shape `run_smollm_quant.py` and
`run_deploy_quant.py` write: top-level named arms, each holding a `styxx.checksum/compare/*` cert
(nested certs such as `h1_held_out.cert` are not graded; only top-level arms are) — and, optionally,
the fingerprints files those runs wrote. Output is a portability cert (`styxx.portability/v1`):

- per arm shared by every machine: the verdict on each machine and whether they agree; for every
  point-estimate leaf of the distance (mean_abs_nats, rdm_r, corr_dist) the values and the largest
  pairwise absolute difference across machines; for the interval leaves (ci_mean_abs, ci_rdm_r) the
  endpoint spreads, reported and NOT graded — a bootstrap percentile has its own Monte Carlo width
  and is not a point to compare against a floor;
- per fingerprint arm, when fingerprints are given: the mean and the largest per-item |Δ mean
  log-prob| across machines; for the base arm the MEAN is the cross-machine null floor, the same
  statistic `checksum.null_floor` uses within a machine, so the two floors are comparable;
- an overall reading: `verdicts` is AGREE when every shared arm's verdict is the same string on every
  machine, else FLIP (which arms); `magnitudes` is WITHIN-FLOOR when every point leaf's spread is at
  or below the grading floor, MOVE when any exceeds it (which, by how much), and UNCOMPARABLE when a
  leaf is finite on one machine and absent or non-finite on another and nothing else moved — an
  absent number is never read as a number that survived;
- a digest over the comparison AND over what was compared: the input certs' own digests and the
  fingerprints' written hashes are inside the digested body, so the portability cert names the
  bytes it graded; labels and file paths ride outside it.

What is refused, so the reading cannot be steered: certs files that share no arm or that carry more
than one canary set; an arm with no string verdict on some machine; a base arm that is not in the
fingerprints, or that is not the `a` side of every cert that names its side (certs from `compare/v1`
up carry `a.rdm_sha256`; v0 certs do not, and the record says the base-arm binding is unverified);
fingerprints whose canary set is not the certs' or whose item count is not the certs' n_items.

What this is not: a verdict on any model, or a tolerance in any statistical sense. It grades the
recipe's portability between the machines it was given, and only those. Two machines give one
difference; the spread it prints is a floor on the tolerance a RESULT may state, never a ceiling —
a third machine can only widen it — and a challenge under BOUNTY.md is decided at the verdict level,
not by landing outside a two-machine spread. Written 2026-09-13, the day the lab's own recipe
reproduced every verdict and no magnitude on a second machine; rewritten the same night after its
own red team (an absent leaf read as survived; the base arm chose the verdict; the labels were in
the digest and the inputs were not).

Exit codes: 0 when verdicts AGREE (whatever the magnitudes did), 3 on a FLIP, 2 on a refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

RESOLUTION_NATS = 1e-4          # styxx.checksum.RESOLUTION_NATS, restated so this module imports nothing heavy
POINT_LEAVES = ("mean_abs_nats", "rdm_r", "corr_dist")
INTERVAL_LEAVES = ("ci_mean_abs", "ci_rdm_r")


def _load(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))


def arms_of(certs: dict) -> dict:
    """The compare certs inside a runner's output: TOP-LEVEL keys whose value carries a distance and a
    compare schema. Nested certs are not graded."""
    out = {}
    for k, v in certs.items():
        if isinstance(v, dict) and isinstance(v.get("distance"), dict) and str(v.get("schema", "")).startswith("styxx.checksum/compare"):
            out[k] = v
    return out


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _leaf(values: list) -> tuple[float | None, bool]:
    """(spread, comparable): the spread of the finite values when every machine has one; otherwise None,
    and comparable=False when at least one machine has a finite value and another does not."""
    finite = [v for v in values if _finite(v)]
    if len(finite) == len(values):
        return (float(max(finite) - min(finite)) if len(finite) >= 2 else None), True
    return None, (len(finite) == 0)      # all absent: nothing to compare, but nothing hidden either


def cross_machine_floor(fingerprints: list[dict], n_items: int | None = None, canary: str | None = None) -> dict:
    """Per arm present in every fingerprints file: the mean and the largest per-item |Δ mean log-prob|
    between any two machines. Refuses fingerprints that do not name the certs' canary set or that do
    not carry n_items values."""
    common = set(fingerprints[0].keys())
    for f in fingerprints[1:]:
        common &= set(f.keys())
    out = {}
    for arm in sorted(common):
        worst, mean_worst, n_diff, n = 0.0, 0.0, 0, None
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                fa, fb = fingerprints[i][arm], fingerprints[j][arm]
                for f, m in ((fa, i), (fb, j)):
                    if canary is not None and f.get("canary_sha256") != canary:
                        raise ValueError(f"fingerprints[{m}][{arm!r}] names canary set {str(f.get('canary_sha256'))[:12]}…, "
                                         f"not the certs' {canary[:12]}…")
                a, b = fa.get("mean_lp"), fb.get("mean_lp")
                if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b) or not a:
                    raise ValueError(f"fingerprints for arm {arm!r} do not carry comparable mean_lp lists "
                                     f"({type(a).__name__} of {len(a) if isinstance(a, list) else '?'} vs "
                                     f"{type(b).__name__} of {len(b) if isinstance(b, list) else '?'})")
                if n_items is not None and len(a) != n_items:
                    raise ValueError(f"fingerprints for arm {arm!r} carry {len(a)} items; the certs grade {n_items}")
                d = [abs(float(p) - float(q)) for p, q in zip(a, b)]
                worst = max(worst, max(d))
                mean_worst = max(mean_worst, sum(d) / len(d))
                n_diff = max(n_diff, sum(1 for v in d if v > 0))
                n = len(d)
        out[arm] = {"mean_abs_diff_mean_lp": mean_worst, "max_abs_diff_mean_lp": worst, "n_items_differing": n_diff, "n_items": n}
    return out


def compare(certs: list[dict], labels: list[str], fingerprints: list[dict] | None = None, base_arm: str = "A") -> dict:
    if len(certs) < 2:
        raise ValueError("portability needs certs from at least two machines")
    if len(labels) != len(certs):
        raise ValueError("one label per certs file")
    arm_sets = [arms_of(c) for c in certs]
    all_arms = set().union(*[set(s) for s in arm_sets])
    common = set(arm_sets[0])
    for s in arm_sets[1:]:
        common &= set(s)
    if not common:
        raise ValueError("the certs files share no compare arms; are they from the same recipe?")
    arms_not_shared = sorted(all_arms - common)
    canaries = set()
    for m, s in enumerate(arm_sets):
        for a in common:
            c = s[a].get("canary_sha256")
            if not isinstance(c, str):
                raise ValueError(f"cert {a!r} on machine {labels[m]!r} names no canary_sha256")
            canaries.add(c)
    if len(canaries) != 1:
        raise ValueError(f"the runs used different canary sets ({len(canaries)}); not the same recipe")
    canary = next(iter(canaries))
    n_items_set = {s[a]["n_items"] for s in arm_sets for a in common if "n_items" in s[a]}
    n_items = next(iter(n_items_set)) if len(n_items_set) == 1 else None

    # the base arm: must be in the fingerprints, and must be the `a` side of every cert that names its side
    floors, base_binding = {}, "no fingerprints given"
    if fingerprints:
        if len(fingerprints) != len(certs):
            raise ValueError("one fingerprints file per certs file")
        if any(base_arm not in f for f in fingerprints):
            raise ValueError(f"base arm {base_arm!r} is not in every fingerprints file; it must be the same-weights "
                             "arm every cert compares against")
        floors = cross_machine_floor(fingerprints, n_items, canary)
        checks = []
        for m, s in enumerate(arm_sets):
            base_hash = fingerprints[m][base_arm].get("rdm_sha256")
            for a in common:
                side = (s[a].get("a") or {}).get("rdm_sha256")
                if side is not None:
                    checks.append(side == base_hash)
        if checks and not all(checks):
            raise ValueError(f"base arm {base_arm!r} is not the `a` side of every cert; the grading floor would come from "
                             "an arm the certs do not compare against")
        base_binding = "verified: the base arm's written rdm hash is the `a` side of every cert" if checks else \
                       "unverified: the certs carry no a.rdm_sha256 (compare/v0)"
    null_floor = floors.get(base_arm, {}).get("mean_abs_diff_mean_lp") if fingerprints else None
    grading_floor = max(null_floor, RESOLUTION_NATS) if null_floor is not None else RESOLUTION_NATS
    grading_source = ("cross-machine null floor of the base arm (mean per-item |Δ|, as checksum.null_floor)"
                      if null_floor is not None else "resolution (no fingerprints given)")

    arms, flips, moves, uncompared = {}, [], [], []
    for arm in sorted(common):
        ds = [s[arm]["distance"] for s in arm_sets]
        verdicts = [d.get("verdict") for d in ds]
        if not all(isinstance(v, str) and v for v in verdicts):
            raise ValueError(f"arm {arm!r} has no string verdict on some machine: {verdicts}")
        # values are recorded per machine IN ORDER (the `labels` list outside the digest names the order), so the
        # digest depends on the comparison and never on what the machines were called
        entry = {"verdicts": verdicts, "verdicts_agree": len(set(verdicts)) == 1, "numbers": {}, "intervals": {}}
        if not entry["verdicts_agree"]:
            flips.append(arm)
        for leaf in POINT_LEAVES:
            vals = [d.get(leaf) for d in ds]
            sp, comparable = _leaf(vals)
            entry["numbers"][leaf] = {"values": [v if _finite(v) else None for v in vals], "max_abs_diff": sp, "comparable": comparable}
            if not comparable:
                uncompared.append(f"{arm}.{leaf}")
            elif sp is not None and sp > grading_floor:
                moves.append(f"{arm}.{leaf}")
        for leaf in INTERVAL_LEAVES:
            ivs = [d.get(leaf) for d in ds]
            if all(isinstance(iv, list) and len(iv) == 2 for iv in ivs):
                lo, _ = _leaf([iv[0] for iv in ivs])
                hi, _ = _leaf([iv[1] for iv in ivs])
                clean = [[x if _finite(x) else None for x in iv] for iv in ivs]
                entry["intervals"][leaf] = {"values": clean, "endpoint_spreads": [lo, hi], "graded": False}
            else:
                entry["intervals"][leaf] = {"values": [iv if isinstance(iv, list) else None for iv in ivs], "endpoint_spreads": None, "graded": False}
        arms[arm] = entry
    magnitudes = "MOVE" if moves else ("UNCOMPARABLE" if uncompared else "WITHIN-FLOOR")
    if flips:
        reading = "a verdict does not survive the move: the recipe is not portable between these machines at the verdict level"
    elif magnitudes == "MOVE":
        reading = ("every verdict survives the move between these machines; the numbers named in magnitudes_moved do not, "
                   "and each max_abs_diff is the observed spread between these machines — a floor on the tolerance a "
                   "RESULT may state, not a ceiling")
    elif magnitudes == "UNCOMPARABLE":
        reading = "every verdict survives, but a number present on one machine is absent or non-finite on another; nothing is said about it"
    else:
        reading = "every verdict and every compared number survive the move between these machines"
    input_digests = [sorted(str(s[a].get("digest")) for a in common) for s in arm_sets]
    fp_hashes = ([{a: {"rdm_sha256": f[a].get("rdm_sha256"), "mean_lp_sha256": f[a].get("mean_lp_sha256")} for a in sorted(f)}
                  for f in fingerprints] if fingerprints else None)
    body = {
        "schema": "styxx.portability/v1",
        "recipe_canary_sha256": canary,
        "n_machines": len(certs),
        "n_items": n_items,
        "input_cert_digests": input_digests,
        "input_fingerprint_hashes": fp_hashes,
        "arms": arms,
        "arms_not_shared": arms_not_shared,
        "cross_machine_floor": floors,
        "base_arm": base_arm,
        "base_arm_binding": base_binding,
        "grading_floor_nats": grading_floor,
        "grading_floor_source": grading_source,
        "verdicts": "AGREE" if not flips else "FLIP",
        "verdict_flips": flips,
        "magnitudes": magnitudes,
        "magnitudes_moved": moves,
        "magnitudes_uncompared": uncompared,
        "intervals_graded": False,
        "reading": reading,
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    body["digest"] = hashlib.sha256(blob).hexdigest()
    body["labels"] = labels                                   # outside the digest
    body["created"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.portability", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("certs", nargs="+", help="two or more certs files from the same recipe on different machines")
    ap.add_argument("--fingerprints", nargs="*", default=[], help="the matching fingerprints files, same order")
    ap.add_argument("--labels", nargs="*", default=[], help="one label per certs file (default: the file names without directories)")
    ap.add_argument("--base-arm", default="A")
    ap.add_argument("--out", default="portability.json")
    a = ap.parse_args(argv)
    labels = a.labels or [os.path.basename(p) for p in a.certs]
    try:
        fps = [_load(p) for p in a.fingerprints] if a.fingerprints else None
        rec = compare([_load(p) for p in a.certs], labels, fps, base_arm=a.base_arm)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    rec["inputs"] = {"certs": a.certs, "fingerprints": a.fingerprints}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1, allow_nan=False) + "\n")
    print(f"verdicts {rec['verdicts']}  magnitudes {rec['magnitudes']}  grading floor {rec['grading_floor_nats']:.3g} nats/token "
          f"({rec['grading_floor_source']})  digest {rec['digest'][:12]} -> {a.out}")
    for arm, e in rec["arms"].items():
        m = e["numbers"]["mean_abs_nats"]
        print(f"  {arm:10s} verdicts {dict(zip(labels, e['verdicts']))}  mean_abs {dict(zip(labels, m['values']))}  max|Δ| {m['max_abs_diff']}")
    if rec["arms_not_shared"]:
        print("  not shared by every machine, not graded:", rec["arms_not_shared"])
    if rec["verdict_flips"]:
        print("  FLIP:", rec["verdict_flips"])
    if rec["magnitudes_moved"]:
        print("  MOVE:", ", ".join(rec["magnitudes_moved"]))
    if rec["magnitudes_uncompared"]:
        print("  UNCOMPARABLE:", ", ".join(rec["magnitudes_uncompared"]))
    return 0 if rec["verdicts"] == "AGREE" else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
