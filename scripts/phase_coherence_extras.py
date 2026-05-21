#!/usr/bin/env python3
"""
phase_coherence_extras.py
=========================

Exploratory companion analyses for the phase-coherence preregistration.

NOT the hypothesis test. The hypothesis test is run by the locked driver
``scripts/phase_coherence_pilot.py`` at commit ``23b7912`` against the
``composite``-only channel at lag 0. The preregistration §3 single-channel
multiple-comparison lock means only that one statistic is admissible as
positive/negative evidence.

This script computes — for the same corpus, below the lock-line — :

  1. **Lag-sweep:** primary CC at lags ±0..±3, to reveal whether one
     agent's pulse leads or lags the other.
  2. **Per-axis CC:** Pearson r at lag 0 on each sub-instrument
     (sycophancy, deception, overconfidence, refusal) independently.
  3. **Hilbert phase-locking value (PLV):** the signal-processing
     definition of phase coherence — agrees with Pearson r when the
     signals are well-aligned amplitude-wise, can diverge otherwise.

All three are diagnostic depth. They do not modify the §6 bar, the
kill-gate, or the outcome categories. The primary scorer's verdict
stands regardless of what this script reports.

Usage
-----
    python scripts/phase_coherence_extras.py \\
        --manifest papers/cooperative-agent-regime/corpus_manifest.json \\
        --output papers/cooperative-agent-regime/results/extras.json

If two manifests are passed (cooperative + noncooperative), the
exploratory comparison is also written.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional


def _ensure_styxx_importable() -> None:
    """Add the styxx repo to sys.path when running from outside an
    editable install. (Editable installs need nothing.)"""
    here = Path(__file__).resolve().parent.parent
    if (here / "styxx").is_dir() and str(here) not in sys.path:
        sys.path.insert(0, str(here))


_ensure_styxx_importable()


from styxx.coherence import (  # noqa: E402
    load_pulse_trace,
    pulse_coherence,
)


def analyze_manifest(manifest_path: Path) -> dict:
    """Run the extras analyses over every conversation in a manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    convs = manifest["conversations"]
    regime = manifest.get("regime", "unspecified")

    per_dyad: list[dict] = []
    for c in convs:
        pulse_a = load_pulse_trace(Path(c["chart_a"]))
        pulse_b = load_pulse_trace(Path(c["chart_b"]))
        result = pulse_coherence(pulse_a, pulse_b)
        per_dyad.append({
            "session_id": c["session_id"],
            "chart_a": c["chart_a"],
            "chart_b": c["chart_b"],
            "n_samples": result.n_samples,
            "primary_cc": result.primary_cc,
            "plv": result.plv,
            "lag_sweep": result.lag_sweep,
            "per_axis": result.per_axis,
        })

    # Aggregate
    primary_ccs = [d["primary_cc"] for d in per_dyad]
    plvs = [d["plv"] for d in per_dyad]

    lag_keys = sorted({k for d in per_dyad for k in d["lag_sweep"].keys()})
    lag_aggregate = {}
    for k in lag_keys:
        vals = [d["lag_sweep"][k] for d in per_dyad if k in d["lag_sweep"]]
        vals = [v for v in vals if v == v]  # drop NaN
        if vals:
            lag_aggregate[str(k)] = {
                "median": statistics.median(vals),
                "mean": statistics.fmean(vals),
                "n": len(vals),
            }

    axis_aggregate: dict[str, dict] = {}
    for ax in ("sycophancy", "deception", "overconfidence", "refusal"):
        vals = [d["per_axis"].get(ax) for d in per_dyad]
        vals = [v for v in vals if v is not None and v == v]
        if vals:
            axis_aggregate[ax] = {
                "median": statistics.median(vals),
                "mean": statistics.fmean(vals),
                "n": len(vals),
            }

    return {
        "regime": regime,
        "n_conversations": len(per_dyad),
        "per_dyad": per_dyad,
        "aggregate": {
            "primary_cc_median": statistics.median(primary_ccs),
            "primary_cc_mean": statistics.fmean(primary_ccs),
            "plv_median": statistics.median(plvs),
            "plv_mean": statistics.fmean(plvs),
            "lag_sweep": lag_aggregate,
            "per_axis": axis_aggregate,
        },
        "preregistration_lock_hash": "3473523",
        "scoring_code_lock_hash": "23b7912",
        "note": (
            "EXPLORATORY ANALYSIS — NOT preregistered. The primary "
            "hypothesis test is scripts/phase_coherence_pilot.py corpus "
            "against the manifest. The extras here add diagnostic depth "
            "(lead/lag structure, signal-processing phase-locking, "
            "per-sub-instrument coherence) but do not modify the §6 bar "
            "or the outcome category."
        ),
    }


def compare_regimes(coop: dict, noncoop: dict) -> dict:
    """Exploratory: side-by-side regime comparison."""
    return {
        "primary_cc": {
            "cooperative_median": coop["aggregate"]["primary_cc_median"],
            "noncooperative_median": noncoop["aggregate"]["primary_cc_median"],
            "delta": coop["aggregate"]["primary_cc_median"]
                     - noncoop["aggregate"]["primary_cc_median"],
        },
        "plv": {
            "cooperative_median": coop["aggregate"]["plv_median"],
            "noncooperative_median": noncoop["aggregate"]["plv_median"],
            "delta": coop["aggregate"]["plv_median"]
                     - noncoop["aggregate"]["plv_median"],
        },
        "note": (
            "Exploratory regime delta. The non-cooperative corpus is NOT "
            "preregistered. A positive delta (cooperative > noncooperative) "
            "is consistent with — not proof of — cooperative-regime "
            "specificity for the observed coherence signal."
        ),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    p.add_argument("--manifest", type=Path, required=True,
                   help="primary (cooperative) corpus manifest")
    p.add_argument("--noncoop-manifest", type=Path, default=None,
                   help="optional non-cooperative control corpus manifest")
    p.add_argument(
        "--output", type=Path,
        default=Path("papers/cooperative-agent-regime/results/extras.json"),
    )
    args = p.parse_args(argv)

    coop = analyze_manifest(args.manifest)
    payload = {"cooperative": coop}

    if args.noncoop_manifest:
        noncoop = analyze_manifest(args.noncoop_manifest)
        payload["noncooperative"] = noncoop
        payload["regime_comparison"] = compare_regimes(coop, noncoop)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Compact stdout summary
    print(json.dumps(
        {
            "cooperative": {
                "primary_cc_median": payload["cooperative"]["aggregate"]["primary_cc_median"],
                "plv_median": payload["cooperative"]["aggregate"]["plv_median"],
                "per_axis": payload["cooperative"]["aggregate"]["per_axis"],
                "lag_sweep_median_lag": {
                    k: v["median"]
                    for k, v in payload["cooperative"]["aggregate"]["lag_sweep"].items()
                },
            },
            "noncooperative_present": "noncooperative" in payload,
            "output": str(args.output),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
