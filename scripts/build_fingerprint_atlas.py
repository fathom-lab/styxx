"""Compile the v0 cognometric-fingerprint atlas from existing ablation runs.

Consumes the outputs of:
  - scripts/drift_feature_scaling.py       -> benchmarks/drift_feature_scaling.json
  - scripts/refusal_feature_scaling.py     -> benchmarks/refusal_feature_scaling.json
  - scripts/refusal_cross_model_feature_scaling.py
                                           -> benchmarks/refusal_cross_model_feature_scaling.json

Produces: benchmarks/cognometry_fingerprint_atlas_v0.json

Fingerprint field convention (see papers/calibration_fingerprints_v0.md):

  instrument        : str         e.g. "drift-v1", "refusal-v1"
  substrate         : str|None    model family or "in-sample"
  n_features        : int
  baseline_auc      : float       full-model AUC (what you'd report today)
  critical_K        : int|None    smallest K where AUC >= CRITICAL_THRESHOLD
  critical_feature  : str|None    feature added at critical K
  delta_auc_at_K    : float       AUC[critical_K] - AUC[critical_K-1]
  final_auc_at_N    : float       AUC at K = N (all features)
  negative_lift     : list of {K, feature, delta} where delta <= -0.10

Threshold: AUC >= 0.80 triggers critical K. Above-chance-by-0.30 might
be more principled for low-ceiling instruments; we document the choice
rather than optimizing it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]

DRIFT_PT_JSON = REPO / "benchmarks" / "drift_feature_scaling.json"
REFUSAL_PT_JSON = REPO / "benchmarks" / "refusal_feature_scaling.json"
REFUSAL_CROSS_JSON = REPO / "benchmarks" / "refusal_cross_model_feature_scaling.json"

OUT_JSON = REPO / "benchmarks" / "cognometry_fingerprint_atlas_v0.json"

CRITICAL_AUC = 0.80
NEGATIVE_LIFT_DELTA = -0.10


def find_critical_k(auc_curve: List[float]) -> Optional[int]:
    """Return smallest K (1-indexed) where AUC curve crosses CRITICAL_AUC.

    auc_curve[0] is AUC at K=0 (chance), auc_curve[K] is AUC with top-K
    features.
    """
    for K in range(1, len(auc_curve)):
        if auc_curve[K] >= CRITICAL_AUC:
            return K
    return None


def find_negative_lift(auc_curve: List[float], feature_names: List[str]) -> List[Dict]:
    """Return list of (K, feature, delta) where adding the K-th feature
    dropped AUC by at least NEGATIVE_LIFT_DELTA (magnitude-wise)."""
    out = []
    for K in range(1, len(auc_curve)):
        delta = auc_curve[K] - auc_curve[K - 1]
        if delta <= NEGATIVE_LIFT_DELTA:
            out.append({
                "K": K,
                "feature": feature_names[K - 1] if K - 1 < len(feature_names) else None,
                "delta": round(delta, 4),
            })
    return out


def build_drift_fingerprints():
    d = json.loads(DRIFT_PT_JSON.read_text(encoding="utf-8"))
    top_k = d["top_k_by_importance"]
    ranking = d["ranking_by_importance"]  # feature names in |coef| order
    n_features = d["methodology"]["n_features_total"]
    fingerprints = []

    # Overall fingerprint using mean_auc across K
    K_values = [row["n_features"] for row in top_k]
    mean_auc_by_K = {row["n_features"]: row["mean_auc"] for row in top_k}
    # Build full 0..N curve by interpolating missing K (linear between known points)
    overall_curve = [0.5]
    for K in range(1, n_features + 1):
        if K in mean_auc_by_K:
            overall_curve.append(mean_auc_by_K[K])
        else:
            # linear interp between previous known and next known
            prev_K = max((k for k in K_values if k < K), default=0)
            next_K = min((k for k in K_values if k > K), default=n_features)
            if prev_K == 0:
                overall_curve.append(mean_auc_by_K.get(next_K, 0.5))
            elif next_K == n_features + 1 or next_K not in mean_auc_by_K:
                overall_curve.append(mean_auc_by_K.get(prev_K, 0.5))
            else:
                a = mean_auc_by_K[prev_K]
                b = mean_auc_by_K[next_K]
                t = (K - prev_K) / (next_K - prev_K)
                overall_curve.append(a + t * (b - a))
    crit_k = find_critical_k(overall_curve)
    fingerprints.append({
        "instrument": "drift-v1",
        "substrate": "in-sample",
        "failure_class": "overall (pooled)",
        "n_features": n_features,
        "baseline_auc": round(overall_curve[-1], 4),
        "critical_K": crit_k,
        "critical_feature": ranking[crit_k - 1] if crit_k else None,
        "delta_auc_at_K": round(overall_curve[crit_k] - overall_curve[crit_k - 1], 4) if crit_k else None,
        "final_auc_at_N": round(overall_curve[-1], 4),
        "negative_lift": find_negative_lift(overall_curve, ranking),
        "notes": f"drift v6.0 22-feature, CV top-K (n_K_points={len(top_k)})",
    })

    # Per-drift-type fingerprints via already-computed phase_transitions list
    pt_by_class = {}
    for pt in d.get("phase_transitions", []):
        cls = pt["drift_type"]
        pt_by_class.setdefault(cls, []).append(pt)

    for cls, pts in pt_by_class.items():
        # Use the first phase transition for this class as the critical one
        first = pts[0]
        from_k = first["from_k"]
        to_k = first["to_k"]
        critical_feature = ranking[to_k - 1] if to_k - 1 < len(ranking) else None
        fingerprints.append({
            "instrument": "drift-v1",
            "substrate": "in-sample",
            "failure_class": cls,
            "n_features": n_features,
            "baseline_auc": None,  # per-class full-baseline requires per-class CV
            "critical_K": to_k,
            "critical_feature": critical_feature,
            "delta_auc_at_K": round(first["delta"], 4),
            "final_auc_at_N": round(first["to_auc"], 4),
            "negative_lift": [],
            "additional_transitions": [
                {"from_k": pt["from_k"], "to_k": pt["to_k"], "delta": round(pt["delta"], 4)}
                for pt in pts[1:]
            ],
            "notes": f"derived from pre-computed phase_transitions list; {len(pts)} transition(s) in this class",
        })
    return fingerprints


def build_refusal_in_sample_fingerprint():
    d = json.loads(REFUSAL_PT_JSON.read_text(encoding="utf-8"))
    top_k = d["top_k"]
    aucs = [row["auc"] for row in top_k]
    names = [row["added"] for row in top_k if row["added"]]
    crit_k = find_critical_k(aucs)
    return {
        "instrument": "refusal-v1",
        "substrate": "in-sample (JBB-Llama-1B)",
        "failure_class": "refusal vs comply",
        "n_features": d["n_features"],
        "baseline_auc": d["full_model_auc"],
        "critical_K": crit_k,
        "critical_feature": names[crit_k - 1] if crit_k else None,
        "delta_auc_at_K": (
            round(aucs[crit_k] - aucs[crit_k - 1], 4) if crit_k else None
        ),
        "final_auc_at_N": d["full_model_auc"],
        "negative_lift": find_negative_lift(aucs, names),
        "notes": f"5-fold CV, n={d['n_samples']}",
    }


def build_refusal_cross_model_fingerprints():
    d = json.loads(REFUSAL_CROSS_JSON.read_text(encoding="utf-8"))
    sweep = d["top_k_sweep"]
    fingerprints = []
    splits = list(d["splits"].keys())
    feature_names = [row["added"] for row in sweep if row["added"]]

    for split in splits:
        aucs = [row["aucs"].get(split, 0.5) for row in sweep]
        crit_k = find_critical_k(aucs)
        pts_for_split = d.get("phase_transitions_per_split", {}).get(split, [])
        fingerprints.append({
            "instrument": "refusal-v1",
            "substrate": split,
            "failure_class": "refusal vs comply",
            "n_features": d["n_features"],
            "baseline_auc": aucs[-1],
            "critical_K": crit_k,
            "critical_feature": feature_names[crit_k - 1] if crit_k else None,
            "delta_auc_at_K": (
                round(aucs[crit_k] - aucs[crit_k - 1], 4) if crit_k else None
            ),
            "final_auc_at_N": aucs[-1],
            "negative_lift": find_negative_lift(aucs, feature_names),
            "phase_transitions_documented": pts_for_split,
            "substrate_n": d["splits"][split]["n"],
            "substrate_class_balance_refuse": d["splits"][split]["refuse"],
            "notes": (
                "Out-of-sample. Train on JBB-Llama-1B, evaluate on XSTest "
                f"{split} completions."
            ),
        })
    return fingerprints


def main():
    print(f"building atlas from existing ablation JSONs ...")
    atlas_entries = []

    if DRIFT_PT_JSON.exists():
        drift_fps = build_drift_fingerprints()
        print(f"  drift:           {len(drift_fps)} fingerprint(s)")
        atlas_entries.extend(drift_fps)
    else:
        print(f"  drift:           SKIP (missing {DRIFT_PT_JSON.name})")

    if REFUSAL_PT_JSON.exists():
        atlas_entries.append(build_refusal_in_sample_fingerprint())
        print(f"  refusal in-sample: 1 fingerprint")
    else:
        print(f"  refusal in-sample: SKIP (missing {REFUSAL_PT_JSON.name})")

    if REFUSAL_CROSS_JSON.exists():
        cross_fps = build_refusal_cross_model_fingerprints()
        print(f"  refusal cross-model: {len(cross_fps)} fingerprint(s)")
        atlas_entries.extend(cross_fps)
    else:
        print(f"  refusal cross-model: SKIP (missing {REFUSAL_CROSS_JSON.name})")

    # Summary
    print()
    print(f"{'='*78}")
    print("FINGERPRINT ATLAS v0 — SUMMARY")
    print(f"{'='*78}")
    print()
    header = (
        f"  {'instrument':<14s} {'substrate':<24s} {'class':<20s} "
        f"{'K*':>3s} {'critical feat':<24s} {'dAUC':>6s} {'final':>6s}  neg-lift"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for fp in atlas_entries:
        k_str = str(fp["critical_K"]) if fp["critical_K"] else "—"
        cf = fp["critical_feature"] or "—"
        delta = (f"+{fp['delta_auc_at_K']:.3f}" if fp['delta_auc_at_K'] else "—")
        final = f"{fp['final_auc_at_N']:.3f}" if fp['final_auc_at_N'] else "—"
        nl = len(fp.get("negative_lift", []))
        nl_s = f"{nl}" if nl else "—"
        print(
            f"  {fp['instrument']:<14s} {fp['substrate'][:24]:<24s} "
            f"{fp['failure_class'][:20]:<20s} {k_str:>3s} {cf[:24]:<24s} "
            f"{delta:>6s} {final:>6s}  {nl_s}"
        )

    atlas = {
        "version": "v0",
        "date": "2026-04-24",
        "methodology": "papers/calibration_fingerprints_v0.md",
        "critical_auc_threshold": CRITICAL_AUC,
        "negative_lift_delta_threshold": NEGATIVE_LIFT_DELTA,
        "n_fingerprints": len(atlas_entries),
        "n_instruments": len({e["instrument"] for e in atlas_entries}),
        "n_substrates": len({e["substrate"] for e in atlas_entries}),
        "fingerprints": atlas_entries,
    }
    OUT_JSON.write_text(json.dumps(atlas, indent=2), encoding="utf-8")
    print()
    print(f"wrote -> {OUT_JSON.relative_to(REPO)}")


if __name__ == "__main__":
    main()
