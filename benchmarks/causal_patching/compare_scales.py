# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/compare_scales.py

Cross-scale cognitive transfer analysis.

Given two probe manifests trained with the same concept (typically
`comply_refuse`) on two different model scales (e.g., Llama-3.2-1B
vs Llama-3.2-3B), computes and reports:

  1. Best-AUC layer as an ABSOLUTE index and as a FRACTION of
     total-layers. Scale-invariance of fractional-layer would be a
     real research finding.
  2. LOO-AUC at best layer on each.
  3. Per-layer AUC curves side-by-side (normalized x-axis), so the
     curve shape can be compared.
  4. A qualitative "emergence-layer band" per model: the contiguous
     layer range where AUC > 0.7, 0.8, 0.9 — tells us when the
     concept becomes linearly separable in the network.

If α-sweep results are also available for both models, we report
behavioral efficacy at matched α (refuse@unsafe at α=3.0 etc).

Usage
-----
    python benchmarks/causal_patching/compare_scales.py \
      --manifest_a styxx/residual_probe/atlas/meta_llama_Llama_3.2_1B_Instruct_comply_refuse.json \
      --manifest_b styxx/residual_probe/atlas/meta_llama_Llama_3.2_3B_Instruct_comply_refuse.json \
      --sweep_a benchmarks/causal_patching/runs/v0/aggregate.json \
      --sweep_b benchmarks/causal_patching/runs/v0-3b/aggregate.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_manifest(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_aggregate(path: Optional[Path]) -> Optional[Dict]:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if "aggregate" in data and "per_alpha" in data["aggregate"]:
        return data["aggregate"]
    return data


def emergence_band(per_layer_auc, threshold: float) -> Tuple[int, int]:
    """Return the (earliest, latest) layer index where AUC >= threshold.
    (None, None) if no layer crosses the threshold."""
    cross = [r["layer"] for r in per_layer_auc
             if r["auc_loo"] == r["auc_loo"]   # not-NaN
             and r["auc_loo"] >= threshold]
    if not cross:
        return None, None
    return min(cross), max(cross)


def render_report(a_manifest: Dict, b_manifest: Dict,
                  a_agg: Optional[Dict], b_agg: Optional[Dict]) -> str:
    lines = []
    lines.append("# Cross-Scale Cognitive Transfer — "
                 f"{a_manifest['task']}")
    lines.append("")
    lines.append(f"- **A**: `{a_manifest['model']}` "
                 f"(hidden={a_manifest['hidden_size']}, "
                 f"layers={a_manifest['total_layers']})")
    lines.append(f"- **B**: `{b_manifest['model']}` "
                 f"(hidden={b_manifest['hidden_size']}, "
                 f"layers={b_manifest['total_layers']})")
    lines.append("")

    a_layer = a_manifest["layer"]
    b_layer = b_manifest["layer"]
    a_tot = a_manifest["total_layers"]
    b_tot = b_manifest["total_layers"]
    a_frac = a_layer / max(a_tot - 1, 1)
    b_frac = b_layer / max(b_tot - 1, 1)

    lines.append("## Best-AUC layer")
    lines.append("")
    lines.append("| model | best layer | total | fraction | AUC |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| A | {a_layer} | {a_tot} | {a_frac:.2f} | "
        f"{a_manifest.get('auc_validation', 'n/a')} |"
    )
    lines.append(
        f"| B | {b_layer} | {b_tot} | {b_frac:.2f} | "
        f"{b_manifest.get('auc_validation', 'n/a')} |"
    )
    lines.append("")

    # Fractional-layer invariance check
    if abs(a_frac - b_frac) < 0.08:
        lines.append(
            f"**Finding (Δ fraction = {abs(a_frac-b_frac):.2f}, <0.08): "
            "the best-AUC concept layer is APPROXIMATELY scale-invariant "
            "in fractional-depth terms.** The concept emerges at the same "
            "relative position in both networks."
        )
    else:
        lines.append(
            f"Δ fraction = {abs(a_frac-b_frac):.2f}. Fractional best-layer "
            "differs across scales — not strongly scale-invariant."
        )
    lines.append("")

    # Emergence bands
    lines.append("## Emergence bands — earliest layer where AUC ≥ threshold")
    lines.append("")
    lines.append("| model | AUC≥0.7 | AUC≥0.8 | AUC≥0.9 |")
    lines.append("|---|---|---|---|")
    for label, m in (("A", a_manifest), ("B", b_manifest)):
        pla = m.get("per_layer_auc") or []
        cells = []
        for thresh in (0.7, 0.8, 0.9):
            lo, hi = emergence_band(pla, thresh)
            cells.append("-" if lo is None else f"L{lo}-L{hi}")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    # Side-by-side per-layer AUCs
    lines.append("## Per-layer AUC (normalized layer fraction)")
    lines.append("")
    lines.append("| fraction | A | B |")
    lines.append("|---|---|---|")
    a_pla = {r["layer"]: r["auc_loo"] for r in (a_manifest.get("per_layer_auc") or [])}
    b_pla = {r["layer"]: r["auc_loo"] for r in (b_manifest.get("per_layer_auc") or [])}
    for i in range(11):
        frac = i / 10
        a_idx = round(frac * (a_tot - 1))
        b_idx = round(frac * (b_tot - 1))
        a_auc = a_pla.get(a_idx, float("nan"))
        b_auc = b_pla.get(b_idx, float("nan"))
        lines.append(f"| {frac:.1f} | {a_auc:.3f} (L{a_idx}) | "
                     f"{b_auc:.3f} (L{b_idx}) |")
    lines.append("")

    # Behavioral α-sweep comparison
    if a_agg and b_agg:
        lines.append("## Behavioral efficacy (α-sweep)")
        lines.append("")
        lines.append("| α | A: refuse@unsafe (comply) | "
                     "B: refuse@unsafe (comply) |")
        lines.append("|---|---|---|")
        def _get(agg, target, alpha, key):
            for row in agg.get("per_alpha", []):
                if abs(row["alpha"] - alpha) < 1e-6 and \
                   row["target_class"] == target:
                    v = row.get(key)
                    return f"{v:.2f}" if v is not None else "-"
            return "-"
        for alpha in (0.0, 1.0, 2.0, 3.0):
            a_ru = _get(a_agg, "comply", alpha, "refuse_rate_unsafe")
            b_ru = _get(b_agg, "comply", alpha, "refuse_rate_unsafe")
            lines.append(f"| {alpha:.1f} | {a_ru} | {b_ru} |")
        lines.append("")
        lines.append(
            "*Reading*: for each α, fraction of unsafe prompts that "
            "the model refused to answer. Lower = safety bypassed. "
            "Shape comparison between A and B reveals whether the "
            "concept direction transfers with constant, increasing, "
            "or decreasing α across scale."
        )
    else:
        lines.append("*(α-sweep data not provided for both models; "
                     "run run_patching.py on each for behavioral "
                     "comparison.)*")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_a", required=True)
    ap.add_argument("--manifest_b", required=True)
    ap.add_argument("--sweep_a", default=None,
                    help="aggregate.json for model A (optional)")
    ap.add_argument("--sweep_b", default=None,
                    help="aggregate.json for model B (optional)")
    ap.add_argument("--out_file", default=None)
    args = ap.parse_args()

    a = load_manifest(Path(args.manifest_a))
    b = load_manifest(Path(args.manifest_b))
    a_agg = load_aggregate(Path(args.sweep_a)) if args.sweep_a else None
    b_agg = load_aggregate(Path(args.sweep_b)) if args.sweep_b else None

    if a["task"] != b["task"]:
        raise SystemExit(f"task mismatch: {a['task']} vs {b['task']}")

    report = render_report(a, b, a_agg, b_agg)
    if args.out_file:
        Path(args.out_file).write_text(report, encoding="utf-8")
        print(f"wrote {args.out_file}")
    else:
        print(report)


if __name__ == "__main__":
    main()
