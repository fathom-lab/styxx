# -*- coding: utf-8 -*-
"""
benchmarks/causal_patching/measure_probe_geometry.py

Probe-direction geometry: pairwise angles between trained concept
probes in the residual stream of the same model.

Motivation
----------
A single linear probe for "refuse vs comply" has an AUC. Fine — it's
a classifier. But when we train probes for multiple concepts on the
same model (refuse, sycophant, confabulate, deceive, ...), a new
question opens up: *how are these directions arranged in the residual
stream?*

Two extreme hypotheses:
  1. **Orthogonal concepts (modular hypothesis)**: cos(refuse, sycophant) ≈ 0.
     Concept directions are independent basis vectors. A steering
     intervention on one does not bleed into the others. The residual
     stream factorizes cleanly.
  2. **Single misalignment axis (collapse hypothesis)**: cos ≈ 1.
     The "concepts" we named (refuse, sycophant, ...) are all
     readouts of one underlying "model-will-not-help" direction.
     Interventions are degenerate — steering refusal is the same
     move as steering sycophancy.

Either result is a finding worth publishing. Real models likely land
somewhere in between, giving a measured *concept geometry* that tells
us how decomposable in-context behavior is.

Method
------
1. Load N trained InterveneProbe checkpoints that share a model +
   hidden_size.
2. Unit-normalize each probe direction (weight vector after L2
   logistic regression training).
3. Compute pairwise cosine similarity and angular separation.
4. Optionally align probe directions to their "positive class
   pushes-toward" polarity (sign convention sanity check).
5. Report the matrix + a compact readout.

Usage
-----
  python benchmarks/causal_patching/measure_probe_geometry.py \
    --probes styxx/residual_probe/atlas/*.json \
    --out_file benchmarks/causal_patching/runs/geometry.json

Output
------
JSON with:
  - probes: [{task, model, layer, positive_class, auc, filepath}, ...]
  - cosine_matrix: N×N float matrix
  - angle_matrix_deg: N×N float matrix, same order
  - interpretation: one-line per pair
  - model_match: True if all probes were trained on the same model
  - layer_match: True if all probes sit on the same layer
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_probe_record(manifest_path: Path,
                      at_layer: Optional[int] = None) -> Dict:
    """Read a probe manifest + sibling .pt weight file.

    If at_layer is given AND the probe was trained with v1 artifact
    (has 'weight_per_layer'), return that layer's weight instead of
    the best-AUC layer's weight. This enables same-layer geometry
    comparison across concepts that picked different best layers.
    """
    import torch

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weight_path = manifest_path.parent / manifest["weight_file"]
    state = torch.load(weight_path, map_location="cpu", weights_only=True)

    resolved_layer = manifest["layer"]
    if at_layer is not None and "weight_per_layer" in state:
        per_layer = state["weight_per_layer"]
        key = str(at_layer)
        if key in per_layer:
            weight = per_layer[key].to(torch.float32)
            resolved_layer = at_layer
        else:
            weight = state["weight"].to(torch.float32)
    else:
        weight = state["weight"].to(torch.float32)

    return {
        "task": manifest["task"],
        "model": manifest["model"],
        "layer": resolved_layer,
        "positive_class": manifest["positive_class"],
        "negative_class": manifest["negative_class"],
        "auc_validation": manifest.get("auc_validation"),
        "hidden_size": manifest["hidden_size"],
        "weight": weight,
        "weight_norm": float(weight.norm()),
        "manifest_path": str(manifest_path),
    }


def cosine(a, b) -> float:
    import torch
    na = a.norm()
    nb = b.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float(torch.dot(a, b) / (na * nb))


def classify_angle(cos_sim: float) -> str:
    """Short interpretive tag for a pairwise angle."""
    if math.isnan(cos_sim):
        return "undefined"
    abs_c = abs(cos_sim)
    if abs_c >= 0.90:
        return "near-parallel (same direction)" if cos_sim > 0 \
            else "near-antiparallel (opposite direction)"
    if abs_c >= 0.60:
        return "strongly correlated" if cos_sim > 0 \
            else "strongly anticorrelated"
    if abs_c >= 0.30:
        return "moderately correlated" if cos_sim > 0 \
            else "moderately anticorrelated"
    if abs_c >= 0.10:
        return "weakly correlated"
    return "approximately orthogonal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+", required=True,
                    help="Paths (or globs) to probe manifest .json files")
    ap.add_argument("--out_file", default=None,
                    help="Optional output JSON path")
    ap.add_argument("--at_layer", type=int, default=None,
                    help="Force same-layer comparison using per-layer "
                         "weights (requires probe_version>=v1). If omitted, "
                         "each probe uses its own best-AUC layer.")
    args = ap.parse_args()

    # Expand globs
    manifest_paths: List[Path] = []
    for p in args.probes:
        matched = glob.glob(p)
        if matched:
            manifest_paths.extend(Path(m) for m in matched)
        elif Path(p).exists():
            manifest_paths.append(Path(p))
    manifest_paths = sorted(set(manifest_paths))

    if len(manifest_paths) < 2:
        print("FATAL: need at least 2 probe manifests for geometry analysis",
              file=sys.stderr)
        sys.exit(2)

    records = [load_probe_record(p, at_layer=args.at_layer)
               for p in manifest_paths]

    model_match = len({r["model"] for r in records}) == 1
    layer_match = len({r["layer"] for r in records}) == 1
    hidden_match = len({r["hidden_size"] for r in records}) == 1

    if not hidden_match:
        print("FATAL: probes have mismatched hidden_size - cannot compare",
              file=sys.stderr)
        sys.exit(2)
    if not model_match:
        print("WARNING: probes were trained on different models; geometry "
              "comparison only meaningful across same-model probes.",
              file=sys.stderr)
    if not layer_match:
        print("WARNING: probes were trained on different layers; geometry "
              "readings cross-layer are suggestive but not definitive.",
              file=sys.stderr)

    N = len(records)
    cos_matrix: List[List[float]] = [[1.0] * N for _ in range(N)]
    ang_matrix: List[List[float]] = [[0.0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            c = cosine(records[i]["weight"], records[j]["weight"])
            cos_matrix[i][j] = cos_matrix[j][i] = c
            ang = math.degrees(math.acos(max(min(c, 1.0), -1.0)))
            ang_matrix[i][j] = ang_matrix[j][i] = ang

    tasks = [r["task"] for r in records]
    print(f"\n=== Probe Geometry ({N} probes) ===")
    print(f"model_match={model_match} layer_match={layer_match}\n")
    print(f"tasks: {tasks}\n")

    print("Cosine matrix:")
    header = "            " + "  ".join(f"{t[:10]:>10s}" for t in tasks)
    print(header)
    for i, t in enumerate(tasks):
        row = "  ".join(f"{cos_matrix[i][j]:>+10.3f}" for j in range(N))
        print(f"  {t[:10]:>10s}  {row}")

    print("\nAngle (degrees):")
    print(header)
    for i, t in enumerate(tasks):
        row = "  ".join(f"{ang_matrix[i][j]:>10.1f}" for j in range(N))
        print(f"  {t[:10]:>10s}  {row}")

    print("\nPairwise interpretation:")
    interps: List[Dict] = []
    for i in range(N):
        for j in range(i + 1, N):
            interp = classify_angle(cos_matrix[i][j])
            print(f"  {tasks[i]} <-> {tasks[j]}: "
                  f"cos={cos_matrix[i][j]:+.3f} "
                  f"angle={ang_matrix[i][j]:.1f}° — {interp}")
            interps.append({
                "a": tasks[i], "b": tasks[j],
                "cosine": cos_matrix[i][j],
                "angle_deg": ang_matrix[i][j],
                "interpretation": interp,
            })

    out = {
        "probes": [
            {k: v for k, v in r.items() if k != "weight"}
            for r in records
        ],
        "cosine_matrix": cos_matrix,
        "angle_matrix_deg": ang_matrix,
        "interpretations": interps,
        "model_match": model_match,
        "layer_match": layer_match,
        "hidden_size": records[0]["hidden_size"],
    }

    if args.out_file:
        out_path = Path(args.out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
