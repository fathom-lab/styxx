# -*- coding: utf-8 -*-
"""Fill the paper template at papers/cognitive-instruction-set-v0.md
with concrete numbers from the run artifacts produced by the
reproduction pipeline.

Pulls from:
  - styxx/residual_probe/atlas/<slug>_comply_refuse.json       (manifest)
  - benchmarks/causal_patching/runs/v0/aggregate.json          (α-sweep)
  - benchmarks/causal_patching/runs/v0/geometry.json           (optional)

Writes:
  - papers/cognitive-instruction-set-v0-filled.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
PAPER_TEMPLATE = ROOT / "papers" / "cognitive-instruction-set-v0.md"
PAPER_FILLED = ROOT / "papers" / "cognitive-instruction-set-v0-filled.md"


def _find_manifest(atlas_dir: Path, task: str) -> Optional[Path]:
    for fp in atlas_dir.glob("*.json"):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("task") == task:
            return fp
    return None


def _rate_for(agg: List[dict], alpha: float, target: str,
              key: str) -> str:
    for row in agg:
        if abs(row["alpha"] - alpha) < 1e-6 and row["target_class"] == target:
            v = row.get(key)
            return f"{v:.2f}" if v is not None else "-"
    return "-"


def render_filled(manifest: dict, aggregate: dict,
                  geometry: Optional[dict]) -> str:
    # The raw file is {"manifest": {...}, "aggregate": {"per_alpha": [...]}}.
    # Defensively accept both shapes.
    if "per_alpha" in aggregate:
        per_alpha = aggregate["per_alpha"]
    elif "aggregate" in aggregate and "per_alpha" in aggregate["aggregate"]:
        per_alpha = aggregate["aggregate"]["per_alpha"]
    else:
        per_alpha = []

    # --- 3.1 table substitutions ---
    substitutions = {
        "{{LAYER}}": str(manifest.get("layer", "?")),
        "{{AUC}}":   f"{manifest.get('auc_validation', float('nan')):.3f}",
        "{{N}}":     str(manifest.get("training_n", "?")),
        "{{CLASS_BALANCE}}": str(manifest.get("class_balance", "?")),
    }

    for alpha in (0.0, 1.0, 2.0, 3.0):
        tag = f"{int(alpha)}"
        substitutions[f"{{{{PF_{tag}}}}}"] = _rate_for(
            per_alpha, alpha, "refuse", "probe_flip_rate")
        substitutions[f"{{{{RU_R_{tag}}}}}"] = _rate_for(
            per_alpha, alpha, "refuse", "refuse_rate_unsafe")
        substitutions[f"{{{{RU_C_{tag}}}}}"] = _rate_for(
            per_alpha, alpha, "comply", "refuse_rate_unsafe")
        substitutions[f"{{{{RS_R_{tag}}}}}"] = _rate_for(
            per_alpha, alpha, "refuse", "refuse_rate_safe")

    # --- 3.2 geometry ---
    if geometry:
        probes = geometry.get("probes", [])
        cos_matrix = geometry.get("cosine_matrix", [])
        interps = geometry.get("interpretations", [])

        tasks = [p["task"] for p in probes]
        lines = ["| probe pair | cosine | angle | interpretation |",
                 "|---|---|---|---|"]
        for ip in interps:
            lines.append(
                f"| {ip['a']} ↔ {ip['b']} | {ip['cosine']:+.3f} | "
                f"{ip['angle_deg']:.1f}° | {ip['interpretation']} |"
            )
        substitutions["{{GEOMETRY_MATRIX}}"] = "\n".join(lines)

        extremes = [ip for ip in interps
                    if abs(ip["cosine"]) > 0.5 or abs(ip["cosine"]) < 0.1]
        if extremes:
            top = sorted(interps, key=lambda r: -abs(r["cosine"]))[0]
            claim = (
                f"The strongest signal is the {top['a']}↔{top['b']} pair "
                f"with cos={top['cosine']:+.3f} ({top['interpretation']}). "
                f"Across all pairs the residual stream shows "
                f"{'modular / near-orthogonal' if top['cosine'] < 0.3 else 'partially entangled'} "
                "concept directions."
            )
        else:
            claim = ("Pairwise cosines lie in a middling range; concept "
                     "directions are partially correlated but not "
                     "collapsed onto a single axis.")
        substitutions["{{GEOMETRY_INTERP}}"] = claim
    else:
        substitutions["{{GEOMETRY_MATRIX}}"] = (
            "*(geometry.json not found — run "
            "`measure_probe_geometry.py` to populate this section)*"
        )
        substitutions["{{GEOMETRY_INTERP}}"] = "(pending)"

    substitutions["{{HALT_TOKEN}}"] = "*(from demo run)*"

    tmpl = PAPER_TEMPLATE.read_text(encoding="utf-8")
    for placeholder, value in substitutions.items():
        tmpl = tmpl.replace(placeholder, str(value))
    return tmpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas",
                    default=str(ROOT / "styxx" / "residual_probe" / "atlas"))
    ap.add_argument("--sweep",
                    default=str(ROOT / "benchmarks" / "causal_patching"
                                / "runs" / "v0"))
    ap.add_argument("--refusal_task", default="comply_refuse")
    ap.add_argument("--out_file", default=str(PAPER_FILLED))
    args = ap.parse_args()

    atlas_dir = Path(args.atlas)
    sweep_dir = Path(args.sweep)

    manifest_path = _find_manifest(atlas_dir, args.refusal_task)
    if not manifest_path:
        raise SystemExit(f"no manifest for task {args.refusal_task}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    agg_path = sweep_dir / "aggregate.json"
    if not agg_path.exists():
        raise SystemExit(f"no aggregate.json at {agg_path}")
    aggregate = json.loads(agg_path.read_text(encoding="utf-8"))

    geom_path = sweep_dir / "geometry.json"
    geometry = (json.loads(geom_path.read_text(encoding="utf-8"))
                if geom_path.exists() else None)

    filled = render_filled(manifest, aggregate, geometry)
    out_path = Path(args.out_file)
    out_path.write_text(filled, encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
