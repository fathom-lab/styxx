"""Null-heuristic baselines for tool-call drift detection — Day 1 floor.

Evaluates five cheap heuristics on the drift_v0 dataset to establish
where the bar sits before building the calibrated detector. If any
heuristic hits AUC > 0.90, we kill the project (trivially solved).
If all are < 0.70, we need the calibrated detector (expected).

Heuristics:
  1. Uniform random (expected AUC 0.50)
  2. Exact tool-name match (tool_called in tools_mentioned_in_prompt)
  3. JSON-schema conformance only (required args present, type matches)
  4. Verbatim arg-value rate (fraction of arg values appearing in prompt)
  5. Embedding cosine (sentence-transformers all-MiniLM-L6-v2)

Report: per-heuristic AUC + per-source stratified AUC.

Output: benchmarks/drift_null_baselines_v0.json
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
DATA_PATH = REPO / "data" / "drift_v0" / "drift_dataset_v0.jsonl"
OUT_PATH = REPO / "benchmarks" / "drift_null_baselines_v0.json"

random.seed(42)
np.random.seed(42)


def load_dataset() -> List[Dict]:
    rows = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------
# Heuristic 1: uniform random
# --------------------------------------------------------------

def h_random(rows) -> np.ndarray:
    return np.random.rand(len(rows))


# --------------------------------------------------------------
# Heuristic 2: exact tool-name match
#
# Score = 1 (high drift) if the called tool name does NOT appear in
#         the prompt text (model invented a tool).
# Score = 0 (low drift)  if the called tool name appears verbatim in
#         the prompt text.
# Also count "the tool was in the available schema" — if the model
# called a tool NOT in the provided schemas, that's maximum drift.
# --------------------------------------------------------------

def h_exact_tool_match(rows) -> np.ndarray:
    scores = []
    for r in rows:
        call_name = (r["tool_call"].get("name") or "").lower()
        prompt = (r["prompt"] or "").lower()
        available = [f.get("name", "").lower() for f in r.get("functions", [])]
        # Score 1.0 if tool is not in available schema
        if call_name not in available:
            scores.append(1.0)
            continue
        # Score 0.5 if present in schema but not named in prompt
        if call_name in prompt:
            scores.append(0.0)
        else:
            scores.append(0.5)
    return np.array(scores)


# --------------------------------------------------------------
# Heuristic 3: JSON schema conformance
#
# Score = fraction of required args missing + fraction of spurious args
# Higher = more drift
# --------------------------------------------------------------

def h_schema_conformance(rows) -> np.ndarray:
    scores = []
    for r in rows:
        call_name = r["tool_call"].get("name") or ""
        call_args = r["tool_call"].get("arguments") or {}
        # Find the matching function schema
        schema = next(
            (f for f in r.get("functions", []) if f.get("name") == call_name),
            None,
        )
        if schema is None:
            # Can't even find the schema — max drift
            scores.append(1.0)
            continue
        props = schema.get("parameters", {}).get("properties", {}) or {}
        required = schema.get("parameters", {}).get("required", []) or []

        call_arg_names = set(call_args.keys())
        spec_arg_names = set(props.keys())

        missing_required = [a for a in required if a not in call_arg_names]
        spurious = call_arg_names - spec_arg_names

        n_req = max(1, len(required))
        n_spec = max(1, len(spec_arg_names))

        miss_frac = len(missing_required) / n_req
        spur_frac = len(spurious) / n_spec

        scores.append(0.5 * miss_frac + 0.5 * spur_frac)
    return np.array(scores)


# --------------------------------------------------------------
# Heuristic 4: verbatim arg-value rate
#
# Score = 1 - fraction_of_arg_values_appearing_in_prompt
# High score = args don't come from prompt = likely drift
# --------------------------------------------------------------

def h_verbatim_arg_rate(rows) -> np.ndarray:
    scores = []
    for r in rows:
        args = r["tool_call"].get("arguments") or {}
        prompt = (r["prompt"] or "").lower()
        if not args:
            scores.append(0.5)
            continue
        matches = 0
        total = 0
        for v in args.values():
            total += 1
            sv = str(v).lower()
            if sv and sv in prompt:
                matches += 1
        if total == 0:
            scores.append(0.5)
        else:
            # Fraction NOT found = drift signal
            scores.append(1.0 - (matches / total))
    return np.array(scores)


# --------------------------------------------------------------
# Heuristic 5: composite (exact + schema + verbatim, unweighted)
# --------------------------------------------------------------

def h_composite(rows, h2, h3, h4) -> np.ndarray:
    # Simple mean — baseline for what a calibrated LR might do
    return (h2 + h3 + h4) / 3.0


# --------------------------------------------------------------
# Metrics
# --------------------------------------------------------------

def auc_by_slice(scores, labels, slice_mask):
    mask = np.asarray(slice_mask, dtype=bool)
    sub_labels = labels[mask]
    sub_scores = scores[mask]
    if len(set(sub_labels)) < 2:
        return None
    return roc_auc_score(sub_labels, sub_scores)


def main():
    rows = load_dataset()
    n = len(rows)
    labels = np.array([r["drift"] for r in rows])
    sources = np.array([r["source"] for r in rows])
    drift_types = np.array([r["drift_type"] for r in rows])

    print(f"loaded {n} samples")
    print(f"  drift balance: {int((1-labels).sum())} no-drift / {int(labels.sum())} drift")

    heuristics = {}
    print("\n--- running heuristics ---")
    h1 = h_random(rows);                heuristics["random"] = h1
    print(f"  h1 random         done")
    h2 = h_exact_tool_match(rows);      heuristics["exact_tool_match"] = h2
    print(f"  h2 exact_tool     done")
    h3 = h_schema_conformance(rows);    heuristics["schema_conformance"] = h3
    print(f"  h3 schema         done")
    h4 = h_verbatim_arg_rate(rows);     heuristics["verbatim_arg_rate"] = h4
    print(f"  h4 verbatim_args  done")
    h5 = h_composite(rows, h2, h3, h4); heuristics["composite_unweighted"] = h5
    print(f"  h5 composite      done")

    # --- Overall AUCs
    print(f"\n{'heuristic':<30s}  {'overall AUC':>13s}")
    print("-" * 50)
    overall_aucs = {}
    for name, scores in heuristics.items():
        auc = roc_auc_score(labels, scores)
        overall_aucs[name] = float(auc)
        bar = "#" * int(auc * 40)
        print(f"  {name:<28s}  {auc:.4f}  [{bar:<40s}]")

    # --- Stratified by source
    print(f"\n--- stratified by source ---")
    sources_list = sorted(set(sources))
    source_aucs: Dict[str, Dict[str, float]] = {}
    for src in sources_list:
        mask = sources == src
        n_pos = int(labels[mask].sum())
        n_neg = int((1 - labels[mask]).sum())
        print(f"\n  {src}  (n={int(mask.sum())}, +{n_pos}/-{n_neg})")
        source_aucs[src] = {}
        for name, scores in heuristics.items():
            a = auc_by_slice(scores, labels, mask)
            source_aucs[src][name] = float(a) if a is not None else None
            if a is not None:
                print(f"    {name:<28s}  {a:.4f}")
            else:
                print(f"    {name:<28s}  n/a (single class)")

    # --- Stratified by drift_type
    print(f"\n--- stratified by drift_type (positives only + all golds) ---")
    type_aucs: Dict[str, Dict[str, float]] = {}
    gold_mask = drift_types == "gold"
    for dt in sorted(set(drift_types)):
        if dt == "gold":
            continue
        # Eval this drift type vs gold negatives
        mask = (drift_types == dt) | gold_mask
        type_aucs[dt] = {}
        for name, scores in heuristics.items():
            a = auc_by_slice(scores, labels, mask)
            type_aucs[dt][name] = float(a) if a is not None else None
        print(f"  {dt:<22s}: " + ", ".join(
            f"{n}={type_aucs[dt][n]:.3f}" if type_aucs[dt][n] is not None else f"{n}=n/a"
            for n in ("exact_tool_match", "schema_conformance", "verbatim_arg_rate", "composite_unweighted")
        ))

    # --- Write results
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "methodology": "Day 1 null-heuristic baselines for drift_v0 dataset (BFCL v3, mutation-based)",
        "n_samples": n,
        "n_drift": int(labels.sum()),
        "n_no_drift": int((1 - labels).sum()),
        "overall_auc": overall_aucs,
        "per_source_auc": source_aucs,
        "per_drift_type_auc": type_aucs,
        "sources": sources_list,
        "drift_types": sorted(set(drift_types)),
    }
    OUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote -> {OUT_PATH.relative_to(REPO)}")

    # --- Kill-criterion check
    best_non_random = max(
        (auc for name, auc in overall_aucs.items() if name != "random"),
    )
    print()
    if best_non_random >= 0.90:
        print(f"  !! KILL-CRITERION: best null heuristic AUC {best_non_random:.3f} >= 0.90")
        print(f"     Task is trivially solved by text matching. No calibrated detector needed.")
    else:
        print(f"  GO: best null heuristic AUC {best_non_random:.3f} < 0.90")
        print(f"     Room for calibrated detector to contribute. Proceed to Day 2.")


if __name__ == "__main__":
    main()
