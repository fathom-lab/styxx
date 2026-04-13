# -*- coding: utf-8 -*-
"""
generate_centroids.py -- Build the styxx calibration artifact from the
Fathom atlas v0.3 captures.

This script is the bridge from 14 months of Fathom research into the
styxx product. It reads the atlas captures on disk, builds the
feature vectors at each phase boundary (token 1, token 5, token 25),
computes per-category z-score parameters and centroids, and writes
everything to a single sha256-pinned JSON file that styxx ships with
at runtime.

Tier 0 uses only closed-weight-compatible signals (entropy, logprob,
top2_margin). D-axis is excluded because closed-weight APIs don't
expose residual stream — that's tier 1 territory.

The feature vector at each phase is:
    [mean_entropy, std_entropy, min_entropy, max_entropy,
     mean_logprob, std_logprob, min_logprob, max_logprob,
     mean_top2,    std_top2,    min_top2,    max_top2]
    computed over tokens [0, phase_end_token).

Usage (one-off, run once before shipping):
    python scripts/generate_centroids.py \
        --captures C:/Users/heyzo/clawd/sae-reasoning-depth/figures/.claude/worktrees/intelligent-brown/atlas/captures \
        --out styxx/centroids/atlas_v0.3.json

Output file is committed to the package and loaded at runtime.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


CATEGORIES = [
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
]

TIER0_SIGNALS = ["entropy", "logprob", "top2_margin"]

PHASE_TOKEN_CUTOFFS = {
    "phase1_preflight":  1,   # token 0 only
    "phase2_early":      5,   # tokens 0-4
    "phase3_mid":       15,   # tokens 0-14
    "phase4_late":      25,   # tokens 0-24
}


def extract_features(probe, n_tokens):
    """Flat feature vector for tokens [0, n_tokens) over tier0 signals."""
    trajs = probe.get("trajectories", {})
    feats = []
    for s in TIER0_SIGNALS:
        t = trajs.get(s, [])
        if len(t) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        window = np.asarray(t[:n_tokens], dtype=float)
        if len(window) == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
            continue
        feats.append(float(window.mean()))
        feats.append(float(window.std(ddof=1)) if len(window) > 1 else 0.0)
        feats.append(float(window.min()))
        feats.append(float(window.max()))
    return np.array(feats, dtype=float)


def load_atlas_captures(captures_dir):
    """Return dict: model_name -> {category -> list of (feature_vec_dict, probe_id)}.

    feature_vec_dict maps phase_name -> feature array, so we compute all
    phases in one pass.
    """
    captures_dir = Path(captures_dir)
    paths = sorted([
        p for p in captures_dir.glob("*.json")
        if "gpt-4o-mini" not in p.name.lower()
    ])
    out = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            capture = json.load(f)
        model = capture.get("model", p.name)
        model_data = {}
        for cat in CATEGORIES:
            cat_items = []
            probes = capture.get("probes", {}).get(cat, {})
            for pid, probe in probes.items():
                if len(probe.get("trajectories", {}).get("entropy", [])) < 1:
                    continue
                phase_feats = {}
                for phase_name, n_tok in PHASE_TOKEN_CUTOFFS.items():
                    phase_feats[phase_name] = extract_features(probe, n_tok)
                cat_items.append((phase_feats, pid))
            model_data[cat] = cat_items
        out[model] = model_data
    return out


def compute_centroids_for_phase(data, phase_name):
    """For one phase, collect all per-category features across ALL models
    (pooled), compute z-score mu/sigma, then compute the centroid of each
    category in z-space.

    This is training-free in the sense that no per-model fit happens;
    it's a pooled cross-architecture calibration.
    """
    # Pool all features
    X = []
    y = []
    for model, by_cat in data.items():
        for cat, items in by_cat.items():
            for phase_feats, pid in items:
                X.append(phase_feats[phase_name])
                y.append(cat)
    X = np.array(X, dtype=float)

    # z-score normalization (per-feature)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma = np.where(sigma > 1e-12, sigma, 1.0)
    Z = (X - mu) / sigma

    # Per-category centroid in z-space
    centroids = {}
    n_by_cat = {}
    for cat in CATEGORIES:
        idx = [i for i, l in enumerate(y) if l == cat]
        n_by_cat[cat] = len(idx)
        if idx:
            centroids[cat] = Z[idx].mean(axis=0).tolist()
        else:
            centroids[cat] = [0.0] * X.shape[1]

    return {
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "centroids": centroids,
        "n_by_category": n_by_cat,
        "n_features": int(X.shape[1]),
        "n_samples_total": int(X.shape[0]),
    }


def generate(captures_dir, out_path):
    print("=" * 72)
    print("STYXX CENTROID GENERATION")
    print("=" * 72)
    print(f"Captures:   {captures_dir}")
    print(f"Out:        {out_path}")
    print()

    print("Loading atlas captures...")
    data = load_atlas_captures(captures_dir)
    models = sorted(data.keys())
    print(f"  {len(models)} models loaded:")
    for m in models:
        total_probes = sum(len(v) for v in data[m].values())
        print(f"    - {m:<38} ({total_probes} probes)")
    print()

    print("Computing per-phase centroids (z-scored, cross-architecture pooled)...")
    phase_centroids = {}
    for phase_name in PHASE_TOKEN_CUTOFFS:
        result = compute_centroids_for_phase(data, phase_name)
        phase_centroids[phase_name] = result
        print(
            f"  {phase_name:<20} n_samples={result['n_samples_total']:>4d} "
            f"n_features={result['n_features']:>2d}"
        )
    print()

    # Build the final artifact
    artifact = {
        "schema_version": "0.1.0",
        "name": "atlas_v0.3",
        "description": (
            "Cross-architecture cognitive state centroids for styxx tier 0. "
            "Built from Fathom atlas v0.3 captures on 12 open-weight models "
            "across 3 architecture families (Gemma, Llama, Qwen), base and "
            "instruct variants. Feature vector: (mean, std, min, max) per "
            "signal in [entropy, logprob, top2_margin] over tokens [0, t) "
            "for each of four phase windows."
        ),
        "categories": CATEGORIES,
        "tier0_signals": TIER0_SIGNALS,
        "phase_token_cutoffs": PHASE_TOKEN_CUTOFFS,
        "feature_vector_layout": [
            f"{stat}_{sig}"
            for sig in TIER0_SIGNALS
            for stat in ("mean", "std", "min", "max")
        ],
        "models": models,
        "n_models": len(models),
        "phases": phase_centroids,
        "atlas_source": "Fathom Cognitive Atlas v0.3 (concept DOI 10.5281/zenodo.19502715)",
        "research_repo": "https://github.com/fathom-lab/fathom",
        "product_repo": "https://github.com/fathom-lab/styxx",
        "honest_specs": {
            "phase1_preflight_adversarial_acc": 0.52,
            "phase1_preflight_reasoning_acc":   0.43,
            "phase1_preflight_creative_acc":    0.41,
            "phase4_late_hallucination_acc":    0.52,
            "phase4_late_reasoning_acc":        0.69,
            "chance_6class":                    0.167,
            "methodology": "leave-one-model-out cross-validation, nearest-centroid classifier, z-scored features",
            "caveat_pre_flight": "confidence gating does not improve accuracy above baseline",
            "caveat_hallucination": "hallucination signal peaks at t=25, not at t=0",
            "caveat_adversarial": "adversarial signal peaks at t=1 and degrades fast",
        },
    }

    # Write artifact
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, default=float)

    # Compute sha256 for reproducibility pinning
    h = hashlib.sha256()
    with open(out_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    sha = h.hexdigest()
    size = out_path.stat().st_size

    print(f"Wrote artifact:  {out_path}")
    print(f"  size:          {size:,} bytes")
    print(f"  sha256:        {sha}")
    print()
    print("Pin this sha in styxx/vitals.py as EXPECTED_CENTROIDS_SHA256")
    print("so any corruption or tampering is caught at package import time.")
    return sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--captures",
        default=(
            "C:/Users/heyzo/clawd/sae-reasoning-depth/figures/.claude/"
            "worktrees/intelligent-brown/atlas/captures"
        ),
    )
    ap.add_argument(
        "--out",
        default="styxx_staging/styxx/centroids/atlas_v0.3.json",
    )
    args = ap.parse_args()
    generate(args.captures, args.out)


if __name__ == "__main__":
    main()
