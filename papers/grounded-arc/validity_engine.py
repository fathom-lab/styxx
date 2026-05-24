# -*- coding: utf-8 -*-
"""
Grounded-arc validity engine — the compute that fills in §3 of
run_bet0_phase1.py. Built 2026-05-24.

This is the embedding -> distance -> validity pipeline for Bet 0's H1
(validity predicts instrument reliability). It is written and self-tested
HERE, on synthetic + real-embedding data, so the confirmatory kill-gate
runs clean the instant the operator locks decisions and constructs the
holdout. NOTHING in this file touches the pre-registered holdout corpora
(they are not yet constructed, per preregistration §10).

Metric alignment (the one non-obvious bit):
  The threshold-law paper (Zenodo 10.5281/zenodo.20278945) measures
  OVERLAP = mean_e max_c cos(e, c) — cosine SIMILARITY, higher = more
  in-distribution — with inflection τ_overlap ≈ 0.31. The per-prompt
  validity in the pre-registration uses min k-NN DISTANCE (lower = more
  in-distribution). Same axis, inverted:  d_cos = 1 − cos_sim, so
  τ_distance = 1 − τ_overlap ≈ 0.69 in cosine-distance units. Then
  validity = sigmoid(α·(τ_distance − d)) is HIGH in-distribution, as
  intended. This alignment is asserted against the published τ in
  Phase-1 curve replication; it is NOT a free parameter.
"""
from __future__ import annotations

import math
from typing import List, Sequence

# Published threshold-law inflection (cosine similarity / overlap units).
TAU_OVERLAP = 0.31
# Same inflection expressed in cosine-distance units for the validity sigmoid.
TAU_DISTANCE = 1.0 - TAU_OVERLAP  # 0.69


def embed(texts: Sequence[str], model: str) -> List[List[float]]:
    """Embed texts with the locked embedding model and unit-normalize so a
    dot product equals cosine similarity.

    Dispatch:
      - 'text-embedding-3-*'  -> OpenAI embeddings API
      - any other id          -> local sentence-transformers (e.g. BAAI/bge-large)
    """
    texts = [t.replace("\n", " ") for t in texts]
    if model.startswith("text-embedding-3"):
        from openai import OpenAI

        client = OpenAI()
        vecs: List[List[float]] = []
        for i in range(0, len(texts), 256):  # batch under request limits
            resp = client.embeddings.create(model=model, input=texts[i : i + 256])
            vecs.extend(d.embedding for d in resp.data)
    else:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(model)
        vecs = [list(map(float, v)) for v in st.encode(list(texts))]

    normed: List[List[float]] = []
    for v in vecs:
        n = math.sqrt(sum(x * x for x in v)) or 1e-12
        normed.append([x / n for x in v])
    return normed


def knn_distance(query: Sequence[float], corpus: Sequence[Sequence[float]], k: int = 10) -> float:
    """Mean cosine DISTANCE (1 − cos) from query to its k nearest neighbours
    in the calibration corpus. Lower = more in-distribution. Vectors assumed
    unit-normalized (dot == cosine)."""
    sims = sorted((sum(q * c for q, c in zip(query, cv)) for cv in corpus), reverse=True)
    top = sims[: max(1, min(k, len(sims)))]
    return 1.0 - (sum(top) / len(top))


def validity(distance: float, *, tau: float = TAU_DISTANCE, alpha: float) -> float:
    """Per-call validity in [0,1] from prompt-to-calibration distance.
    Locked sigmoid form from preregistration §6. α is fit on the validation
    slice in Phase 1 — never on the holdout."""
    return 1.0 / (1.0 + math.exp(-alpha * (tau - distance)))


def self_test(model: str = "text-embedding-3-large") -> int:
    """Validate the full compute pipeline on SYNTHETIC + real-embedding data
    with known ground truth. Proves correctness without touching any
    pre-registered holdout. Returns 0 on PASS, 1 on FAIL.

    Three checks:
      (1) directional — in-distribution prompts get higher validity than
          out-of-distribution prompts (the mechanism works at all);
      (2) recovery — Spearman ρ + permutation null recover a planted
          validity↔(−error) correlation;
      (3) null control — a shuffled pairing yields ρ≈0.
    """
    import random

    # stats live in the scaffold; load by file path (scripts/dogfood isn't a
    # package) using the same pattern the repo's tests use.
    import importlib.util
    import sys as _sys
    from pathlib import Path

    _scaffold = Path(__file__).resolve().parents[2] / "scripts" / "dogfood" / "run_bet0_phase1.py"
    _spec = importlib.util.spec_from_file_location("run_bet0_phase1", _scaffold)
    _h = importlib.util.module_from_spec(_spec)
    _sys.modules["run_bet0_phase1"] = _h  # register so @dataclass can resolve annotations
    _spec.loader.exec_module(_h)  # type: ignore[union-attr]
    spearman_rho, permutation_p = _h.spearman_rho, _h.permutation_p

    print(f"── grounded-arc validity-engine self-test (model={model}) ──")
    print("   (no pre-registered holdout is touched)\n")

    # (1) directional validity on a tiny synthetic calibration corpus
    cal = [
        "how do I bake sourdough bread at home",
        "best flour to use for pizza dough",
        "proofing times for a rye loaf",
        "kneading technique for baguettes",
        "hydration ratio for ciabatta dough",
        "scoring patterns on a boule before baking",
        "sourdough starter feeding schedule",
        "using oven steam for a crisp crust",
    ]
    in_dist = ["what temperature should I bake a baguette at", "how to revive a neglected sourdough starter"]
    ood = ["how do I file quarterly estimated taxes", "explain the offside rule in football"]

    e_cal = embed(cal, model)
    e_in = embed(in_dist, model)
    e_ood = embed(ood, model)
    alpha = 8.0  # demo slope; real α is fit on the validation slice in Phase 1

    v_in = [validity(knn_distance(q, e_cal, k=4), alpha=alpha) for q in e_in]
    v_ood = [validity(knn_distance(q, e_cal, k=4), alpha=alpha) for q in e_ood]
    print(f"   in-distribution validity:     {[round(x, 3) for x in v_in]}")
    print(f"   out-of-distribution validity: {[round(x, 3) for x in v_ood]}")
    ok_dir = min(v_in) > max(v_ood)
    print(f"   [1] directional (in-dist > OOD): {'PASS' if ok_dir else 'FAIL'}\n")

    # (2) planted-correlation recovery + (3) null control
    rng = random.Random(0)
    val = [rng.random() for _ in range(150)]
    neg_err = [v + rng.gauss(0, 0.35) for v in val]  # validity ~ −error, with noise
    rho = spearman_rho(val, neg_err)
    p = permutation_p(val, neg_err, n_permutations=2000, rng_seed=1)
    ok_rec = rho > 0.40 and p < 0.01
    print(f"   [2] planted-correlation recovery: ρ={rho:+.3f}  p={p:.4f}  "
          f"{'PASS' if ok_rec else 'FAIL'}")

    shuf = list(neg_err)
    rng.shuffle(shuf)
    rho0 = spearman_rho(val, shuf)
    ok_null = abs(rho0) < 0.30
    print(f"   [3] null control (shuffled):      ρ={rho0:+.3f}  "
          f"{'PASS' if ok_null else 'FAIL'}")

    passed = ok_dir and ok_rec and ok_null
    print(f"\n   SELF-TEST: {'PASS — pipeline correct, kill-gate is runnable' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    import sys

    raise SystemExit(self_test(sys.argv[1] if len(sys.argv) > 1 else "text-embedding-3-large"))
