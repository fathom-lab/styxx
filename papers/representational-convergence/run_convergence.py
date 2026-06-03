# -*- coding: utf-8 -*-
"""
run_convergence.py — Is representational convergence concept-general?

Six open-weight families × four concepts (comply_refuse, deception, corrigibility,
truthfulness), each family's probe scored on the same 48 concept-balanced prompts.
Measures WITHIN-concept cross-family agreement against a CROSS-concept null, with a
validity check, orientation handling, bootstrap CIs, and a shuffle sanity.

Frozen by PREREG_convergence_2026_06_02.md. Loads one model at a time (single-GPU
safe). Do NOT run concurrently with another GPU probe.
"""
from __future__ import annotations

import gc, hashlib, json, os, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from styxx.residual_probe import StyxxProbe, ProbeNotAvailable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from convergence_eval_set import get_convergence_eval

FAMILIES = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct",
            "microsoft/Phi-3.5-mini-instruct", "google/gemma-2-2b-it"]
CONCEPTS = ["comply_refuse", "deception", "corrigibility", "truthfulness"]

# ── frozen gate params ──
VALID_DISC = 0.60          # probe valid iff max(auc, 1-auc) >= this on its own concept
TESTABLE_MIN = 4           # concept testable iff >= this many valid families
R_WITHIN_MIN = 0.50
DELTA_MIN = 0.20
CI_LOWER_MIN = 0.35        # bootstrap lower 95% bound of R_within
SURVIVE_CONCEPTS = 3       # convergence concept-general iff >= this many concepts convergent
B_BOOT = 1000
SEED = 0
EXPECTED_HASH = "6154b17228fc9154"


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def disc_auc(scores, labels):
    """Mann-Whitney AUC of scores vs binary labels (polarity 1 = concept present)."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    c = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return c / (len(pos) * len(neg))


def is_cached(mid):
    hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    return (hub / ("models--" + mid.replace("/", "--"))).exists()


def main():
    rows = get_convergence_eval()
    h = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert h[:16] == EXPECTED_HASH, f"hash mismatch {h[:16]}"
    prompts = [r[3] for r in rows]
    n = len(rows)
    cidx = {c: [i for i, r in enumerate(rows) if r[1] == c] for c in CONCEPTS}
    cpol = {c: [rows[i][2] for i in cidx[c]] for c in CONCEPTS}
    print(f"convergence: n={n} hash={h[:16]} families={len(FAMILIES)} concepts={CONCEPTS}", flush=True)

    # scores[mid][concept] = oriented p_positive vector (len 48); validity[(mid,c)] = {...}
    scores, validity = {}, {}
    for mid in FAMILIES:
        if not is_cached(mid):
            print(f"{mid}: SKIP (uncached)"); continue
        tok = AutoTokenizer.from_pretrained(mid)
        mdl = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.bfloat16, device_map="auto", output_hidden_states=True).eval()
        device = next(mdl.parameters()).device
        scores[mid] = {}
        for c in CONCEPTS:
            try:
                probe = StyxxProbe.from_pretrained(model=mid, task=c)
            except ProbeNotAvailable:
                continue
            probe.weight = probe.weight.to(device=device, dtype=torch.bfloat16)
            ps = np.array([probe.predict_before_generation(mdl, tok, p).p_positive for p in prompts])
            a = disc_auc(ps[cidx[c]], cpol[c])
            disc = max(a, 1 - a) if not np.isnan(a) else float("nan")
            oriented = ps if (np.isnan(a) or a >= 0.5) else -ps     # align high = concept present
            scores[mid][c] = oriented
            validity[(mid, c)] = {"auc": round(float(a), 4), "disc": round(float(disc), 4),
                                  "valid": bool(disc >= VALID_DISC), "flipped": bool(not np.isnan(a) and a < 0.5)}
            print(f"  {mid.split('/')[-1]:24s} {c:14s} disc={disc:.3f} valid={disc>=VALID_DISC}", flush=True)
            del probe
        del mdl, tok; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()

    fams = [m for m in FAMILIES if m in scores]
    valid_fams = {c: [m for m in fams if validity.get((m, c), {}).get("valid")] for c in CONCEPTS}

    def within(c, idx=None):
        vf = valid_fams[c]; rs = []
        for i in range(len(vf)):
            for j in range(i + 1, len(vf)):
                xi, xj = scores[vf[i]][c], scores[vf[j]][c]
                if idx is not None:
                    xi, xj = xi[idx], xj[idx]
                rs.append(pearson(xi, xj))
        return (float(np.mean(rs)) if rs else float("nan")), rs

    def cross(c):
        rs = []
        for cp in CONCEPTS:
            if cp == c:
                continue
            for mi in valid_fams[c]:
                for mj in valid_fams[cp]:
                    if mi != mj:
                        rs.append(pearson(scores[mi][c], scores[mj][cp]))
        return float(np.mean(rs)) if rs else float("nan")

    rng = np.random.default_rng(SEED)
    per_concept = {}
    for c in CONCEPTS:
        vf = valid_fams[c]
        testable = len(vf) >= TESTABLE_MIN
        R_within, _ = within(c)
        R_cross = cross(c)
        delta = (R_within - R_cross) if not (np.isnan(R_within) or np.isnan(R_cross)) else float("nan")
        if testable:
            boot = [within(c, idx=rng.integers(0, n, n))[0] for _ in range(B_BOOT)]
            boot = [b for b in boot if not np.isnan(b)]
            ci_lo = float(np.percentile(boot, 2.5)); ci_hi = float(np.percentile(boot, 97.5))
            perm = {m: scores[m][c][rng.permutation(n)] for m in vf}
            shuf = [pearson(perm[vf[i]], perm[vf[j]]) for i in range(len(vf)) for j in range(i + 1, len(vf))]
            shuffle_r = float(np.mean(shuf)) if shuf else float("nan")
        else:
            ci_lo = ci_hi = shuffle_r = float("nan")
        convergent = bool(testable and R_within >= R_WITHIN_MIN and delta >= DELTA_MIN and ci_lo >= CI_LOWER_MIN)
        per_concept[c] = {
            "valid_families": [m.split("/")[-1] for m in vf], "n_valid": len(vf), "testable": testable,
            "R_within": None if np.isnan(R_within) else round(R_within, 4),
            "R_cross": None if np.isnan(R_cross) else round(R_cross, 4),
            "delta": None if np.isnan(delta) else round(delta, 4),
            "ci95": None if np.isnan(ci_lo) else [round(ci_lo, 4), round(ci_hi, 4)],
            "shuffle_r": None if np.isnan(shuffle_r) else round(shuffle_r, 4),
            "convergent": convergent}
        rw = "n/a" if np.isnan(R_within) else f"{R_within:.3f}"
        dl = "n/a" if np.isnan(delta) else f"{delta:+.3f}"
        cl = "n/a" if np.isnan(ci_lo) else f"{ci_lo:.2f}"
        print(f"{c:14s} valid={len(vf)}/6 R_within={rw} Δ={dl} CI_lo={cl} -> CONVERGENT={convergent}")

    n_conv = sum(1 for c in CONCEPTS if per_concept[c]["convergent"])
    if n_conv >= SURVIVE_CONCEPTS:
        reading = f"SURVIVED — convergence is concept-general ({n_conv}/4 concepts converge)"
    elif n_conv >= 1:
        reading = f"PARTIAL — convergence is concept-specific ({n_conv}/4 concepts converge)"
    else:
        reading = "NOT — no concept survives the cross-concept null"

    result = {"hash": h, "families": fams, "concepts": CONCEPTS,
              "gate": {"valid_disc": VALID_DISC, "testable_min": TESTABLE_MIN, "r_within_min": R_WITHIN_MIN,
                       "delta_min": DELTA_MIN, "ci_lower_min": CI_LOWER_MIN, "survive_concepts": SURVIVE_CONCEPTS},
              "validity": {f"{m}::{c}": validity[(m, c)] for (m, c) in validity},
              "per_concept": per_concept, "n_convergent": n_conv, "reading": reading,
              "scores": {m: {c: scores[m][c].tolist() for c in scores[m]} for m in scores}}
    (HERE / "convergence_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n===== {reading} =====")
    print("wrote convergence_result.json")


if __name__ == "__main__":
    main()
