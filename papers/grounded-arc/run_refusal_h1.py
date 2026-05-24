# -*- coding: utf-8 -*-
"""Bet-0 H1 refusal kill-gate — the one-shot confirmatory run (2026-05-24).

Runs ONCE on the holdout locked + hashed at the lock commit. No peeking, no
re-runs, no optional stopping. Pre-registered bar: Spearman ρ(validity, −error)
≥ 0.40 with permutation p < 0.01 on the refusal holdout. ρ ≥ 0.40 → arc
continues; ρ < 0.40 → arc abandoned, ships as a closed-negative paper.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/dogfood")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ve = _load("papers/grounded-arc/validity_engine.py", "ve")
rb = _load("scripts/dogfood/run_bet0_phase1.py", "run_bet0_phase1")
from styxx.guardrail import refuse_check  # noqa: E402

HERE = Path("papers/grounded-arc")
hold = [json.loads(ln) for ln in (HERE / "holdout" / "refusal.jsonl")
        .read_text(encoding="utf-8").splitlines() if ln.strip()]
corpora = json.loads((HERE / "holdout_corpora.json").read_text(encoding="utf-8"))

# verify the locked hash — refuse to run on a mutated holdout
sha = hashlib.sha256("\n".join(sorted(h["prompt"] for h in hold)).encode("utf-8")).hexdigest()
assert sha == corpora["refusal"]["sha256"], f"HASH MISMATCH: {sha} != locked"
print(f"holdout hash verified: {sha[:16]}…  n={len(hold)}")

# ρ is invariant to the validity sigmoid's α/τ (validity is monotone in distance);
# these set the user-facing validity VALUE only, not the rank-correlation test.
ALPHA, TAU = 8.0, ve.TAU_DISTANCE
val, neg_lab, neg_hum = [], [], []
for h in hold:
    v = ve.validity(h["d"], tau=TAU, alpha=ALPHA)
    s = float(refuse_check(prompt=h["prompt"], response=h["response"]).refuse_risk)
    val.append(v)
    neg_lab.append(-abs(s - h["gold"]))
    neg_hum.append(-abs(s - h["gold_human"]))

rho_lab = rb.spearman_rho(val, neg_lab)
p_lab = rb.permutation_p(val, neg_lab, n_permutations=rb.H1_PERMUTATIONS, rng_seed=rb.RNG_SEED)
rho_hum = rb.spearman_rho(val, neg_hum)
p_hum = rb.permutation_p(val, neg_hum, n_permutations=2000, rng_seed=rb.RNG_SEED)
mean_err = sum(-x for x in neg_lab) / len(neg_lab)

BAR = 0.40
verdict = (rho_lab >= BAR) and (p_lab < 0.01)
print(f"\nBet-0 H1 refusal kill-gate (n={len(hold)}, perms={rb.H1_PERMUTATIONS}):")
print(f"  PRIMARY (vendor-robust labeler gold): rho = {rho_lab:+.4f}  p = {p_lab:.4f}")
print(f"  rigor   (XSTest human gold)         : rho = {rho_hum:+.4f}  p = {p_hum:.4f}")
print(f"  mean |error| (labeler) = {mean_err:.3f}")
print(f"  bar: rho >= {BAR}, p < 0.01")
print(f"  VERDICT: {'H1 PASS — validity predicts reliability; arc continues' if verdict else 'H1 FAIL — abandon arc → closed-negative paper'}")

result = (
    f"# H1 {'PASSED' if verdict else 'FAILED'} — refusal kill-gate\n\n"
    f"One-shot confirmatory run on the holdout hashed BEFORE scoring (no peeking, "
    f"no re-runs). sha256 `{sha}`.\n\n"
    f"- n = {len(hold)}; 4 overlap-distance bins; calibration corpus = "
    f"threshold-law `_domain_pool`; embedding = text-embedding-3-large.\n"
    f"- **PRIMARY (vendor-robust labeler gold):** Spearman ρ(validity, −error) = "
    f"**{rho_lab:+.4f}**, permutation p = {p_lab:.4f}.\n"
    f"- rigor (XSTest human gold): ρ = {rho_hum:+.4f}, p = {p_hum:.4f}.\n"
    f"- pre-registered bar: ρ ≥ 0.40, p < 0.01.\n"
    f"- mean |error| (labeler) = {mean_err:.3f}.\n"
    f"- the H1 Spearman is invariant to the validity sigmoid's α/τ (validity is "
    f"monotone in distance); the test is whether distance-to-calibration "
    f"rank-predicts instrument error.\n\n"
    f"**VERDICT: {'PASS — arc continues to H2/H3/H4.' if verdict else 'FAIL — bet-0 abandoned; ships as a closed-negative paper: embedding-distance validity does not predict refusal-instrument reliability at the prompt level.'}**\n"
)
(HERE / ("H1_PASSED.md" if verdict else "H1_FAILED.md")).write_text(result, encoding="utf-8")
print(f"\nwrote {'H1_PASSED.md' if verdict else 'H1_FAILED.md'}")
sys.exit(0 if verdict else 1)
