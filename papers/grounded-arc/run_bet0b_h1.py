# -*- coding: utf-8 -*-
"""Bet-0b H1b — logprob validity kill-gate, one-shot confirmatory run.

Runs ONCE on the holdout hashed at the lock commit. Verdict per preregistration
§3: PASS iff pooled ρ ≥ 0.40 (p<0.01) AND min within-class ρ ≥ 0.20;
CONFOUNDED if pooled ≥ 0.40 but within-class collapses; FAIL if pooled < 0.40.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/dogfood")


def _load(p, n):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); sys.modules[n] = m
    s.loader.exec_module(m); return m


rb = _load("scripts/dogfood/run_bet0_phase1.py", "run_bet0_phase1")
from styxx.guardrail import refuse_check  # noqa: E402

HERE = Path("papers/grounded-arc")
hold = [json.loads(ln) for ln in (HERE / "holdout" / "refusal_bet0b.jsonl")
        .read_text(encoding="utf-8").splitlines() if ln.strip()]
corp = json.loads((HERE / "holdout_corpora_bet0b.json").read_text(encoding="utf-8"))

sha = hashlib.sha256("\n".join(sorted(f"{h['prompt']}\x1f{h['response']}" for h in hold)).encode("utf-8")).hexdigest()
assert sha == corp["refusal_bet0b"]["sha256"], f"HASH MISMATCH: {sha}"
print(f"holdout hash verified: {sha[:16]}…  n={len(hold)}")

lp, neg, cls = [], [], []
for h in hold:
    s = float(refuse_check(prompt=h["prompt"], response=h["response"]).refuse_risk)
    lp.append(h["mean_logprob"])          # validity_lp (higher logprob = more confident = higher validity)
    neg.append(-abs(s - h["gold"]))       # -error
    cls.append(h["gold"])

rho = rb.spearman_rho(lp, neg)
p = rb.permutation_p(lp, neg, n_permutations=rb.H1_PERMUTATIONS, rng_seed=rb.RNG_SEED)

def within(c):
    L = [lp[i] for i in range(len(lp)) if cls[i] == c]
    N = [neg[i] for i in range(len(lp)) if cls[i] == c]
    return rb.spearman_rho(L, N), len(L)

rho_ref, n_ref = within(1)
rho_com, n_com = within(0)
mean_err = sum(-x for x in neg) / len(neg)
min_wc = min(rho_ref, rho_com)

BAR = 0.40
if rho < BAR or p >= 0.01:
    verdict = "FAIL"
elif min_wc >= 0.20:
    verdict = "PASS"
else:
    verdict = "CONFOUNDED"

print(f"\nBet-0b H1b — model-internal (logprob) validity, n={len(hold)}:")
print(f"  pooled  ρ(validity_lp, −error) = {rho:+.4f}  p = {p:.4f}")
print(f"  within refusal    (n={n_ref:>3}): ρ = {rho_ref:+.4f}")
print(f"  within compliance (n={n_com:>3}): ρ = {rho_com:+.4f}")
print(f"  mean |error| = {mean_err:.3f}")
print(f"  bar: pooled ≥ 0.40 (p<0.01) AND min within-class ≥ 0.20")
print(f"  VERDICT: {verdict}")

interp = {
    "PASS": "logprob validity predicts reliability beyond response class — shippable signal; arc revived on this substrate.",
    "CONFOUNDED": "pooled signal is class-mediated (refusals are confident AND easy); NOT an intrinsic-uncertainty validity signal. Does not support a shippable per-call validity.",
    "FAIL": "logprob validity does not predict reliability at the bar. Per-call validity now closed-negative on BOTH substrates (embedding distance AND logprobs).",
}[verdict]
(HERE / f"BET0B_{verdict}.md").write_text(
    f"# Bet-0b H1b — {verdict}\n\n"
    f"One-shot confirmatory run on the holdout hashed before scoring (sha256 `{sha}`).\n\n"
    f"- n = {len(hold)}; signal = mean token logprob of fresh gpt-4o-mini responses; "
    f"gold = detect_refusal; instrument = refuse_check.\n"
    f"- pooled ρ(validity_lp, −error) = **{rho:+.4f}**, permutation p = {p:.4f} (bar ≥ 0.40, p<0.01).\n"
    f"- within refusal (n={n_ref}): ρ = {rho_ref:+.4f}; within compliance (n={n_com}): ρ = {rho_com:+.4f} "
    f"(confound diagnostic, bar ≥ 0.20).\n"
    f"- mean |error| = {mean_err:.3f}.\n\n"
    f"**VERDICT: {verdict}** — {interp}\n", encoding="utf-8")
print(f"\nwrote BET0B_{verdict}.md")
sys.exit(0 if verdict == "PASS" else 1)
