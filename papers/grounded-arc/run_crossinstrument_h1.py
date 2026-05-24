# -*- coding: utf-8 -*-
"""Cross-instrument H1d kill-gate — does logprob-validity predict the HALLUCINATION
instrument's reliability? One-shot on the locked holdout. PASS = pooled ρ ≥ 0.40
(p<0.01) AND min within-class ρ ≥ 0.20."""
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
from styxx.guardrail import check  # noqa: E402

HERE = Path("papers/grounded-arc")
hold = [json.loads(ln) for ln in (HERE / "holdout" / "halluc_h1d.jsonl")
        .read_text(encoding="utf-8").splitlines() if ln.strip()]
corp = json.loads((HERE / "holdout_corpora_crossinstrument.json").read_text(encoding="utf-8"))
sha = hashlib.sha256("\n".join(sorted(f"{h['question']}\x1f{h['response']}" for h in hold)).encode("utf-8")).hexdigest()
assert sha == corp["halluc_h1d"]["sha256"], f"HASH MISMATCH {sha}"
print(f"hash verified {sha[:12]} n={len(hold)}; scoring with hallucination check(use_nli=True) …", flush=True)

lp, neg, cls = [], [], []
for i, h in enumerate(hold):
    risk = float(check(prompt=h["question"], response=h["response"], reference=h["knowledge"], use_nli=True).risk)
    lp.append(h["mean_logprob"]); neg.append(-abs(risk - h["gold"])); cls.append(h["gold"])
    if (i + 1) % 150 == 0:
        print(f"  scored {i+1}/{len(hold)}", flush=True)

rho = rb.spearman_rho(lp, neg)
p = rb.permutation_p(lp, neg, n_permutations=rb.H1_PERMUTATIONS, rng_seed=rb.RNG_SEED)

def w(c):
    L = [lp[i] for i in range(len(lp)) if cls[i] == c]
    N = [neg[i] for i in range(len(lp)) if cls[i] == c]
    return rb.spearman_rho(L, N), len(L)

r1, n1 = w(1)   # hallucinated responses
r0, n0 = w(0)   # correct responses
min_wc = min(r1, r0)
mean_err = sum(-x for x in neg) / len(neg)
BAR = 0.40
verdict = "FAIL" if (rho < BAR or p >= 0.01) else ("PASS" if min_wc >= 0.20 else "CONFOUNDED")

print(f"\nBet H1d — logprob-validity on the HALLUCINATION instrument (n={len(hold)}):")
print(f"  pooled  ρ(validity_lp, −error) = {rho:+.4f}  p = {p:.4f}")
print(f"  within hallucinated (n={n1}): ρ = {r1:+.4f}")
print(f"  within correct      (n={n0}): ρ = {r0:+.4f}")
print(f"  mean |error| = {mean_err:.3f}")
print(f"  bar: pooled ≥ 0.40 AND min within-class ≥ 0.20")
print(f"  VERDICT: {verdict}")

interp = {
    "PASS": "GENERALIZES — logprob-validity predicts a 2nd, different instrument's reliability; the 'model-internal confidence grounds cognometry' thesis holds across instruments, not just refusal.",
    "CONFOUNDED": "pooled signal is class-mediated; within-class doesn't survive. Does not cleanly generalize to hallucination at the bar.",
    "FAIL": "INSTRUMENT-SPECIFIC — logprob-validity was a refusal property; it does not transfer to the hallucination instrument. The map gains a sharp boundary.",
}[verdict]
(HERE / f"H1D_{verdict}.md").write_text(
    f"# H1d (cross-instrument, hallucination) — {verdict}\n\n"
    f"One-shot on the holdout hashed before scoring (sha `{sha}`).\n\n"
    f"- instrument = styxx hallucination check(use_nli=True); n={len(hold)}; gold via gpt-4o judge (val 0.90).\n"
    f"- pooled ρ(validity_lp, −error) = **{rho:+.4f}**, p = {p:.4f}.\n"
    f"- within hallucinated (n={n1}): ρ={r1:+.4f}; within correct (n={n0}): ρ={r0:+.4f}.\n"
    f"- bar: pooled ≥ 0.40, p<0.01, min within-class ≥ 0.20.\n\n"
    f"**VERDICT: {verdict}** — {interp}\n", encoding="utf-8")
print(f"\nwrote H1D_{verdict}.md")
sys.exit(0 if verdict == "PASS" else 1)
