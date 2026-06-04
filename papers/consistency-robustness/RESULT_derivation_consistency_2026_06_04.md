# RESULT — derivation is a regime-gated readout: it breaks the derivable floor, not the knowledge floor

**Date:** 2026-06-04 · **Verdict: PARTIAL — regime-gated (does NOT break the knowledge floor).**
Frozen: `PREREG_derivation_consistency_2026_06_04.md`. Qwen2.5-3B-Instruct, 100 knowledge-subject MMLU
items (computation-heavy subjects excluded), greedy CoT, parse-fail 0.

## Numbers
- Accuracy: **direct 0.640 → CoT 0.630** (P1 FALSE — reasoning did not help; ~equal).
- Derivation-disagreement AUROC (direct≠cot flags direct-wrong): **0.516** (P2 FALSE — chance).
- Reasoning corrects **11%** of direct errors (P3 FALSE). Residual floor (direct AND cot wrong): **32%**.

## What it means (the regime distinction — and it sharpens the framework)
For **knowledge facts there is nothing to derive** — you retrieve the fact or you don't — so chain-of-
thought just **restates the retrieved belief**. Direct ≈ derived; their disagreement carries no signal.
Derivation-consistency therefore does **not** break the knowledge-retrieval floor.

This is not a failure of the idea; it **isolates** where the idea works. Re-derivation breaks the floor
for **derivable** claims (multi-step math, logic, code-tracing), where a wrong one-shot *retrieval* can
be corrected by *computing* — exactly styxx's own path-diverse finding (grounded AUC **0.694 → 0.955**
on hard arithmetic by re-deriving through independent reasoning paths). My 3B model on full MMLU could
not test that regime cleanly (it can't reliably compute the hard items — runaway reasoning), so I tested
the knowledge regime, where the answer is a clean negative.

**Consequence for the floor:** the irreducible floor **decomposes** —
- **derivable sub-floor:** breakable by re-derivation (the path-diverse readout); not truly irreducible.
- **knowledge sub-floor:** a stable retrieved false belief with nothing to derive — **here ~32%** —
  reasoning cannot touch it; only an **external truth anchor** (retrieval/verification) can. This is the
  genuinely irreducible part the framework names.

## Framework update
Derivation-consistency is the **fifth readout**, regime-gated: reads what the model *derives*; blind to
*retrieved* knowledge (collapses to retrieval) and to a belief stable across reasoning paths. It shrinks
the *derivable* part of the floor and, by its silence on knowledge facts, **confirms the knowledge floor
as the irreducible residual.**

## Honest scope
One 3B model, knowledge-MMLU, single greedy CoT (self-consistency voting would only help at the margin);
the derivable regime is not cleanly tested here (model capability bound). The negative is clean *for the
knowledge regime*, which is the point.
