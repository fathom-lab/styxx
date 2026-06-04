# PREREG — does rhythm buy anything attention can't? (the necessity question, controlled)

**Date (frozen before data):** 2026-06-04. The genuinely-open core of the 2,500-year question made
testable: is oscillation NECESSARY/special for ordered memory, or just one efficient mechanism among
others? The clean test is a **matched-parameter, same-task** architecture showdown — not pretrained
LLMs (confounded by training), but three nets trained identically on the established ordered-copy task.

## Method (frozen)
- Task: ordered copy (V=12, KMAX=20), the rig from `run_rhythm_rescue.py`. 3 seeds. Identical training
  (STEPS=4000, BATCH=64, LR=2e-3, Adam, grad-clip 1.0). kcap = largest K with mean acc ≥ 0.80.
- Three arms, **matched total parameter count (within ~15%, reported exactly):**
  - **LRU-FREE** — complex-diagonal LRU, eigenvalue phase θ learnable → oscillatory dynamics.
  - **LRU-CLAMPED** — same, θ≡0 → real eigenvalues, decay only, NO oscillation.
  - **TRANSFORMER** — small causal (decoder-only) transformer: attention, NO rhythm at all.

## Hypotheses & predictions (frozen)
- **P1 (oscillation helps recurrence — replicates rhythm-rescue):** LRU-FREE kcap > LRU-CLAMPED kcap by
  ≥ 2 items.
- **P2 (THE necessity test — my bet):** the rhythm-free **TRANSFORMER kcap ≥ LRU-FREE kcap** (within
  noise or better) — attention achieves oscillation's benefit *without* rhythm, at matched parameters.
  This would show oscillation is NOT necessary or special for ordered memory — a clean, rigorous nail
  in "rhythm underlies cognition," beyond the qualitative transformer counterexample.

## Decision rule (frozen)
- **RHYTHM NOT SPECIAL** iff P1 ∧ P2 — oscillation helps a *recurrent* substrate, but a rhythm-free
  attention substrate matches/beats it at equal cost → oscillation is one efficient mechanism, not a
  requirement. (Direct support for the honest, demarcated reading of the frequency-mind question.)
- **OSCILLATION HAS A REAL EDGE** iff P1 holds but TRANSFORMER kcap < LRU-FREE kcap by ≥ 2 items —
  surprising: rhythm buys something attention can't at this scale. Would be the genuinely novel result.
- **MIXED** otherwise; report the three kcaps.

## Caveats (frozen)
- A matched-parameter CONTROLLED comparison on ONE synthetic task (ordered copy), not a pretrained-LLM
  benchmark — by design, because pretrained Mamba-vs-Transformer is confounded by training. This is the
  clean mechanistic test; it does NOT speak to large-scale capability. Param-matching is approximate
  (~15%); exact counts reported. The transformer is well-suited to copy by construction — that is the
  point (it is the strong rhythm-free competitor).
