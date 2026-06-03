# RESULT — Format-dependence: real, and model-specific

**Date:** 2026-06-03 · **Reading: PARTIAL — the native-tool-calling collapse is REAL (not
a labeling artifact), and the signal under native calling is MODEL-SPECIFIC.**
Frozen gate: `PREREG_open_toolcall_confirm_2026_06_02.md`. Same 40 scenarios
(`b1303559f2261c8a`), sampled-native labels (N=9), 3 seeds, HM CIs.

## Per model (sampled-native vs the prior greedy)

| model | sampled-native mean AUC | CI | margin | greedy was | verdict |
|---|---|---|---|---|---|
| **Qwen-1.5B** | **0.841** | [0.71, 0.97] | +0.51 | 0.819 | **robust** — no format problem |
| Llama-1B | 0.621 | [0.44, 0.80] | +0.20 | — | weak (least-legible everywhere) |
| **Llama-3B** | **0.530** | [0.34, 0.72] | +0.11 | 0.514 | **genuine collapse — did NOT recover** |
| Qwen-3B | unbalanced (7–8/40) | — | — | "degenerate" | **cautious** (declines destruction) |

Gate: direction holds (3–4/4 every seed) but magnitude clears on only **1** model
(Qwen-1.5B) → **PARTIAL**.

## What it means (honest)

The re-test was worth running — it converted a single-greedy-label verdict into two real
findings:

1. **Llama-3B's collapse is real, not an artifact.** Under stable sampled labels across 3
   seeds, Llama-3B's pre-emission destructive signal stayed at **~chance (0.530)**, CI
   floor 0.34. The "small-n / unstable greedy label" hypothesis is **falsified** for
   Llama-3B — under native tool-calling its residual genuinely does not carry the
   destructive-choice signal it carried under the menu harness (0.912). The interface
   really does break it for this model.
2. **Qwen-3B was misread as "degenerate" — it's cautious.** It declines the destructive
   action under native calling (7–8/40 prone, unbalanced), so there's no class to predict.
   A safety behavior, not a probe failure. The re-test corrected the record.
3. **Qwen-1.5B is robust** under native calling (0.841, CI off the floor, margin +0.51) —
   for this model the menu result transfers cleanly to the deployment interface.

**The deployment-relevant conclusion:** white-box action-gating under the *real* native
tool-calling interface is **model-specific** — robust for some models (Qwen-1.5B),
genuinely degraded for others (Llama-3B). The board's FORMAT-DEPENDENT verdict stands,
now on multi-seed CI evidence, and sharpened: *don't assume the action gate transfers to
native calling — validate per-model under the deployment interface.* This is exactly the
kind of bound that keeps the action-gate claim honest, and it's why we re-ran it instead
of leaving a single greedy label to carry the verdict.

## Caveats (frozen)

- Same 40 scenarios by design (labeling-regime comparison, not fresh data).
- n=40, native parsing depends on the model emitting parseable tool calls; Qwen-3B's
  abstention is itself a (benign) form of format-sensitivity.
- Open-weight, simulated tool schemas.
