# RESULT — open / native tool-calling replication

**Date:** 2026-06-02
**Pre-registration:** [PREREG_open_toolcall_2026_06_02.md](PREREG_open_toolcall_2026_06_02.md)
**Verdict:** **FORMAT-DEPENDENT — did NOT replicate** (per the frozen gate).
The earlier 4/5 emitted-action result leaned, in part, on the presented-menu format.

## Result

Native function-calling (`tools=` schemas, structured `tool_call`), everything
else identical to the parent study.

| Model | dest/safe | residual LOCO | text (bow) | margin | gate |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 15/24 | **0.819** @L19 | 0.269 | +0.550 | ✅ |
| Llama-3.2-3B | 11/29 | **0.514** @L1 | 0.160 | +0.354 | ✗ (floor) |
| Qwen2.5-3B | 7/31 | — | — | — | not balanced |

Frozen gate (residual LOCO ≥ 0.70 **and** margin ≥ 0.15, on ≥2 balanced models):
**only 1 of 2 balanced models passes → did NOT replicate.**

## Honest interpretation

- **Qwen-1.5B held.** 0.819 under native tool-calling (vs 0.917 on the menu) —
  slightly lower but still strong, still well above its 0.269 text baseline. On
  this model the capability is **real and survives the realistic interface.**
- **Llama-3B collapsed.** 0.912 (menu) → **0.514** (native), best layer at L1
  (near chance, near the input). Its earlier signal was **largely tied to the
  presented-menu format**, not a format-invariant commitment.
- **Qwen-3B went degenerate.** Under native calling it chose destructively only
  7/40 — below the balance floor — so it couldn't be tested. (Native calling
  made it *more* cautious.)

**The blunt conclusion:** the pre-output action signal is **model-specific and
partly format-dependent.** It robustly survives native tool-calling on Qwen-1.5B;
it does not on Llama-3B. The headline 4/5 SURVIVED was, for at least one model,
inflated by the elicitation format. Per the pre-registered gate, this is a
**non-replication**, and I will not move the goalpost to rescue it.

## What it changes

1. **The claim is now properly bounded:** "predicts the chosen destructive
   action before emission" holds **robustly on some open-weight models under
   realistic tool-calling (Qwen-1.5B), not universally.** Not "4/5 models" once
   you use the real interface.
2. **ActionGuard does NOT get promoted to the package.** It works on Qwen-1.5B
   under native calling; it is not validated broadly. Shipping it now would have
   shipped Llama's format-dependence to users. **The reason we ran this before
   promoting, not after — it did its job.**
3. The steering / specificity / operating-curve results inherit the same bound:
   demonstrated on the menu harness; their native-calling robustness is unproven
   beyond Qwen-1.5B.

## Why Llama collapsed (hypotheses, not claims)

The native `tools=` prefill renders the toolset very differently (JSON schemas,
special tokens) than a system-prompt menu — that changes the residual geometry
at the read point. Llama-3.2-3B's destructive-vs-safe distinction may not be
linearly present in *that* representation; Qwen-1.5B's is. Untested; the honest
next step is to find out rather than assume.

## Next (re-scoped honestly)

- **Per-model validation under the deployment interface** — fit and test the
  direction in the *same* native-calling representation it will run in, model by
  model. (The guard is per-model anyway.)
- Understand the Llama collapse (layer/representation analysis under `tools=`).
- More scenarios + larger models before any universality claim.
- Hold the package promotion until native-calling robustness is shown on ≥2
  models.

This is the discipline paying its rent: an exciting claim, tested against the
realistic interface, found to be **bounded** — caught by us, before a user, and
on the record.

— scored 2026-06-02 against the frozen gate; reported against, not around, it.
