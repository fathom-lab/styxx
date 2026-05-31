# FINDING — "representation beats blind output" did NOT replicate on fresh data — VOID (the confirmatory caught an exploratory false-positive)

**VOID (precondition failed; no representational advantage either way).** PREREG
`PREREG_residual_confab_strict_2026_05_31.md`, **committed before the data** (`2105c1b`, pushed to
origin). Qwen2.5-3B-Instruct, **1612 FRESH disjoint** TriviaQA items (`--skip 800`), nested-CV,
`probe_strict_confirmatory.py`; receipt `residual_probe_strict_result.json`.

## The hook, and the kill

EXPLORATORY (`probe_strict_diagnostic.py`, on the BASE residuals): at bottom-12% entropy the output
looked **blind** (surface AUC 0.474) while the probe held (0.799) — contrast **+0.325**. Exciting. So
it was pre-registered as a confirmatory on **fresh disjoint data**, with a **precondition** that the
output must actually be blind (surface ≤ 0.55) and a **nested-CV** estimator.

Result (fresh, n=1612; bottom-12% subset = 194: **31 confident-wrong, 163 right**, powered):

| check | value | bar | pass |
|---|---|---|---|
| **PRECONDITION** surface entropy-AUC | **0.698** | ≤ 0.55 | **FAIL** (output NOT blind) |
| PROBE nested-CV AUC | 0.697 | ≥ 0.70 | fail |
| **CONTRAST** probe − surface | **−0.001** | ≥ 0.20 | fail |

**RESULT = VOID.**

## The claims that land

1. **The exploratory signal was a small-sample artifact — it did not replicate.** The base run's
   "blindness" (surface 0.474) came from 84 items / 16 wrong — noise near chance. On 1612 fresh items
   the same threshold gives surface **0.698**: the output is **not** blind among the most-confident
   answers, and the residual probe (0.697) carries **no advantage** (contrast −0.001).
2. **The pre-registered confirmatory caught an exploratory false-positive — the method working on us.**
   Reporting the exploratory +0.325 would have been a false "representation beats output" claim. Fresh
   disjoint data + the blind-output precondition + nested CV killed it. This is exactly what
   pre-registration is for, and it fired on its own authors.
3. **Substantive negative:** a linear probe on Qwen-3B's first-token commitment residual tracks the
   **same uncertainty the output reads** (both ≈ 0.70); the representation does **not** see confident
   confabulation better than the output here. The "representation > output" hope is unsupported.
4. **"Confident" via entropy-quantile never goes truly blind:** even the bottom entropy decile retains
   an output gradient (surface 0.698) — there is always residual uncertainty the output can read.

## What it bounds

Consistent with the whole arc: output and representation both track generation uncertainty; **neither
cleanly catches confident confabulation** (the wall), and external **retrieval** remains the only
demonstrated lever. The representation-level hope, tested properly on fresh data, does not hold.

## Honest scope / what could still be true

Linear probe + **first-token** commitment position only. A nonlinear probe, span-aggregated residuals,
a different layer/position, or a blind subset built by something other than entropy-quantile could
differ — but the *registered* "representation beats blind output" claim, on fresh data, is **dead**.
Single model, one run, 31 confident-wrong (thin). Receipt and prereg are public.

## One line

The exciting exploratory "representation beats blind output on confident confab" (+0.33) did NOT
replicate on 1612 fresh items — the output was not blind (0.698) and the residual probe (0.697) carried
no advantage (−0.001) — VOID; the pre-registered confirmatory caught an exploratory false-positive,
which is the discipline working on us, in public, with the receipts.
