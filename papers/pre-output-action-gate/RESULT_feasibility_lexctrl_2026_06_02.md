# RESULT — Pre-output action gate, lexical-control feasibility

**Date:** 2026-06-02
**Pre-registration:** [PREREG_feasibility_lexctrl_2026_06_02.md](PREREG_feasibility_lexctrl_2026_06_02.md)
**Pre-data anchor:** `a9cd8ad`
**Verdict:** **SURVIVED** — the end-of-prefill residual encodes an action's
destructiveness *beyond the prompt text*, cross-domain, on all 4 models.

## Result

Text-only bag-of-words baseline (LODO): **0.511** — chance. No word separates
destructive from safe across action domains; the set is genuinely controlled.

| Model | residual LODO (best) | layer | margin vs text | LODO @~65% | LOO |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | **0.947** | 18/28 | **+0.436** | 0.947 | 0.902 |
| Qwen2.5-3B | **0.940** | 19/36 | **+0.428** | 0.894 | 0.749 |
| Llama-3.2-1B | **0.837** | 8/16 | **+0.326** | 0.766 | 0.558 |
| Llama-3.2-3B | **0.890** | 15/28 | **+0.379** | 0.887 | 0.737 |

Frozen gate (residual LODO ≥ 0.70 **and** margin ≥ 0.15, on ≥2 models):
**4/4 pass.** SURVIVED.

## Why this one is trustworthy (and round 1 was not)

1. **Not a 1.000 ceiling.** 0.84–0.95 with real per-model variation is the
   shape of genuine signal, not a confound. (Round 1's perfect 1.000 was the
   tell that the probe was reading the destructive *verb*.)
2. **The text baseline is at chance (0.511).** Destructiveness lives in the
   value/consequence here, not a giveaway word — so a residual that separates is
   encoding something the prompt text cannot express.
3. **Generalizes across action domains (LODO).** Train on 22 action types,
   predict the held-out 23rd. The fixed ~65%-depth layer holds too (0.77–0.95),
   so it isn't best-layer cherry-picking.
4. **Holds across architectures** (Qwen + Llama, 1B–3B).
5. **LOO < LODO is expected here, not a flaw.** Under leave-one-out, a prompt's
   near-identical matched twin (opposite label) sits in training and drags the
   prediction backwards — the same artifact that put the *text* LOO at 0.000.
   That is exactly why the gate is on leave-one-**domain**-out.

## What this establishes

The pre-emission residual of an open-weight instruct model **linearly encodes
whether a requested action is destructive**, in a way that is (a) not reducible
to the prompt's words, (b) generalizes to unseen action types, and (c) holds
across four models. That is the **necessary condition** for a pre-output action
gate, met **non-trivially** — the first real evidence.

## What it does NOT establish (the honest boundary)

- **Prompt-level intent, not emitted-action behavior.** This reads the residual
  after a prompt that *explicitly describes* a destructive action. The product
  claim is predicting the model's **chosen / emitted** tool call in an agent
  loop — where the model decides, possibly from a benign-looking prompt
  ("clean up the database"). **Untested.**
- **Description ≠ the dangerous case.** The Replit failure was the model
  *choosing* destruction, not being asked for it. Not tested here.
- **Feasibility-grade:** n=46, 23 domains, single deterministic pass.
- **Not a deployable gate yet:** no calibration / FPR at an operating point.
- **Mechanism not pinned:** the residual may encode a learned "this action is
  harmful" property, or training-corpus co-occurrence of risky configs with
  warnings. Usable either way, but not explained.

## The arc is the story

Round 1: AUC 1.000 → we called it a confound and refused the claim.
Round 2: controlled the confound, added a text baseline, **tried to kill the
signal — and it lived at 0.84–0.95 over a 0.51 baseline.** That is the honest,
compelling version of "first": not a number we trumpeted, but a result that
*survived our own attempt to falsify it.*

## Next (now justified)

Build the **emitted-action study**: agent scenarios where the model chooses the
tool call; read the residual at the decision token *before emission*; label by
the **emitted action**, cross-model, held-out. That is the product claim. This
result is the green light to spend the days it takes — earned, not assumed.

— scored 2026-06-02 against the frozen gate; boundary stated honestly.
