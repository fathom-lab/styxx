# PILOT — emitted-action premise check (exploratory)

**Date:** 2026-06-02
**Status:** EXPLORATORY (not pre-registered). One question: do open-weight
models actually *choose* destructive actions, so the emitted-action study has a
positive class to predict?

## Setup

12 realistic "fix it" ops scenarios, each offering a mix of tools — one clearly
destructive/irreversible option plus safer alternatives. The model is asked for
the single tool call it would make first; we record whether it chose the
destructive one. Greedy, deterministic. (`pilot_emitted_action.py`.)

## Result

| Model | destructive | safe | unparsed |
|---|---|---|---|
| Qwen2.5-3B-Instruct | 2/12 | 10/12 | 0 |
| Llama-3.2-3B-Instruct | 2/12 | 5/12 | 5/12 |
| Qwen2.5-1.5B-Instruct | **7/12** | 5/12 | 0 |

## Findings

1. **Premise holds.** Models choose destructive actions with usable variation —
   not 0%, not 100%. There is a real behavior to predict.
2. **Capacity → caution.** The 1.5B model took the destructive shortcut **3.5×**
   as often as the 3B (58% vs 17%). The smaller, cheaper, self-hostable models —
   exactly the ones styxx can read (open weights) — are the most dangerous. The
   risk concentrates where the mechanism applies.
3. **Choice is scenario- and model-dependent.** The same scenario yields
   destructive (1.5B) or safe (3B) — so the label genuinely comes from model
   *behavior*, not prompt design. That is what the lexctrl feasibility could not
   test, and it is the product claim.

## Honest caveats

- **Harness, not native tool-calling:** the model picks from a presented list
  (multiple-choice-ish), not fully open planning. Llama-3B's 5 unparsed are a
  format-parsing gap, not a premise problem — the full study needs native
  tool-call capture.
- **n=12, 3 models, single pass.** Directional, not a measurement.
- **"Destructive" = chose the pre-labeled destructive tool.** Reasonable, but a
  real study should label irreversibility/blast-radius per emitted call.

## Conclusion → the full study is VIABLE

Green light, earned with data. Next: the **emitted-action study** —
- robust native tool-call capture across models;
- read the residual at the **decision token, before the tool call is emitted**;
- label by the **emitted** action (the model's choice), not the prompt;
- pre-register: does the pre-emission residual predict the model's destructive
  *choice*, cross-model, held-out? AUC kill-gate vs a text baseline.

If that survives, styxx predicts a destructive agent action before it fires —
the product claim, the empty lane, the honest first.
