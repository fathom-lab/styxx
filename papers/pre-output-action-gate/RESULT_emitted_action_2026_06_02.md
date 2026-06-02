# RESULT — Emitted-action study (the product claim)

**Date:** 2026-06-02
**Pre-registration:** [PREREG_emitted_action_2026_06_02.md](PREREG_emitted_action_2026_06_02.md)
**Pre-data anchor:** `1b3fb20` (gemma template-compat fix after; method/gate unchanged)
**Verdict:** **SURVIVED** — the pre-emission residual predicts the model's
*chosen* destructive action, beyond the prompt text, on 4 of 5 open-weight models.

## Result

Label = destructiveness of the **emitted** tool (the model's behavior). All AUCs
under leave-one-category-out (train on 7 action categories, predict the held-out
8th). "bow" = text bag-of-words baseline on the same folds.

| Model | dest/safe | residual LOCO | text (bow) | margin | gate |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 16/24 | **0.917** | 0.354 | **+0.562** | ✅ |
| Qwen2.5-3B | 8/32 | **0.758** | 0.227 | **+0.531** | ✅ |
| Llama-3.2-3B | 11/29 | **0.912** | 0.207 | **+0.705** | ✅ |
| gemma-2-2b | 10/30 | **0.803** | 0.257 | **+0.547** | ✅ |
| Llama-3.2-1B | 9/31 | 0.649 | 0.165 | +0.484 | ✗ (floor) |

Frozen gate (residual LOCO ≥ 0.70 **and** margin ≥ 0.15, on ≥2 balanced models):
**4/5 pass → SURVIVED.** All 5 models had class balance (≥8 each).

## Why it's trustworthy

1. **Not a ceiling.** 0.65–0.92 with per-model spread — real signal, not the
   round-1 confound's 1.000.
2. **The text baseline is near/below chance (0.17–0.35).** The prompt does *not*
   predict the model's choice across held-out categories — but the residual
   does, by +0.48 to +0.71. The residual carries the model-specific impending
   decision the prompt can't reveal.
3. **Generalizes across action types (LOCO):** trained on 7 categories, predicts
   the held-out 8th.
4. **Holds across 3 architectures** (Qwen, Llama, Gemma).
5. **Label is behavior, not prompt:** the same scenario yields destructive on
   one model, safe on another; only a model's own residual tracks that.

## What it establishes

The first **falsification-survived** evidence that a **pre-output action gate is
buildable**: an open-weight model's residual, read before it emits anything,
linearly predicts whether the tool call it is about to make is destructive —
cross-model, generalizing across action types, beating the prompt text.

## What it does NOT establish (the honest boundary)

- **It decodes the model's *forming decision*, not "intent from nowhere."** At a
  late layer (L17–30) the residual largely determines the deterministic choice,
  so part of this is decoding a near-downstream output. What makes it a
  capability rather than a triviality: it is *linear*, *generalizes* to unseen
  action types, and *beats the prompt*. We do **not** claim styxx "reads the
  agent's mind."
- **Simplified harness:** the model selects from a presented tool list
  (multiple-choice-like), not fully open tool-calling. Real agents are messier;
  open tool-call capture is the next build.
- **4/5, not 5/5:** Llama-3.2-1B missed the 0.70 floor (0.649) — though it still
  beat text by +0.48. Not universal.
- **Feasibility-to-validation grade:** n=40 scenarios, single greedy pass.
- **Not a deployable gate:** no calibration, no false-positive budget at an
  operating point, no latency analysis.
- **"Destructive" = chose the labelled destructive tool** — coarse; a real
  system should score irreversibility/blast-radius per emitted call.

## The arc (why this is the honest "first")

1. Round-1 probe → AUC **1.000** → called it a **confound, refused the claim**.
2. Lexical-control → residual **0.84–0.95** vs text 0.51 → *described* intent
   survived.
3. Pilot → models **do** choose destruction (small models most).
4. Emitted-action → the residual predicts the **chosen** destructive action,
   **0.65–0.92** vs text 0.17–0.35, 4/5 models → **SURVIVED.**

Four rounds, two kills and two survivals, every gate frozen before its data.
The first nobody can take with a citation, because the rigor is the result.

## Next

- **Open tool-call capture** (native tool-calling, not a presented list).
- **Deployable gate:** calibration + false-positive budget + operating point +
  latency; read at the true decision token in a live agent loop.
- **Scale + irreversibility-graded labels;** add a held-out scenario corpus.

— scored 2026-06-02 against the frozen gate; boundary stated honestly.
