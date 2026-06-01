# PREREG — is there a SHARED cave direction across formats, hiding under the failed transfer?

**REGISTERED 2026-05-31, before the shared-subspace probe is scored.**
**SIGN-OFF:** Flobi — *"use things noone has thought to use, prove everyone wrong"* (2026-05-31).

## The reframe

The frozen MCQ probe reads free-form at **0.50** (chance) — so the *full* MCQ direction is not the
free-form direction. The lazy conclusion is "format-specific, done." The hacker question: is there a
**shared subspace** that *neither single-format probe finds alone* but that **joint training reveals**? If
one direction, trained on both, separates lie-from-mistake on **held-out from both formats**, the
"format-specific" verdict is wrong and a universal cave lives in the intersection.

## Design (Qwen2.5-3B, same d=2048, $0, CPU, existing residuals)

- MCQ lie/mistake = `full + bc + bc2`; free-form lie/mistake = `ff`. Same model, same dims, all layers.
- Per layer L: stratified 70/30 split of each format. Train **one** probe on the **pooled** train
  (MCQ-train ∪ ff-train); evaluate **separately** on MCQ-test and ff-test. Also the two cross-transfers
  (MCQ-only→ff-test, ff-only→MCQ-test) for context.
- Report the layer maximizing `min(MCQ-test, ff-test)`.

## Bars (FIXED)

| Bar | Statement | Threshold |
|---|---|---|
| **SHARED** *(key)* | one direction serves both formats | at some layer, **min(MCQ-test, ff-test) ≥ 0.65** |
| **NON-TRIVIAL** | it's a real shared direction, not a lucky union | the joint probe beats the failed cross-transfer (MCQ→ff 0.50) on ff-test by **≥ 0.10** |

**RESULT = SHARED DIRECTION EXISTS iff SHARED ∧ NON-TRIVIAL.**

## Honest scope

- Not confidence-matched (this tests geometry existence, not the matched-confidence claim); single model;
  raw cave direction. A **fail** confirms the formats use genuinely distinct cave geometry (the §10 negative
  is robust) — reported. A joint probe that works only because it memorizes each format separately is the
  failure mode the NON-TRIVIAL bar guards against. Correlational.

## One line

A failed transfer isn't proof of no shared direction — it's proof the *whole* direction differs. Train on
both, test on both: if one vector still reads the cave in MCQ *and* prose, "format-specific" was wrong.
