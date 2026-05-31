# FINDING — the residual stream CARRIES a confident-confabulation signal (probe AUC 0.74) but does not clear the margin over the output — REPORT_AS_LANDED

**REPORT_AS_LANDED.** PREREG `PREREG_residual_confab_probe_2026_05_31.md` (signed *"go deeper, push the
tech farther than was ever possible"*). Qwen2.5-3B-Instruct, TriviaQA (verified answers, hashed),
700 items extracted, held-out test, unit-tested `_evallib` matcher. `extract_residuals.py` +
`probe_residuals.py`; receipts `residual_probe_result.json`, `residuals_meta.json`.

## Result

- N = 700 (accuracy 0.41). **Confident subset** (first-token entropy < median 0.975) = 350
  (**139 confident-wrong, 211 confident-right**). Powered ✓.
- Best layer **34 / 36** (top CV layers 32–36), TRAIN 5-fold CV AUC 0.750.
- Held-out TEST (n = 105: 42 wrong / 63 right):

| metric | value | bar | pass |
|---|---|---|---|
| **PROBE** residual test-AUC | **0.739** | ≥ 0.70 | **PASS** |
| surface output-signal AUC | 0.663 (entropy 0.663 / −margin 0.625) | — | — |
| **CONTRAST** probe − surface | **+0.077** | ≥ 0.20 | **FAIL** |

**RESULT = REPORT_AS_LANDED** (SURVIVED required PROBE ∧ CONTRAST; CONTRAST missed).

## The claims that land

1. **The representation carries a confident-confab signal — a real, standalone positive.** A linear
   probe on the **late-layer (34/36) residual** at the commitment position separates confident-wrong
   from confident-right at **held-out AUC 0.74**. Confident factual confabulation **is** partially
   linearly decodable from the residual stream — the first time this program has read a confabulation
   signal off the *wiring* rather than the output distribution.
2. **But it does NOT beat the output by the registered margin — because the output was not blind.** The
   design assumed the median-split "confident" subset would neutralize the output signal
   (entropy AUC → 0.5). It did not: entropy alone scored **0.663** on the same confident items. The
   bottom-half-entropy subset still spans enough uncertainty for the surface signal to work, so the
   residual's advantage is only +0.077. **The CONTRAST bar was the right test; the confident-subset
   definition was too permissive to run it.** Owned, not spun.
3. **Late layers carry it** (top CV layers 32–36) — consistent with knowledge/truth representations
   concentrating in late depth.

## What it bounds

This run does **not** establish "the representation sees what the output cannot," because the output
was not made blind. The headline ambition — representation > output on confident confabulation — is
**unproven here** (REPORT_AS_LANDED). What is shown: confident confabulation is decodable from the
representation at AUC 0.74, and the output signal is *not* useless within a median-split confident set.

## The next bet (pre-registerable, deliberately NOT run post-hoc)

A **strict-confidence variant**: define "confident" as the **bottom entropy decile** (or a fixed low
threshold) where surface AUC → 0.5 *by construction*, and test whether the residual probe **holds
~0.74** there. If it does, CONTRAST is large and "representation beats blind output" survives. Changing
the threshold *now*, after seeing this result, would be p-hacking — so it is a fresh pre-registration,
not an edit to this one.

## Honest scope

Single model (Qwen2.5-3B), TriviaQA, **linear** probe + first-token commitment position, one run.
Held-out test corrects layer-selection inflation. AUC 0.74 means **"a linear direction separates
confident-wrong from confident-right"**, **NOT** "the model knows it fabricates" — representation,
never mind. The probe may read familiarity/topic rather than confabulation per se (disclosed). Label
noise from exact-match aliasing. n_test = 105. Does NOT revive the universal-oracle or cross-vendor
claims (both CLOSED).

## One line

A linear probe on Qwen-3B's late-layer residual detects confident factual confabulation at held-out
AUC 0.74 — the representation carries the signal — but it does not clear the +0.20 margin over the
output, because median-split "confident" left the output partly able to discriminate; the clean
"representation beats blind output" test needs a stricter confidence threshold, pre-registered fresh.
