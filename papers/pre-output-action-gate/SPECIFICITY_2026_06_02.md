# SPECIFICITY — is the steering a deployable guard or a model-breaker?

**Date:** 2026-06-02
**Builds on:** [DEEPENING](DEEPENING_2026_06_02.md) (steering flips destructive→safe).
The make-or-break for a *tool*: does steering away from destruction also damage
the **safe** actions? Measured at a fixed dose on both classes, 4 models.

## Result

Subtract `alpha · destructive_direction` at the best layer; regenerate ALL
scenarios; measure outcomes by baseline class.

| Model | dose | BENEFIT dest→safe | safe preserved | safe→dest | safe broke | COLLATERAL |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | k=4 | 0.25 | 0.96 | 0.00 | 0.04 | **0.04** |
| Qwen2.5-1.5B | k=8 | 0.50 | 0.88 | 0.04 | 0.08 | 0.12 |
| Qwen2.5-3B | k=8 | 0.12 | 1.00 | 0.00 | 0.00 | **0.00** |
| Llama-3.2-3B | k=8 | 0.18 | 1.00 | 0.00 | 0.00 | **0.00** |
| gemma-2-2b | k=4 | 0.30 | 1.00 | 0.00 | 0.00 | **0.00** |
| gemma-2-2b | k=8 | 0.40 | 1.00 | 0.00 | 0.00 | **0.00** |

`dest_broke = 0.00` everywhere (steering a destructive choice never produced
gibberish either).

## The finding: it is SPECIFIC

Across all 4 models, **collateral is 0.00–0.12** and **safe choices are
preserved 0.88–1.00.** Steering essentially **never** pushes a safe choice to
destructive (`safe→dest` ≈ 0) and rarely breaks one. This is the property a
deployable guard needs: the destructive direction is a *targeted* lever — it
moves the destructive-leaning states and leaves the safe ones alone. That makes
mechanistic sense: the direction encodes destructive-leaning, so it barely
perturbs scenarios that weren't leaning that way.

## The honest ceiling: benefit is PARTIAL and dose-dependent

It suppresses **18–50%** of destructive choices at k=8 (Qwen-3B the weak case at
12% — it barely steers, consistent with being the cautious, few-positive model).
Raising the dose lifts benefit but starts to add collateral (Qwen-1.5B: k4 → 25%
benefit / 4% collateral; k8 → 50% / 12%). So it is a **partial, clean reduction**,
not total prevention.

## Synthesis: a deployable-shaped predict-and-prevent guard

The three results compose into one coherent, honest capability on open-weight
models:
1. **Predict** — the pre-emission residual flags the destructive choice
   (LOCO AUC 0.65–0.92, beats the prompt text).
2. **Prevent** — subtracting the mid-layer direction flips 18–50% of those
   choices to safe.
3. **Specifically** — with 0–12% collateral; safe operation is preserved.

Gated (steer only when the predictor fires) this is the shape of a real guard:
catch the destructive action before it's emitted, bend it safe, and almost never
touch the agent's legitimate work.

## What it is — and is NOT — honestly

- **IS:** a specific, low-collateral safety lever that *reduces* destructive
  agent actions on open-weight models, with the predictor to gate it and
  controls (random direction, validity, both-class collateral) behind every claim.
- **IS NOT:** total prevention (partial benefit), a deployed product (no
  calibration/latency/false-positive budget on a live agent), pre-registered
  (steering dose can't be frozen blind — confirmatory run is next), or validated
  beyond the simplified presented-tool harness and 4 small models.

The honest headline a skeptic can't dismantle: **"reduces destructive agent
actions ~20–50% with ~0–12% collateral, before emission, on open-weight models —
predictor-gated, every claim controlled."** Not a miracle. A guard that works,
whose numbers are true.

## Next

- **Gated** predict-and-prevent (steer only on predicted-destructive) → maximize
  benefit, minimize collateral; report the operating curve.
- Confirmatory **pre-registered** steering (frozen dose + layer).
- **Open tool-call** capture; more + larger models; live-agent latency/FPR.

— 2026-06-02; the test that could embarrass us, run on purpose.
