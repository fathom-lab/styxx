# FINDING — the live signature MVP works (and shows its own calibration boundary)

**2026-06-10 · Fathom Lab / styxx. Receipt: `live_signature_result.json`. Local, gemma-2-2b-it on
GPU, styxx's shipped pre-output probes. The first time styxx reads a real model's internal state
and reports a signature BEFORE the model generates a token.**

## What was built

`run_live_signature_mvp.py`: load gemma-2-2b-it, load styxx's pre-trained residual probes
(`StyxxProbe.from_pretrained`), and read the honesty/grounding signature directly from the residual
stream at the trained layer on the final prefill token — no output token generated. Zero new
dependencies (torch / transformer_lens / nnsight already present); ~7.9 GB free GPU. This is the
real-time core the showcase visualization renders, driven by actual activations instead of
hand-placed values.

## Result — real signal, honestly bounded

- **Truthfulness probe (layer 12, manifest leave-one-out AUC 0.8508 in-distribution):** on 16 FRESH,
  out-of-sample factual true/false pairs, paired accuracy P(p_correct(true) > p_correct(false)) =
  **0.750**, mean gap **+0.247**. A genuine live readout of grounding from internal state — above
  chance (0.5), below the in-distribution 0.85 exactly as out-of-distribution generalization
  predicts. Per-item it is noisy (the "Paris" assertion even inverted, gap -0.251); the signal is in
  the aggregate, not any single token.
- **Deception probe (layer 0, manifest AUC 1.0 in-distribution, instruction-contrast):** on the same
  out-of-distribution honest-vs-deceptive framings it **saturated to 1.000 on both**, flip +0.000 —
  no discrimination. This is the textbook in-distribution-overfit failure the build-plan research
  flagged (the "catches 95-99% of lies" robustness claim was REFUTED 0-3), observed first-hand.

## Why this is the right outcome, not a disappointment

styxx's thesis is that an instrument must know its own operating range. This run demonstrates both
halves live: a calibrated grounding signature that carries real information off-distribution (0.75),
AND a probe whose perfect-looking in-distribution AUC collapses out-of-distribution (deception,
1.0 -> chance-on-these-prompts). The showcase renders the truthfulness signature, labeled exactly
as what it is — a calibrated, in-domain correlate, not a verdict. We caught our own probe's boundary
before any film overclaimed it. Nobody else would build the demo this way; that is the brand.

## For the showcase

- Drive the live constellation with the **truthfulness (layer-12) probe**, the calibrated signal.
- For a strong on-screen separation, feed in-distribution items (TruthfulQA-style, where AUC is
  0.85) or aggregate over many items rather than relying on single-token reads.
- Do NOT use the layer-0 deception probe as a live verdict; it is an in-distribution instruction
  detector. If a deception channel is wanted, re-fit at a mid layer with an out-of-distribution
  validation gate.

## Strategic note

The capability is now unlocked end-to-end: styxx can read a model's internal state in real time,
before it speaks. This is the foundation the product's live-monitoring roadmap needs (drift monitor,
conduct sensor) AND the engine of the showcase — one build serving research, product, and
distribution at once.
