# PREREG — Universal cross-model thought/control transfer

**Frozen 2026-06-06, before any transfer steering number was read. Fathom Lab / styxx.**
Gates hardened by the design red-team (`tasks/wptyjvpjs.output`).

## Claim

A concept STEERING direction computed only in model A can be installed in model B through the
**unsupervised zero-anchor map** (Wasserstein–Procrustes over concept clouds, target concept held
out of the map) and produce a **concept-specific** steering effect in B — graded by the pair's
isometry.

## Apparatus (reused, verified)

`styxx_transfer.TransferMap` (PCA→orthogonal Q→inverse-PCA; Sinkhorn-annealed WProc; GW warm-start;
structure-only inits, **no identity init**). Injection: rms all-position hook at 0.60-depth, dose
re-locked in B. Steering metric: MiniLM concept-sim gain of B's generation minus clean (the
`steering_gain` form). v_B′ L2-renormalized to native norm (norm not a confound).

## Held-out / anti-leakage

Leave-out protocol: the map's PCA bases and Q are fit on TRAIN concepts only; transferred concept
directions are HELD OUT (never seen by the map). Steering measured on held-out concepts.

## Frozen kill-gates (per model pair)

- **G0 — instrument (interpretability gate):** real-geometry transfer positive control (A→A-rotated,
  zero-anchor) mean |cos| ≥ **0.80**. If it fails → **ALIGNER_LIMITED**, a null is *not* interpretable
  (the anti-"false-falsify-Plato" guard).
- **G1 — effect:** mean transfer gain TG ≥ **0.15**.
- **G2 — vs random/raw:** TG − max(TG_randomQ, TG_rawA) ≥ **0.10** (paired sign test p<0.05).
- **G3 — ceiling:** NTE = TG_transfer / TG_native ≥ **0.40** (recovers ≥40% of B's native steering).
- **G4 — specificity:** TG_transfer(c→c) − TG_wrong(c′→c) ≥ **0.10**.
- **G5 — map-not-anything:** TG_transfer − TG_randomQ ≥ **0.10**.

## Verdict logic

- G0 fails → **ALIGNER_LIMITED** (uninterpretable).
- G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5 → **TRANSFER WORKS** (zero-anchor cross-model control transfer).
- G1 ∧ G2 but G3<0.40 or G4 weak → **PARTIAL** (real but lossy transfer).
- G1 or G2 fails → **REPORT_AS_LANDED null** (no zero-anchor transfer at this isometry).

## Pre-registered prediction (from the verified isometry→recovery curve)

- Llama-3.2-3B → Llama-3.2-1B (RSA 0.97): **SURVIVES**.
- Llama-3.2-3B → gemma-2-2b (RSA 0.79, cross-family): **PARTIAL**.
- Any pair RSA < 0.70: **NULL**.
- If even the same-family RSA-0.97 pair fails G1, the transfer thesis is **FALSIFIED**.

## Honest ceiling

A SURVIVED = "a control computed once in one model steers another with zero shared data, in the
near-isometric regime" — a real universal-conscience-mount primitive, NOT "any model can be
controlled from any other." Cross-family is expected partial/lossy; shared training data is not
controlled (this is the practical, not data-independent, frontier).
