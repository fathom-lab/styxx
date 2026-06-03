# PREREG — Geometry scaling law: does convergence reach full universality (and recovery emerge) with faithfulness?

**Date:** 2026-06-03
**Status:** PRE-REGISTERED (gate frozen before data).
**Completes the geometry half.** The Disjoint-Worlds result was **UNIVERSAL but partial**
(same-structure RSA 0.42 at embedding-faithfulness 0.51) and the unsupervised
correspondence-recovery (Gromov-Wasserstein) **failed at that scale.** The result said the
convergence *tracks faithfulness*. This experiment tests that directly: as two independent
representations become more faithful to the shared latent structure, (a) does their RSA rise
toward 1, and (b) does **unsupervised recovery emerge** above some faithfulness — reproducing
the vec2vec phenomenon and explaining why GW failed at 0.5?

## Design (frozen)

Shared latent geometry z (N=100, dim 8). Two representations of it, built to a CONTROLLED
faithfulness by varying noise: `E_A = normalize(z R_A) + σ·ξ_A`, `E_B = normalize(z R_B) +
σ·ξ_B`, with **independent random projections R_A (8→32), R_B (8→24)** (different dims) and
**independent noise** ξ. Sweep σ to span **faithfulness ∈ [~0.2, ~0.99]** (faithfulness ≡
RSA(D(E_A), D(z)), measured at each level). At each level measure:
- **same-structure RSA** = RSA(D(E_A), D(E_B))  (do the two converge?)
- **control RSA** = RSA(D(E_A), D(E_B′)) with E_B′ from a DIFFERENT latent z′  (must stay ≈0)
- **unsupervised recovery** = Gromov-Wasserstein top-1 correspondence accuracy (chance 1/N)
3 latent seeds. **Anchor:** the real SGNS-trained models from the prior run sit at
faithfulness ≈ 0.51, same-RSA ≈ 0.42 — plotted on the same curve.

## KILL-GATE (frozen)

- **SCALING CONFIRMED:** same-structure RSA **increases monotonically with faithfulness and
  reaches ≥ 0.9** near faithfulness ~0.95; control stays ≈ 0; **AND unsupervised recovery
  rises from chance to ≥ 0.5** above some faithfulness threshold (recovery *emerges*). →
  convergence scales to full universality and the vec2vec recovery phenomenon is reproduced;
  the partial synthetic result is one point on this curve. Explains the GW failure (faithfulness
  0.5 is below the recovery threshold).
- **PLATEAU:** RSA rises with faithfulness but plateaus well below 0.9, or recovery never
  emerges even near faithfulness 1 → the convergence is intrinsically partial / recovery needs
  more than faithfulness (report the ceiling and the threshold).
- **FLAT:** RSA does not track faithfulness → the prior "tracks faithfulness" reading was wrong
  (would contradict the prior result; treat as a red flag to diagnose).

## Honest prior / caveats (frozen)

- Honest prior: SCALING CONFIRMED — RSA→1 and recovery emerges as noise→0 (at faithfulness ~1
  the two are near-isometric, where GW already scored 0.68–1.0 in validation). The load-bearing
  measurements are the **recovery-emergence threshold** (at what faithfulness does unsupervised
  recovery cross chance→useful?) and where **real SGNS models sit** on the curve.
- This is a controlled-faithfulness characterization (noise-controlled embeddings), anchored to
  the real SGNS-trained point — it isolates the *faithfulness → convergence* law cleanly. It is
  not itself a new "models converge" claim (that's the prior result); it explains *how* the
  convergence scales and *why* recovery needs scale.
- GW is the same validated aligner (identical→1.0, isometry+noise→0.68, different→0.03); its
  weakness at low faithfulness is exactly what this maps.

— frozen 2026-06-03
