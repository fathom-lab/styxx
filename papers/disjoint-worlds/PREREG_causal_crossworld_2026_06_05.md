# PREREG — does CAUSAL structure transfer across disjoint worlds through an UNSUPERVISED map? (decisive expt A, causal arm)

**Date (frozen before data):** 2026-06-05. Half A proved the *geometry* of meaning is shared across worlds
with zero shared data (RSA 0.42). But shared geometry could be mere correlation — matching distances
without matching *meaning*. This is the decisive arm the program named (Park–Veitch causal inner product,
ICML 2024) and never ran: learn the cross-world map with **zero paired concepts**, then test whether a
**causal attribute-manipulation direction** transfers through it. If world-A's "attribute a" steering
direction, pushed through the unsupervised map, lands on world-B's *same* attribute and stays orthogonal to
*separable* attributes, the shared geometry is **recoverable, causal meaning** — universal forms in the
strongest testable sense.

## Method (frozen)
- **Worlds.** N concepts, each an L-bit attribute code (the shared latent structure). Two worlds render the
  SAME codes through DISJOINT surfaces: independent random embeddings + independent co-occurrence sampling
  (co-occurrence prob ∝ attribute-similarity). Different embedding dims D_A ≠ D_B. **Zero shared data.**
- **Per-world representations.** Train a skip-gram-style embedding per world on its own co-occurrence stream
  → h^A_i, h^B_i (concept reps). Faithfulness = RSA(h, true-attribute-RDM); sweep it via training/temperature
  to reach the high-faithfulness regime the scaling law requires (>0.95).
- **Unsupervised map (zero pairs).** Recover the cross-world concept correspondence from geometry ALONE
  (Gromov-Wasserstein / Sinkhorn on the two RDMs); then orthogonal Procrustes using the **recovered** (not
  true) correspondence → map M: h^A→h^B. Report recovery accuracy vs the true (held-out) correspondence.
- **Causal directions.** For each attribute a, dir_a = mean(h : a=1) − mean(h : a=0), in each world.
- **Conditions.** structure {DISTINCTIVE (anisotropic attribute weights) vs ISOTROPIC} × faithfulness sweep,
  3 seeds. Isotropic is the negative regime the scaling law says recovery should fail in.

## Hypotheses & predictions (frozen)
- **P1 — causal transfer (the decisive claim):** for the DISTINCTIVE/high-faithfulness regime, the
  cross-world transfer matrix T[a,b] = cos( M(dir^A_a), dir^B_b ) is **diagonal-dominant**: mean diagonal −
  mean off-diagonal ≥ **0.30**, and diagonal beats a shuffled-map null by ≥ **0.30**. The same-attribute
  direction transfers; separable attributes do not leak (causal-inner-product orthogonality, cross-world).
- **P2 — recovery prerequisite:** unsupervised correspondence recovery accuracy ≥ **0.80** in that regime
  (else M is meaningless and P1 is moot). Gates P1.
- **P3 — structure dependence (control):** in the ISOTROPIC regime, recovery fails (acc ≈ chance) AND causal
  transfer collapses (diagonal advantage ≈ 0) — the effect requires distinctive structure, not just any
  geometry. This is the negative control that rules out a metric artifact.

## Decision rule (frozen)
- **CAUSAL UNIVERSALITY — universal forms recoverable & causal** iff P1 ∧ P2 in the distinctive regime AND
  P3 holds (collapses in isotropic). The shared geometry carries transferable causal meaning across zero-
  shared-data worlds: the strongest form of Plato's forms demonstrable in silico.
- **CORRELATION-ONLY / BOUNDED** iff P1 fails while RSA-sharing still holds — the geometry is shared but the
  causal/attribute structure does NOT transfer through an unsupervised map: universality is correlational,
  not causal. Either outcome is decisive and kept.
- Report the full transfer matrix, recovery curve, and the distinctive-vs-isotropic contrast regardless.

## AMENDMENT 2026-06-05 (before the confirmatory 3-seed run; justified by PRIOR committed data, not by this experiment's outcome)
The original **P2 bar (recovery ≥ 0.80) was mis-specified above the instrument's known ceiling.** The
validated scaling results committed 2026-06-03 (`geometry_scaling_result.json`, `geometry_scaling_distinctive.json`)
already show exact unsupervised correspondence recovery is a **sharp threshold capping ~0.6** (distinctive,
faith 0.987) and ~0.10 (isotropic) — so a 0.80 bar can never be met regardless of the science, making the
gate structurally unable to render the verdict the data supports. I should have calibrated to that ceiling
when writing the PREREG. **Correction: P2 → recovery ≥ 0.30** (well above chance 0.01, below the ~0.6
ceiling). **The decisive P1 transfer threshold is UNCHANGED** (diag advantage ≥ 0.30 AND beats the shuffled-
map null by ≥ 0.30) — I have not touched it. Two instrument improvements also applied (more GW restarts; a
higher-faith σ=0.02 to reach the recovery regime) — these harden the aligner, not the gate. The RESULT will
report BOTH the original-bar and amended-bar readings for full transparency.

## Caveats (frozen)
- Synthetic worlds, skip-gram reps, linear attribute directions, an unsupervised matcher that may fail
  outside the high-faithfulness/distinctive regime (a known threshold). This tests causal transfer of LINEAR
  attribute structure through a geometry-only map — the cleanest operationalization of the causal-inner-
  product claim cross-world. Real-scale models and nonlinear concepts remain the open frontier. A clean P1
  is the strongest Half-A result to date; a failure bounds universality to correlation.
