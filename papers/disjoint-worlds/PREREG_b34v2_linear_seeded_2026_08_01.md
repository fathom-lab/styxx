# PREREG — B34-v2: label-free nonlinear, seeded by the committed linear machinery

Fathom Lab · 2026-08-01 · frozen before the scored run. Successor to
`PREREG_b34_labelfree_nonlinear_2026_08_01.md`, which returned `INVALID__pipeline_broken`
under its own G0 (`b34_result.json`): the same-family cell read 0.1571 against the 0.29
floor, so the gemma observation (0.2571, 18× chance, nulls clean) is UNLICENSED and is not
claimed. **The measured mechanism:** the v1 pipeline seeded from ONE raw entropic-GW plan;
its same-family seed accuracy (0.066) was no better than cross-family (0.071/0.026) — far
below what the committed linear label-free machinery achieves, since `_fit_Q` refines its GW
warm start with Sinkhorn-annealed Procrustes over restarts. v1 broke at the initializer, not
the thesis.

## The one change (frozen)

Replace the initializer only: stage 1 = the COMMITTED linear label-free pipeline itself
(`TransferMap.fit` on the seeded-shuffled fit rows — GW warm start + annealed Procrustes +
restarts, unchanged); the initial pseudo-pairing is the assignment induced by the fitted
linear map in full space (`linear_sum_assignment` on transfer-mapped distances). Stage 2 =
the v1 nonlinear iteration verbatim (b31v2 MLP, 8 refinement iterations, seed 34). No other
change; labels still touch nothing but held-out scoring and the reported diagnostic.

## Arms and gates (identical to v1, restated)

M-LF (linear-seeded iterative MLP), N0 (random pairs, no iterations), R0 (random init + 8
iterations, reported not gated).

- **G0 (machinery):** llama_1b M-LF ≥ 0.29. Fail → `INVALID__pipeline_broken` (and the
  family is parked pending a redesign prereg — no third same-day patch).
- **G1 (the bar):** gemma_2b M-LF ≥ 0.143 → `TELEPATHY_BAR_CLEARED__pairing_discoverable`.
- **G2 (null):** N0 ≤ 2× chance everywhere.
- **G3 (pre-committed negative):** G0+G2 pass, G1 miss →
  `PAIRING_NOT_DISCOVERABLE__at_this_class`.

The v1 INVALID stays on the record beside this run either way. Same data, same seeds, same
metric, CPU-from-cache, smoke INVALID-only.
