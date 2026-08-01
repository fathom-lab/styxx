# PREREG — B34: label-free nonlinear content transport (the actual telepathy bar)

Fathom Lab · 2026-08-01 · frozen before any scored cell. Spawned by
`FINDING_b31v2_door_opens_2026_08_01.md`: a paired two-layer MLP reads cross-family content at
49–55× chance where the label-free linear class read 1–8× — the information is present and
alignable. This experiment asks the remaining question, the one the word *telepathy* actually
requires: **can the correspondence itself be discovered with no labels anywhere in fitting —
and still read held-out content across families?**

The program's linear pipeline is already fully label-free (`_fit_Q`: "rows = same concepts,
hidden order" — entropic-GW warm start + Sinkhorn-annealed Procrustes recovers the pairing
unsupervised). Its cross-family failure (gemma at exact chance) is therefore a label-free
correspondence-plus-linear-lens failure. B34 keeps the unsupervised correspondence discovery
and makes the lens nonlinear, iterating between the two.

## Design (frozen)

- **Data:** the b31v2 banked extractions VERBATIM (`_b31v2_ptsA.npz`, `_b31v2_pts_*.npz`) —
  462 concepts, the committed split (392 fit / 70 held-out). The true correspondence is used
  ONLY to score held-outs and for a per-iteration pseudo-pair-accuracy diagnostic (reported,
  never fed back). Before fitting, the target's 392 fit rows are SHUFFLED by a seeded
  permutation so aligned-order leakage is impossible by construction.
- **M-LF (the pipeline):** (1) per-side distance matrices (`distmat`, the committed metric)
  over the 392 fit points; (2) `entropic_gw(D_A, D_B)` → assignment → pseudo-pairs; (3) for
  t = 1..8 (frozen): fit the b31v2 MLP (same architecture, training, seed discipline; seed
  34) on current pseudo-pairs, recompute the assignment by `linear_sum_assignment` on
  mapped-A-to-B distances, update pseudo-pairs; (4) final MLP scores the 70 held-outs with
  the b31v2 read metric verbatim.
- **N0 (null):** random initial assignment, zero iterations — an MLP on random pairs (the
  b31v2 N1 shape, re-run inside this pipeline). Must sit ≤ 2× chance everywhere.
- **R0 (secondary arm, reported not gated):** random initial assignment WITH the full 8
  iterations — does the iteration alone find the alignment without the GW seed? Either
  outcome is informative and neither is gated.
- **Targets:** llama_1b (G0 machinery), gemma_2b (decisive), qwen_1p5b (context).

## Gates (frozen; no optional stopping)

- **G0 (machinery):** same-family M-LF held-out top-1 ≥ 0.29 (the measured label-free linear
  0.3429 minus 0.05 — the nonlinear label-free pipeline must not materially degrade the
  linear label-free read where that read already works). Fail → `INVALID__pipeline_broken`.
- **G1 (the bar):** gemma_2b M-LF ≥ 0.143 (10× chance; the b31v2 bar, now with zero labels
  anywhere in fitting). Pass → `TELEPATHY_BAR_CLEARED__pairing_discoverable`.
- **G2 (null):** N0 ≤ 2× chance on every target. Fail → `INVALID__pipeline_artifact`.
- **G3 (the pre-committed negative):** G0+G2 pass, G1 miss →
  `PAIRING_NOT_DISCOVERABLE__at_this_class`: the content is alignable (b31v2) but the
  correspondence is not label-free-discoverable at this scale/objective — a measured
  boundary, full result. Successors (larger anchor sets, cycle-consistency objectives,
  distribution-level matching) each need their own prereg.

## Compute & discipline

CPU/GPU-light, zero model loads (all extractions banked) — immune to the shard-load kill
class by construction. Deterministic seeds; smoke = 40 fit / 10 held-out concepts, writes
`_smoke` files, INVALID-only. Every number in the FINDING re-derives from `b34_result.json`;
OATH-certified before commit.
