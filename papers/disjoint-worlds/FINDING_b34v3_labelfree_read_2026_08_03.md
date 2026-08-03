# FINDING — the label-free telepathy READ clears the bar: two model families aligned with zero labels, content read across the gap

Fathom Lab · 2026-08-03 · scored MECHANICALLY by `styxx.protocol` against the frozen
`gates` block in `PREREG_b34v3_labelfree_read_2026_08_03.md` (committed `aea406f` before the
apparatus `75684e9` existed). Receipts: `b34v3_result.json`, `b34v3_fresh_split_addendum.json`.
Prior-run comparators: `../synthesis_minds_addendum.json`.

## Verdict: `TELEPATHY_READ_BAR_CLEARED__labelfree_pairing_reads_crossfamily`

The parked family's third attempt, redesigned from what v2 measured: drop the MLP iteration
loop that degraded the same-family read, keep the committed linear machinery as the pairing
*discoverer*, fit ONE MLP on the discovered pseudo-pairs, read once. **Zero labels anywhere in
fitting** — the correspondence between two model families is recovered from geometry alone.

| target | seed_acc (discovery) | read top-1 | × chance | null |
|---|---:|---:|---:|---:|
| llama_1b (same family, G0) | 0.7959 | 0.6857 | 48× | 0.0143 |
| **gemma_2b (decisive, G1)** | **0.5918** | **0.5714** | **40×** | 0.0143 |
| qwen_1p5b (context) | 0.0536 | 0.1429 | 10× | 0.0286 |

`styxx.protocol` evaluated the gates and walked the frozen outcome table: G0_discovery ✓
(0.7959 ≥ 0.30), G1_bar ✓ (0.5714 ≥ 0.143), G2_null ✓ — verdict computed, not chosen.
**gemma — a model that read at EXACT chance through a linear map — reads content from a
different family at 40× chance with no labels in the loop.** The read half of the
telepathy-shaped claim, label-free.

## The correction the prereg forced (stated as loud as the result)

**The frozen prereg claimed the seed-343 held-out set is "disjoint in membership" from v1/v2's.
Verification falsified that: 13 of 70 concepts overlap** (`b34v3_fresh_split_addendum.json`). A
random 70-of-462 draw overlaps another such draw by ~11 on average, so the overlap is
expected-by-chance, not a bug — but I asserted disjointness in a frozen document, and it was
wrong. The prereg stays frozen and wrong; this erratum carries the correction (the v31.1
lineage rule). **Recomputed on the 57 genuinely-unseen concepts alone (still scored 70-way,
chance 1/70): gemma 30/57 = 0.5263 (37× chance), llama 38/57 = 0.6667.** The 13 shared concepts
were not carrying the result — the claim stands on items no map has ever seen. And nothing was
tuned per-concept: the method (linear-discover + single MLP) was fixed by mechanism, so the
shared items were never a fitting surface, only test items in both runs.

## The nuances that keep it honest

1. **G2's null passed at exactly the boundary.** qwen's shuffled-pairing read was 0.0286 = 2 of
   70 = 2× chance, and the gate is ≤ 2× chance. Two coincidental hits (Poisson expects ~1) — a
   knife-edge pass, disclosed. llama and gemma nulls sat at exact chance (0.0143), so the
   decisive gemma cell is not near the edge; qwen's context cell is.
2. **Cross-family discovery is gemma-strong, not uniform.** gemma's pairing was discovered at
   seed accuracy 0.5918; qwen's at only 0.0536 — barely above pairing-chance. qwen's 10× read
   therefore rests on a *poor* discovery and is context, not headline. The clean, licensed claim
   is the gemma cell.
3. **This is the READ half only.** b36 settled that control does NOT cross even with this
   machinery (`READ_NEQ_WRITE_SURVIVES_CAPACITY`). This finding does not touch write.

## What it establishes — and its scope

Between two different model families (Llama-3.2-3B → gemma-2-2b), the correspondence needed to
read one from the other is **discoverable from representational geometry with no labels**, and a
nonlinear lens reads held-out content through the discovered pairing at 37–40× chance on unseen
concepts, nulls at (or one hit above) chance. This is stronger on the label-free axis than
b31v2, which used 392 *true* pairs — here the pairing is found, not given. Bounded to: this map
class, ≤3B models, one source family, one fresh seed, 70-way identification, one strong-discovery
target (gemma) with a second (qwen) whose discovery was weak. The label-free *write* channel
remains closed (b36); label-free read across families is now open.

## Method-discipline note

First finding whose verdict was produced by `styxx.protocol` from a machine-readable gates block
frozen in git — the agent reported the verdict, it did not choose it. And the first where the
frozen prereg carried a factual error that verification caught: the process worked on its author.
