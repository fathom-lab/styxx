# FINDING -- the ladder prices, confirmed: fresh-seed gate matched to the mechanism

date: 2026-07-21
subject: part-2c pricing confirmation -- the "price not drop" mechanism on FRESH seeds
(12013-12024), the gate the drop-based X3 could not credit, frozen before the run.
receipts: `papers/anchored-validity/stage_b_crossmodel_pricing_result.json`,
`p2c_crossmodel_cache.jsonl`.
prereg: `PREREG_part2c_pricing_confirm_2026_07_21.md` (frozen; motivating seeds 12001-12012
held disjoint).
verdict: **SURVIVED -- all three frozen PC gates pass.** The heterogeneous kill reproduces on
fresh seeds and the ladder recovers a calibrated prevalence by pricing every judge's error rate.

## the claim, now gated on data it did not shape

Part 2c measured that same-generator ladder anchors recover pi from a mostly-garbage
heterogeneous panel by measuring and subtracting each judge's organic error rate rather than
discarding bad judges -- but the frozen X3 gate had asked for dropping, so the pricing result
shipped only as an unbarred characteristic. This confirmation freezes the matched gate BEFORE
the run and tests it on twelve seeds disjoint from the ones that motivated it. On those fresh
seeds:

- PC1, the kill reproduces: blatant gold anchors yield zero coverage across all twelve, at a median
  prevalence error of 0.158 -- the two capable-looking judges (1.5B, 3B) pass the sanity checks and drag the audit.
- PC2, the pricing claim, both halves: ladder coverage of all twelve ESTIMATED replicates,
  ladder median error 0.027, a margin of 0.131 below the blatant arm on the same replicates --
  clearing the preregistered 0.08 margin with room. The ladder did not need to identify which
  of four heterogeneous judges to trust; it measured each judge's error rate on representative
  items and the moment system did the rest.
- PC3, the refusal reality: every deaf replicate VOID.

## what this establishes

The sharpest new property in the arc is now a gated result, not a post-hoc reading: on a
genuinely heterogeneous, mixed-capability, less-correlated panel -- the realistic deployment
that gold checks cannot audit -- same-generator ladder anchors recover a calibrated prevalence
by pricing, at an error an order of magnitude below the gold-anchor audit. Combined with the
part-2c kill result (gold anchors license nothing on the same panel, gated), the model-
generality residual is substantially closed: anchor non-transfer and the ladder's corrective
power both hold beyond persona-correlated panels, on four different base models spanning a 14x
range. Scope unchanged: one task family, one model family across four sizes, constructed
oracle-certified corpora, every gate preregistered -- and the pricing gate confirmed on seeds
it did not shape, which is the whole point of running it separately. The paper's next version
gains a gated pricing claim; the frontier-panel arm still waits on credits.
