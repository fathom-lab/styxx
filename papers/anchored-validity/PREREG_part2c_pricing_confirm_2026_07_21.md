# PREREG — part 2c pricing confirmation: the "price not drop" mechanism on FRESH seeds

date: 2026-07-21
status: FROZEN before the confirmation run; committed with the seed extension.
motivation, disclosed: part 2c's X3 gate (drop-based) was CLOSED_NEGATIVE, and the realized
mechanism was ladder anchors PRICING the bad judges (coverage 12/12, median err 0.019 vs
blatant 0.165). Re-scoring the SAME seeds with a pricing gate would be post-hoc rescue -- adding
a favorable claim on the data that motivated it, which the program forbids. This confirmation
runs FRESH seeds (12013-12024, disjoint from the motivating 12001-12012) so the pricing gate is
tested on data it did not shape.

## frozen design

Identical to part 2c in every respect except the seeds: same four-model heterogeneous panel
(Qwen2.5 {0.5B,1.5B,3B}-Instruct fp16 + 7B 4-bit), same attr family, same both-true phrasing,
same blatant + ladder arms over the same per-model verdicts, same deaf arm, n_organic 200,
K 80, pi 0.35, R = 12 (seeds 12013-12024), audit_panel n_boot 300 null_sims 200.

## frozen gates

- **PC1 (kill replicates):** blatant coverage <= 3/12 -- the heterogeneous kill reproduces on
  fresh seeds.
- **PC2 (the pricing claim, the load-bearing gate):** ladder coverage >= 10/12 among ESTIMATED
  (>= 8 ESTIMATED required) AND ladder median prevalence error at least 0.08 below the blatant
  arm's median error on the same replicates. Both halves required. This is the gate matched to
  the mechanism: the ladder recovers a calibrated prevalence from a mostly-garbage panel by
  measuring and subtracting each judge's organic error rate, without needing to identify which
  judges are good.
- **PC3 (deaf reality):** deaf VOID >= 11/12.

Characteristics unbarred: per-judge ladder-measured alpha vs true organic false-fire (the
pricing fidelity), kept masks, misfit flags. A missed gate is CLOSED_NEGATIVE verbatim; no bar
moves after the first scored token; diagnostics/motivating seeds stay disjoint.
