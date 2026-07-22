# FINDING -- the ladder prices on a SECOND task family: pricing is the instrument, not the corpus

date: 2026-07-22
subject: part-2d -- the part-2c pricing result re-run on the numeric family, same heterogeneous
cross-model panel, every bar inherited unchanged from the attr confirmation.
receipts: `papers/anchored-validity/stage_b_fam2_numeric_pricing_result.json`,
`p2c_fam2_numeric_cache.jsonl`, `corpus_label_oracle_receipt.json`.
prereg: `PREREG_part2d_pricing_second_family_2026_07_22.md` (frozen before any judge ran; gates
copied verbatim from the attr pricing confirmation).
verdict: **SURVIVED -- all three frozen PD gates pass on numeric.** The "price not drop"
mechanism is a property of the instrument, not an artifact of the attribution task.

## the question this closes

Part 2c established, on a heterogeneous four-model panel (Qwen2.5 0.5B/1.5B/3B/7B-4bit), that
gold anchors license nothing and that same-generator ladder anchors recover a calibrated
prevalence by pricing each judge's error rate -- but only on the **attr** family. That left one
honest reading open: maybe pricing works because attribution items happen to expose judge error
in a ladder-friendly way. This cycle re-runs the identical apparatus -- byte-identical judge,
prompt, decoding, panel, and gates -- on the **numeric** family and asks whether the same three
gates still fire. They do, on twelve seeds disjoint from every attr seed the arc has scored:

- PD1, the kill transfers: blatant gold anchors yield zero coverage across all twelve replicates,
  at a median prevalence error of 0.140 -- three judges sit near chance and one is perfect, and
  the gold check certifies the broken majority.
- PD2, the pricing claim, both halves: every one of the twelve ladder replicates ESTIMATED and
  covered, at a ladder median error of 0.016 -- a margin of 0.124 below the blatant arm on the
  same replicates, clearing the preregistered 0.08 bar. No ladder replicate refused; the ladder
  did not need to know which judge to trust, it priced all four.
- PD3, the refusal reality: every deaf replicate VOID.

## what this establishes

The sharpest property of the arc -- ladder anchors correct a mostly-garbage heterogeneous panel
by measuring and subtracting each judge's organic error rate rather than discarding judges -- now
holds on two task families, with the numeric error even lower than the attribution one and the
margin comparable. Pricing is behaviour of the auditor, not a feature of the corpus it happened
to be discovered on. Combined with the part-2c attr result, the model-generality residual named
at cycle 58 is closed on its last open sub-item: anchor non-transfer and the ladder's corrective
power both reproduce across four base models spanning a 14x size range AND across two independent
task families. Corpus labels for both scored numeric arms were re-derived from item text by the
family oracle with zero mismatches (`corpus_label_oracle_receipt.json`). Scope, stated plainly:
one model family across four sizes, two constructed oracle-certified families, twelve fresh seeds,
every gate frozen before the panel ran. The remaining residuals are unchanged and named -- a
genuine frontier panel (blocked on Fable credits) and an in-the-wild eval -- neither of which this
run claims to have touched.
