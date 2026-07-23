# FINDING -- the ladder REFUSES on the third task family: pricing is scoped, refusal is not

date: 2026-07-23
subject: part-2e -- the part-2c/2d pricing result re-run on the temporal family, same
heterogeneous cross-model panel, every bar inherited unchanged from the numeric confirmation.
receipts: `papers/anchored-validity/stage_b_fam2_temporal_pricing_result.json`,
`p2c_fam2_temporal_cache.jsonl`, `corpus_label_oracle_receipt.json`.
prereg: `PREREG_part2e_pricing_third_family_temporal_2026_07_23.md` (frozen before any judge ran;
gates copied verbatim from the numeric/attr pricing confirmation).
verdict: **CLOSED_NEGATIVE -- PD2 missed: the ladder does NOT price on temporal, it REFUSES.**
Pricing is scoped to families with recoverable informative structure; on the smooth-violation
family the honest ladder correctly declines to certify. Reported verbatim per the frozen rule.

## the question this answers

Parts 2c (attr) and 2d (numeric) established, on a heterogeneous four-model panel (Qwen2.5
0.5B/1.5B/3B/7B-4bit), that gold anchors license nothing and that same-generator ladder anchors
recover a calibrated prevalence by pricing each judge's error rate. Both families share a property:
an informative judge exists for the honest anchors to find. The prereg picked **temporal** as the
adversarial third family precisely because it does NOT obviously share that property -- temporal is
the smooth-violation family, the one that on the same-model 3B panel (cycle 51) produced the worst
gold-anchor errors in the program and SILENCED the misfit flag (0/15, versus numeric's 15/15). The
pre-named kill path was explicit: if temporal violations are smoothly wrong in a judge-consistent
way, the honest ladder anchors inherit the same blindness, find no informative judge, and VOID
rather than price -- failing PD2. That is exactly what happened.

- PD1, the kill transfers: **PASS.** Blatant gold anchors yield coverage 0/12 across all twelve
  replicates, at a blatant median prevalence error of 0.652 -- the largest gold-anchor error the
  program has recorded. The gold check does not merely fail to help; on every replicate the gold
  anchors certify the broken panel with maximal confidence, driving the estimate to pi 1.0 against
  a true prevalence near 0.31 (everything a contradiction). Gold anchors license nothing on
  temporal too, and here they are most dangerous.
- PD2, the pricing claim: **MISS.** The ladder returns VOID_PANEL__uninformative on 12/12
  replicates: zero ESTIMATED, so no coverage and no error margin to compare. The honest
  same-generator anchors reveal that no judge clears the informativeness gate on temporal -- the
  four judges score at or below chance on the before/after task, and the ladder, refusing to
  average garbage, declines to certify. This misses the frozen PD2 bar (needs >=8 ESTIMATED and
  ladder coverage >=10/12) and is logged as CLOSED_NEGATIVE, not rescued.
- PD3, the refusal reality: **PASS.** Every deaf replicate VOID, 12/12.

## what this establishes

The pricing mechanism is **scoped, and the scope is honest.** On attr and numeric the ladder
prices a mostly-garbage panel; on temporal it refuses -- and refusal is the ladder doing the OTHER
right thing, not a defect. Temporal now joins chain (cycle 53, ladder VOID 14/15) as the second
family where the honest ladder correctly returns a refusal rather than a number. The four families
partition cleanly: where an informative judge exists (attr, numeric) the ladder PRICES; where none
does under honest anchors (chain, temporal) the ladder REFUSES -- and it NEVER certifies garbage.
The sharper reading is the alignment of the two failure modes: the two families where gold anchors
are most dangerous (temporal and chain -- silent misfit flags, worst errors) are exactly the two
where the honest ladder refuses. The instrument's refusal fires precisely where the sanity check's
false confidence is most damaging. The paper's model-generality claim tightens to what is measured:
the kill (gold anchors license nothing) holds on four families and a heterogeneous cross-model
panel; the ladder's corrective POWER holds on the two families with recoverable structure and is
replaced by a correct refusal on the two without -- not a universal repair, a repair-or-refuse
instrument. Corpus labels for both scored temporal arms were re-derived from item text by the
temporal oracle with zero mismatches (`corpus_label_oracle_receipt.json`, 4320 blatant + 4320
ladder items, 0 undecidable). Scope, stated plainly: one model family across four sizes, three
constructed oracle-certified families, twelve seeds, every gate frozen before the panel ran. The
motivating attr and numeric pricing results are untouched; no bar was moved to convert this miss
into a pass.
