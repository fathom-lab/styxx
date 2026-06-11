# FINDING — the portable conscience survives OUT-OF-DISTRIBUTION (OOD-PORTABLE)

**2026-06-11 · Fathom Lab / styxx. Pre-registered: `PREREG_portable_conscience_ood_v2_2026_06_11.md`
(frozen pre-run; the v_ood_1 prereg + its VOID receipt are part of the record). Receipts:
`portable_conscience_ood_v2_result.json` (the answer) and `portable_conscience_ood_result.json`
(the VOID that forced the correct null). This extends the in-distribution v2 result
(`FINDING_portable_conscience_v2_2026_06_10.md`) across distribution shift.**

## Result — one honesty direction, fit on one family-set, reads truth in unseen models on unseen families

Leave-families-out: gemma-2-2b's layer-12 difference-of-means honesty direction AND the label-free
cross-model map were fit ONLY on four train fact-families (capitals, chemical-elements, arithmetic,
biology-classification). They were then tested on four DISJOINT out-of-distribution families with
different domains and templates (historical-dates, comparatives, geography-location,
definitions-properties) — which never touched the direction, the map, or layer/alpha selection.

| target | OOD AUROC | permutation-null p95 | p-value | drop-geography OOD | drop-geo p |
| --- | --- | --- | --- | --- | --- |
| Llama-3.2-3B | **0.923** | 0.830 | **0.003** | 0.898 | 0.009 |
| Qwen2.5-3B | **0.849** | 0.802 | **0.009** | 0.766 | 0.017 |

Both primary targets clear the 0.65 bar AND beat the **label-permutation null** — honesty directions
built from random source-label bipartitions pushed through the SAME map (k = 1000). The true-label
direction transfers out-of-distribution significantly better than chance bipartitions (p 0.003 / 0.009),
and the effect SURVIVES removing the one trivially-separable family (geography = 1.0): on the remaining
three families it still beats the null (drop-geography p 0.009 / 0.017). Two smaller secondary models
concur (Llama-3.2-1B p 0.004; Qwen2.5-1.5B p 0.002). gemma's own OOD self-readout is 0.929 (the source
direction itself generalizes across families in-model), so the test is valid, not void.
**Verdict per the frozen gate: OOD-PORTABLE.**

## Why this is the honest positive, not an overclaim

- **The map really does transport broad truth structure.** The permutation null sits HIGH (0.830 /
  0.802) and so does the random-direction floor (0.838 / 0.798): once the label-free map is fit on the
  train families, OOD true/false becomes broadly linearly readable in the aligned space along many
  directions. We do not hide this — it is exactly why v_ood_1 was VOID. The claim is narrower and
  earned: the SPECIFIC honesty direction adds significant signal ON TOP of that transported structure.
- **Modest margins.** Llama 0.923 vs null 0.830; Qwen 0.849 vs 0.802. Significant (p < 0.01) and
  robust, but a margin, not a chasm. The random-MAP control collapses (0.530 / 0.435), so the LEARNED
  alignment is necessary.
- **No OOD degradation.** Retention (OOD / matched in-distribution) is 1.086 (Llama) and 1.080 (Qwen):
  honesty transfer does not decay under this distribution shift — if anything the small in-distribution
  test slice reads lower than the larger OOD set.

## Why v_ood_1 was VOID, and why fixing it is discipline not goalpost-moving

v_ood_1 returned OOD AUROC 0.923 / 0.829 but a frozen guard fired: Llama's random-direction floor_p95
hit 0.865 (>= 0.78). The random-direction floor is the WRONG null for a difference-of-means direction —
it asks "does ANY direction separate transported truth" (yes), not "is the TRUE-label honesty direction
special." v2 replaced it with the correct label-permutation null (refit the direction on shuffled
labels) and pre-registered that the elevated random-direction floor is expected and is now descriptive,
not the gate. The bar did not move — the null became correct. The random-direction floor is still
reported (0.838 / 0.798), and the honesty direction beats THAT too. This is the v0 -> v1 -> v2 pattern
applied to the null itself.

## What this means for the North Star

The in-distribution result (v2) showed a single honesty direction transfers across minds when the test
matches the fit. This shows it still transfers when the test is a DIFFERENT KIND of fact than anything
the direction or map ever saw — temporal, relational, locational, definitional — across four models it
was never trained on, beating the correct null and surviving the drop of the easy family. That is the
property a cross-model conscience needs to be more than a per-distribution probe: one instrument, many
minds, unseen inputs.

## Honest bounds (what is NOT claimed)

Linear DiM source, linear ridge map, one task (truthfulness). OOD here means UNSEEN FACT-FAMILIES; it
does NOT test adversarial inputs, jailbreaks, paraphrase attacks, or non-factual honesty. Local open
models only (gemma-2-2b source; Llama-3.2 + Qwen2.5 targets) — closed frontier models are blocked on
credits. The gate is statistical significance against the correct null (the honesty direction beats
chance label-bipartitions), an EXISTENCE-and-significance claim, not a deployed-accuracy guarantee; the
margins are modest. This establishes that linear honesty-direction transfer is real out-of-distribution
across these minds and families — the next rungs are adversarial-OOD, non-linear maps, a portable
values basis beyond truth, and a single shared direction for a whole family.
