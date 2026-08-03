# FINDING — two generality tests, two INVALIDs, two mechanisms: the sanity gates earned their keep

Fathom Lab · 2026-08-03 · both runs scored MECHANICALLY by `styxx.protocol` against frozen
gates blocks (`PREREG_b35c_open_vocab_2026_08_03.md` @ `6d88a80`;
`PREREG_b35b_second_source_2026_08_03.md` @ `a3d981f`). Receipts: `b35c_result.json`,
`b35c_null_replication.json`, `b35c_collapse_diagnostic.json`, `b35b_result.json`.

Neither generality claim is licensed. Both runs produced something more useful than a claim:
a named mechanism for why the test could not be read, and a precise remedy. Recorded at full
volume because the alternative — reporting the headline numbers these runs *also* produced —
would have been the exact failure this program exists to prevent.

## B35-c — open-vocabulary readout: `INVALID__null_artifact`

Removing the 70-way shortlist (queries scored against all 462 concepts, trained anchors
included as distractors) produced the largest reads in program history: **llama 0.3143 = 145×
chance, gemma 0.2000 = 92× chance** at a 1/462 floor. **These are UNLICENSED and are not
claimed.** The frozen G2 null gate failed: a shuffled-pairing null scored one hit across the
70 queries, above the 3×-chance floor.

The prereg pre-wrote the remedy for exactly this branch — *"a re-run with a second null seed
reported beside it, not a bar move"* — so the null was re-drawn under five independent seeds
per target (`b35c_null_replication.json`):

| target | null hits across 5 seeds | mean |
|---|---|---:|
| llama_1b | 0, 0, 1, 0, 0 | 0.00286 |
| gemma_2b | **1, 1, 1, 1, 1** | 0.01429 |
| qwen_1p5b | 1, 2, 0, 1, 1 | 0.01429 |

**gemma's null scored exactly one hit on five of five independent seeds.** Under the prereg's
Poisson assumption (0.152 expected hits per draw) that is a ~4×10⁻⁵ coincidence. It is not a
coincidence — it is a mechanism, and **the prereg's null model was simply wrong.**

**My first hypothesis was full mode collapse — every query mapping to one centroid point — and
the diagnostic FALSIFIED it** (`b35c_collapse_diagnostic.json`, formal verdict `NO_COLLAPSE`
against its pre-set ≤3-distinct criterion). The real mechanism is the weaker one: **partial mode
concentration.** An MLP fit on shuffled pairs has no correspondence to learn, so it regresses
toward the target centroid without degenerating:

| target | null distinct preds | null modal share | real distinct | null spread vs real |
|---|---:|---:|---:|---|
| llama_1b | 22 / 70 | 0.30 | 56 | 0.2564 vs 0.654 |
| gemma_2b | 22 / 70 | 0.3143 | 51 | 8.12 vs 19.2845 |
| qwen_1p5b | 21 / 70 | 0.2857 | 44 | 1.6694 vs 3.4136 |

The null makes ~22 distinct predictions where the real map makes 44–56, with ~30% of queries
landing on a single modal entry and output spread roughly a third of the real map's. That
explains the exactly-one-hit signature precisely: **when the modal entry happens to be one of
the 70 queries, exactly one query — the one that *is* that entry — scores, reproducibly, every
seed.** The 462-way null is therefore **structured and over-dispersed, not Poisson**, so a floor
derived from Poisson expectation is not a valid gate for it.

**Remedy (successor prereg, not a bar move):** the open-vocabulary null must be specified as a
*rank-based or concentration-corrected* comparison — e.g. gating on mean reciprocal rank, or on
the read's margin over the empirical null distribution measured across seeds — rather than on a
raw hit-rate floor derived from independence. Until such a prereg exists and passes, the
open-vocabulary reads (llama 145× chance, gemma 92× chance) remain unlicensed observations.

## B35-b — second source family: `INVALID__second_source_discovery_broken`

Swapping the source to Qwen2.5-3B produced discovery accuracy of **0.0 (qwen_1p5b, the
same-family control), 0.0204 (gemma_2b), 0.0 (llama_1b)** — total failure of the label-free
discovery step, including on the same-family control that should be easiest. G0 fired.

**The diagnosis is an operating-point error, and it is mine.** The Llama-3B source layer was not
chosen by a fraction rule — it was **locked by an explicit G0 optimization** (layer 11, k 150,
pc_cos 0.9131, `g0clear_result_llama3b.json`). This prereg transplanted the resulting *fraction*
(11/28) to Qwen2.5-3B's 36 layers → layer 14, with **no equivalent search**. The frac rule was
validated for choosing *target* layers, never for choosing a *source* read layer, and I applied
it outside its established scope.

So the licensed reading is **not** "the read is a Llama-source artifact." It is: **no valid
operating point was ever established for the Qwen-3B source, so the second-source question
remains open and untested.** This is the same confound class the write-layer decouple run had to
kill in the b36 lineage — a null at an unvalidated operating point licenses nothing.

**Remedy (successor prereg):** run the G0-style layer/k search on the Qwen-3B source to
establish its own locked read point with a positive control, then re-run the pipeline unchanged.
Only a failure *at a G0-cleared operating point* would license "source matters."

## What this cycle establishes

- The label-free cross-family read stands exactly where b35-a left it: **seed-stable, 70-way,
  one source family, one strong-discovery target.** Neither generality extension succeeded, and
  neither failed in a way that reflects on the b34-v3/b35-a result.
- **Two sanity gates (G2, G0) caught two un-established preconditions** before either could be
  written up as a generality claim. In both cases the run's headline numbers pointed the
  *opposite* direction from the verdict — spectacular reads in B35-c, total failure in B35-b —
  and in both cases the gate, not the number, decided.
- The b35-c null model error is now on the record: **Poisson independence is the wrong null
  model for a many-way readout whose null map mode-concentrates.** That correction applies to
  any future open-vocabulary rung in this arc.

*The gates rejected our best number and our worst one in the same session. That is what they are for.*
