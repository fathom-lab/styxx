# FINDING — B49: the island arc survives the control we built after publishing it — and my fifth metric mis-specification is in the verdict name

Fathom Lab · 2026-08-07 · prereg: `PREREG_b49_amplitude_reaudit_2026_08_07.md` (frozen before
the re-analysis) · receipt: `b49_result.json` · scored by `styxx.protocol` under
`require_power_basis=True`, the first run in this program to do so.

**Machine verdict: `PARTIAL__b45_holds_b47_verdict_moves`.** The first half is real. The second
half is my metric comparing two different fields, and the correction is below.

## The science: both published findings survive, and one strengthens

An adversarial audit showed that shared per-item **amplitude**, with no shared geometry at all,
can drive frame affinity far above a random-frame null. We added the control, applied it to the
human data, and did not apply it to our own island arc — findings already sealed, cited in the
connection-of-minds synthesis, and staged in the arXiv paper. B49 applies it.

| | published (uncontrolled) | with the amplitude control |
|---|---|---|
| b45 clique median affinity | 0.848 | **0.862** |
| b45 margin over random-null p95 | 0.7914 | **0.8054** |
| b45 island below clique | 5 of 5 seeds | **5 of 5 seeds** |
| b47 verdict through its frozen gates | `SINGLE_LEGIBLE_CLIQUE` | **`SINGLE_LEGIBLE_CLIQUE`** |

**The shared cross-family concept-frame geometry is not an amplitude artifact.** Stripping every
item's magnitude and keeping only direction leaves the clique's co-alignment *higher* than
published, the island still lowest in every seed, and the ten-model cohort's verdict unchanged.
The claim in synthesis §3 and in the staged arXiv abstract stands as written.

## The correction: G3 compared the wrong two things

`G3_b47_verdict_unchanged` matched `survey.verdict` — the instrument's internal label,
`UNIMODAL_COHORT` — against b47's **protocol** verdict, `SINGLE_LEGIBLE_CLIQUE__no_islands_in_this_cohort`.
Those are different fields and were never equal, including in the published run, where
`b47_result.json` records both. The gate could not have passed under any data.

Scored properly — the normalised numbers through b47's own frozen gates — every gate fires
identically (`G0` true, `G1` true, `G2` false) and the verdict is unchanged. **b47 did not move.**

That is the fifth metric or bar mis-specification in this program this week, after b37 G2, b48
G2, C5 G1 and the protocol-v2 harness. It occurred in a preregistration that carries a
`power_basis` on every gate — including on G3, where I wrote *"exact string match against the
sealed verdict; binary by construction."* The declaration was accurate about the statistic's
shape and silent about whether I had identified the right string. **A power basis constrains the
bar, not the metric.** That gap is now demonstrated rather than hypothesised, and it belongs in
the successor to the protocol change.

## What is not done here

The verdict stands as computed; the run is not repeated with a corrected G3, because a metric
fixed after seeing the data is not a preregistered metric. The correction is stated, the
underlying comparison is shown, and a successor prereg may re-ask the b47 question with the
field named unambiguously.

*Frozen before the re-analysis; the science survived; the process error is mine and is reported
in the same document as the result it did not affect. Every number grounds in `b49_result.json`.*
