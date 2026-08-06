# FINDING — H1b: there are no unreadable minds in this cohort — the worst-read subject sits at 212 times chance

Fathom Lab · 2026-08-06 · prereg: `PREREG_h1b_human_readability_2026_08_06.md` (frozen before
the scored run) · receipt: `h1b_result.json` · scored by `styxx.protocol`.

## Verdict (machine-computed)

**`READABILITY_IS_CONTINUOUS__no_switch_at_this_cohort`** — the branch we preregistered as
expected.

| gate | frozen bar | measured | pass |
|---|---|---|---|
| G0_cohort | ≥ 8 subjects | 8 | ✅ |
| G1_group_reads_above_chance | median ≥ 5× chance | 264.001× | ✅ |
| G2_bimodal_readability | gap-screen p ≤ 0.05 | 0.6523 | ❌ |

## The number that matters

H1 predicted that "a subpopulation sits near the decoding floor while their within-subject
decoding is normal." Measured, on eight subjects, with a template built only from the other
seven and no fitting anywhere in the pipeline:

| subject | readability | × chance |
|---|---|---|
| subj02 | 0.8128 | 369 |
| subj01 | 0.7687 | 349 |
| subj05 | 0.7048 | 320 |
| subj03 | 0.5925 | 269 |
| subj06 | 0.5705 | 259 |
| subj04 | 0.5441 | 247 |
| subj07 | 0.5132 | 233 |
| subj08 | 0.4670 | **212** |

Chance is 0.002203. **The least readable subject in the cohort is read at two hundred and twelve
times chance.** There is no floor population here — there is not even a subject in the
neighbourhood of a floor. The spread is real and roughly two-fold, and it is entirely a spread
among the well-read.

## What this settles about our own prediction

This is the variable H1 actually names, and it is the fourth independent negative in one day:
the model-side precursor (B47), the published human literature, human alignment (H1a), and now
human readability. **H1 as written predicted a subpopulation near the decoding floor. The
measured minimum is 212× chance.** We are not going to describe that as anything other than what
it is.

## And the switch did not transfer

H1b existed because our own b46 result says alignment and legibility are joined by a switch —
flat, then nearly vertical — which would let a continuous alignment spread produce a bimodal
readability distribution. It did not. Alignment and readability instead track each other almost
proportionally (Pearson r 0.8825 across eight subjects, with the two orderings agreeing at both
ends: subj08 lowest and subj02 highest on both). Reported ungated, because eight points license
no fit — but the qualitative shape is what a *ramp* looks like, not a cliff.

That is a genuine dissociation between the model result and the human result, and it is the most
interesting thing in this document. Between two language models, correcting a frame rotation
moved legibility from near-zero to near-perfect through a near-vertical knee. Between eight human
brains, alignment and readability move together smoothly. Either the switch is a property of that
model pair rather than of minds in general, or the human cohort's alignment range simply never
approaches the knee. **We cannot distinguish those two readings with eight subjects, and we are
not going to pretend the model finding generalized.**

## Confirmed against the reference statistic

The gate above uses this module's gap screen. **Hartigan's dip test — the statistic the methods
literature names — was run afterwards on the same vector and returns p 0.7553**, agreeing with
the screen and sitting further from the bar. Receipt: the diptest addendum committed beside this
finding. The screen is the more liberal of the two here, so its non-flag is the stronger
statement.

## Limits, fixed before the run

- **n = 8 is exactly the instrument's floor.** A gap screen on eight points has very little
  power; this is weak evidence of absence, not proof that no human cohort contains islands.
- Identification accuracy is one readability measure. A trained decoder could rank subjects
  differently; this one was chosen because it involves no fitting and therefore could not be
  tuned toward either verdict.
- Same dataset, ROI, modality, and seed as H1a — these are **not** independent runs, and the
  r 0.8825 between them is partly a shared-data artifact by construction.
- The alignment-versus-readability relation is ungated, reported for successors at larger n.

## Why this was worth running anyway

The claim "some minds are unreadable" is doing real work in public right now — in privacy
arguments, in funding pitches, in the framing of what non-invasive decoding will become. It is
repeated because per-subject distributions are described in words and low performers are
routinely excluded before anyone looks. On the eight subjects of this dataset, including the
four that the reconstruction literature drops for incomplete sessions, **every single one is
legible to a template built from strangers**, and the least legible is legible at 212× chance.

*Prereg frozen before the run with the losing branch named first, and it is the one that landed.
Every number grounds in `h1b_result.json`. Sealed before commit.*
