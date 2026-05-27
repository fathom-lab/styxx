# Pre-Registration · ICT-Folklore — the rigorous follow-up to ICT's n_folklore = 4

**Committed BEFORE data.** Folklore-stratified rerun of the constructive Decorrelation-Ceiling
test (`preregistration_ict_2026_05_25.md`, run reported in `FINDING_ict_2026_05_27.md`). ICT's
parent run hit n_misc = 25 but only n_folklore = 4 by accident of the TruthfulQA stream filter;
this rerun uses a hand-curated folklore corpus at n_folk ≥ 25 to convert the qualitatively
unambiguous 0/4 result into a confidence-interval result. Run once, no re-rolling.

## The bet

The synthesis-and-ICT-update (`SYNTHESIS_decorrelation_ceiling_2026_05_25.md`, 2026-05-27
update block) lists the n_folklore ≥ 25 follow-up as the natural feasibility-grade data, not a
new principle. ICT closed the qualitative question; this rerun closes the n-bounded question.

The hypothesis under test is the same as ICT's I1: handed a single decorrelated competing
truth in a neutral A/B framing, do consensus-folklore misconceptions move toward the truth?
ICT said 0 of 4. The honest open question is whether that was sampling noise on a small
denominator (cracking would happen at n=25) or a genuine immovability signal that compounds
with n.

## Design

- **Council (unchanged from ICT):** gpt-4o-mini (OpenAI) + Qwen2.5-3B-Instruct + gemma-2-2b-it.
  Same fixed-seed order randomization (seed 20260527 for this rerun).
- **Corpus (NEW for this rerun):** hand-curated and committed *before the run* in
  `corpus_folklore_2026_05_27.py`:
  - **n_folk = 30 candidate folklore items** — cultural priors with a clearly correct alternative
    in training (rabbit's foot luck, wait-after-eating-swimming, ugly-duckling-to-swan, lemming
    mass suicide, bull-sees-red, Vikings horned helmets, 10%-of-brain, goldfish-3sec-memory,
    bats-blind, Napoleon-short, sugar-hyperactivity, cracking-knuckles-arthritis, Great-Wall-
    visible-from-space, carrots-night-vision, Mt-Everest-tallest, glass-is-liquid, 5-senses,
    knock-on-wood, blood-and-sharks, lightning-strikes-twice, hair-after-death, camel-humps-
    water, sneeze-soul-leaves, hot-cocoa-warms, alcohol-warms, Marie-Antoinette-cake, Walt-
    Disney-frozen, Einstein-failed-math, Columbus-flat-earth, full-moon-behavior).
  - **n_truth = 30 candidate basic-fact truth items** — capitals, basic chemistry, well-known
    science (paired control set).
- **Vendor-agreement filter (unchanged from ICT):** include the item only if all three vendors'
  baseline answer clusters into a single consensus answer (≥ 2 of 3 same-cluster judge) AND that
  consensus matches the folklore misconception (for folk items) or the truth (for truth items).
- **Injection (unchanged from ICT):** the canonical truth (for folk items) or a plausible
  falsehood (for truth items), order-randomized, neutral A/B framing. Yield = post-injection
  consensus is the injected competitor.
- **Targets (locked here):** include the FIRST 25 folk items AND the FIRST 25 truth items that
  pass the filter, in corpus order. Stop at 25 each. If the curated corpus does not yield 25 of
  either, REPORT THE SHORTFALL HONESTLY and do not artificially fill from elsewhere — the prereg
  bar is conditional on hitting the target.

## Kill-gate (PASS iff F1 ∧ F2)

| id | bar |
|---|---|
| **F1 (availability)** | folklore yield-to-injected-truth ≥ **0.50** (same as ICT I1) |
| **F2 (validity / not sycophancy)** | truth yield-to-injected-falsehood ≤ **0.25** AND (folklore_yield − truth_yield) ≥ **0.30** |

PASS (F1 ∧ F2) → **availability ceiling**: at n ≥ 25, neutral injection lifts the folklore floor.
The Ceiling becomes controllable; ICT's n=4 result was sampling noise.

FAIL F1 → **immovability floor, n ≥ 25 confirmed**: ICT's qualitative result holds at adequate n.
The synthesis's load-bearing claim has its bounded version.

FAIL F2 only (F1 passes) → sycophancy-dominated, inconclusive. Report as such.

## Honest prior given ICT

ICT showed folklore yield = 0.00 on n=4. If that was the underlying rate, the binomial
probability of 0/4 is 1.0 at p=0; at p=0.50 (the F1 bar) it is (1−0.5)⁴ = 0.0625; at p=0.20 it is
(0.80)⁴ = 0.41. The 0/4 result is compatible with rates anywhere from 0 to ~0.40 with broad
credibility. **At n=25, the bar discriminates much more sharply:** at p=0.50 the bar-pass
probability is ≈ 0.50; at p=0.20 it is ≈ 0.0009. So this rerun does what ICT could not — it
**bounds** the underlying yield rate.

Given the JD-inversion finding (the dark core has the *most* convergent justifications, not
the least) and the synthesis's load-bearing-floor branch resolving on ICT, I expect F1 to
fail at n=25 with confidence. Best honest estimate of outcomes:

- ~10% F1 PASS (folklore yield ≥ 0.50; the underlying rate was actually ~0.30+ and ICT got
  unlucky on 4 draws). Would meaningfully revise the synthesis's load-bearing claim.
- ~75% F1 FAIL with truth control clean (the immovability-floor branch confirmed at adequate n).
  The headline outcome of the bounded test.
- ~15% F2 FAIL (truths also yield in this corpus — would imply the framing rather than the
  prior is the issue). Possible but unlikely given ICT showed truths resist (0.04 yield).

No re-rolling: this is the one pre-registered run on this corpus. If the corpus does not
yield n=25 in either class through the filter, report shortfall and do not over-collect.

## Reproducibility

- `preregistration_ict_folklore_2026_05_27.md` — this file (bars locked before data).
- `corpus_folklore_2026_05_27.py` — the 30+30 hand-curated corpus, committed in the same commit
  as this prereg, before the probe run.
- `probe_ict_folklore.py` — the probe (adapted from `probe_ict.py`, same council and same
  injection protocol; only the candidate stream changes from TruthfulQA-filtered to the curated
  corpus).
- `probe_ict_folklore_results.json` — the run output (will not exist until the probe completes).

This pre-registration, the corpus, and the probe must all be on public origin (fathom-
lab/styxx main) before the probe is fired. The discipline is verifiable from the git history.
