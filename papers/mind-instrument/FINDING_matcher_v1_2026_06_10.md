# FINDING — matcher v1 loses; the label-free objective is miscalibrated (DESIGN-INSUFFICIENT)

**2026-06-10 · Fathom Lab / styxx. Pre-registered: `PREREG_matcher_v1_2026_06_10.md` (frozen
pre-run, one shot, no tuning loop). Receipt: `matcher_v1_result.json` vs baseline
`gavagai_scale_result.json`. The loss ships as promised.**

## Result

The anatomy-blocked two-stage matcher with label-free consensus FAILED gate M1: mean accuracy at
N=192 of 0.1018 vs the v0 baseline's 0.1073 (and below baseline at every world size: 0.1349 vs
0.1693 at N=48; 0.1352 vs 0.1573 at N=96); 16 of 40 pairs improved; sign test p = 0.33678.

## The diagnostic that matters

The consensus step chose the block-seeded candidate **100% of the time at every N** — including
everywhere it was worse on ground truth. The label-free objective (mean rowwise correlation of the
mapped RDMs) is **miscalibrated**: block-coherent mappings inflate it while being less correct.
The objective rewards smooth structural agreement, and the block seed manufactures exactly that.
This is a finding about WHY unsupervised matching is hard, beyond this design: candidate selection
without labels cannot use the most natural structural score, because plausible-looking structure
is precisely what a wrong-but-coherent alignment fakes best. (The same lesson the program's
text-only deception axis taught about minds: smooth surface agreement is the cheapest thing to
counterfeit.)

## Status of the wall

The matcher bottleneck named by GAVAGAI-SCALE stands, now with one fortified flank: block-seeding
plus correlation-consensus is ruled out as frozen here. Next attempts (each requiring its own
prereg, one shot each): seeded entropic Gromov-Wasserstein; soft-assignment EM; consensus
objectives penalized for block-coherence (e.g. held-out-row prediction rather than in-map
correlation); multi-restart with truth-free stability selection across random seeds.

## Bounds

One design as frozen; the components were not ablated (the prereg forbade post-hoc variants), so
whether block-seeding or the consensus rule is the dominant failure is for the next prereg to
separate — the 100% pick rate strongly implicates the consensus rule.
