# RESULT — Phase-clamp ablation: oscillation is a capacity-extending MECHANISM, not a hard requirement

**Date:** 2026-06-03 · **Reading: ADVANTAGE** (oscillation helps substantially but is not
strictly necessary). Frozen gate: `PREREG_rhythm_rescue_2026_06_03.md`. Two complex-diagonal
LRU nets, identical init except eigenvalue phase θ (FREE = can rotate/oscillate; CLAMPED =
θ≡0, real eigenvalues, no rotation), matched state & parameters, ordered-copy task, 3 seeds.

## The numbers

| K (items) | FREE acc | CLAMPED acc |
|---:|---:|---:|
| 1 | 1.000 | 1.000 |
| 2 | 0.988 | 0.959 |
| 3 | 0.988 | 0.823 |
| 4 | 0.974 | 0.723 |
| 6 | 0.895 | 0.594 |
| 8 | 0.727 | 0.463 |
| 10 | 0.635 | 0.385 |
| 12 | 0.593 | 0.318 |
| 20 | 0.331 | 0.205 |

- **Capacity (kcap, acc ≥ 0.80):** FREE **6.0** vs CLAMPED **2.67** → oscillation **≈ doubles**
  ordered-memory capacity (gap 3.3 items).
- **FREE kept its oscillation:** osc_use = **0.62** after training (it did *not* learn θ→0) —
  the rotation was genuinely used, not abandoned as useless.
- FREE > CLAMPED at **every** K ≥ 2, and the gap **widens with K** (the signature of a
  capacity-specific benefit, not a constant offset).

## What it means (honest)

**Oscillation is a powerful, capacity-extending MECHANISM for recurrent ordered memory —
used when available, partially substitutable when not.** Three honest claims:

1. **Not epiphenomenal.** Removing only the rotation (everything else identical) roughly
   *halves* capacity, and the free net actively *keeps* its oscillatory modes. Phase-coding
   does real work for holding ordered items — direct, controlled support for the theta-gamma
   intuition (Lisman & Jensen 2013).
2. **Not strictly necessary.** The clamped net still solves short sequences and stays above
   chance everywhere — real-eigenvalue (multi-timescale decay) dynamics *partially rescue*
   the function. So oscillation is not a hard requirement, exactly as the substrate-flexibility
   literature predicts (Sussillo & Barak 2013; transformers do memory via attention with no
   rhythm at all).
3. **This refines our own earlier finding.** "Rhythm = mechanism not function" was too clean.
   The controlled answer: rhythm is a **mechanism that confers a large, real advantage** for
   the function it's credited with — the honest middle the field actually lives in
   ("necessary-in-tissue, not in-principle"), now shown in a clean ablation.

## A suggestive hint (flagged, not claimed)

FREE's capacity (~6) lands inside the **7±2** range; the no-rhythm net (~3) is far below.
This is *consistent* with the hypothesis that the magic-number capacity bound is a
**fingerprint of the oscillatory mechanism** — but the absolute capacity depends on hidden
size, task difficulty, and the 0.80 threshold, so we do **not** claim 7±2 here. It is a
prediction to test by scaling d and threshold and checking whether FREE's bound stays in
range while CLAMPED's tracks elsewhere.

## How it fits the program

Half B of the ancient-question program asked: *is rhythm necessary for cognition, or a
substrate-specific mechanism?* The decisive in-silico ablation answers, cleanly:
**a substrate-specific mechanism that materially extends capacity — not a requirement.**
Combined with our transformer result (memory achieved with *no* oscillation, via attention)
and Mamba-1 (real eigenvalues, commits), the picture is coherent: **recurrent substrates get
a real capacity boost from oscillation; attention substrates get the same function another
way.** Rhythm is fundamental to *recurrent* cognition's *efficiency*, not to cognition.

## Caveats (frozen)

- One function (ordered copy), one architecture (LRU), n=3 seeds, capacity at a 0.80
  threshold, in-silico. The capacity *magnitudes* are task/size-dependent — read the *gap*,
  not the absolute numbers, and do not over-read the 7±2 coincidence.
- Binding and routing arms (which transformers already do without rhythm) are pre-registered
  follow-ups; the capacity-vs-d scaling is the natural next run.
- This is the in-silico arm of "phase-clamp ablation with rescue"; the in-vivo
  (rate-clamped optogenetics) arm is out of scope.
