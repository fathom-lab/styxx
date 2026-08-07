# The Recorder's Signature: a failure class that permutation nulls cannot see

Fathom Lab · 2026-08-06. Three independent adversarial audits, of three different instruments,
run on the same day, broke all three. Every break was the same failure wearing a different mask.
This document names the class, states why the standard defence does not work against it, and
gives the test that does.

## The observation

| instrument | what was planted | what the instrument reported | verified by |
|---|---|---|---|
| `styxx.coupling` | two independent streams on one irregular clock | coupled, RV 0.3704 at p 0.0033 | **this lab, on real agent telemetry** |
| `styxx.coupling` | two independent streams both drifting over the window | coupled on every seed | red team |
| `styxx.sense` | two recorders sharing a stall clock | coupled on every seed, far above the run's own power floor | red team |
| `styxx.sense` | counter deltas measured over the recorder's own loop period | agent coupled to a network whose true rate was constant | red team |
| `styxx.sense` | two logs whose write moments were gated by one busy machine | a majority of runs licensed at the highest stall rates | red team |
| `styxx.islands` | members sharing only a per-item amplitude profile | "shared frame" far above the random null | **this lab, reproduced at 7.0×** |
| `styxx.islands` | a Monte-Carlo p estimated from 1000 draws | our published 0.0779 against a re-run 0.0634 | **this lab** |

Rows marked *red team* are the auditors' measurements, reported here as attributed claims rather
than as this lab's own: we acted on them and fixed the code, but we did not independently
reproduce every figure, and stating them as our measurements would be precisely the transcription
error this program exists to catch. (The verifier caught a draft of this document doing exactly
that.) The rows marked *this lab* were reproduced here, and their receipts are committed.

In every row the two streams share **nothing about the world**. What they share is the
apparatus: when it sampled, how long its loop took, when it stalled, which items it recorded
loudly, which random seed it drew. The measured dependence is real. Its cause is the recorder.

## Why the standard defence fails

The defence in all of these instruments is a permutation or shift null: destroy the pairing,
rebuild the statistic, see whether the observation stands out. Restricted permutation preserves a
named confound; circular shift preserves autocorrelation; both are textbook and both are correct
for what they were built for.

Neither can see this class, and the reason is structural:

> **A permutation null destroys row alignment. The recorder's signature *is* row alignment.**

When bin *i* of stream A and bin *i* of stream B are both quiet because the recorder stalled at
*i*, that is a genuine bin-wise dependence. Shuffling the rows destroys it — which is exactly why
the null sits low and the observation stands out. The null is not failing to model the artifact;
**the null is defined by removing it.** The more faithfully a null destroys the pairing, the more
confidently it certifies the artifact.

This is why adding nulls did not help. `styxx.coupling` ended the day with four of them —
confound-matched, circular-shift, leverage, sampling-density — and each new one closed exactly
one mask while the class stayed open.

## The test that does work

Not another null. A **counterfactual on the apparatus**: hold the world fixed and vary only the
recorder.

- **Time-reversal.** Pair a stream against its own reverse. Marginals identical, autocorrelation
  identical, world-alignment destroyed. Any dependence that survives is the recorder. This is how
  the sampling-density channel was found, on real agent telemetry, in one command.
- **Amplitude normalisation.** Remove per-item magnitude and re-measure. What survives is
  geometry; what vanishes was the recorder deciding which items to record loudly. This is how
  the h1a headline was defended — and it could as easily have been how it died.
- **Rate not count.** Any counter delta is `rate × dt` where `dt` belongs to the recorder. Divide
  by measured elapsed time or the loop period is in your data.
- **Freshness.** A value re-read because nothing new was written is not a measurement. Duplicate
  rows inflate *n* without inflating effective *n*, and the null is drawn at the inflated *n*.
- **Seed variance.** Re-run the statistic across seeds. If the verdict moves, the verdict was
  partly the seed. Our own published p moved from 0.0779 to 0.0634 on a twenty-fold longer run,
  and the audit reports a real fraction of seeds flipping the verdict outright.

The common shape: **vary the apparatus, hold the world constant, and see if the finding moves.**
A null varies the world and holds the apparatus constant, which is precisely the wrong axis for
this class.

## The uncomfortable part

Every instrument above was *built* to refuse. Each one carries explicit refusals, earned from
earlier failures, and each names this exact hazard in its own opening paragraph —
`styxx.sense` says a recorder's duty cycle must not become "I feel my body," and then scored a
stall clock as a sense. Writing the warning did not implement it.

A smaller instance of the same thing happened while this document was being written: a draft
quoted seven figures taken from the audit reports as though this lab had measured them. The OATH
verifier refused the document until each was either reproduced here or attributed. That refusal
is the only reason the table above distinguishes the two.

And the failures were not found by the discipline. They were found by **adversaries pointed at
the discipline**. Three instruments passed their own test suites, their own frozen gates, their
own validation exams, and were publicly announced — and then broke within hours under attack. The
gates were necessary and they were not sufficient. What closed the gap was: *give someone the
explicit job of breaking this, before you announce it.*

That is now a standing rule here, and it is a rule about process rather than statistics:
**no instrument is announced before an adversarial pass.** Today it cost two recalls, three
errata, and a withdrawn application claim to learn — all published, all with receipts.

## What is claimed, and what is not

**Claimed:** that recorder-state contamination is a distinct failure class; that permutation and
shift nulls are structurally incapable of detecting it; and that apparatus-counterfactuals detect
it cheaply.

**Not claimed:** that the five tests above are complete, or that they are new individually —
time-reversal, surrogate testing and seed sensitivity are all standard practice in their own
literatures. The contribution is naming the class they jointly address and showing that a
composition of nulls, however careful, cannot substitute for them.

**Not claimed:** that our instruments are now clean. `styxx.coupling` is currently known-broken
in *both* directions — it licensed independent autoregressive streams before its refusal existed
and refuses genuine intersubject correlation after it. That is stated in its docstring and it is
why the mind↔brain application is withdrawn rather than softened.

*The recorder is part of the world you are measuring. Nothing in a null tells you where the
instrument ends and the world begins.*
