# PRE-FREEZE PANEL SYNTHESIS -- coupling attempt 3. Verdict: **NO_GO on freezing as drafted.**

**Fathom Lab - papers/calib-poison-general - 2026-07-16. Status: NOTHING FROZEN. No artifact of
attempt 3 is committed; no scored run has launched; the GPU is idle. Receipt:
`_coupling_v3_prefreeze_review.json` (22 merged findings, 5 lenses, 16 recovered refutation votes).
Raw agent returns: `_coupling_v3_prefreeze_review_raw.json`.**

This document is the synthesis the panel's own synthesis agent owed and could not deliver: 23 of 45
panel agents died on an API session limit mid-run, including that agent and 22 refutation votes. It
is written by the main loop from the recovered journal. **Verification is INCOMPLETE and this
document does not pretend otherwise:** findings carrying no recovered refutation vote are
UNRESOLVED, not cleared. Of 16 recovered votes, 2 refuted a finding; 14 let one stand.

## Verdict

**NO_GO.** Two FATAL findings survived adversarial refutation, and they are not patchable in place:
they are the same architectural defect seen from two sides. Freezing this prereg would repeat
cycle 43 with the sign flipped.

| severity | count |
|---|---|
| fatal | 2 |
| major | 6 |
| minor | 12 |
| wording | 2 |

## The two fatals, and why they are one defect

**F1 -- no sensitivity control anywhere** (raised independently by 3 of 5 lenses:
channel-isolation, statistics, sensitivity-asymmetry). The design gates instrument admissibility
only against false-COUPLED. Nothing requires the multiple-choice battery to demonstrate it CAN
register genuine capability loss, yet failure-to-fire is mapped straight onto a positive claim
(`DECOUPLED__read_neq_write_BROKEN...`, scope: "capability held"). Three levels reinforce it:
argmax over four letter groups is invariant to any perturbation that shrinks margins without
flipping a winner; the sub-tasks are ceiling-trivial by selection (clean floor of 0.90, attempt 2's
analogue landed every selected sub-task at 1.0); and `eval_knowledge` reads through the *voided*
True/False channel, so the MC battery is the sole gating capability channel. The per-seed rule is a
one-sided 1.47-sigma test (PAIR_DROP of 0.15 against a paired SE up to 0.102): power is 0.50 at a
drop exactly on the bar, 0.31 at 0.10, 0.16 at 0.05. The only capability price this program has
ever measured -- the dose run's coupled-seed knowledge_drop of 0.0909 -- sits *below* the bar.
Because all five seeds share one true effect, the majority rule concentrates on the more-probable
side instead of adding power: P(DECOUPLED majority | true drop of 0.10) is about 0.82; under pure
noise it is 0.997.

**F2 -- the specificity gate's population is not the verdict's population** (channel-isolation,
statistics). The gate estimates the false-fire rate on READ-INTACT pairs; the verdict fires at a
READ-CROSSED point. Attempt 2's own receipts show the fire rate covaries with read status (0.75 on
accumulate admissible vs 0.7115 read-intact; 0.5833 fixed vs 0.5741 read-intact), so the two
populations are not exchangeable and the gate systematically mis-estimates the null *at the point
where it matters*. Worse, the gate's stated premise -- "read intact means nothing has been removed"
-- is false: the dose run shows the read RECOVERING above threshold at ranks 10-24, so the null
pool contains the run's HIGHEST-dose checkpoints. In the most natural coupled world (price scales
with accumulated rank, not with whether the read happens to sit below 0.70 at that checkpoint) the
paired rule fires hardest exactly there, the rate blows SPEC_MAX, and the run returns
`VOID_COUPLING__battery_lacks_dose_specificity` -- with a stated cause that is the precise opposite
of the truth.

**The single root.** Both fatals are the same mistake: **the design validates a POINT verdict using
a POPULATION that is not that point, and validates it in only ONE direction.** Every downstream
pathology follows. A deaf battery passes the specificity gate *because* it is deaf (F1). A real
dose-graded price VOIDs the battery *because* the gate calls high-dose evidence "noise" (F2). The
net asymmetry: COUPLED is structurally near-unreachable, DECOUPLED is the default in both the null
world and the genuinely-priced-below-bar world, and an unearned COUPLED remains reachable through
the mis-estimated gate. **The design can produce either headline claim unearned.** That is a strict
regression on attempt 2, which could only produce one.

## The uncomfortable symmetry with cycle 43

Attempt 2 shipped COUPLED from a rule never shown to be SPECIFIC -- it fired under the null.
Attempt 3 would ship DECOUPLED from a rule never shown to be SENSITIVE -- it was never shown to
fire under the alternative. Cycle 43's lesson was recorded as "put a control-arm noise floor in
every prereg that gates a verdict on a measured invariant." **That lesson, applied literally and
alone, produced this design -- and a noise floor alone actively selects FOR deaf instruments,
because the deafest instrument passes it best.** The rung was real but half a rung.

## The rung this panel actually buys (worth more than the run, again)

**Two-sided instrument admissibility, measured with the verdict's own statistic on the verdict's
own population.** An instrument may gate a verdict only if it is shown BOTH:

- **specific** -- it does not fire when nothing was done (the noise floor; cycle 43's rung), AND
- **sensitive** -- it DOES fire when something known-destructive was done (the positive control;
  this panel's rung),

and both demonstrations must use **the same statistic and the same population as the verdict**,
or they certify a different experiment than the one that ships. Specificity alone is not
conservative -- it is biased toward whichever claim silence implies. Which claim silence implies is
a property of the *verdict mapping*, not of the instrument, and that is what makes one-sided
validation invisible: cycle 43's silence implied nothing, so specificity looked sufficient;
attempt 3 mapped silence onto its headline claim, and the same gate became a rubber stamp.

This generalizes cleanly past this program and is shippable as a `styxx.ladder` rung. It is the
first methodological result in this arc that neither run could have produced -- both attempts had
to fail, in opposite directions, for the symmetry to be visible.

## Recommended redesign (v4) -- one change dissolves both fatals

**Score the dose-response, not a point; validate the instrument in both directions with the
verdict's own statistic.**

1. **Statistic: the paired slope, not the paired point.** Per seed, at every matched step compute
   the paired MC delta (fixed minus accumulate) and regress it on the accumulate arm's
   `erased_rank`. The fixed arm is pinned at rank 2, so at matched step the shared fine-tuning
   drift differences out and what remains is dose. Coupling = slope reliably above zero. This uses
   about 12 checkpoints x 5 seeds instead of 5 points (kills F1's power problem), makes high-dose
   read-intact checkpoints EVIDENCE rather than contamination (kills F2's premise), and handles the
   transient recovery natively -- the recovery is part of the curve, not an embarrassment to it.
2. **Specificity: a K-averaged permutation null on that same slope** -- shuffle rank labels within
   seed, K draws, per this program's own standing lesson ("K-averaged permutation nulls, never
   single-draw controls at small n", learned on the sentiment foundation run). The null is then
   exchangeable with the verdict statistic BY CONSTRUCTION, which is exactly what F2 says the
   read-intact pool is not.
3. **Sensitivity: a positive-control arm with known capability destruction** -- the lambda=0
   (no knowledge-replay) regime, which this program already knows collapses `eval_knowledge`. The
   paired MC rule MUST fire there, or the battery is inadmissible for ANY verdict:
   `VOID_COUPLING__battery_insensitive`. Cost is one extra arm per seed; it is the only thing that
   makes a DECOUPLED claim mean anything.
4. **Verdict strings** must match what the instrument carries: no `capability held` (a positive
   assertion) where the evidence supports "no price at or above the bar visible to this battery",
   and no `read_neq_write_BROKEN` headline unless the sensitivity control passed.

The six MAJOR findings (unenforced `>= 3 survivors` selection guard; PAIR_DROP chosen with
attempt-2's deltas in hand; direction-correlated `VOID_crossing_not_dose_attributable`; unbuffered
r* attributability with no multiplicity control; under-powered gate at its own floor; contaminated
null pool) are mostly dissolved by items 1-3. The two that survive the redesign and must be fixed
explicitly: **the unenforced selection guard** (a real code-vs-prereg conformance bug -- `MIN_DISJOINT`
is decorative, the calibration receipt is written even when `ok=False`, and `base_model` is never
verified) and **threshold provenance** (any bar must be derived from a pre-committed minimum effect
of interest with a stated power target, not from a number that leaves the run pre-decided; the
prereg must disclose the anticipated verdict under reproduced dynamics).

## What this costs, honestly

The v4 arm structure is 3 arms x 5 seeds against attempt 3's 2 x 5 -- roughly 1.5x the GPU time,
which at the arc's measured rate is an overnight run, not a new budget line. The paper stays
blocked on the coupling side for that long. The alternative on offer -- freeze attempt 3 tonight --
buys a verdict tomorrow that the program would have to refuse for the third time, and this one
would be refusing its own headline.

## Owed before any freeze

1. Re-run the 23 dead panel agents (session limit resets 16:00 America/New_York) so the UNRESOLVED
   findings are actually resolved rather than assumed. `Workflow({scriptPath: ...,
   resumeFromRunId: 'wf_8228537d-028'})` replays the 22 cached agents and re-runs only the failures.
2. Operator decision on v4 vs patch (below).
3. A v4 prereg + harness, re-panelled pre-freeze, THEN frozen, THEN calibrated, THEN run.

## The decision that is not the agent's to make

Attempt 3's design is unfrozen and cheap to abandon; that is the whole value of having panelled it
before the freeze rather than after the run. But **v4 is a redesign, not a patch** -- it adds an
arm, replaces the verdict statistic, and pushes the coupling answer at least a day right. The
standing position (`PARTIAL__coupling_seed_split`) is unaffected either way, and the static /
adaptive / 3B SURVIVES results remain untouched: they gate on nothing in this battery.

---
*Attempt 1 split. Attempt 2 built an instrument that could not tell dose from noise and said COUPLED.
Attempt 3 built an instrument that cannot hear a price and would have said BROKEN. The panel caught
it while every file was still untracked and the card was still cold. The rung is that a noise floor
alone is not conservatism -- it is a bias toward whatever silence happens to mean, and this design
made silence mean the loudest claim in the program.*
