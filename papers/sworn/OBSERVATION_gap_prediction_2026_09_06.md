# OBSERVATION — can a reader predict where an instrument is blind? Once yes, once not

Fathom Lab · 2026-09-06 · **Post-hoc, not preregistered.** This compares two completed runs that
were not designed to be compared, and it exists because a claim was made from the first of them
before the second was in. It is an observation and a correction, not a result.

## The claim being corrected

After `RESULT_mutation_coverage_2026_09_05.md`, this lab said — twice, in a session summary and in
the RESULT itself — that *an agent reading the source predicted the blind spots better than 150000
random cases could find them*. The evidence was one number: a completeness critic, asked to name
what nobody had covered, proposed ten mutations and the differential harness caught only one.

The suite-power leg ran the same design against a different instrument. **It did not replicate.**

## Both numbers

In each leg, "readers" are the per-region or per-layer agents, and "the critic" is the completeness
agent that read the same code afterwards and was asked to name what the readers had missed. The
measure is the share of each group's mutations that the instrument **failed to detect**.

| leg | instrument | readers undetected | critic undetected | odds ratio | two-sided Fisher |
|---|---|---|---|---|---|
| mutation coverage | a differential harness | 20/60 = 33% | **9/10 = 90%** | 18.0 | **p = 0.0011** |
| suite power | a test suite | 23/41 = 56% | **3/10 = 30%** | 0.3 | p = 0.173 |

The first effect is large and unlikely to be chance. The second points the other way and **is not
significant**: it establishes that the first did not replicate, not that the direction reverses.
Absence of evidence, and it should not be read as more.

## The confound, stated before any interpretation

The critic was **told** to find what nobody had covered, and it could see the other agents'
proposals. So it was not predicting blind spots from the code alone — it was differentiating
against a list. Any advantage it shows is partly the instruction working, which is worth knowing
but is a weaker claim than the one that was made.

## A hypothesis for the difference, labelled as one

The two instruments are blind in different ways, and the two kinds of blindness may not be equally
legible from source:

- The differential harness's blind spots are **aperture** — which inputs its generator cannot
  construct. That is a property of the code the critic was reading: a fixed list of ten payload
  literals is visible on the page, and "no BOM in that list" is an observation, not a prediction.
- A test suite's blind spots are **assertion coverage** — which behaviours no test checks. That is
  a property of code the critic was *not* asked to read. It read `styxx/sworn.py`, not the tests,
  and reasoned about what looked load-bearing. Load-bearing code is exactly what tends to be
  tested, which would explain the critic's mutations being caught *more* often than the readers'.

If that is right, the useful rule is narrower than the one this lab published: **reading finds an
aperture, because an aperture is written down. It does not find a missing assertion, because a
missing assertion is written nowhere.** Testing that would take a third leg in which the critic is
given the tests and asked the same question.

## What this does not say

That the critic was not useful in the suite-power leg — it was: it named `verify_receipt`'s
`digest_ok and reproduces` join, which turned out to be defended by exactly one test, and both
sidecar injection boundaries, which turned out to be defended by none. Being caught more often than
average is not the same as being uninformative.

That either number generalises beyond these two runs, ten mutations each on the critic side. Ten is
a small sample and the confidence intervals are wide in both directions.

That "reading beats fuzzing" is false. It is unestablished, which is a different and more boring
thing, and the earlier claim should have said so.

---

*One result, published as a finding, did not survive its first replication. The replication was
cheap and the finding was load-bearing to how this lab planned to aim its next legs, which is the
only reason it was checked at all.*
