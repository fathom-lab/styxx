# PROBE E outside our own code: an existence proof, and why it is not a ranking

**Date:** 2026-08-13. **Runs:** `probe_e_styxx_v2_joined.json`,
`probe_e_numpy_v3_joined.json`, both from the instrument corrected after adversarial
review (`FINDING_adversarial_review_2026_08_13.md`). sklearn is re-running and is not
reported here.

## The measurement

| repository | terms | exercised | dead / powered | **dead / decisions** | value-position dead | no-population files |
|---|---:|---:|---:|---:|---:|---:|
| styxx | 4,995 | **37.1%** | 43.3% | **40.4%** | 186 | 0/129 |
| numpy | 6,396 | 53.5% | 19.0% | **18.1%** | 53 | 30/178 |
| sklearn | 9,484 | **82.5%** | 16.6% | **16.2%** | 80 | 2/247 |

**The comparison tool refuses to compare these rows, and it is right to.** Exercised
fractions span 37.1%–82.5%, a factor of 2.22 against a stated limit of 1.5.
`dead_rate` is computed over each suite's exercised minority, so a row where the suite
drives 82.5% of the decision terms and a row where it drives 37.1% are not answering the
same question. The per-row numbers stand; the pairwise reading does not, and the tool
now blocks it rather than printing a paragraph asking the reader to be careful.

**The coverage column is the more actionable finding.** sklearn's own suite exercises
**82.5%** of its decision terms to the observation floor; styxx's exercises **37.1%**.
Whatever the dead rates mean, that gap is a direct statement about how much of each
package its tests actually drive, and it needs no cross-repo inference to read.

The headline column is decision terms only — `if`/`while`/`assert`/`return` positions.
The pooled column includes value-position operands (`float(x or 0.5)` picking a
default), which adversarial review showed are not gates and made up 21.9% of styxx's
originally published dead count.

Excluding terms that clear the observation floor on *process count* rather than
population variety: styxx **40.35%**, numpy **17.4%** (numpy carries 41 such terms
against styxx's 2, because 148 interpreters contributed).

## What is licensed

**A dead-decision rate of this magnitude is measurable by this instrument in code we did
not write.** That is the whole claim, and it is worth something: it means styxx's 40.4%
is not self-evidently an ecosystem norm, and it means the instrument works on a codebase
whose conventions nobody here chose.

## What is not licensed, and why — with the numbers

**This pair is not a quality ranking and must not be quoted as one.** `dead_rate` is a
joint property of code composition, suite design, and observations per term, and all
three differ by construction. Two measurements make that concrete, both computed here
rather than asserted:

| | styxx | numpy |
|---|---:|---:|
| `@pytest.mark.parametrize` per 1,000 test lines | 1.3 | **9.8** |
| powered terms reading dispatch vocabulary (`dtype`, `axis`, `out`, `shape`, `order`, `casting`, `subok`) | 0.6% | **16.6%** |

numpy's suite parametrises **7.5× more densely**, and 16.6% of its powered terms read
exactly the parameters those sweeps vary — against 0.6% in styxx. **A suite built to
sweep the parameters its gates read will move those terms by construction.** That is not
numpy gaming anything; it is what a numerical dispatch library's tests are *for*. But it
means a large part of the gap measures test design rather than code health.

Three further asymmetries, each of which cuts the same way:

- **numpy's branching logic is largely in C** and invisible to a Python AST prober. The
  figure describes a Python veneer, not the library.
- **30 of numpy's 178 test files never ran** (f2py, needing a Fortran compiler). Their
  rows are now correctly discarded rather than merged — an earlier run merged them and
  produced a confident 69% for numpy off import-time observations alone.
- **n=1.** This replaces a withdrawn comparison that pooled nine libraries. There is no
  between-repository variance estimate here at all.

## The static screen generalises worse than the gap does

The census miss and false-positive rates, verified identical under a collision-free join
key and therefore quotable as **"at least"**:

| | styxx | numpy | sklearn |
|---|---:|---:|---:|
| dead terms the static screen never flagged | at least 68.8% | at least **92.6%** | — |
| flagged-and-adjudicable functions that execution shows are live | 15.8% | **39.4%** | **44.4%** (28/63) |

It is tempting to read this as a second axis on which numpy looks worse. **It is not, and
using it that way would be the same error one page later.** `falsifiability_census.py`
defines its at-risk vocabulary from *styxx's own* failure modes, so a higher miss rate on
numpy measures the screen's parochialism. And the false-positive rate is mechanically
coupled to the very gap just declared unlicensed: numpy's powered terms are 81.0% live
(2,772/3,424) against styxx's 56.7% (1,052/1,855), so a flagged numpy function is
likelier to come out all-live for exactly the same reason the headline differs.

The honest statement is about the screen, not about numpy: **a static falsifiability
screen written against one codebase misses at least 92.6% of the real dead terms in
another.** Whatever else the census is, it is not portable.

## What would make this a comparison

The comparator set and the adjustment plan fixed **before** the runs. sklearn and scipy
ship runnable suites and were censused the same afternoon; sklearn is running now. Even
then, a defensible comparison needs a way to hold suite design constant, and
parametrisation density is the obvious covariate — 1.3 against 9.8 is not a nuisance
term, it is plausibly most of the effect.

Until then this is one number from one foreign repository, and it is reported as that.
