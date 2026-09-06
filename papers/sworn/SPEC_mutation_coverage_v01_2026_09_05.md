# SPEC — mutation coverage: what can the differential test actually see?

Fathom Lab · 2026-09-05 · **A spec, not a result.** Frozen in its own commit before any mutation is
run and before any catalogue is consulted. It makes no numeric claim; the numbers it exists to
produce are produced by running it, and the gates below are written down first so the run cannot be
scored after the fact.

## Why this exists

`RESULT_differential_agreement_2026_09_05.md` reports 150000 generated documents on which two
independent implementations of the sworn format produced the same verdict core digest, and zero on
which they differed. That sentence has two readings and the result cannot tell them apart:

1. the two implementations agree, or
2. the generator cannot produce the inputs on which they differ.

Every differential test in the world has this problem, and the usual response is to report the
agreement number and move on. The agreement number is the flattering half. **A differential test
with no measure of its own detection power is an instrument with no calibration**, and this lab has
already refused that shape once — the coverage estimate withdrawn on 2026-09-02 was withdrawn for
exactly this reason, because its denominator measured the detector rather than the thing.

This is not hypothetical here. Before this spec was written, an ad-hoc probe mutated four constants
in the JavaScript verifier and ran the guard against each. Three were caught. **One was not**:
removing the BOM refusal from the strict JSON parser produced zero disagreements in 5000 cases,
because the harness builds every manifest as a Python object and serialises it, so no manifest JSON
it generates is ever BOM-prefixed. The frozen `SPEC_differential_agreement_v01` D2 lists BOM among
the byte-level hazards the grammar composes — and for the document bytes that is true, and for the
manifest JSON it is not. That miss is the reason this spec exists, and it is named here rather than
discovered again later, because a study prompted by a known failure should say so.

The question, then: **for what changes to either implementation would the differential harness raise
an alarm, and for what changes would it stay silent?** The silent set is the finding.

## The method

A **mutant** is one shipped implementation with a single localised edit applied to a scratch copy.
The tree is never written. The harness's own comparison — the same code path the standing guard
runs — is executed with the mutant substituted for one side, at the guard's own size and seed, and
the mutant is **caught** if it produces at least one disagreement and **missed** if it produces
none.

Missing is the interesting outcome. It means a real behavioural difference between the two
implementations of this format could exist today, in that exact place, and every number this lab has
published about their agreement would look the same.

## The rules, each with its attack

**M1 — a mutant must load and run, or it is not a measurement.** A mutation that introduces a syntax
error, or that makes the implementation raise on every input, tells us nothing about detection: the
harness would "catch" it for a reason that has nothing to do with the format. Mutants that fail to
load are counted and reported separately and are **excluded from the denominator**.
*Attack:* padding the catalogue with broken files to inflate the detection rate. *Answer:* the
non-viable count is published beside the rate, and a mutant that raises on more than half the cases
is reported as `degenerate` rather than caught.

**M2 — the catalogue is written from the source, not from the coverage.** Mutations are proposed by
reading the two implementations, region by region, without running the harness and without seeing
which regions it exercises. The proposers are told explicitly to include mutations they *suspect*
the generator does not reach, because those are the ones that expose holes.
*Attack:* proposing only mutations already known to be caught, so the instrument flatters itself.
*Answer:* the instruction above, and the region census below — a region with a high detection rate
and one proposed mutation is reported as such rather than averaged away.

**M3 — every run carries controls, and a caught control voids it.** A control is a deliberately
semantics-preserving edit: a reworded comment, a renamed local, whitespace. A guard that fires on a
control is not detecting divergence, it is detecting editing, and its detection rate means nothing.
*Attack:* a comparison keyed on the implementation's bytes rather than its behaviour. *Answer:*
G-K below makes a caught control VOID the run, not merely a note.

**M4 — both sides are mutated.** A study that only mutates JavaScript measures whether the harness
notices JavaScript changes. The Python verifier is the reference implementation and is mutated too,
loaded from a scratch copy under its own module name.
*Attack:* mutating only the side that is easy to swap. *Answer:* G-S below sets a floor on each
side independently.

**M5 — the miss list is published in full, whichever way it falls.** Every missed mutation is named
in the receipt with its region, its anchor, and what behaviour it changed. There is no bar on the
detection rate and no threshold that makes the run a success: this is a measurement of an
instrument, and an instrument that turns out to be weak is a finding, not a failure to be reworded.
*Attack:* reporting "detection rate 84%" and omitting which 16%. *Answer:* the misses are the
result; the rate is the summary.

**M6 — the run writes one receipt and never rewrites it.** `mutation_coverage.json` carries the
seed, the size, every mutation with its verdict, the controls, the non-viable list and the region
census. A second run is a second file. The receipt does not wear a name another sweep claims.
*Attack:* regenerating the receipt after strengthening the grammar, so the instrument looks better
than it was. *Answer:* the rule this lab already pays — a receipt is history — and if the grammar is
later strengthened, that is a second grammar, a second run and a second file, and the pair of
numbers is the interesting thing.

## The frozen gates

Written here, before the catalogue is read and before anything is run:

| gate | quantity | bar |
|---|---|---|
| G-M | viable mutants measured | ≥ 25 |
| G-K | controls caught | **exactly 0** — a single caught control makes the run VOID and it reports nothing about detection |
| G-S | viable mutants per side | ≥ 5 on `js` and ≥ 5 on `python` independently |
| G-R | regions with at least one viable mutant | ≥ 5 |
| G-D | detection rate, caught / viable | **no bar** — reported overall and per region, with every miss named in full |

A run that fails G-M, G-S or G-R is under-powered and says so in its own headline. A run that fails
G-K is VOID. G-D cannot be failed, only reported, which is the point of it.

## What this spec does not say

That a caught mutation means the format is correctly implemented — it means a difference in that
place would be visible, nothing more. That a missed mutation is a bug: it is a region where a bug
would be invisible to this harness, which is a statement about the harness. That the mutation
catalogue is exhaustive over the implementations' behaviour; it is what the proposers found in the
regions named, and the completeness critic's gap list is published beside it precisely because it is
not exhaustive. That detection rates from this study transfer to any other differential test, or
that this seed and this size are the right ones — a larger run would catch strictly more, and the
size used is the size the standing guard actually runs, which is the number a reader cares about.

---

*A test that has never been seen to fail is not known to be a test. This one has one hundred and
fifty thousand agreements behind it and, until now, no evidence at all that it could have told us
otherwise.*
