# RESULT — half the changes to the layers nothing can reach would ship green

Fathom Lab · 2026-09-06 · Spec: `SPEC_suite_power_v01_2026_09_06.md`, frozen with its five gates
before any mutant ran. Catalogue: `conformance/sworn/suite_power_catalogue_audited.json`, committed
before the run. Receipt: `conformance/sworn/suite_power.json`. **This document is itself sworn.**

## What this measures and why it had to be this

`RESULT_mutation_coverage_2026_09_05.md` found six blind spots no generator can ever reach. They are
not gaps in a grammar — they are whole layers the differential does not touch, because the
JavaScript verifier has no repository and the sidecar and receipt layers sit outside the compared
verdict core. `RESULT_aperture_closure_2026_09_06.md` then showed what a blind spot can hide: two
real defects, one verdict-changing, in a region two independent instruments called clean.

These layers are in exactly that condition and have no second implementation to compare against. The
only instrument left is the test suite, and the question is the one anybody about to change them
actually has: **if I broke this line, would anything fail?**

## The result

<sworn r="path:conformance/sworn/suite_power.json#/counts/viable" k="numeric">51 viable mutants were measured</sworn>
against a bar of twenty-five, across all four layers, and
<sworn r="path:conformance/sworn/suite_power.json#/counts/killed" k="numeric">25 were killed.</sworn>
<sworn r="path:conformance/sworn/suite_power.json#/counts/survived" k="numeric">26 survived</sworn>
— a kill rate of
<sworn r="path:conformance/sworn/suite_power.json#/gates/G-S/value/rate" k="numeric">0.4902.</sworn>

The catalogue was clean, so nothing left the denominator quietly:
<sworn r="path:conformance/sworn/suite_power.json#/counts/anchor_missing" k="numeric">0 anchors failed to match</sworn>
and
<sworn r="path:conformance/sworn/suite_power.json#/counts/non_viable" k="numeric">0 mutants failed to load.</sworn>
<sworn r="path:conformance/sworn/suite_power.json#/counts/controls_killed" k="numeric">0 controls were killed,</sworn>
without which the run would be void and this number would mean nothing.

**Half of what can be broken in these layers can be broken silently.**

## Where, and the order is the finding

| layer | killed / viable | |
|---|---|---|
| coverage | 8/11 | 72.7% |
| receipt | 7/13 | 53.8% |
| sidecar | 6/13 | 46.2% |
| **tree** | **4/14** | **28.6%** |

The weakest layer is the one carrying the entire `UNRESOLVED` reason vocabulary — the code that
decides whether a receipt was found, was too large, was not a blob, or names a commit this handle
was not taken at. It is also the layer with no second implementation anywhere.

The survivors are not evenly distributed within a layer, either. Both of the sidecar's **injection
boundaries** survived: `load_sidecar`'s tag-grammar guard on the `receipt` and `kind` attributes,
which are interpolated straight back into document bytes by `render`, and the upper bound on a span's
end offset, where Python's slicing clamps rather than raising. Both live on the standalone `render`
path, where `verify()`'s re-scan reconciliation never runs to catch what they let through.

## How thin the defence is where it exists

A kill rate says a layer is defended. It cannot say by how much.

Of the twenty-five kills,
<sworn r="path:papers/sworn/suite_power_concentration.json#/killed_by_exactly_one_test" k="numeric">9 rest on a single test,</sworn>
and
<sworn r="path:papers/sworn/suite_power_concentration.json#/killed_by_five_or_fewer" k="numeric">20 rest on five or fewer.</sworn>
Each of those nine is one deleted assertion from joining the survivor list, and the assertion is
named in the receipt.

The starkest is `verify_receipt reports VERIFIED on the digest alone`. That mutation removes
`reproduces` from the conjunction that decides whether a receipt verifies — the join to which the
whole *trust neither the author nor the verifier* doctrine reduces — and exactly one test notices.

## Who does the defending

<sworn r="path:papers/sworn/suite_power_concentration.json#/top_killing_file_kills" k="numeric">tests/test_sworn_dogfood.py accounts for 105 kills,</sworn>
more than any other file. That test does one thing: it re-derives every sworn document committed
under `papers/` and fails if any receipt stops re-deriving. **The lab's practice of swearing its own
documents turns out to be its strongest single test**, which is a better argument for dogfooding
than an essay about dogfooding.

## Two repairs, and what they cost to find

Two mutations from the pilot survived the whole oracle, and both are instructive because neither is
visible to line coverage — every line involved executes either way.

**The cap constant could be raised a thousandfold.** `test_an_entry_over_the_cap_...` builds its
fixture as `sworn.MAX_RECEIPT_BYTES + 1`, so the fixture rises with the cap. *A test whose fixture
derives from the constant under test cannot detect a change to that constant.*

**The size clause could be deleted outright.** The gate is a disjunction, and every fixture in that
test reaches `receipt_too_large` through the first disjunct — `big` and `small` both carry
`bytes: None`, and `at_cap` sits at the cap rather than over it. There was no entry with bytes
present and size over the cap, which is the only shape the size clause decides. *A test can be named
for a behaviour, assert on that behaviour, pass, and never exercise the branch it is named for.*

Both are repaired here, and the repair is checked the way this corpus checks everything: re-running
the two mutations against the strengthened tests kills both, and the control still survives.

## What this does not say

**That a killed mutant means the code is correct.** It means a change there would be noticed.

**That a survivor is a bug.** It is a place where a bug would be invisible to this oracle. Some
survivors are behaviours nobody has decided are load-bearing; the point is that the decision has not
been made, not that it has been made wrongly.

**That this is the repository's whole defence.** The oracle is five named files, and S1 says so. A
mutant that survived here may be killed elsewhere — and one from the pilot was: `receipt_too_large`
is exercised by `test_sworn_conformance.py`, outside the oracle, which is how the two traps above
were diagnosed rather than merely counted.

**That the rate is a quality score.** 0.4902 is one catalogue against one oracle. The survivor list
is the deliverable; the rate is how it was summarised.

**That the catalogue is exhaustive.** The critic's gap list, published in the catalogue file, names
regions still unmutated after its own additions — including `_coverage`'s advisory binding, which
no mutation can measure because `coverage_reproduces` is computed and then never consulted by
`status`.

---

*Two legs measured what an instrument could see. This one measured what the repository can defend
over the ground no instrument reaches, and the useful half of the answer is the half where it
cannot.*
