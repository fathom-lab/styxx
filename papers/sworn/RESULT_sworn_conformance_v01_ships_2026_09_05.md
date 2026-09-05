# RESULT — sworn conformance vectors v0.1 ship: the tests are bytes now, under one digest

Fathom Lab · 2026-09-05 · Spec: `SPEC_sworn_conformance_vectors_v01_2026_09_05.md`, frozen in its
own commit before any code, with a dated ERRATA section appended in the commit that carries this
document. Module addition: `styxx.sworn.SnapshotTree`. Set: `conformance/sworn/`. Tests:
`tests/test_sworn_conformance.py` and `tests/test_sworn.py::TestSnapshotTree`. Leg 3, item 2 of
`papers/PLAN_the_next_level_2026_09_02.md`, under the plan's own label: **the precondition for any
second verifier; no claim.** **This document is itself sworn**: every count in it is a leaf of
`conformance/sworn/index.json` at the commit the sidecar names, a file the generator wrote and the
author did not edit, and its sidecar and verdict receipt sit beside it. Nothing here is a
measurement of anything, and nothing here says a second verifier exists.

## What the set is

<sworn r="path:conformance/sworn/index.json#/label" k="quote">The index carries the plan's label verbatim: `the precondition for any second verifier; no claim`.</sworn>
<sworn r="path:conformance/sworn/index.json#/vector_count" k="numeric">The set holds 3618 vectors.</sworn>
<sworn r="path:conformance/sworn/index.json#/family_count" k="numeric">They fall into 20 families, each named by the test class that produced its lowest source id.</sworn>
<sworn r="path:conformance/sworn/index.json#/blobs/count" k="numeric">Together they name 3977 byte objects, every one keyed by its own digest and carried in the blob store.</sworn>
<sworn r="path:conformance/sworn/index.json#/families/fuzz/count" k="numeric">The seeded fuzz corpus is carried in full, 3158 vectors, because a capped corpus with the same seed would be a different set.</sworn>
<sworn r="path:conformance/sworn/index.json#/outcomes/core" k="numeric">Of the outcomes, 2067 are verdict cores pinned by the digest of their canonical text,</sworn> <sworn r="path:conformance/sworn/index.json#/outcomes/refused" k="numeric">and 1256 are refusals pinned by a code from the table the index carries.</sworn>
<sworn r="path:conformance/sworn/index.json#/requires/tree" k="numeric">Only 116 vectors need a tree snapshot; every other vector is one a verifier for harness receipts and embedded blobs must pass.</sworn>
<sworn r="path:conformance/sworn/index.json#/modes/receipt_check" k="numeric">The 16 receipt checks carry the issuing build's verifier block inside their inputs by construction, which the spec's errata records.</sworn>
<sworn r="path:conformance/sworn/index.json#/unvectored/skipped_count" k="numeric">The recorder listed 3 calls the set cannot carry, each with its test id and its reason, and dropped nothing silently.</sworn>
<sworn r="path:conformance/sworn/index.json#/unvectored/verdicts/0" k="quote">The one verdict no vector produces is `WITHHELD`, which has no producer in the verifier.</sworn>
<sworn r="path:conformance/sworn/index.json#/clock" k="quote">Every manifest and every fixture commit in the set was minted at `2026-09-01T00:00:00Z`, the pinned clock.</sworn>
<sworn r="path:conformance/sworn/index.json#/set_sha256" k="quote">One digest pins every byte transitively: `96dfe15981209b847523687ca3e4854ad9086216fecdc1c055a9c2d38e60797e`.</sworn>

## What this does not say

That agreement on these vectors makes a verifier correct: it makes a verifier agree with this one
on inputs two test files chose. That the vectors cover the format: they cover what the tests
exercise, and the tests were written by the builder, the weakest attacker there is. That the fuzz
family is adversarial: it is a seeded random walk over a fixed list of atoms. That any second
verifier exists, or is easier to write because of this. That any number above is a measurement of
anything: each is a count of what a generator wrote, sworn so that the count cannot drift from the
file.

## Owed

The spec's own list, unchanged: a `committed` family from `tests/test_sworn_dogfood.py`; a `refusal`
attribute on the verifier's `SystemExit` if a second verifier needs codes at the source; the
`verdict_core()` split of `verify()`; the lone-surrogate row for the next attack pass; and the
Python-versus-JavaScript semantics the vectors pin and item 5 must implement.

---

*A verifier that only its author has run is a verifier whose behaviour lives in one head. These
bytes are that behaviour, under one digest, so that the next verifier can be shown where it
disagrees before anyone is asked to trust it — and so that this set, once sworn to, is never
regenerated in place.*
