# RESULT — sworn output v0.2 ships: the format is attacked, pays four rules, and withdraws the number it printed beside every verdict

Fathom Lab · 2026-09-02 · Spec: `SPEC_sworn_output_v02_2026_09_02.md`, frozen before any line of
`styxx/sworn.py` changed. Battery: `ATTACKS_sworn_v01_battery_2026_09_02.md`. Harness:
`harness_pytest.py`, turn `turn_2026_09_02_sworn_v02`, rung L1. **This document is itself sworn**:
every count in it is bound to a receipt the author could not have written — the manifest the
harness minted from the test run, or a file at the commit the sidecar names — and its sidecar and
verdict receipt sit beside it. Nothing here is a measurement of sworn output. That is owed item 3
of the v0.1 spec, redesigned in `DESIGN_sworn_measurement_v2_2026_09_02.md`, and still owed.

## What was built

<sworn r="prereg:d3f4f852acf3cb8acf7c3aa769cc28b0f2d1574a282b6ae3aa3aee6cf867f4b5" k="quote">The spec's opening rule reads `The rules, each with its attack`</sworn>, and there are nine of them.
Four repair the battery: fragments on manifest receipts (`rN#/pointer`, `rN#Ln-Lm`); a tag inside
an HTML comment is MALFORMED `hidden_commitment`; a short `quote` needle over a whole receipt is
MALFORMED `short_needle`, with pointer leaves, line anchors and `absent` exempt; the span cap
counts code points. Three print the trust boundary: an `attestation` source kind whose bytes are
a receipt and whose signature is never checked; a manifest that declares its rung, L1 or L2, L3
reserved; provenance on every span and a rung count on every receipt. One withdraws a number: the
coverage estimate, whose denominator was a diff-claim detector fired on result-shaped prose, is
replaced by two counts that cannot flatter. One moves the receipt: schema v1 digests the core
without coverage, so a receipt re-derives wherever the observer differs, and every committed v0
receipt has been re-issued under it in a new commit with no span verdict moving.

<sworn r="path:styxx/sworn.py" k="hash">At the commit this document names the verifier's bytes hash to f75aff8a47656ce74fbc34b6a566e17bb9e23c56da45eed3d28ca55503ca22f1.</sworn>
<sworn r="r5#/sworn_lines" k="numeric">It is 1789 lines long.</sworn>
<sworn r="r1" k="numeric">The harness ran the three sworn suites and 358 tests passed.</sworn>
<sworn r="r4" k="numeric">0 failed.</sworn>
<sworn r="r2" k="absent">pytest's output carries no `failed`.</sworn>
<sworn r="r3" k="quote">ruff printed `All checks passed!`</sworn>
<sworn r="r5#/python" k="quote">The interpreter was `3.12.10`.</sworn>
<sworn r="r5#/rung" k="quote">The manifest declares rung `L1`.</sworn>

Two of those receipts are the new form at work: `r5#/sworn_lines` names a leaf inside the
harness's own JSON capture, where v0.1 needed a one-number receipt per number. The rung is the
honest part: this harness is a committed script on the author's machine, and every span above
that rests on it prints L1 beside its verdict.

## The re-issue

<sworn r="path:papers/sworn/RESULT_sworn_v01_ships_2026_09_01.sworn-receipt.json#/schema" k="quote">The v0.1 RESULT's receipt now carries schema `styxx.sworn.verdict-receipt/v1`.</sworn>
<sworn r="path:papers/sworn/RESULT_sworn_v01_ships_2026_09_01.sworn-receipt.json#/rungs/committed" k="numeric">Of its spans, 9 rest on committed objects</sworn>
<sworn r="path:papers/sworn/RESULT_sworn_v01_ships_2026_09_01.sworn-receipt.json#/rungs/undeclared" k="numeric">and 4 on a manifest that declared no rung</sworn> —
the v0.1 harness predates rungs, and the receipt says `undeclared` rather than guessing.
<sworn r="path:papers/sworn/RESULT_sworn_v01_ships_2026_09_01.sworn-receipt.json#/coverage/sentence_share" k="numeric">Its coverage floor reads 0.1667</sworn>
where its v0.1 receipt printed an estimate of 0.9286 — the same document, the same spans, two
denominators, and the census in the battery document says which one was measuring what.
<sworn r="path:papers/sworn/coverage_census_v01_result.json#/documents" k="numeric">All 12 committed sworn documents were re-issued</sworn>
by `reissue_receipts_v1.py`, which refuses if any span verdict moves; none did. The v0 receipts
remain in history and still check on their core.

## What this does not say

That the format works: no panel has read a sworn document, and the six attacks the battery could
not repair are priced only by a measurement that has not run. That the floor is coverage: it
treats every narrative sentence as load-bearing and is printed as a floor. That any rung above L1
exists in this tree: the only harness is this script, and L2 is what a CI runner would be. That
the attacker was independent: the builder ran the battery, which is the weakest attacker there
is. That anyone can `pip install` this: `styxx.sworn` is unreleased and a stranger needs a clone
at a named commit.

## Owed

The measurement, redesigned (`DESIGN_sworn_measurement_v2_2026_09_02.md`) and waiting on a
signature. Conformance vectors before any second verifier. The prior-art survey before any
"we know of no other". Harness adapters at L1 and L2, each printing its rung. A release.

---

*The first sworn document about v0.2 swears to what the test runner printed, to the hash of its
own verifier, and to the rung it stood on — and says, beside every number, how little that
proves.*
