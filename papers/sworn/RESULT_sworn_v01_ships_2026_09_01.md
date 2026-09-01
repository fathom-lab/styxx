# RESULT — sworn output v0.1 ships: the author declares, the receipt disposes

Fathom Lab · 2026-09-01 · Spec: `SPEC_sworn_output_v01_2026_09_01.md`, frozen before any code.
Module: `styxx/sworn.py`. Tests: `tests/test_sworn.py`. Harness: `harness_pytest.py`, turn
`turn_2026_09_01_sworn_v01`. **This document is itself sworn**: every count in it is bound by `sworn/0.1` to a
receipt the author could not have written — the manifest the harness minted from the test run,
or a file at commit `10b051b6608d` — and its sidecar and verdict receipt sit beside it. Nothing here
is a measurement of sworn output. That is owed item 3 of the spec and it is still owed.

## What was built

<sworn r="prereg:cd29034153d2aa81c6f38bfb46557fcd99414ee542eed1adab29fdd26d7af8f1" k="quote">The spec's invariant reads `THE AUTHOR CHOOSES WHAT TO SWEAR; THE AUTHOR CANNOT CHOOSE WHAT THE RECEIPT SAYS.`</sworn>
The module implements it as a byte-exact lexer, a canonicalizer whose round trip is asserted
before a sidecar is written, three receipt forms resolved as pure functions of bytes, four check
kinds with no float, no percent conversion, no normalisation and no search over leaves, and a
content-addressed verdict receipt that re-derives in the parrhesia manner.

<sworn r="path:styxx/sworn.py" k="hash">At commit `10b051b6608d` the verifier's own bytes hash to `628eea7a03b5d40715153252ae4b62fc5b2d8fc9d96832623871c6698deb5600`.</sworn>
<sworn r="path:papers/sworn/turn_2026_09_01_sworn_v01.test_run_result.json#/sworn_lines" k="numeric">It is 1541 lines long.</sworn>
<sworn r="r1" k="numeric">The harness ran the suite and 296 tests passed.</sworn>
<sworn r="r4" k="numeric">0 failed.</sworn>
<sworn r="r2" k="absent">pytest's output carries no `failed`.</sworn>
<sworn r="r3" k="quote">ruff printed `All checks passed!`</sworn>
<sworn r="path:papers/sworn/turn_2026_09_01_sworn_v01.test_run_result.json#/python" k="quote">The interpreter was `3.11.15`.</sworn>

The four receipts above marked `rN` were minted by `harness_pytest.py` from what pytest and ruff
wrote, with `kind_of_source` naming where each came from; the harness recorded no
`authored_sha256`, because a script that runs after the agent has finished cannot see what the
agent wrote during the turn, so invariant 2 rests on `kind_of_source` alone for this manifest.
The verdict receipt says so.

## The two sworn documents

<sworn r="path:papers/closed-model-frontier/DECLARATION_h_mapping_2026_09_01.sworn-receipt.json#/counts/HELD" k="numeric">The h declaration binds 25 counts to its census receipt, and all of them held.</sworn>
<sworn r="path:papers/closed-model-frontier/DECLARATION_h_mapping_2026_09_01.sworn-receipt.json#/document_verdict" k="quote">Its document verdict reads `SWORN-HELD`.</sworn>
<sworn r="path:papers/closed-model-frontier/DECLARATION_h_mapping_2026_09_01.sworn-receipt.json#/counts/UNRESOLVED" k="numeric">0 of its spans were UNRESOLVED.</sworn>
This document is the second. `tests/test_sworn_dogfood.py` renders every committed sidecar back to
the committed bytes and re-derives every committed receipt at the commit it names; a receipt that
stops re-deriving fails the suite rather than sitting quietly in the tree.

## What the design refuses, mechanically

The receipt-shopping attack in `papers/charon/RECON_state_2026_09_01.md` — an OATH verdict on
fixed bytes moves with the receipt set the author supplies, because the verifier value-matches
over every leaf — cannot reach a sworn span: the author named the leaf, so a larger pool has
nothing to offer. <sworn r="path:tests/test_sworn.py" k="quote">The suite pins it as `test_receipt_shopping_moves_oath_but_cannot_move_sworn`.</sworn>

The OATH `_NUM` trailing-period defect — `precision of 0.55.` certifying with zero tokens
examined — cannot occur: <sworn r="path:tests/test_sworn.py" k="quote">the suite pins `test_a_number_followed_by_a_sentence_period_is_still_the_number`</sworn>,
and the number grammar cuts a span into maximal tokens of which exactly one may carry a digit,
so a fragment of a number is never extracted and an identifier glued to digits is MALFORMED
rather than guessed about.

A broken tag is never narrative. A document that swore nothing is UNSWORN, never "no failures".
A receipt the verifier cannot see is UNRESOLVED and accuses nobody. Every decision the frozen
spec left open is named once in `DECISIONS`, carried in every receipt, and pinned by a test.

## What this does not say

That the format works. No panel has read a sworn document; no author outside this session has
written one; the coverage figure printed beside every verdict is counted by an instrument at a
documented ceiling and is advisory by construction. That the right sentences were bound: the
spans above are the counts this author chose to bind, and the narrative around them is exactly
as unverified as the spec says narrative is. That the manifest is honest: it is as trustworthy as
the committed script that minted it, and no more.

## Owed

The measurement (spec item 3). The price of trivial swearing against published coverage (item 4).
The eleven-instrument census (item 2). A pointer form for `rN` receipts, so a numeric span can
name a leaf inside a harness capture instead of needing a one-number capture. A rule for tags
inside HTML comments, which v0.1 recognises because the spec's lexical rules are closed. The
script-aware cap (item 5).

---

*The first sworn document about sworn output swears to what the test runner printed and to the
hash of its own verifier, and says, in the same breath, how little that proves.*
