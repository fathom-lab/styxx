# RESULT — the browser verifier ships: a second implementation, in another language, agrees on every vector in scope

Fathom Lab · 2026-09-05 · Spec: `SPEC_sworn_browser_verifier_v01_2026_09_05.md`, frozen before any
code, with the acceptance bar written into it before the verifier existed. Verifier:
`styxx/_data/sworn_verify.js`. Harness: `conformance/sworn/replay_js.js` →
`conformance/sworn/replay_js_report.json`. Tests: `tests/test_sworn_verify_js.py`,
`tests/test_capsule_sworn.py`. **This document is itself sworn**, at the commit its sidecar names.
Leg 3, item 5 of `papers/PLAN_the_next_level_2026_09_02.md`, under the label the plan writes and
this document repeats wherever the artifact is described:

> re-derives sworn span verdicts offline; a forger controlling the whole file passes both browser
> layers; the package at the named commit is the check

## What was built, and what it is held to

`conformance/sworn/` was built so a second verifier could be shown where it disagrees, byte by
byte, before anyone was asked to trust it. Until now nothing had been held to it. This is the
first thing that has been.

The verifier is one JavaScript file: the lexer, the receipt grammar, the manifest, the four kinds
and the core assembly, ported so that `sha256(utf8(jcs(core)))` equals what `styxx.sworn` computes
over the same bytes. It reads no file, fetches nothing, and touches no clock.
<sworn r="path:styxx/_data/sworn_verify.js" k="hash">At the commit this sidecar names it hashes to e633626063e8be9b9c22b1c57c18a542e4751dc69e468ccffd1e50420b9a334b.</sworn>

## The number

The bar the spec froze before the code existed was every in-scope vector, and it named how many
there were.
<sworn r="path:conformance/sworn/replay_js_report.json#/ran" k="numeric">1689 vectors are in scope — mode `inline`, with `requires` a subset of `{manifest}`.</sworn>
<sworn r="path:conformance/sworn/replay_js_report.json#/passed" k="numeric">1689 of them reproduce the verdict core digest.</sworn>
<sworn r="path:conformance/sworn/replay_js_report.json#/failed" k="numeric">0 disagree.</sworn>
<sworn r="path:conformance/sworn/replay_js_report.json#/skipped" k="numeric">1929 vectors are out of scope and are skipped,</sworn>
each counted with its own mode and `requires` as the reason: every mode but `inline`, and
everything that needs a git tree.
<sworn r="path:conformance/sworn/replay_js_report.json#/vectors_total" k="numeric">The committed set holds 3618 vectors in all,</sworn>
and the harness names the set it ran against:
<sworn r="path:conformance/sworn/replay_js_report.json#/set_sha256" k="quote">`85e2d3b95b778ad61402225e41727e7844ba42bd80daa36b453bc0247d6e371f`</sworn>.
<sworn r="path:conformance/sworn/replay_js_report.json#/families/fuzz/ran" k="numeric">1452 of the vectors run are the seeded fuzz corpus,</sworn>
which is the lexer's conformance and was not capped.

A test asserts that the number actually run equals the number the spec froze, so a set that grows
fails rather than quietly re-fitting the bar to the run.

## What the vectors caught

Five disagreements, every one found by a vector and repaired in the JavaScript, never in the
verifier:

| what disagreed | what the vector said |
|---|---|
| the class name Python prints for a parsed JSON object | `_Obj`, not `dict`, in `leaf_not_scalar`'s detail |
| `absent` over a receipt whose completeness the harness never declared | UNRESOLVED `manifest_no_completeness`, not a verdict on the bytes |
| `absent` over a receipt declared incomplete | MALFORMED `absent_over_partial`, with the resolution's provenance carried |
| a manifest's numbers | plain JSON, as `Manifest.from_dict` receives them, while receipt bytes are read with the decimal-exact reader |
| the last line of an `L`-anchor | its terminating newline is excluded from the haystack |

Each was a place where the two languages differ and the format decides with the difference. That
is what the set was built to surface, and it surfaced them in one run.

## The capsule

The sworn capsule profile seals the document bytes, the manifest, the verdict receipt and the
verifier's own bytes, and the page is written LF-only so the copy sealed as bytes and the copy the
browser runs are byte-identical on disk. It refuses to mint on five named conditions:
`sworn_document_mismatch`, `sworn_tree_receipt`, `sworn_no_manifest`, `sworn_manifest_mismatch`,
`sworn_receipt_mismatch`. The first one is committed beside the document it seals.
<sworn r="path:papers/sworn/CANNED_harness_junit_2026_09_05.capsule.html" k="hash">It hashes to c92934807ca36e3bc924cb6eb11f7daa46ccba663b03ddcc3a536ed796610093 at the commit this document names.</sworn>

**And the label is demonstrated, not asserted.** The tamper battery builds the forger the label
describes: a page whose inlined verifier quietly un-does the change made to the document, so that
layer 1 re-derives the sealed core and agrees with itself while the reader is shown something else.
Layer 1 believes it. Layer 2 names the inlined copy as not the sealed one. That is the whole
meaning of *a forger controlling the whole file passes both browser layers*, and it is now a test
rather than a sentence.

INSTRUMENT SKEW is kept apart from tamper: a receipt issued by another build whose core still
re-derives is reported as skew beside a verdict that holds, never as a lie.

## What this does not say

That agreement on these vectors makes either verifier correct — they agree on what two test files
exercise, and the same hands wrote both, which is the objection the next attacker should press.
That the format is covered: 1929 vectors are outside this subset, including every sidecar, every
canon and everything needing a tree. That `path:` receipts can be checked offline — they cannot,
here, and the profile refuses to seal a document that carries one rather than sealing a span the
browser could only call UNRESOLVED. That a capsule is self-verifying, tamper-proof or immutable:
it is none of those, and the page says so in the reader's own view. That this verifier has run in
any browser but the harness that drives it here.

---

*The vectors were built so a second verifier could be held to something. The first one was held to
1689 of them, disagreed five times, and the five were the places where two languages read the same
bytes differently — which is the only reason the set exists.*
