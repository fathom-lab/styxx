# SPEC — the browser verifier v0.1: a second implementation, held to the vectors before anyone is asked to trust it

Fathom Lab · 2026-09-05 · **A spec, not a result.** Frozen in its own commit before any code.
Leg 3, item 5 of `papers/PLAN_the_next_level_2026_09_02.md`, under the label the plan writes and
this document repeats wherever the artifact is described:

> re-derives sworn span verdicts offline; a forger controlling the whole file passes both browser
> layers; the package at the named commit is the check

It makes no numeric claim. The one number this leg produces is how many conformance vectors the
second implementation reproduces, and it is produced by running it.

## Why this exists

`conformance/sworn/` exists so a second verifier can be shown where it disagrees, byte by byte,
before anyone is asked to trust it. Nothing has yet been held to it. A capsule's browser layer
today compares digests — it can say *these bytes are not the bytes that were sealed*, and it
cannot say *this span does not hold against its receipt*. A reader with no Python, no clone and no
network can therefore check integrity and not verdicts, which is the half of the promise the
format does not yet keep offline.

This spec builds the other half: one JavaScript file that re-derives the verdict core from the
same bytes the Python verifier reads, and is held to the committed vector set for the subset it
implements. It changes no verdict, adds no kind, and moves nothing in `styxx/sworn.py`.

## The rules, each with its attack

**B1 — the subset is declared, and everything outside it is counted, not hidden.** The verifier
implements exactly:

| in scope | out of scope for v0.1 |
|---|---|
| mode `inline`: document bytes + an optional manifest → the verdict core | modes `canon`, `sidecar`, `load`, `manifest`, `receipt_check` |
| receipts `rN` and `rN#/pointer` against an embedded manifest, resolved from the manifest's own `bytes` | a live git tree: any vector whose `requires` contains `tree` |
| receipts `path:` and `prereg:` **when no tree is given** — which is UNRESOLVED `no_repository`, and must be reproduced as such | resolving those receipts against a tree |
| all four kinds (`numeric`, `quote`, `hash`, `absent`) and every MALFORMED reason the subset can reach | the coverage block: it is outside the core by R9 and the observer is not portable |

The harness prints, per family, how many vectors it ran and how many it skipped, and the skip
reason is the vector's own `requires`, never a judgement of the vector.
*Attack:* a verifier that passes by skipping what it cannot do. *Answer:* the skip count is
printed beside the pass count, the acceptance bar below names an exact number of vectors that must
pass, and a skipped vector inside the declared subset is a failure of the run.

**B2 — the bar is the core digest, and nothing weaker.** For every in-scope vector the verifier
computes `core_sha256 = sha256(utf8(jcs(core)))` over its own core, and it must equal the
vector's `expect.core_sha256`. Comparing verdict strings, counts or span lists instead is
refused: they are the parts a wrong implementation gets right first.
*Attack:* a verifier that matches the digest by copying the expectation. *Answer:* the harness
computes the digest from the verifier's own object and never passes the expectation into it; the
verifier's entry point takes document bytes and a manifest, and returns a core.

**B3 — the acceptance bar, frozen here.** The set at `conformance/sworn/index.json` carries 1689
vectors in mode `inline` whose `requires` is a subset of `{manifest}`. **All 1689 must reproduce
their core digest.** A run that reproduces fewer does not ship; the count is reported either way,
and if the bar is not met the RESULT says so under that title.
*Attack:* moving the bar after the run. *Answer:* the number is in this frozen file, in a commit
that precedes the verifier.

**B4 — the semantics the two languages do not share are ported deliberately, and each is named.**
Python and JavaScript disagree on exactly the things this format decides with, and the vector set
pins every one. The verifier implements the Python side of each, and says so at the point of
implementation:

| what | Python | JavaScript | what the verifier does |
|---|---|---|---|
| `\w`, `\d` in the token grammar | Unicode | ASCII | Unicode property escapes, so `٣.٥` tokenises as Python does |
| `\s` in the sentence splitter | ASCII over bytes | Unicode | not reached: the splitter lives in the coverage block, outside the core |
| JSON numbers | `Decimal` of the source digits | IEEE double | a decimal carrying the source digits and exponent, never a JS number |
| `NaN`, `Infinity`, `-Infinity` in a receipt | accepted, kept as `Decimal` | `JSON.parse` throws | a reader that accepts them and marks the leaf non-finite |
| duplicate JSON keys | remembered; a pointer through one is `pointer_ambiguous` | last wins, silently | the reader records duplicates per object |
| a leading BOM | `ValueError` → `receipt_not_json` | stripped silently | refused explicitly |
| `str(Decimal)` in `detail.receipt` | scientific above the plain range | `String(number)` | Python's `to-scientific-string`, ported |
| quantize to N places | `ROUND_HALF_EVEN` | `toFixed` (half-away-from-zero, and lossy) | half-even on integers, exact |
| a lone surrogate in a leaf | not encodable → `leaf_not_string` | encodes to U+FFFD | detected and refused as Python does |
| byte offsets | native | UTF-16 code units | every offset is computed over a `Uint8Array`, never a string index |

*Attack:* a verifier that is right on the lab's own documents and wrong on the tenth row of this
table. *Answer:* every row above is pinned by at least one vector in the set, and B3 admits no
exceptions.

**B5 — the verifier is a pure function, and the file says so.** `sworn_verify.js` exports one
entry point taking `(documentBytes, manifestObject|null)` and returning the core object. It reads
no file, fetches nothing, touches no clock and no global state, and it is loaded by the node
harness and by a capsule alike. It never decides whether to *believe* a document — it re-derives
what the Python verifier would say.
*Attack:* a browser bundle that reaches for the network to resolve a `path:` receipt. *Answer:*
there is no I/O in the file; `path:` with no tree is UNRESOLVED by B1 and by the vectors.

**B6 — the capsule's sworn profile seals four things and fails closed.** A sworn capsule embeds
the document bytes, the manifest, the verdict receipt issued by the Python verifier, and this
verifier's own bytes. Layer 1 (browser) re-derives the core with `sworn_verify.js` and compares
its digest to the sealed receipt's; layer 2 (`python -m styxx.capsule verify`) re-runs
`styxx.sworn` over the same bytes. The profile refuses to mint, by name:

| refusal | when |
|---|---|
| `sworn_no_manifest` | a document with an `rN` receipt and no manifest to seal |
| `sworn_receipt_mismatch` | the supplied receipt's core digest does not re-derive from the sealed bytes |
| `sworn_manifest_mismatch` | the receipt names a manifest digest the sealed manifest does not have |
| `sworn_tree_receipt` | a span carries `path:`/`prereg:` — v0.1 seals no tree, so the browser could only ever call it UNRESOLVED |
| `sworn_document_mismatch` | the document bytes do not hash to the receipt's `inline_sha256` |

and on verification distinguishes **INSTRUMENT SKEW** (the sealed receipt was issued by a
different `styxx.sworn` build) from **tamper** (the bytes moved under a build that matches).
*Attack:* calling the result self-verifying. *Answer:* the label in the epigraph is printed in the
capsule itself and in the README; a forger who controls the whole file controls both browser
layers, and the package at the named commit is the check.

**B7 — no verdict moves.** `styxx/sworn.py` is not edited by this leg. If a `verdict_core()`
split is wanted it must be a zero-verdict-change refactor that preserves the emitted key order and
leaves `conformance/sworn/index.json`'s `set_sha256` where it is; if the digest moves, it is
reverted.
*Attack:* a convenient refactor that re-keys the vector set. *Answer:* `gen_vectors.py --check`
after any touch, and this rule.

## Tests this spec commits to

`tests/test_sworn_verify_js.py`: node is invoked over the committed set; the run reports pass,
fail and skip per family; the test asserts zero failures, and that the number of in-scope vectors
run equals B3's 1689. `tests/test_capsule_sworn.py`: a capsule minted over a committed sworn
document with `rN` receipts verifies at both layers; each of B6's five refusals fires on its own
input; a tamper battery flips a document byte, a receipt verdict, the manifest digest and the
embedded JavaScript, and each is caught by the layer this spec names; a re-sealed forgery passes
layer 1 and is caught by layer 2, and the RESULT says so in those words.

## What this spec does not say

That agreement on the vectors makes either verifier correct — they agree on what two test files
exercise, both written in this lab. That the browser layer is a check on the author: a forger
controlling the file controls the layer. That `path:` receipts can be verified offline: they
cannot, here, and are UNRESOLVED by construction. That anything is self-verifying, tamper-proof or
immutable. That a second *independent* implementation exists: the same hands wrote both, and the
next attacker should be somebody else.

---

*The vectors were built so that a second verifier could be held to something. This is the first
one, and it is held to 1689 of them or it does not ship.*
