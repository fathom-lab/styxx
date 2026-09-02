# SPEC — Sworn output v0.2: the format hardens before it leaves the builder's hands

Fathom Lab · 2026-09-02 · An engineering spec (not a measurement prereg): frozen before any
code changes, with the attack that motivates each rule named beside it. Extends
`SPEC_sworn_output_v01_2026_09_01.md`, which is **not edited** — a spec is history too.
Module: `styxx/sworn.py`. Format identifier stays `sworn/0.1` for the document grammar the author
writes (nothing an author wrote under v0.1 becomes ill-formed here except the hidden-commitment
case below, which no committed document exercises); the manifest moves to `sworn/manifest/0.2`
and the verdict receipt to `styxx.sworn.verdict-receipt/v1`.

## Why this exists

v0.1 shipped with its owed items listed and no adversarial pass recorded. Before any sentence about
the format leaves this tree, the lab's standing rule applies: *no instrument is announced before
an adversarial pass.* That pass was run on 2026-09-02 against the module as committed at
`320b303`, in memory, on a `MemoryTree`, touching no file. Twelve attack shapes were tried; the
battery and its outcomes are `ATTACKS_sworn_v01_battery_2026_09_02.md`, and every row there is
pinned by `tests/test_sworn_attacks.py`. The rules below are the repairs the battery licenses.
The attacks it does not repair are listed in their own section, because a spec that names only
what it fixes is the half-truth this format exists to reject.

One finding is not an attack but a defect in the shipped instrument, and it is the reason the
coverage rule below is the largest change in this spec. **The coverage denominator was measured
on a different idiom from the documents it was printed beside.** `styxx.claimdetect` (STRUCT-1)
is a diff-claim detector for agent pull-request prose: action verb, concrete object, not stative,
no other actor. Its own docstring says result-shaped sentences — test totals, CI verdicts,
measured rates — "are never claims". Every sworn document in this tree is a result-shaped
document. Across the twelve committed `*.sworn-receipt.json` files at `320b303`, the printed
`coverage_estimate` ran from 0.6667 to 1.0 with between 0 and 8 narrative sentences counted as
claims; the grain synthesis printed 0.9412 with exactly one counted claim — a sentence fragment
about a mapping column — against roughly fifty narrative sentences. Probed directly, STRUCT-1
returns `is_claim=False` for "Precision was 0.16 on 100 items.", "We measured 0.9515 on n=165."
and "Every accusation upheld. Zero false positives. Ships today." The number beside every verdict
was therefore near-vacuous by construction: a rule written for one idiom, fired in another —
M1/M2 inside the format built to escape them. The spec's invariant 4 (coverage travels with every
verdict) stands; what travelled was the wrong number.

## The rules, each with its attack

**R1 — fragments on manifest receipts.** `rN#/pointer` (RFC 6901) and `rN#Ln[-Lm]` are legal,
with exactly the fragment grammar `path:` already has. `absent` and `hash` over a partial `rN`
are MALFORMED, as they are over a partial `path:`. *Why:* the v0.1 RESULT needed one-number
captures (`r1` = the passed count as ASCII) so a numeric span had a scalar to name; a harness
capture is usually a whole object, and the author should name the leaf, not the harness invent a
receipt per number. Probed at `320b303`: `<sworn r="r1#/passed" k="numeric">` is MALFORMED
`receipt_form` — the owed item, confirmed absent.

**R2 — hidden commitments.** A tag-shaped candidate inside an HTML comment (`<!--` … `-->`,
scanned outside fenced regions and inline code spans, comments never nesting) is MALFORMED
`hidden_commitment`. Never HELD, never narrative. *Why (A4):* a hidden `<sworn>` that HELDs
inflates the numerator of every coverage figure while rendering as nothing; v0.1 admitted this in
`DECISIONS.html_comments` and named it owed. Census at `320b303`: no committed document carries a
tag inside a comment, so no committed verdict moves.

**R3 — short needles.** For `quote` against a *whole* receipt, a needle under 16 bytes is
MALFORMED `short_needle`. Three exemptions, all stated: a `quote` against a JSON-pointer string
leaf is exempt, and so is one against a line anchor (in both the author narrowed the haystack by
naming it, and the comparison is against that alone); `absent` is exempt (a short absent needle is
the *conservative* direction — absence of a short string over a complete object is a stronger
oath, not a weaker one). *Why (A10):* a one-byte needle against a 70-byte receipt HELDs today;
against a 5 KB stdout almost any short needle would. The attack is the size of the haystack, not
the needle, which is why a haystack the author narrowed is exempt. Census at `320b303`: four
committed needles are under 16 bytes — `SHIPPED` (pointer leaf), `3.11.15` (pointer leaf),
`SWORN-HELD` (pointer leaf), `failed` (`absent` over `r2`) — all exempt, so no committed verdict
moves. The number 16 is a decision, not a measurement; it is carried in `DECISIONS` and can be
argued with.

**R4 — the cap counts code points.** The inner text is capped at 300 Unicode code points, not 300
bytes. Still a pure function of bytes: invalid UTF-8 is already document-level MALFORMED before
any span is measured. *Why:* v0.1 disclosed its own inequity (three-byte scripts got a third of
the cap). The "at most one sentence" rule is unchanged and still authorial. Census: no committed
span exceeds 300 bytes, so none exceeds 300 code points.

**R5 — the `attestation` source kind.** Added to the closed external vocabulary for DSSE / in-toto
envelope bytes a harness recorded. The kind names the *shape* of the bytes. No signature is
verified, no key material is consulted, no trust root is pinned — exactly the boundary
`styxx.evidence` states under "WHAT `binding` IS NOT". A span may swear a `hash`, `quote` or
`absent` over attestation bytes; it may not swear that the attestation is genuine, because the
verifier does not know.

**R6 — manifest `sworn/manifest/0.2` and the trust ladder.** The manifest gains `rung` and an
optional per-receipt `harness_note`. The rungs are a closed set and every receipt prints the one
its manifest declares:

| rung | who minted | what it is worth |
|---|---|---|
| L0 | the author, or bytes in `authored_sha256` | MALFORMED `receipt_author_minted` — unchanged from v0.1 |
| L1 | a local hook sharing a filesystem and a shell with the agent (a Claude Code hook, an OpenClaw skill) | **weak**: the agent's shell can write what the hook later reads; the manifest is as trustworthy as that isolation |
| L2 | a runner that minted after the agent's turn ended and that the agent could not write to (a CI job on the base branch) | the strong rung this version reaches |
| L3 | a signed attestation whose signature this verifier checked | **reserved**; no producer in v0.2 |

A `manifest/0.1` file still loads; its rung is reported as `undeclared`, never as L2. The ladder
is SLSA-shaped and `authored_sha256` is the in-toto *products* set, unsigned — both credited here
rather than claimed. *Why (A5, A7):* the battery showed a manifest minted by the agent itself, with
an honest-looking `kind_of_source`, verifies HELD, and a `path:` receipt carries no statement of who
committed the blob. The verifier cannot detect either; it can refuse to hide them.

**R7 — provenance is printed on every span.** Each span verdict carries `provenance`: for `rN`,
`{harness, rung, kind_of_source}`; for `path:` and `prereg:`, the literal
`"committed object at <commit>; authorship unchecked"`. The document receipt carries a `rungs`
summary (count of spans per rung). The `certifies` sentence gains the words *"at the rung the
manifest declares"*.

**R8 — the coverage estimate is withdrawn; counts are printed instead.** The `coverage` block
becomes schema `sworn/coverage/1`, advisory always, and carries:

- `sworn_total` — spans of every verdict;
- `narrative_sentences` — the diffgate splitter's count over the narrative (canonical text minus
  sworn spans minus fenced regions), non-empty after stripping;
- `sentence_share = sworn_total / (sworn_total + narrative_sentences)`, null on 0/0 — **a floor,
  not an estimate**: it treats every narrative sentence as if it were load-bearing, so it cannot
  flatter a document and it says nothing about which sentences mattered;
- `diff_claim_sentences` and `diff_claim_share` — STRUCT-1's count, labelled with its idiom
  ("agent pull-request prose"), its ceiling (0.4211 precision, n=38, one model family, in-house)
  and the sentence that it does not read result-shaped prose as claims;
- `unsworn_claims` — STRUCT-1's flagged sentences, kept for the one idiom it was measured on.

The headline prints `sentence-share` and the two raw counts. `estimate` no longer exists; a reader
who wants "how much of what mattered was bound" is pointed at the measurement design
(`DESIGN_sworn_measurement_2026_09_01.md`, Q1), which is the only instrument that can answer it,
and which has not run.

**R9 — verdict receipt `styxx.sworn.verdict-receipt/v1`: coverage leaves the digest.** The
content-addressed core is `{document, commit, manifest_digest, spans, counts, sworn_total,
unresolved, document_verdict, document_malformed, rungs, certifies, verifier}`; `coverage` travels
beside it under its own `coverage_sha256`. `verify_receipt` re-derives the core and reports
coverage reproduction separately (`coverage_reproduces`, advisory). *Why:* v0.1 digested the
claimdetect block, so a receipt could not re-derive anywhere the observer differed — a build
without `claimdetect`, or a browser. A receipt of schema `/v0` is re-derived on its core minus
coverage minus verifier, with that note printed; its digest is still checked over its full body,
so tampering with a v0 receipt is still caught. **Every committed v0 receipt is re-issued under
v1 in a new commit; the v0 receipts remain in history.** A receipt is history too.

**Unchanged, deliberately:** the four invariants; the kinds and their checks (no float, no percent
conversion, no normalisation, no search over leaves); the canonical form and its asserted round
trip; the `path:` and `prereg:` resolvers; `exec` reserved; exit codes — sworn reports, it never
gates.

## The attacks v0.2 does not repair, named

| attack | why it stands | where it is priced |
|---|---|---|
| A1 rider clause — a HELD number painting an unchecked qualifier in the same sentence | by design: the span is a sentence, the check is one token; a smarter rule would be a claim detector | the pairing-validity cell of the measurement (does the leaf evidence the sentence?) |
| A2 trivial swearing | no verifier rule can tell a trivial oath from a load-bearing one without reading meaning | Q2 of the measurement; `sentence_share` makes the padding visible but cannot judge it |
| A3 the stative dodge | `sentence_share` counts every sentence, so the dodge no longer shrinks the denominator; STRUCT-1's blind spots remain inside `diff_claim_share`, which is why that share is labelled | — |
| A6 prereg precedence | `prereg:` proves content, not order; order is the ledger's and `styxx.protocol`'s job, and honest-signal's firewall is the prior art | `papers/LEDGER.md`, `_first_commit` |
| A7 a lying or compromised harness | the trust boundary; R6 prints the rung, nothing verifies it | L3, reserved |
| A11 post-hoc tagging | undetectable from bytes by construction (spec invariant 1 is a rule, not a check); a CI-side verifier narrows it because it reads the body as submitted | the Action, report-only |
| A12 wrong leaf, coincident value | a leaf that happens to hold the printed value HELDs; the author named it, so no search occurred, but the naming can be wrong | the pairing-validity cell |

## What this spec does not say

That the format has been measured: it has not (owed item 3 of v0.1, `DESIGN_sworn_measurement`).
That a SWORN-HELD document is true, or that the right sentences were bound. That `sentence_share`
is coverage — it is a floor computed by a sentence splitter, and the splitter is the lane's
largest false-flag source. That any rung above L1 exists in this tree today: the only committed
harness is `papers/sworn/harness_pytest.py`, which runs after the agent's turn and records no
`authored_sha256`; it declares L1. That "we know of no other format" — the survey that would price
that sentence is owed and is not this document.

## Owed after v0.2, recorded as owed

1. The measurement, with the repairs the 2026-09-02 review named: a receipt-seeing panel for the
   pairing-validity cell; a seeded-canary arm so the verifier's FAILED precision is a count with a
   denominator rather than a vacuous zero; Q1 reported in three cells (bindable and bound /
   bindable and unbound / unbindable under these kinds); seats from at least two model families.
2. Conformance vectors generated from `tests/test_sworn.py`, before any second verifier exists.
3. A browser verifier held to those vectors, for `rN` and embedded blobs only (`path:` and
   `prereg:` need a tree), and a capsule profile that fails closed.
4. Harness adapters at L1 (Claude Code hooks; OpenClaw) and L2 (a runner after the turn), each
   printing its rung.
5. The prior-art survey behind any "we know of no other".

---

*A format that was only ever written by its builder has not been attacked. This one now has
been, twelve ways, and the four rules above are what it paid. The number it printed beside every
verdict was measured on a different kind of sentence from the ones it stood beside; it is
withdrawn, and two counts that cannot flatter stand in its place.*
