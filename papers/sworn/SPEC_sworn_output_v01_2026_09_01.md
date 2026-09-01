# SPEC — Sworn output v0.1: the author declares, the receipt disposes

Fathom Lab · 2026-09-01 · An engineering spec (not a measurement prereg): frozen before any
code exists, with the threat model and the honest boundary stated first. Module: `styxx/sworn.py`.
Format identifier: `sworn/0.1`.

## Why this exists

Every instrument this lab has built to find claims in prose has been measured against blind
panels and has come back short. The ones with receipts in this tree:

- diffgate, in-house 0.95 and **0.23** on agent-authored pull requests it had never seen
  (`RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md`, receipt
  `external1_summary.json`, with `CORRECTION_external1_cause_2026_08_31.md` attached).
- The same class after two cycles of repair aimed at named defects: held-out precision
  **0.16** against a floor of **0.95**, and the accusing verdict stays disabled
  (`RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md`, receipts `v14_gates.json`,
  `v14_adjudication.json`).
- OATH outside its own contract: a false-accusation rate of **0.2596** across 140
  repositories, against an earlier in-house reading that said the instrument was nearly
  silent outside the contract (`RESULT_oath_external_corpus_2026_08_27.md`, as recorded in
  `OATH_CONTRACT.md`).
- OATH's obligation predicate: **0.5811** of its verifications — 3,458 of 5,951 — were
  volunteered rather than obligated, and **0.3399** were value match alone with the receipt
  path never compared (`RESULT_unobligated_oath_2026_08_28.md`).
- STRUCT-1, the one structural repair in the lane that beat its null: precision **0.4211**
  against a frozen bar of **0.2061**, n=38 per arm, no significance claimed
  (`RESULT_struct1_beats_the_null_2026_08_31.md`, receipt `stage2_result.json`).

**UNVERIFIED.** The brief that commissioned this spec states the tally as *eleven* prose-
claimhood instruments measured and eleven failures. No receipt in this tree enumerates those
eleven, so the count is recorded here as unverified rather than repeated as a finding. The
census that would settle it is owed and listed below. What the receipts above do establish is
narrower and sufficient: the approach has been measured repeatedly, in-house and in the wild,
and has not once cleared a bar set before the measurement.

The failure has one shape. A verifier that reads prose is handed its target *by the text it
judges*. It must decide, from vocabulary and shape alone, which spans are assertions — and
that decision is where every measured error lives: the quoted `9` accused as a claim, the
command-line flag verified because its value matched a receipt field, the abstained median
that was a real claim and got no verdict at all.

So stop finding claims.

## The invariant

    THE AUTHOR CHOOSES WHAT TO SWEAR; THE AUTHOR CANNOT CHOOSE WHAT THE RECEIPT SAYS.

Sworn output is a format in which the speaker binds a claim, at write time, to bytes it could
not have authored. Everything the author does not bind is narrative **by definition** — not
by a verifier's judgement, not by a threshold, not by a template. The verifier is never handed
a target. It is handed commitments, and it checks them.

This does not make the author honest. It makes a specific kind of dishonesty mechanically
visible, and it moves the remaining dishonesty into a place where it can be counted.

## The model

A document is a sequence of spans.

- A **NARRATIVE** span is the default. It is unverified and it is **never accused**. There is
  no verdict for it, no band, no colour. The instrument has nothing to say about it and says
  nothing.
- A **SWORN** span is bound to exactly one receipt and exactly one check kind.

Sworn spans do not nest, do not overlap, and are at most one sentence. The length cap is
**300 bytes** of the span's inner text, measured over its UTF-8 encoding.

**Why a cap exists at all.** The verdict attaches to the span. If a span may be a paragraph,
an author binds one receipt to a mass of prose and a HELD verdict paints over everything
inside it — the binding becomes decorative, and coverage becomes uncountable because one
sworn span no longer corresponds to one asserted thing. The cap bounds the ratio of asserted
content to checked content, and it keeps the coverage denominator commensurable with its
numerator.

**Why a byte count and not a sentence parser.** This lane's own sentence splitter is measured
as the single largest source of false flags in its claim detector
(`RESULT_struct1_beats_the_null_2026_08_31.md`, class 3: splitter fragments). A format whose
legality depended on that splitter would inherit its defects. A byte count is a pure function
of bytes, which is the same discipline the capsule layer runs on. "At most one sentence" is
the authorial rule; 300 bytes is the mechanical enforcement, and the two are not identical.

**Disclosed inequity.** 300 bytes is roughly 300 characters in Latin script and roughly 100 in
scripts encoded at three bytes per character. The cap is therefore materially tighter for
authors not writing in Latin script. This is a real defect of v0.1, named here rather than
discovered later, and a script-aware cap is v0.2 work.

## Inline form (generation)

    <sworn r="RECEIPT" k="KIND">…one sentence…</sworn>

Markdown-safe: the tags are HTML, they survive every markdown renderer this lab uses, and a
renderer that strips unknown tags degrades to the plain sentence rather than to garbage.

Lexical rules, deliberately not a markdown parse:

- Tags are recognised only **outside** fenced code regions (a line whose first non-space
  character run is ```` ``` ```` at indent 0–3) and **outside** inline backtick spans. Inside
  them, `<sworn>` is literal text. A lane whose measured failure class is *"the verb is inside
  a code span"* does not get to make the same mistake in its own syntax.
- If fence delimiters are unbalanced, the document is MALFORMED at document level. The scanner
  refuses to guess rather than scanning on an assumption.
- A `<sworn>` opened inside another open `<sworn>`, or a `</sworn>` with no opener, is
  MALFORMED — both spans in the nesting case.
- Attribute order is fixed (`r` then `k`), values are double-quoted, no other attributes are
  permitted. Anything else is MALFORMED. The tag is a commitment, not markup to be styled.

A byte-exact canonicalizer converts inline form to the sidecar form and back with zero loss.

## Canonicalization

- **Canonical text** is the document with every `<sworn …>` and `</sworn>` tag deleted and
  *nothing else changed*. No whitespace normalization. No newline normalization.
- **Offsets** are UTF-8 byte offsets into the canonical text.
- **Newlines are never normalized.** The capsule layer learned this the expensive way: a
  certificate that hashed text read with universal newlines did not describe the bytes on a
  CRLF checkout (`styxx/capsule.py`, the byte-faithfulness comment on `create_capsule`).
  A canonicalizer that normalised newlines would put every offset in the sidecar at odds with
  the document on half the world's checkouts.
- **Round trip is asserted, not assumed.** Re-inserting the recorded tags at their recorded
  offsets, in descending offset order, must reproduce the original document byte-for-byte. An
  implementation that cannot assert this **refuses to emit a sidecar**, in the same spirit as
  `create_capsule` refusing to mint a capsule around a certificate it cannot reproduce.

## Receipts

`RECEIPT` is one of exactly three forms.

**`rN`** — an id from the **turn manifest the harness minted**. It resolves to
`{id, sha256, kind_of_source, captured_at, bytes}`, plus `complete: true|false`, which v0.1
adds to the shape named in the brief because the `absent` kind cannot be adjudicated without
it. The manifest is a separate object with its own spec string (`sworn/manifest/0.1`) written
by the harness, never by the agent.

**`path:PATH`** — a committed file, optionally suffixed with an RFC 6901 JSON pointer
(`path:v14_gates.json#/gates/2/observed`) or a line anchor (`path:FILE.md#L13`, `#L13-L20`).
Resolved **at the commit the document names**, not in the working tree. A sworn document
therefore carries a `commit` field in its sidecar header; without it, every `path:` receipt is
UNRESOLVED. Resolving against a working tree would make the verdict a function of somebody's
checkout, and the verdict must be a function of bytes. This is the OATH form: cite your
receipts and ship them.

**`prereg:SHA256`** — the span states a bar or a rule from a frozen preregistration. It
resolves by locating a file in the repository whose bytes hash to that digest. This is the one
receipt form that is content-addressed rather than located, because the point of a prereg
citation is that the bar was frozen before the measurement — a path can be rewritten, a digest
cannot.

**Anything else is MALFORMED.** A missing, unparseable, or unknown receipt reference is a
parse error, and the span is **not silently downgraded to narrative**. Silent downgrade is how
a format gets gamed: if a broken tag became ordinary prose, and ordinary prose is never
accused, then the cheapest route to a clean document would be to write tags that break. A
malformed tag renders MALFORMED and counts against the author.

## Kinds — a closed set, v0.1

**`numeric`** — exactly one number in the span; the pointer resolves to exactly one scalar;
equality within the precision the span prints.

- More than one number in the span → MALFORMED. A pointer resolving to an array, an object,
  or nothing → MALFORMED.
- Comparison is decimal, on the printed digits. Let `d` be the count of fractional digits the
  span prints; the span HELDs iff the receipt scalar rounded to `d` fractional digits equals
  the printed value, compared as decimal strings. Not as floats: OATH needed a ULP escape
  clause to survive float comparison (`styxx/certify.py`, `V07_ULP_ESCAPE`), and an escape
  clause is a repair, not a design.
- **No percent/fraction conversion.** If the span prints `42%` and the leaf holds `0.42`, the
  span FAILS. OATH is percent-aware; sworn refuses to be. An automatic conversion is the
  verifier guessing what the author meant, and guessing what the author meant is the entire
  failure class this format exists to leave behind. Write what the receipt says, or point at a
  leaf that holds what you wrote.

*Why one token and one leaf dissolves two measured classes by construction rather than by
repair.* **Mention versus use** was a decision the verifier had to make — which numerals on a
line are assertions — and it made it from vocabulary. The eleventh catalogued instance in this
lane fired on a quoted `"9 new tests."` inside a paper about telling mention from use
(`RESULT_struct1_beats_the_null_2026_08_31.md`). Under sworn there is no such decision: a
quoted number is not inside a `<sworn>` tag, so it is narrative, so no verdict exists to be
wrong. **The vacuous pass** — a number "verifying" against a thousand-row per-item array where
almost any two-decimal value in [0,1] would match, and against receipt leaves that merely
happen to hold the value (OATH rule 9, and the 0.3399 of verifications that were value match
alone) — cannot occur, because there is no search over leaves. The author named the leaf. The
binding is either right or wrong, and no coincidence is available to it.

**`quote`** — the receipt bytes contain the span's quoted text verbatim. Byte comparison over
the UTF-8 encoding, no whitespace collapsing, no unicode normalization. Permitted against any
receipt form.

**`hash`** — the span states a hash; it must equal the receipt's `sha256`. Case-insensitive
hex comparison; any other digest algorithm is MALFORMED in v0.1.

**`absent`** — the receipt bytes do **not** contain a stated string. Permitted **only** when
the receipt is a complete object: an `rN` receipt with `complete: true` in the manifest, or a
`path:`/`prereg:` receipt naming a whole file (a pointer or line anchor makes it partial). A
partial receipt with kind `absent` → MALFORMED.

*What `absent` buys, plainly.* It is how an agent swears a negative: *the deposited record
carries no erratum*, *the released notes name no CVE*, *the manifest lists no second author*.
That class of statement stayed unverifiable until somebody fetched the bytes, and a negative
asserted over a partial fetch is not a measurement at all — absence of evidence is never a
contradiction. The completeness requirement is that law written into the grammar: you may
swear a negative only over an object the harness attests it fetched whole.

**`exec`** — **excluded from v0.1**, named as v0.2 work. Re-execution belongs to the capsule
layer, not to the sentence. A sentence cannot carry a runtime: verifying an exec claim needs
an environment, a clock, and a sandbox, which makes the verdict non-deterministic and destroys
the property that makes the capsule's layer 1 worth anything — that any reader re-checks it
offline, in a browser, in a second (`SPEC_oath_capsule_v01_2026_08_31.md`). Execution claims
belong in a capsule that carries the trace, and the sworn span then swears against *that*
trace's bytes with `quote` or `numeric`.

## Sidecar form (storage)

A JSON object:

```json
{
  "spec": "sworn/0.1",
  "commit": "<sha, or null>",
  "document": { "name": "...", "sha256": "<sha256 of the canonical text bytes>" },
  "text": "<the canonical text, tags stripped>",
  "spans": [ { "start": 0, "end": 0, "receipt": "r1", "kind": "numeric" } ],
  "manifest": { "spec": "sworn/manifest/0.1", "receipts": { } }
}
```

`spans` is ordered by `start`, ascending, and the ordering is part of the canonical form.
`text` and the offsets together are sufficient to regenerate the inline document exactly; the
`sha256` is over the canonical text bytes, so a sidecar and a capsule agree on what was said.

## Verdicts

**Per span**

| verdict | meaning |
|---|---|
| `HELD` | the receipt resolved and the check passed |
| `FAILED` | the receipt resolved and the check did not pass |
| `UNRESOLVED` | the receipt did not resolve — **never an accusation** |
| `MALFORMED` | the declaration is ill-formed; decidable from the document bytes alone |
| `WITHHELD` | reserved for the confab gate; the span's text is unchanged |

`UNRESOLVED` is the EXTERNAL-1 lesson written into the grammar. When the instrument cannot see
the evidence, it reports that it cannot see the evidence. It does not conclude wrongdoing. A
receipt file that is missing, a manifest that is gone, a commit not present in this checkout —
all UNRESOLVED, all silent about the author.

`MALFORMED` is the one verdict that *does* count against the author, and the asymmetry is
deliberate: MALFORMED is decidable from the document's own bytes with no receipt access at
all, so it is a statement about the declaration and not about the world.

`WITHHELD` is reserved and has **no producer in v0.1**. It is the slot for the confab gate —
`styxx/single_pass.py`'s detect-and-abstain primitive — where an author's own gate declines to
swear a span it was about to swear. The span's text is unchanged when this happens: withholding
an oath is not redacting a sentence. The slot exists now so its arrival is not a schema break.

**Per document**

- `counts`: one integer per verdict.
- `sworn_total`: the sum.
- `coverage_estimate`: `sworn / (sworn + narrative claim-bearing sentences)`, the denominator's
  second term counted by `styxx/claimdetect.py`. **LABELED ADVISORY**, always.
- `unsworn_claims_estimate`: the count of narrative sentences `claimdetect` reads as claims.
  Also advisory.

**Document verdict**

- `SWORN-HELD` iff `FAILED == 0` and `MALFORMED == 0` and `sworn_total > 0`.
- `SWORN-FAILED` otherwise, when `sworn_total > 0`.
- `UNSWORN` when `sworn_total == 0` — **never "no failures"**. A document that swore nothing
  has not passed anything. An absence is not a measurement.

`SWORN-HELD` always prints its coverage estimate beside it, at the same prominence. A
`SWORN-HELD` carrying `UNRESOLVED > 0` prints that count beside it too: the reader is entitled
to know how much of the oath went unchecked because the verifier could not see.

## The four invariants

**1. No post-hoc tagging, ever.** A pass that reads plain text and proposes sworn spans is
claim detection with extra steps, and it would import every measured failure in the receipts at
the top of this document — 0.23 in the wild, 0.16 held out after repair, 0.4211 at the best
structural attempt. Tags are emitted by the author at write time or they do not exist. A tool
that adds them later is out of contract, and a document whose tags were added later is not a
sworn document, whatever it renders as.

**2. Receipts are harness-minted, never author-minted.** A receipt the agent wrote in the same
turn is not a receipt; it is the agent swearing to itself, which is the condition the whole
program exists to escape. The manifest is written by the harness, carries the harness's own
spec string, and an `rN` reference that resolves into anything the agent authored in that turn
is MALFORMED. This is the same reason `parrhesia` says the honest deployment has a party other
than the author issue the receipt.

**3. An absence of sworn spans is not "clean".** `UNSWORN` is a distinct verdict for exactly
this reason. A document with no sworn spans has made no commitments, so it cannot have kept
any, and the only honest report is that it swore nothing. Rendering it as a pass would let
silence buy the same badge as evidence.

**4. Coverage travels with every verdict.** The obvious way to game this format is to swear
only the trivially true — the date, the version string, a hash you just printed — and let every
load-bearing sentence sit in narrative where nothing may accuse it. The countermeasure is not a
threshold. It is publication: unsworn claim-bearing sentences are counted and printed next to
the verdict, so a reader sees a `SWORN-HELD` at coverage 0.04 for what it is.

The countermeasure has a ceiling and it is named. `claimdetect` (STRUCT-1) measures at
precision **0.4211** on n=38 with two known recall misses left unpatched
(`RESULT_struct1_beats_the_null_2026_08_31.md`, receipt `stage2_result.json`). Its false flags
inflate the denominator and bias coverage **low**; its recall misses shrink the denominator and
bias coverage **high**. The net direction on prose of this kind has never been measured. That
is why coverage is advisory, why it is never a gate, and why no rate computed from it may be
quoted as a measurement of anything.

## Threat model — what a sworn document proves, exactly

**Proves.** For each span the author bound: that a specific check, of a named kind, passed
against specific bytes the author did not write, at a commit or a manifest entry named in the
document. A reader re-derives all of it from the sidecar and the receipts, without trusting the
author and without contacting anyone.

**Does not prove.** That the receipts truthfully record reality. The chain from a receipt back
to the run that produced it lives in harness provenance and repository history, exactly as the
capsule spec says of itself, and sworn inherits that boundary unchanged.

**Does not prove.** That the author bound the sentences that matter. This is the important one
and it has its own section.

**Does not defend against.** A compromised harness. If the harness that mints the manifest is
lying, every `rN` receipt is worthless, and no amount of author-side discipline recovers it.
Sworn moves trust from the author to the harness; it does not eliminate trust. The worklog spec
makes the same admission about the same component (`SPEC_worklog_v01_2026_08_31.md`: *"only as
trustworthy as the harness that wrote it"*), and the two are honest about it for the same
reason.

## What it is not

A sworn document can be perfectly `SWORN-HELD` and completely wrong about everything it chose
not to swear.

The format proves the author kept its word on the sentences it bound, against bytes it did not
write. It does not prove the author bound the sentences that matter. An author who swears the
version string, the file count and the date, and leaves the finding in narrative, gets a clean
verdict from this instrument. That is not a bug to be repaired by a smarter verifier; a smarter
verifier that decided which unsworn sentences *should* have been sworn would be a claim
detector, and the receipts at the top of this document are what claim detectors measure at.

Coverage is the reader's number for that gap, and it comes from an instrument with a documented
ceiling of 0.4211 precision on 38 sentences. So the honest summary is: sworn output converts an
unbounded question — *is this document truthful?* — into two bounded ones, *did it keep the
oaths it swore?* and *how much of it did it swear?* The first is answered mechanically. The
second is answered by an instrument that is not very good, and says so.

We know of no other format that makes the second question answerable at all. That is a claim
about our reading of the field, not a measurement, and the survey that would price it is owed.

## Relationship to `parrhesia`

`styxx/parrhesia.py` exists and ships. It does a different job, and the difference is the
cleanest way to say what sworn is:

- **`parrhesia` proves you need not trust the auditor.** An external instrument scores a whole
  message and records the verdict in a content-addressed receipt that a third party re-derives
  by re-running the auditor. Auditor-side, whole-message, register (tone).
- **`sworn` proves you need not trust the author.** The author binds individual spans to bytes
  it could not have written. Author-side, per-span, binding.

They are orthogonal and they compose. **Sworn emits a parrhesia-discipline receipt rather than
replacing one:** the verification result is issued as a content-addressed, re-derivable receipt
under its own schema string `styxx.sworn.verdict-receipt/v0`, carrying the canonical text's
`sha256`, the per-span verdicts, the counts, the advisory coverage, and a `certifies` string
that states its own boundary in the manner `parrhesia._CERTIFIES` already does — *the spans the
author bound were checked against bytes the author did not write; NOT a claim that the document
is correct, and NOT a claim that the right sentences were bound.*

`styxx/parrhesia.py` is not modified. Its verdict is a register verdict and must never be
printed as though it were a sworn verdict, or the reverse; a run that issues both prints both,
separately labelled. The naming follows from this: the adjacent name is occupied by a shipped
module doing a different job, so the module is `styxx/sworn.py`.

## Out of scope for v0.1, named

Signatures and harness attestation (a sworn document is re-derivable, not sender-authenticated).
The `exec` kind and re-execution of any sort. Cross-document receipts. Non-JSON, non-text
receipt bodies. Any blocking behaviour anywhere — sworn reports, it does not gate. Automatic
percent/unit conversion, permanently rather than temporarily. Multi-sentence spans. A
script-aware length cap. Any tooling that proposes tags for existing prose, which invariant 1
forbids outright rather than deferring.

## What is owed, recorded as owed

1. **The arc-question declaration.** `papers/INDEX.md` does not exist in this tree — `papers/`
   carries only `INDEX_behavioral_knowledge_boundary_2026_05_25.md`, and the program map lives
   as `INDEX_program_map` on unmerged PR #51. The declaration this spec owes the index is
   therefore recorded as owed rather than written against a file that is not here.
2. **The eleven-instrument census.** Stated in the commissioning brief, unverified above, with
   no receipt in this tree that enumerates it. Either the census gets built and the number gets
   a receipt, or the number stops being said.
3. **A measurement of sworn output.** There is none. This spec is frozen before any code and
   before any panel. Nothing here is evidence that authors will bind the sentences that matter,
   and the design argument above is a design argument.
4. **A price for the gaming countermeasure.** Invariant 4 rests on coverage; coverage rests on
   an instrument at 0.4211 precision, n=38. The study that prices trivial-swearing against
   published coverage has not been designed, let alone run.
5. **The script-aware cap**, and the survey behind the "we know of no other" in this document.

---

*The instruments that read prose to find claims were measured against strangers and did not
survive it. This format does not read prose. It reads what an author was willing to put its
name to, checks that against bytes the author could not write, and prints how little of the
document that was.*
