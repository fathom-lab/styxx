# SPEC — a numeric span's sign is never silently dropped (v0.1)

**Frozen 2026-09-06, before the code.** One rule, N1, and the guards that must fail before it is
written. Found by the eight-dimension adversarial audit (workflow `wf_9466dcba-f49`, dimension
`numeric`), reproduced by a skeptic and then re-derived independently before this spec was written.

## The defect

`_TOKEN = [\w.,+\-−%/±:]+` (styxx/sworn.py:1277) admits exactly two dash code points, `U+002D
HYPHEN-MINUS` and `U+2212 MINUS SIGN`. Every other dash is outside the class, so it does not join
the digits that follow it — it **splits the token and disappears**. `_number_token` then finds
exactly one digit-bearing token, `_GRAM` matches it, and the span is adjudicated on a number whose
sign is gone.

Measured, on this repository at `f6179c6e`:

```
sentence asserts -0.42, receipt r1 = 0.42

U+002D HYPHEN-MINUS            -> FAILED   printed_token='-0.42'
U+2212 MINUS SIGN              -> FAILED   printed_token='-0.42'
U+2010 HYPHEN                  -> HELD     printed_token='0.42'
U+2011 NON-BREAKING HYPHEN     -> HELD     printed_token='0.42'
U+2012 FIGURE DASH             -> HELD     printed_token='0.42'
U+2013 EN DASH                 -> HELD     printed_token='0.42'
U+FF0D FULLWIDTH HYPHEN-MINUS  -> HELD     printed_token='0.42'
U+00AD SOFT HYPHEN             -> HELD     printed_token='0.42'

document_verdict SWORN-HELD, receipt VERIFIED, verdict_reproduces True
```

A reader sees a dash on the baseline immediately before the digits. The verifier reads a positive
number. This is a **false sentence adjudicated HELD** — the outcome the format exists to prevent,
and it is reachable through the supported CLI with no manifest trickery, no tree handle and no
second implementation involved.

It is not a documented decision. `DECISIONS` has no entry for sign, dash, hyphen or normalisation;
no docstring mentions them; no test asserts the behaviour; and the twelve-row attack battery has no
row for it. `styxx/_data/sworn_verify.js` (`TOKEN_RE`, l.453) hand-mirrors the same two dashes, so
both implementations are wrong in the same way and the differential harness cannot see it.

## N1 — a dash-like code point binds to the number it precedes

Both implementations extend their token class to every **Unicode `Pd` (Dash_Punctuation)** code
point plus **`U+00AD SOFT HYPHEN`** — 26 additions beyond the two already accepted (`U+002D` is
itself `Pd`; `U+2212` is `Sm` and is already in the class).

    U+00AD U+058A U+05BE U+1400 U+1806 U+2010 U+2011 U+2012 U+2013 U+2014 U+2015 U+2E17 U+2E1A
    U+2E3A U+2E3B U+2E40 U+2E5D U+301C U+3030 U+30A0 U+FE31 U+FE32 U+FE58 U+FE63 U+FF0D U+10EAD

`_GRAM` is **not** extended. A dash therefore joins the digits into one token, that token fails the
number grammar, and the span is `MALFORMED` with the existing reason `number_grammar`.

### Why refuse rather than read it as a minus

Reading `U+2010 HYPHEN` as a minus sign would make the span `FAILED`, which is more informative when
the author meant a minus. The verifier cannot know that. `U+2010` is a hyphen; `U+2013` is a range
dash; `U+00AD` is invisible. Guessing which of 26 code points was meant as arithmetic negation is
exactly the judgement this format refuses to make elsewhere, and a wrong guess would manufacture a
`FAILED` accusation out of a typographic artifact. **The verifier declines and says so**, which is
the same posture as `number_count` and `number_grammar` today.

The document does not pass either way: `MALFORMED` counts in `sworn_total` and makes the document
`SWORN-FAILED`. The author is told to write an unambiguous sign.

### The reason code does not change

No new member is added to `REASONS`. `number_grammar` already means "the one digit-bearing token is
not a number this format reads", which is exactly true of `-0.42` written with a hyphen. Adding a
reason would move nothing that matters and would widen the change for no gain.

## What moves, and what does not

- **No committed verdict moves.** Of the 517 sworn spans in the 46 committed `.md` files carrying a
  tag, 410 are `kind="numeric"`, and **0** contain a dash-like code point before their digits.
  Measured at `f6179c6e` and re-measured after the change.
- Spans that were `number_count` MALFORMED because a dash split one token into two (`3-4` with an
  en dash) become `number_grammar` MALFORMED. Both are MALFORMED; no verdict flips.
- The conformance set is expected to be **unchanged**. If any vector moves, that is a finding about
  the change and stops it — it is not a reason to re-record the set.

## Guards, watched to fail before the code is written

| # | guard | must fail before, pass after |
|---|---|---|
| G1 | every one of the 26 code points, placed before a numeric span's digits against a receipt of the opposite sign, is not HELD | fails now: all 26 are HELD |
| G2 | `U+002D` and `U+2212` still adjudicate normally — a matching number HELD, a mismatched one FAILED | passes now and after; catches over-reach |
| G3 | an ordinary numeric span with no dash is unaffected | passes now and after |
| G4 | Python and the JS verifier agree, span verdict and document verdict, on all 26 | fails now only if one side is fixed first; is the parity gate |
| G5 | the conformance set's `set_sha256` is unchanged by the edit | — |

G1 is the guard that must be seen red. G4 is why the two implementations are edited in the same
commit: fixing one alone creates a parity defect of exactly the kind this audit found elsewhere.

## What this does not claim

That the numeric channel is now correct. This closes one route by which a sign disappears. The audit
that found it also found a short-needle bypass, a manifest rung that is outside its own digest, an
invariant-2 bypass through an empty `receipts` map, `refs/replace` subverting `GitTree`, and a
Python/JS document-verdict divergence on `U+0085`. Each is its own spec and its own repair. None is
addressed here.

---

## ERRATA 2026-09-06 — G5 was the wrong bar, and it failed

**G5 as frozen above says "the conformance set's `set_sha256` is unchanged by the edit". That bar
is impossible for any edit to `styxx/sworn.py`, and it failed on the first run.** The spec is not
edited; the correction is recorded here.

**What happened.** `--check` refused: four families (`cli`, `gaming`, `receipt_v1`, `rules`) and
`blobs.json` drifted, with every count unchanged. The control was run first — on clean `main` the
set regenerates to its own digest, so the drift was caused by this change and not pre-existing.

**Why it drifted.** Every verdict receipt carries `verifier.sworn_sha256`, which is the sha256 of
`styxx/sworn.py` itself. Vector generation mints receipts, so **any** byte-level edit to the
verifier — a comment would do it — changes 14 receipt blobs and the `check`-family vectors that
reference them. Confirmed field by field: the only differing key in the `verifier` block is
`sworn_sha256`, old `223b1d0c…` and new `b4111a5c…`, which are exactly the file's hash at `HEAD`
and after the edit.

**Why it is not evidence of a behaviour change.** `_TOKEN` is reachable only from `_number_token`,
which is called only under `kind == "numeric"`, and of the 2441 conformance document blobs **zero**
carry a bound dash in a numeric span. Measured, not argued.

**The bar that replaces G5.** Regenerate and require:

- the same **3618** vectors, and
- the **multiset of expected outcomes identical** between the committed and regenerated sets.

Measured after the edit: 3618 = 3618, multisets identical. Only the build pin moved.

**And the set is re-recorded**, in its own commit, naming the cause. That is this repository's
established practice for a verifier change, not an improvisation — `git log` on the set carries
`regenerate the conformance set: a CLI improvement has a byte-pinned blast radius`, `regenerate the
conformance set: the verify headline changed, so the cli vectors did`, and `regenerate the
conformance set after the warning repair`. The rule this does **not** break is the one about a moved
*core*: a vector whose expected verdict changes is a finding and stops the work. None did.

**The author's error, recorded.** G5 was written by predicting what the set would do instead of
checking what the repository already does with it. That is the same mistake — asserting rather than
measuring — that this leg's own finding was about.
