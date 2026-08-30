# RESULT — the unobligated oath: 0.5811 of our verifications were volunteered

Fathom Lab · 2026-08-28 · Invariant: `INVARIANT_epistemics_annotation_2026_08_28.md`, committed
before the implementation. Receipt: `oath_unobligated_oath_census.json`. Follows
`RECON_v13_not_frozen_the_ladder_2026_08_28.md`.

## What was built

The ladder RECON established that **obligation gates accusation, not verification**: a value match
produces `VERIFIED` whether or not anything obligated the verifier to look. But the certificate
discarded the evidence — which ladder branch produced each status was thrown away at the moment it
was known.

So the verifier now records it. Every ledger entry carries machine-readable epistemics: the
**branch** that produced the status, whether the token was **obligated** when the ladder ran, the
**obligation source** (`vocabulary`, `n-glued`, `range-correlation`, `precision`, `range-sanity`),
and for value matches whether the **path was checked** (integers only — the float path binding is
`CLOSED_NEGATIVE` since v0.8).

**The invariant was frozen first and verified after**: an A/B of the annotated verifier against the
pre-change verifier, per token, over all `192` committed certificates — `192` compared, `192`
identical, `0` moved, `0` errors. The annotation is observation only. Stored historical
certificates are untouched.

## The number

Live re-certification of all `192` documents at the pinned verifier, `5951` verifications:

| verified split | count |
|---|---|
| **unobligated, path unchecked** — value match alone, nothing required the look | **2023** |
| obligated, path unchecked | 1909 |
| unobligated, path filter ran (integers) or derived | 1435 |
| obligated, path filter ran | 584 |

> **Unobligated oath rate: `0.5811`** — `3458` of `5951` verifications in this laboratory's own
> corpus were sworn without anything obligating the verifier to examine the token.

`2023` of them — `0.3399` of *all* verifications — are the weakest attestation the instrument
produces: unobligated **and** path-unchecked. Six documents' verifications are `1.0` unobligated,
including `RESULT_oath_v08_float_binding_CLOSED_NEGATIVE` — the paper that closed the float
path-binding repair is itself certified entirely by volunteered oaths.

## What this means, and what it does not

It does **not** mean those verifications are wrong. Most will be true claims whose lines happen to
carry no trigger vocabulary; the number binds and the receipt is real.

What it means is narrower and worse for the certificate's semantics. `OATH-HELD`'s verified count
is a **mixture** of oaths the obligation predicate required and oaths the value-match volunteered,
and until today the mixture was invisible. The companion number from 2026-08-27's blind panels
says how volunteered oaths behave on foreign text: roughly **one in five** land on things that are
not claims at all — command-line flags, link labels, hardware specs.

Two sentences that must now be kept apart:

* *"The verifier checked what it was obligated to check"* — true, and the obligated 42% is that.
* *"The verifier's verified count reflects its obligation policy"* — **false**, and it was
  published as if true. The larger half of the count is policy-free.

## The instrument catching its own construction, again

Three defects surfaced *while building the measurement*, each caught by a guard this lane built
earlier:

1. My first invariant check compared the annotated verifier against **stored certificates from ten
   verifier versions** and reported 117 violations — verifier-version drift wearing my change's
   clothes. The check as frozen was wrong; the A/B against the pre-change verifier at the same
   commit is the check that means what the invariant says. Amended openly.
2. The rewritten checker then reported **"INVARIANT HOLDS" over zero certificates** — every
   comparison had errored out and the green was vacuous. The denominator guard added after the
   silent-pass work is what refused it.
3. The loader bug behind those errors: `import styxx.certify as after` binds the **function**
   `certify` exported by `styxx/__init__`, not the submodule — a name shadow that produced
   `'function' object has no attribute 'certify_doc'`. Every prior script here dodged it by
   accident with `from styxx.certify import ...`.

And the A/B itself caught a fourth: two early `ledger.append` sites (the v0.11 row-ordinal and
v0.12 formula-constant silencers) bypass the ladder and had no epistemics until the per-entry
assertion failed on a real document.

## What is owed

1. **Certificate semantics must expose the split.** A future certificate should carry the
   obligated/unobligated verified counts at the top level, not only per token. That is a schema
   change and gets its own cycle.
2. **The contract must say it.** `OATH_CONTRACT.md`'s account of what `OATH-HELD` attests is now
   measurably incomplete in a second direction, and the amendment the ladder RECON already
   promised should fold this number in.
3. Whether obligation *should* gate verification is a design question this measurement informs and
   does not answer. Making it gate would demote thousands of true bindings to silence; leaving it
   is the status quo this RESULT makes visible. Either choice needs a preregistration that reads
   the ladder first.

---

*The certificate now says, for every token, which door it came through. More than half came
through a door nobody was watching.*
