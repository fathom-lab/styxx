# RESULT — a second implementation verifies the real log, 25 checks of 25

Fathom Lab · 2026-09-09 · **A result about agreement between two implementations, not about the
model or the log's contents.** The log verified here is the one whose central number is worthless
(`../vacuous_floor_2026_09_09/`); that is deliberate. Structural agreement and substantive value are
different questions, and this document answers only the first. Script: `cross_verify.js`, run as
`node cross_verify.js <log dir>`; machine output in `cross_verify_report.json`. Not sworn.

## What was compared, and why it is not another green check

`styxx/_data/v8_verify.js` was written from the specification and the RFCs by an author who was
forbidden to open any file under `styxx/v8/` or `tests/`. Until now it had only ever run against
**synthetic vectors**: the RFC 8785 appendix example, the Certificate Transparency roots for tree
sizes one to eight, the RFC 8032 signature vectors. Agreement there shows two authors read the same
standards the same way. It does not show they produce the same bytes for a real artifact, because
published vectors are small, hand-chosen, and contain none of the shapes that break canonicalizers
in practice.

This run pointed that implementation at a log the Python produced, and asked the questions Appendix D
asks a stranger to ask, using none of the Python.

| what was checked | result |
|---|---|
| every entry's cert id recomputes from its own bytes | 7 of 7 |
| every cert's signature verifies under its issuer key, over the domain-separated digest | 7 of 7 |
| every entry's stored bytes **are** their own canonical form | 7 of 7 |
| the entries reproduce the root the tree head signs | yes, tree size 7 |
| the tree head's signature verifies under the out-of-band pinned log key | yes |
| an inclusion proof generated and verified entirely in JavaScript | yes |
| the same proof is **refused** for the wrong leaf (negative control) | refused |

**25 of 25.**

## The size of the aperture

The reason this is a stronger statement than the vector agreement is what the real certs contain:

| | |
|---|---|
| entry bytes reproduced byte-for-byte | 489,431 |
| floating-point values re-rendered independently | 9,843 (1,908 distinct) |
| of those, values requiring exponent notation | 350 |
| strings canonicalized | 2,253 |
| maximum nesting depth | 7 |
| largest single cert | 94,174 bytes |

Number rendering is where two canonicalizers actually diverge. RFC 8785 delegates it to the
ECMAScript number-to-string algorithm, and reimplementing that faithfully is the hard part of the
standard — the transitions into and out of exponent form, the shortest-representation rule, negative
zero. This log carries values like `-1.1920928244535389e-07` and `0.09090909090909091` alongside
`-17.6875`. A single digit rendered differently in any one of 9,843 values changes the canonical
bytes, which changes the digest, which changes the cert id, and the check fails. None did.

The same is true of the constructions this project invented rather than inherited: the
`styxx.v8/cert/1` and `styxx.v8/sth/1` domain tags, the byte layout of the tagged preimage, and the
rule that the entry bytes stored on disk must be exactly the canonical form. Two independent readings
of those agreed too.

## What this does and does not establish

It establishes that the format is implementable twice, and that the second implementation was not
guided by the first. That is the property the whole design rests on — a content address that two
honest readers compute differently is not an address — and it had never been tested on a real
artifact until now.

It does **not** establish that either implementation is correct, only that they agree. Two authors
reading the same specification can share a misreading, and this check cannot see one. It says nothing
about the log's contents: the fingerprints in it are structurally perfect and the floor among them is
empty, which is exactly the point of keeping the two questions apart. And it is still in-house. The
author of the JavaScript could not read the Python, but both were commissioned by the same operator
in the same session, which is weaker than a stranger and should not be described as one.

The check that remains unperformed is the one that matters most: a party with no relationship to this
lab, on hardware we have never touched, reproducing a cert and filing a disagreement. Nothing here
substitutes for it.

## Limits

One log, seven entries, one tree head, one session. The negative control covers a proof presented for
the wrong leaf; it does not cover a forged signature, a tampered entry, or a rolled-back tree, all of
which the Python mirror was separately shown to catch and none of which the JavaScript was asked
about here. `cross_verify.js` reads the log directory layout directly and would need updating if that
layout changes, which makes it a check on today's format rather than a permanent one.
