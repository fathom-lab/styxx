# Pre-registration — Portable Content Address (verify in any language)

**Stated 2026-05-28, BEFORE the portable digest was written.**
Author: styxx coding agent. Methodology: recursive-discipline (pre-register →
kill-gate → build → run → report whichever way it lands).

## Context

The standalone verifier (`scripts/styxx_verify_standalone.py`, commit `b9d43ce`)
removed the "trust styxx" dependency from verification — but not the "be Python"
dependency. The content address is `sha256` over
`json.dumps(core, sort_keys=True, separators=(",",":"), ensure_ascii=False)`,
and Python's `json` number repr is not language-portable. I MEASURED this on the
same artifact in two real runtimes:

```
python sha256: 9a734e78078ce6d0772fdacfd3380d61f5cbefadb9854f4d523376b90c9833d7
node   sha256: 6885993616518a4f8726752b44a79b6b1e4c798005fb1d22dfdd95baed77142b
```

Cause: a saturating vitals score (or `coverage`) serializes as `1.0` in Python
and `1` in JavaScript. So today a styxx receipt can only be re-verified in
Python. That is the boundary the previous prereg documented as future work.

## The thesis (what is new)

Every number in a styxx attestation is a finite double in `[0, 1]` (vitals
scores, rounded to 12 decimals; `coverage`, rounded to 4) plus integer counts.
A canonical-number serialization following **RFC 8785 / ECMAScript
`Number::toString`** is therefore tractable and lets the content address
reproduce in ANY language.

Add an **additive, versioned `digest.portable`** computed over a JCS-style
canonical payload (RFC 8785 numbers; keys sorted; strings JSON-escaped). The
existing `digest.value` is left BYTE-FOR-BYTE UNCHANGED — every 7.7.11 / 7.7.12
receipt already issued stays valid; nothing is invalidated. A second-language
implementation (`web/styxx_verify.js`, runs in Node and the browser, zero deps)
reproduces `digest.portable` and the (already-hex-only, already-portable) Merkle
chain. Result: **verify a styxx attestation in a browser, in any language, with
zero install and zero trust.**

**Design constraint discovered before building (recorded honestly):** RFC 8785
sorts object keys by UTF-16 code units and escapes strings by an exact rule.
styxx artifact keys are all ASCII and values carry no control characters, so
Python code-point sort == JS UTF-16 sort and `json.dumps(s, ensure_ascii=False)`
== `JSON.stringify(s)`. The portable scheme is specified for the styxx artifact
domain (ASCII keys, finite doubles, no NaN/Inf); a fully general JCS
implementation is a superset and out of scope.

## Pre-registered predictions

- **P1 — Python↔JS byte-for-byte agreement (headline, decisive).** Over (a) a
  fuzz corpus of finite doubles (random in [0,1], saturating 0.0/1.0, tiny
  1e-12, plus integers) AND (b) the 4 real artifact shapes (plain / pinned /
  vitals / chain), the Python ES-number serializer and the JS `String(n)` agree
  on every number, and `digest.portable` computed in Python equals the JS
  recomputation byte-for-byte — INCLUDING the saturating-score artifact that
  diverged under the legacy scheme.

- **P2 — legacy digest untouched (no break).** `digest.value` (the legacy
  Python-canonical address) is identical before and after adding
  `digest.portable`. Existing receipts and the existing standalone verifier keep
  passing unchanged. The portable digest is purely additive.

- **P3 — tamper-evidence on the portable address.** `verify_attestation`
  recomputes `digest.portable` and a flipped substrate (or a flipped portable
  digest) is caught on the portable axis exactly as on the legacy axis. The JS
  verifier alone (no Python, no styxx) catches a tampered portable digest and a
  broken Merkle chain.

- **P4 — portable chain.** Each link carries `attestation_portable_digest`; a
  portable head `head_chain_portable_digest` rolls them with the SAME hex-only
  Merkle rule (already language-agnostic). The JS verifier reproduces the
  portable head from genesis.

## Kill-gate (stated before the build)

KILLED (reported negative, not reframed) if ANY of:

- **K1** (decisive) — Python and JS disagree on the portable canonical string or
  `digest.portable` for ANY value in the fuzz corpus or ANY of the 4 real
  artifact shapes. Then the scheme is not actually language-portable and the
  thesis fails.
- **K2** — adding `digest.portable` changes `digest.value` (legacy address), OR
  any existing attestation test / the standalone verifier regresses. A break of
  already-issued receipts kills the "additive" claim.
- **K3** — the JS verifier reports portable-OK on a tampered portable digest or a
  broken chain, OR asserts a semantic property (claim/vitals) it cannot check.

K1 is decisive: the entire value ("verify in any language") rests on an
independent JS reimplementation reproducing the address byte-for-byte, on the
very saturating-score case that breaks the legacy scheme.

I commit to reporting whichever way it lands.

## Results (recorded 2026-05-28, AFTER the build — reported as-landed)

**Thesis SURVIVED.** Kill-gate held on all three axes. 11 gate tests
(`tests/test_portable_attestation.py`, all pass; full scoped suite green, zero
regressions). Live receipt: `scripts/dogfood/portable_attestation_self_2026_05_28.json`
(RESULT SURVIVED; Node v24.13.0).

- **P1 — Python↔JS byte-for-byte agreement: HELD (K1, decisive).** The Python ES
  number serializer and JS `String(n)` agreed on **40,019 / 40,019** fuzz values
  (random [0,1], saturating 0.0/1.0, 1e-12, 1e21/1e22, integers). Over the 4 real
  artifact shapes (plain / pinned / vitals / chain), Python `digest.portable`
  equals the Node recomputation byte-for-byte — including the **saturating
  `coverage = 1.0`** artifact that diverged under the legacy scheme (legacy:
  python `9a734e78…` vs node `68859936…`; portable: identical both sides).

- **P2 — legacy digest untouched: HELD (K2).** `digest.value` is byte-identical
  with `digest.portable` added (the legacy canonicalization excludes the whole
  `digest` key). All 39 prior attestation + standalone-verifier tests pass
  unchanged; nothing already issued is invalidated.

- **P3 — tamper-evidence on the portable address: HELD.** `verify_attestation`
  recomputes `digest.portable`; a flipped portable digest yields
  `portable_ok: false` and `ok: false`. The JS verifier alone (no Python, no
  styxx) reports FAIL on a tampered portable digest and on a broken chain.

- **P4 — portable chain: HELD.** Each link carries `attestation_portable_digest`;
  `head_chain_portable_digest` rolls them with the hex-only Merkle rule and the
  Node verifier reproduces the portable head from genesis.

- **K3 honest scope: HELD.** The JS verifier reports claim verdicts and vitals
  scores as `NOT CHECKED`, never asserting them; a re-sealed chain passes
  structure and is caught only with an external expected head.

**What this is, without inflation:** an additive, cross-language content address
for styxx attestations, with an independent zero-dependency JavaScript verifier
(and a paste-and-check browser page, `web/verify.html`) that reproduces it
byte-for-byte. You can now verify a styxx attestation in any language, in a
browser, with zero install and zero trust. It is NOT a full verifier — semantic
claims and vitals remain `NOT CHECKED`, and a re-sealed chain still needs an
external anchor. The portable scheme is specified for the styxx artifact domain
(ASCII keys, finite doubles); a fully general JCS implementation is a superset.
