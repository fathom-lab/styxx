# Pre-registration — Cognometric Transparency Log (append-only, no silent suppression)

**Stated 2026-05-28, BEFORE the transparency-log code was written.**
Author: styxx coding agent. Methodology: recursive-discipline (pre-register →
kill-gate → build → run → report whichever way it lands).

## Context

The attestation arc so far is *trust subtraction*: re-derive verdicts (don't
trust the agent's word) → Merkle chain (don't trust ordering) → standalone
verifier (don't trust styxx) → portable digest (don't trust Python). Every step
is about verifying **one receipt you were handed**.

A trust gap remains that none of it closes: an agent can attest only its
*flattering* runs and silently drop the bad ones. A receipt proves what it says;
it cannot prove that **nothing was suppressed**. There is no append-only,
third-party-checkable record of an agent's cognometric history where a deletion,
edit, or reorder of a *past* entry is detectable.

## The thesis (what is new)

A **Cognometric Transparency Log** — Certificate-Transparency (RFC 6962) applied
to styxx attestations. Leaves are attestation portable digests. The log is a
Merkle tree; an append-only history yields two checkable proofs:

- **inclusion proof** — "attestation X is recorded at index i in the log whose
  root is R" (audit path; O(log n)).
- **consistency proof** — "the log with root R_n (size n) is an append-only
  extension of the earlier log with root R_m (size m < n)" — i.e. no past leaf
  was edited, deleted, or reordered (RFC 6962 §2.1.2).

With consistency proofs, anyone who has *witnessed* an earlier signed tree head
(STH = {size, root, timestamp}) can detect after the fact if the operator
rewrote history. Both proofs verify with hex-only SHA-256, so they extend the
portable verifier (`web/styxx_verify.js`) — **checkable in any language, in a
browser, with zero install and zero trust.**

**Design choices recorded before building (honest):**
- Hashing uses RFC 6962-style domain separation but with ASCII string tags
  (`styxx-tlog-leaf:` / `styxx-tlog-node:`) instead of the 0x00/0x01 byte tags,
  so the same pure-JS string SHA-256 used by the portable verifier works
  unchanged across languages. This is a deliberate, documented deviation from
  RFC 6962's byte tags — functionally equivalent domain separation, not the
  literal CT tree (a styxx log is not submittable to a CT log and vice versa).
- Leaf entry = the attestation's `digest.portable.value` (a hex string), so the
  log inherits the cross-language content address.

## Pre-registered predictions

- **P1 — inclusion is sound and complete (headline).** For a log of n
  attestations, EVERY leaf i produces an inclusion proof that verifies against
  the root; and a leaf that is NOT in the log (or a tampered leaf hash) FAILS to
  verify. The Python prover/verifier and the JS verifier agree byte-for-byte on
  the root hash and on every inclusion-proof verification.

- **P2 — consistency catches rewrite (decisive).** Given an STH at size m, ANY
  non-append mutation of the first m leaves (edit / delete / reorder) makes the
  consistency proof from m→n FAIL, while a pure append (m leaves unchanged, new
  leaves added) PASSES. This is the property the whole thesis rests on: silent
  suppression of a *past* entry is detectable to anyone holding the earlier STH.

- **P3 — cross-language portable.** The JS verifier (no styxx, no Python, no
  deps, bundled SHA-256) reproduces the root and verifies inclusion +
  consistency proofs byte-for-byte against the Python implementation over the
  real log.

- **P4 — honest boundary (stated scope, NOT a kill).** The log proves
  append-only-ness *relative to a witnessed STH*. It does NOT, by itself, stop an
  operator who never publishes an STH from equivocating (showing different logs
  to different parties) — that needs STH gossip/witnessing, exactly as in CT.
  The artifact and docs state this; claiming the data structure alone defeats
  equivocation would be the overclaim, and we will not make it.

## Kill-gate (stated before the build)

KILLED (reported negative, not reframed) if ANY of:

- **K1** — inclusion unsound/incomplete: a real leaf's proof fails to verify, OR
  a non-member leaf's proof verifies as included.
- **K2** (decisive) — consistency does not catch a rewrite: a consistency proof
  from m→n reports OK after the first m leaves were edited / deleted / reordered;
  OR a genuine append reports inconsistent.
- **K3** — cross-language divergence: Python and JS disagree on the root hash or
  on any inclusion/consistency proof verification over the real log.

K2 is the decisive bar: the entire value ("no silent suppression of past
entries") rests on consistency proofs detecting any rewrite of witnessed history.
P4 is the stated SCOPE (witness/gossip requirement), not a kill.

I commit to reporting whichever way it lands.

## Results (recorded 2026-05-28, AFTER the build — reported as-landed)

**Thesis SURVIVED.** Kill-gate held on all three axes. 16 gate tests
(`tests/test_transparency.py`, all pass; full scoped suite green, zero
regressions). Live receipt over styxx's own HEAD:
`scripts/dogfood/transparency_log_self_2026_05_28.json` (RESULT SURVIVED;
Node v24.13.0; 8-leaf log of real attestation portable digests; root
`66e75f50…`; tree head witnessed at size m=5, root `4ed9fbc0…`).

- **P1 — inclusion sound and complete: HELD (K1).** Every leaf of every log
  (sizes 1..33, 64, 100, 129) produced an inclusion proof that verifies; a
  tampered leaf hash, a wrong root, and a forged non-member leaf all FAIL.
  Out-of-range indices raise. Python and Node agree on the root over all sizes.

- **P2 — consistency catches rewrite: HELD (K2, decisive).** With a tree head
  witnessed at size m, a pure append (first m leaves untouched) PASSES the m→n
  consistency proof, while **edit / delete / reorder / truncate** of any of the
  first m witnessed leaves all FAIL. The truncate case is the headline: dropping
  a past receipt (silent suppression) is caught by anyone holding the earlier
  tree head.

- **P3 — cross-language portable: HELD (K3).** The zero-dependency JS verifier
  (`web/styxx_verify.js`, no styxx, no Python, bundled SHA-256) reproduces the
  root byte-for-byte and verifies every inclusion + consistency proof; it accepts
  the genuine append and rejects the rewrite against the witnessed root. The
  browser page (`web/verify.html`) verifies a pasted inclusion/consistency proof
  client-side.

- **P4 — honest boundary: HELD (stated scope, NOT a kill).** Append-only-ness is
  proven RELATIVE to the witnessed tree head. Defeating equivocation by an
  operator who never publishes a tree head needs tree-head gossip/witnessing,
  exactly as in CT — out of scope, stated in the artifact and docs, not claimed.

**What this is, without inflation:** Certificate-Transparency (RFC 6962, with
documented string-tag domain separation) applied to styxx attestations, with an
independent zero-dependency JavaScript verifier that reproduces the root and
both proofs. It closes the *completeness* gap the receipt arc could not: prove an
agent never silently dropped its bad receipts — relative to a witnessed tree
head. It does NOT, by the data structure alone, defeat equivocation; that needs
the witness/gossip layer, which is explicitly out of scope.
