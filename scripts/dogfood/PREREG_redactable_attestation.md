# Pre-registration — Redactable Cognometric Attestation (selective disclosure)

**Stated 2026-05-28, BEFORE the redaction code was written.**
Author: styxx coding agent. Methodology: recursive-discipline (pre-register →
kill-gate → build → run → report whichever way it lands).

## Context

The attestation arc is *trust subtraction*: re-derive verdicts (don't trust the
agent) → Merkle chain (don't trust ordering) → standalone verifier (don't trust
styxx) → portable digest (don't trust Python) → transparency log (don't trust
the operator's *selection* — no silent suppression of past receipts).

One axis is still unsubtracted: **confidentiality**. Every proof so far forces
you to publish the underlying text — to re-derive a digest or a score, a
verifier must see the whole (prompt, response). An agent that wants to prove a
single fact ("my sycophancy score on this interaction was 0.08", "claim C3
PASSED") must reveal the entire conversation to do it. There is no way to
disclose *one attested fact* and prove it is bound to the public receipt while
keeping the rest of the response private.

## The thesis (what is new)

A **Redactable Cognometric Attestation** — a salted Merkle commitment over the
*individual fields* of an attestation, so an agent can later disclose a chosen
subset and prove each disclosed (field, value) is exactly the one committed into
the receipt's public `digest.redactable.root` (and thus into the transparency
log), revealing nothing about the undisclosed fields beyond their count.

Construction (pure hashing — same zero-dependency, any-language ethos as the
portable verifier):

- Flatten the attestation core to a canonical, pointer-sorted list of
  `(pointer, value)` leaves (JSON-pointer paths; values serialized with the
  same RFC 8785 / JCS rule as `digest.portable`).
- Each leaf gets a fresh **256-bit salt**:
  `leaf = SHA-256("styxx-redact-leaf:" + salt + ":" + jcs(pointer) + ":" + jcs(value))`.
- Roll the salted leaf hashes into an RFC 6962-style Merkle tree (string node
  tag `styxx-redact-node:`); the root is `digest.redactable.root`.
- The salts are the agent's secret and are NOT in the public artifact. A
  **disclosure** reveals, per chosen field, `{pointer, value, salt, leaf_index,
  audit_path}`; the verifier recomputes the salted leaf and checks its inclusion
  path against the root. Undisclosed fields appear only as opaque sibling hashes
  in audit paths.

This is **selective disclosure**, like SD-JWT / redactable Merkle credentials.
Composed with the transparency log (the redactable root becomes a leaf), it
yields a *tamper-evident, append-only, selectively-disclosable* cognometric
history: show one score, prove it is the committed one, prove it could not have
been revised after the fact — without revealing the conversation.

**Design choices recorded before building (honest):**
- The salt is load-bearing: a low-entropy field (a verdict in {PASS,FAIL,ERROR},
  a 0–1 score) would be brute-forceable from an unsalted leaf hash by anyone who
  knows the domain. A 256-bit per-leaf salt makes that infeasible.
- This is NOT zero-knowledge. It does not prove a *predicate* (range / threshold)
  over a HIDDEN value, and it does not re-derive scores — a disclosed value is
  trusted as the *committed* value (it inherits the commit-time / re-seal
  boundary, caught only via the transparency log + an external witness). Calling
  this a ZK range proof would be the overclaim; we will not make it.
- Leakage stated up front: a disclosure leaks the total field **count**
  (tree size) and the disclosed pointers + values; it hides every undisclosed
  pointer and value.

## Pre-registered predictions

- **P1 — disclosure is sound (headline).** Every disclosed (pointer, value, salt)
  verifies against the redactable root via its audit path; a tampered disclosed
  value, a wrong salt, or a wrong root FAILS. The Python and JS implementations
  agree byte-for-byte on the redactable root and on every disclosure
  verification.

- **P2 — confidentiality holds, and the salt is why (decisive).** Given a
  disclosure of a strict subset, (a) no undisclosed field's value (notably the
  full response text) appears anywhere in the disclosure, AND (b) the salt is
  necessary: an UNSALTED leaf of a small-domain value is recovered by brute force
  over the domain, while the SALTED leaf of the same value is NOT recoverable
  without the (undisclosed) salt.

- **P3 — additive, legacy + portable untouched.** Adding `digest.redactable`
  leaves `digest.value` (legacy) and `digest.portable.value` byte-identical;
  every previously issued receipt stays valid and every existing test passes.
  Redactable mode is opt-in (`attest(..., redactable=True)`) and is, by design,
  non-deterministic (the salts are the confidentiality).

- **P4 — composes with the transparency log.** The redactable root is a valid
  transparency-log leaf; a disclosed fact is provably included in the public,
  append-only history (binds confidentiality to the no-suppression guarantee),
  and the JS verifier checks the disclosure client-side.

## Kill-gate (stated before the build)

KILLED (reported negative, not reframed) if ANY of:

- **K1** — disclosure unsound: a tampered disclosed value / wrong salt / wrong
  root verifies as OK, OR a genuine disclosure fails to verify.
- **K2** (decisive) — confidentiality broken: an undisclosed field's value
  (esp. the response text) is recoverable from a disclosure, OR the construction
  is brute-forceable because the salt is absent / not binding (i.e. the salted
  leaf of a small-domain value can be recovered without the salt).
- **K3** — additivity broken (legacy or portable digest changes, or any existing
  attestation test regresses), OR cross-language divergence: Python and JS
  disagree on the redactable root or any disclosure verification.

K2 is the decisive bar: the entire value ("disclose one fact, keep the rest
private") rests on undisclosed fields staying hidden AND the salt actually
preventing brute-force recovery of low-entropy disclosed-adjacent values.

I commit to reporting whichever way it lands.

## Results — SURVIVED (2026-05-28)

Kill-gate run as `tests/test_redact.py` (14 tests, all pass) and the live
self-dogfood `scripts/dogfood/redactable_attestation_self_2026_05_28.py` over
styxx's own HEAD (`7c214e0`). Frozen receipt:
`scripts/dogfood/redactable_attestation_self_2026_05_28.json`. Node v24.13.0.

- **P1 — disclosure sound: HELD.** A genuine disclosure of one field verifies
  against the redactable root; a tampered value, wrong salt, wrong root, and a
  swapped leaf index all FAIL. Python and JS agree on the root and on every
  disclosure verification (`web/styxx_verify.js`, zero styxx, zero deps).

- **P2 — confidentiality + salt load-bearing: HELD (decisive).** Disclosing only
  `vitals/scores/sycophancy` (= 0.0188…) leaves the full response text absent
  from the disclosure blob and reveals exactly one pointer/value. The salt is
  decisive: the UNSALTED leaf of a verdict in {PASS,FAIL,ERROR} is recovered by
  brute force over the domain, while the SALTED leaf of the same value is NOT
  recoverable without the (undisclosed) 256-bit salt. Two commitments to the same
  object have different roots and salts (unlinkable).

- **P3 — additive: HELD.** Adding `digest.redactable` leaves `digest.value`
  (legacy) and `digest.portable.value` byte-identical; the public artifact carries
  only `{alg, version, root, tree_size}` — never the salts. A non-redactable
  attestation raises on `.disclose(...)`.

- **P4 — composes with the transparency log: HELD.** The redactable root is a
  valid transparency-log leaf; an inclusion proof binds the confidential receipt
  to the append-only history, a consistency proof passes against an external
  witness's earlier root, and a rewrite of that earlier history is caught.

**RESULT: SURVIVED.** Honest scope preserved: selective DISCLOSURE, not
zero-knowledge — no predicate/range over a hidden value; a disclosed value is the
COMMITTED value (commit-time / re-seal boundary, caught only via the transparency
log + an external witness). Leaks field count + disclosed pointers/values; hides
everything undisclosed.
