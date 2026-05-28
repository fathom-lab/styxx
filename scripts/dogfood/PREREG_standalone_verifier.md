# Pre-registration — Standalone Verifier (trust-minimized, zero-styxx-import)

**Stated 2026-05-28, BEFORE the standalone verifier was written.**
Author: styxx coding agent. Methodology: recursive-discipline (pre-register →
kill-gate → build → run → report whichever way it lands).

## Context

`styxx.attestation` produces a content-addressed (SHA-256), commit-pinned,
Merkle-chained, optionally vitals-bearing artifact, and `verify_attestation` /
`verify_chain` re-derive everything from the recorded substrate rather than
trusting the embedded values (commits `ef897fa`, `63f9da4`, `7670bd9`, and the
cognometric-vitals work). Every verification path so far runs **inside styxx**.

That is a trust asymmetry the rest of the library exists to eliminate. To check
a styxx attestation today you must `import styxx` — i.e. trust the same code
that produced it. A tamper-evident receipt whose ONLY checker is the issuer is
weaker than it looks: "verifiable" should not mean "verifiable by the thing that
signed it."

## The thesis (what is new)

The artifact's structural integrity is defined by a **content-addressing scheme**
small enough to specify in prose and reimplement from scratch:

- per-attestation digest = `sha256(canonical_payload)` where
  `canonical_payload = json.dumps(core, sort_keys=True, separators=(",",":"),
  ensure_ascii=False)` and `core` is the artifact minus the `generated_at` and
  `digest` keys;
- chain link digest = `sha256(f"{prev}|{att_digest}")`, genesis
  `prev = "styxx-attestation-chain-v1"`, head = last link digest.

If that scheme is real and complete, then a **standalone verifier that imports
NOTHING from styxx** (Python stdlib only — `hashlib`, `json`, `sys`) can verify
the structural integrity (digest + Merkle chain + head) of any styxx attestation
or chain, and an independent reimplementation will agree with the library
**byte-for-byte** on every digest. That is the operational form of "you don't
have to trust styxx to verify a styxx attestation."

**Design constraint discovered before building (recorded honestly):** the
canonical payload uses Python's `json.dumps` float repr. Python emits `1.0` as
`"1.0"` and small floats in `e`-notation; JavaScript's `JSON.stringify` and
other languages differ. So byte-identical agreement is claimed for an
**independent PYTHON reimplementation**, NOT cross-language. A cross-language
(browser/JS) verifier would need a canonical-number scheme (JCS / RFC 8785).
That is a stated boundary and explicit future work, NOT part of this thesis.

## Pre-registered predictions

- **P1 — cross-implementation digest agreement (headline).** Over a corpus of
  4 artifact shapes — (a) plain attestation, (b) commit-pinned, (c) vitals-
  bearing, (d) a multi-link chain — the standalone verifier recomputes every
  per-attestation digest and every chain-link digest and matches the
  library-recorded value byte-for-byte. The standalone verifier shares NO code
  with `styxx.attestation` (separate file, stdlib-only imports, asserted in
  test by inspecting its imports).

- **P2 — tamper detection without styxx.** The standalone verifier, given an
  artifact whose embedded `digest.value` was flipped, or a chain whose a link's
  `chain_digest` / `prev_chain_digest` / ordering was altered, reports
  `structural integrity: FAIL` — using only the recomputation, no styxx.

- **P3 — honest scope boundary (stated, NOT a kill).** The standalone verifier
  checks **structural integrity only**: digest reproduction, chain linkage,
  head. It does NOT re-run claim checkers (those need git + the repo tree) and
  does NOT re-derive vitals (those need styxx's scoring instruments). It says so
  in its own output (`semantic: NOT CHECKED — needs styxx + repo`). A verifier
  that *claimed* to fully verify a claim verdict or a vitals score without styxx
  would be lying about its own reach; refusing that claim is the honest outcome.

- **P4 — re-sealed tamper is honestly NOT caught by structure alone.** If an
  attacker edits the substrate (e.g. a claim or a vitals score) AND re-seals the
  digest, the standalone structural verifier reports OK — because the digest is
  now internally consistent. This is the SAME honest boundary the chain has
  (re-seal is caught only against an externally-anchored `expected_head`). The
  standalone verifier therefore accepts an optional `--expected-head` and only
  then can flag a re-sealed chain. Structure-without-an-anchor cannot catch a
  re-seal, and the verifier must not pretend otherwise.

## Kill-gate (stated before the build)

The standalone-verifier thesis is **KILLED** (reported negative, not reframed)
if ANY of:

- **K1** (decisive) — cross-implementation divergence: on the 4-shape corpus the
  standalone verifier's recomputed digest differs from the library-recorded
  digest on ANY artifact or chain link. Then the scheme is not actually
  specifiable/portable and the standalone receipt is not the same object.
- **K2** — the standalone verifier reports structural OK on a naively tampered
  digest or a broken/reordered chain (anchored by `expected_head`). Then it does
  not verify what it claims to.
- **K3** — scope leak: the standalone verifier asserts a semantic property
  (claim verdict true / vitals reproduce) that it cannot actually check without
  styxx. Over-reading its own reach is a kill even if every digest matches.

K1 is the decisive bar: the value ("verify a styxx attestation without trusting
styxx") rests entirely on an independent reimplementation reproducing the
content address byte-for-byte. P3/P4 are stated SCOPE — reporting "structure
only, semantic NOT CHECKED, re-seal needs an anchor" is the honest outcome, not
a failure.

I commit to reporting whichever way it lands.

## Results (recorded 2026-05-28, AFTER the build — reported as-landed)

**Thesis SURVIVED.** Kill-gate held on all three axes. 10 cross-validation gate
tests (`tests/test_standalone_verifier.py`, all pass; full scoped suite green,
zero regressions). Live receipt:
`scripts/dogfood/standalone_verifier_self_2026_05_28.json` (RESULT SURVIVED on a
real chain over styxx's own HEAD `5c8735b`).

- **P1 — cross-implementation digest agreement: HELD (K1, decisive).** Over the
  4-shape corpus (plain / commit-pinned / vitals-bearing / multi-link chain) the
  standalone verifier recomputes every per-attestation and chain-link digest and
  matches the library byte-for-byte; the standalone `verify_chain` reports OK
  anchored to the library head. (`K1_all_digests_agree: true`.)

- **P2 — tamper detection without styxx: HELD (K2).** A flipped embedded digest
  and a broken / reordered chain link are reported FAIL by the standalone
  verifier alone, no styxx imported. (`K2_flipped_digest_caught: true`.)

- **P3 — honest scope boundary: HELD as stated scope (not a kill).** The
  standalone verifier shares no code with styxx — it imports only
  `{argparse, hashlib, json, sys}` (asserted in test by AST-inspecting its
  imports; `K3_no_styxx_import: true`) — and reports claim verdicts and vitals
  scores as `NOT CHECKED` rather than asserting them, since neither is
  re-derivable without styxx (+ the repo, for claims).

- **P4 — re-seal honestly NOT caught by structure alone: HELD.** A fully
  re-sealed chain (substrate edited, digests recomputed) passes structural
  verification — the inherent content-addressing boundary — but is caught the
  moment an external `--expected-head` anchor is supplied.
  (`P4_reseal_passes_structure: true`, `P4_reseal_caught_by_anchor: true`.)

**What this is, without inflation:** an independent, stdlib-only reimplementation
of the styxx content-addressing scheme (spec: `docs/attestation-content-address.md`)
that verifies the structural integrity of any styxx attestation or chain without
importing styxx, cross-validated byte-for-byte against the library. It is NOT a
full verifier — semantic claims and vitals are reported `NOT CHECKED`, and a
re-sealed chain needs an external head anchor. Cross-LANGUAGE agreement is out of
scope (Python json number repr is not language-portable; JCS/RFC 8785 is the
documented future-work path).
