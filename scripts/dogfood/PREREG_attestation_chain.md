# Pre-registration — Attestation Chain (cognometric provenance ledger)

**Stated 2026-05-28, BEFORE the chain code was written.**
Author: styxx coding agent. Methodology: recursive-discipline (pre-register →
kill-gate → build → run → report whichever way it lands).

## Context

`styxx.attestation` produces a content-addressed, third-party-reproducible
artifact for ONE agent self-report (commit `ef897fa`), and commit-pinning binds
each artifact to a fixed point in history (commit `63f9da4`). What is still
missing is **order**: a sequence of attestations an agent emits over its work
history is just a bag of files — nothing proves the sequence is complete, in
order, and untampered.

## The thesis (why this is a new structure, not a list)

An **attestation chain** Merkle-links a sequence of attestations into a
tamper-evident ledger. Each link carries the per-attestation SHA-256 digest
plus a rolling chain digest:

    chain_digest[0] = sha256("styxx-attestation-chain-v1" || att_digest[0])
    chain_digest[n] = sha256(chain_digest[n-1] || att_digest[n])

A third party verifies, WITHOUT trusting the agent, that (1) every attestation
independently reproduces against its pinned commit (the existing per-link
guarantee), AND (2) the links are intact — any insertion, deletion, or reorder
changes a chain digest and the head digest no longer matches. The result is a
verifiable claim-trajectory: "here is everything this agent attested, in order,
each true as-of its commit, and the sequence is tamper-evident." This is the
black-box recorder / cognometric provenance ledger the monitor framing implies.

## Pre-registered predictions

- **P1 — determinism.** The same attestations in the same order produce the
  same `head_chain_digest`. Each per-attestation digest is already deterministic
  (commit-pinned), so the chain over them is too. `generated_at` is the only
  volatile field and is outside every hash.

- **P2 — order is tamper-evident (the headline), with an HONEST tamper model.**
  A naive reorder / insert / delete (links moved without re-sealing the chain
  digests) makes `verify_chain` report the chain BROKEN at the first divergent
  link. A sophisticated attacker who re-seals every chain digest produces an
  internally-consistent chain — that is detectable ONLY against a head digest
  that was anchored externally (committed to git, timestamped, published)
  BEFORE the tamper, supplied as `verify_chain(..., expected_head=...)`.
  Predicted: 100% of naive mutations caught outright; 100% of re-sealed
  mutations caught against an anchored `expected_head`; and — stated as the
  boundary, NOT a guarantee — a re-sealed mutation with NO external anchor is
  NOT detectable by the chain alone (same tamper-evident-not-tamper-proof
  property the single attestation has; per-link substrate reproduction still
  holds, but order is not substrate-grounded).

- **P3 — per-link reproduction is preserved.** `verify_chain` re-runs
  `verify_attestation` on every link against its pinned commit; a link whose
  embedded verdict no longer reproduces makes the chain not-ok even if the
  Merkle links are intact. Composition does not weaken the per-link guarantee.

- **P4 — real-history round-trip.** A chain over styxx's OWN history — attest
  "version is 7.7.10" pinned to the 7.7.10 commit, then "version is 7.7.11"
  pinned to HEAD — verifies end-to-end, with the per-commit verdicts correct
  (PASS at each respective commit) and the chain intact.

## Kill-gate (stated before the build)

The chain thesis is **KILLED** (reported negative, not reframed) if ANY of:

- **K1** — non-determinism: two chains over identical inputs differ in
  `head_chain_digest`.
- **K2** (decisive) — a naive reorder / insertion / deletion survives
  `verify_chain` undetected, OR a re-sealed mutation survives when checked
  against the anchored `expected_head` (P2 fails). Then the chain is a
  decorated list, not a provenance structure. (A re-sealed mutation with no
  anchor evading detection is NOT a kill — it is the pre-stated boundary.)
- **K3** — composition leak: `verify_chain` reports ok while a per-link
  attestation no longer reproduces against its pinned commit (P3 fails).

K2 is the decisive bar: the entire value ("verifiable, ordered, tamper-evident
claim history") rests on order being bound into the digest. I commit to
reporting whichever way it lands.

## Results (recorded 2026-05-28, AFTER the build — reported as-landed)

**Thesis SURVIVED.** Kill-gate held on all three axes; the headline (K2) is met
under the honest tamper model stated above. Frozen receipts:
`scripts/dogfood/chain_self_history_2026_05_28.json` (real-history run) and the
6 chain kill-gate tests in `tests/test_attestation.py` (23/23 attestation tests
pass; full scoped suite 1168 passed, 8 skipped — +6 over the pre-chain baseline,
zero regressions).

- **P1 — determinism: HELD (K1 clear).** Rebuilding the same 2-link chain over
  identical inputs reproduces `head_chain_digest`
  `47b1e751…26f5` byte-for-byte. `generated_at` is the only volatile field and
  sits outside every hash. (`K1_determinism_ok: true`; test
  `K1_chain_determinism`.)

- **P2 — order is tamper-evident: HELD (K2, the decisive bar).** A naive reorder
  (swap links 0/1 without re-sealing) is caught outright — `verify_chain` reports
  BROKEN at `broken_at: 0` (`K2_naive_reorder_caught: true`). Tests confirm naive
  reorder AND forged-link insertion both break the chain, and a *sophisticated*
  re-sealed mutation (every chain digest recomputed) is caught when checked
  against an externally-anchored `expected_head` (`K2_resealed_reorder_caught_
  against_anchored_head`). The pre-stated boundary holds and is NOT a kill: a
  re-sealed mutation with no external anchor is internally consistent and not
  detectable by the chain alone — same tamper-evident-not-tamper-proof property
  the single attestation has.

- **P3 — per-link reproduction preserved: HELD (K3 clear).** `verify_chain`
  re-runs `verify_attestation` on every link against its pinned commit; a link
  whose embedded verdict no longer reproduces makes the chain not-ok even with
  intact Merkle links (`K3_per_link_reproduction_preserved`). Composition does
  not weaken the per-link guarantee.

- **P4 — real-history round-trip: HELD.** A chain over styxx's OWN history —
  "The version is 7.7.10." pinned to the `v7.7.10` commit
  (`e05d9ec…`), then "The version is 7.7.11." pinned to HEAD
  (`63f9da4…`) — verifies end-to-end (`verify_ok: true`) with each per-commit
  verdict PASS as-of its pinned commit. The commit-pin flip is the proof the
  pinning is load-bearing: the SAME "version is 7.7.10" claim evaluated against
  HEAD (7.7.11) returns FAIL (`commit_pin_flip_verdict_at_HEAD: "FAIL"`) — the
  ledger records what was true *then*, not what reproduces *now*.

**What this is, stated without inflation:** a verifiable, ordered,
tamper-evident claim-trajectory — "here is everything this agent attested, in
order, each true as-of its commit, and the sequence is tamper-evident relative
to an anchored head." It is NOT tamper-PROOF without an external anchor; that
boundary is the honest core of the result, not a footnote.
