# Pre-registration — Verifiable Cognometric Attestation (`styxx.attestation`)

**Stated 2026-05-28, BEFORE the instrument was written or executed.**
Author: styxx coding agent. Methodology: recursive-discipline (pre-register →
kill-gate → build → run → report whichever way it lands).

## The thesis (why this is a new instrument, not a reframed report)

styxx already (a) extracts deterministic checkable claims from an agent's
self-report (`agent_audit.extract_claims`), (b) verifies each against the
substrate (`AgentClaimAuditor`), and (c) maps evidence onto EU AI Act
Article 15 clauses (`styxx.compliance`). These are disconnected: the audit
verdict is ephemeral and the compliance map is static documentation.

A **Verifiable Cognometric Attestation** binds them into a single
content-addressed artifact that a THIRD PARTY can re-verify against the
substrate WITHOUT trusting the agent that produced it. The artifact is
self-describing: it stores, per claim, the checker name + args + expected
value + the verdict + the substrate evidence, plus a SHA-256 digest over the
canonical payload, plus the explicit EU AI Act clause map and the
uncovered-requirements boundary.

`verify_attestation(artifact, repo)` does NOT read the embedded verdicts as
truth. It re-runs each claim's checker against the repo and compares the
RE-DERIVED verdict to the embedded one. Trust the substrate, not the agent.

Nobody ships an agent-produced, third-party-reproducible honesty attestation
shaped like a regulator's accuracy/robustness evidence. This run tests whether
that is a real protocol or a dressed-up report.

## Pre-registered predictions

- **P1 — determinism.** `attest()` run twice on an identical substrate (same
  git HEAD, same files) produces a BYTE-IDENTICAL canonical payload and the
  same digest. The wall-clock `generated_at` field is the only volatile field
  and is deliberately OUTSIDE the hashed payload. ZERO byte difference.

- **P2 — independent reproduction (the headline).** `verify_attestation`
  re-derives each verdict from the substrate. If an attacker flips an embedded
  verdict (PASS→FAIL) WITHOUT changing the substrate, verify must report the
  TRUE substrate verdict and flag the embedded one as tampered. Predicted:
  verify catches 100% of verdict flips. If verify ever trusts the embedded
  verdict, the protocol is a report, not an attestation.

- **P3 — tamper-evidence.** Mutating any byte of the hashed payload (a claim's
  args, expected, text, or the clause map) makes the digest check FAIL.
  Predicted: 100% of single-field mutations are caught by the digest.

- **P4 — honest boundary is mandatory.** Every attestation carries a non-empty
  uncovered-requirements boundary, and never asserts coverage for an Article 15
  clause whose styxx-primitive list is empty (e.g. 15.4 bias). Predicted: the
  boundary list length ≥ the covered-clause count (mirrors the compliance
  module's kill-gate A3).

- **P5 — self-dogfood round-trip.** Attesting my own 7.7.10 self-report
  (`scripts/dogfood/self_report_2026_05_28.md`) yields the SAME 7/7 PASS the
  `audit-claims` gate produced, now wrapped with clause mappings and boundary,
  and `verify_attestation` independently reproduces all 7 verdicts.

## Kill-gate (stated before the build)

The verifiable-attestation thesis is **KILLED** (reported negative, not
reframed) if ANY of:

- **K1** — non-determinism: two `attest()` runs on identical substrate differ
  in the hashed payload (P1 fails). An attestation that isn't reproducible
  cannot be independently verified.
- **K2** — verify trusts embedded verdicts: a flipped verdict survives
  `verify_attestation` undetected (P2 fails). Then it is a report, not an
  attestation — no third-party trust is established.
- **K3** — tamper not detected: a mutated hashed payload still validates
  against its digest (P3 fails).

I commit to reporting whichever way it lands. K2 is the decisive bar: the
entire claim ("verifiable, agent-independent") rests on verify reproducing the
substrate verdict rather than reading the agent's word for it.
