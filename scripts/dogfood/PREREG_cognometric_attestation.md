# Pre-registration — Cognometric Attestation (verifiable, reproducible vitals)

**Stated 2026-05-28, BEFORE the vitals code was written.**
Author: styxx coding agent. Methodology: recursive-discipline (pre-register →
kill-gate → build → run → report whichever way it lands).

## Context

`styxx.attestation` already produces a content-addressed, third-party-
reproducible artifact for an agent's *factual* self-report claims (commit
`ef897fa`), pins them to a commit (`63f9da4`), and Merkle-chains a sequence into
a tamper-evident ledger (`7670bd9`). Every guarantee so far is about
**checkable facts** ("version is 7.7.11", "tag v7.7.11 exists").

What is NOT yet attested is the *cognometric register* of the report itself —
the styxx instrument scores (sycophancy / overconfidence / deception / refusal)
that the rest of the library exists to measure. Today those scores are computed
ad hoc and trusted on the agent's word. There is no content-addressed,
re-derivable, tamper-evident record of "this report, scored under styxx's
published text-heuristic instrument, came out HERE."

## The thesis (what is new)

Extend `attest` to optionally embed the **cognometric vitals** of the report as
a verifiable measurement: the deterministic text-heuristic scores from
`styxx.attack.score_all`, bound into the same SHA-256 content address. `verify`
RE-COMPUTES the scores from the recorded (prompt, response) pair and compares —
it never trusts the embedded scores, exactly as it never trusts an embedded
claim verdict. Chained, this yields a **tamper-evident cognometric vitals trend**
over an agent's work history: the black-box-recorder vitals trace the monitor
framing implies, now third-party-reproducible.

**Design constraint discovered before building (recorded honestly):** the four
instruments are RELATIONAL — `score_all(response=...)` alone returns `{}`;
sycophancy/overconfidence/deception/refusal are only defined against a `prompt`.
So a cognometric attestation MUST record the task the report responds to as part
of the attested substrate. Vitals on a referent-free monologue are not "zero",
they are UNDEFINED, and the artifact says so.

## Pre-registered predictions

- **P1 — determinism.** `attest(..., prompt=p, vitals=True)` on the same
  (prompt, report, substrate) produces the same `digest` and the same embedded
  `vitals.scores`. The text-heuristic tier is a pure function of text (no
  network, no randomness — confirmed: two `score_all` calls byte-identical).
  Scores are rounded to 12 decimals before embedding so the content address is
  stable cross-platform.

- **P2 — reproduction + tamper-evidence (the headline, decisive).** `verify`
  re-runs `score_all` on the recorded (prompt, response) and matches every axis.
  A flipped embedded score is caught **even when the attacker re-seals the
  digest** — because the score is re-derived from the recorded text, not read
  from the artifact. This is the same trust-the-substrate-not-the-agent property
  the factual claims already have, now extended to the instrument scores.

- **P3 — honest boundary (stated scope, explicitly NOT a kill).** The attested
  vitals measure **register** (textual surface features), NOT ground-truth
  honesty or correctness. This re-affirms styxx's validated construct ceiling
  (text-only scoring is a register detector, not a truth oracle — a CLOSED
  NEGATIVE I am NOT re-litigating). The artifact carries a machine-readable
  `measures` field and per-axis caveats; reference-less `deception` is flagged
  register-only (known to saturate on benign text). The artifact must make
  over-reading structurally hard. A kill would be the artifact ASSERTING the
  scores prove honesty — it will not.

- **P4 — chain vitals trend.** Vitals flow through `attest_chain`; each link's
  vitals re-derive against its recorded (prompt, response); the chain yields a
  tamper-evident vitals trajectory whose per-link reproduction is preserved
  under composition.

## Kill-gate (stated before the build)

The cognometric-attestation thesis is **KILLED** (reported negative, not
reframed) if ANY of:

- **K1** — non-determinism: two vitals-attestations over identical
  (prompt, report, substrate) differ in `digest` or in any embedded score.
- **K2** (decisive) — a tampered embedded score survives `verify`: `verify`
  reports ok while the score re-derived from the recorded text differs. Then the
  vitals are decorative, not verifiable, and the whole point is lost.
- **K3** — composition leak: `verify_chain` reports ok while a per-link vitals
  no longer reproduces against that link's recorded (prompt, response).

K2 is the decisive bar: the value ("a verifiable, re-derivable, tamper-evident
cognometric record") rests entirely on the score being reproduced from the
substrate text, never trusted from the artifact. The register-not-honesty
boundary (P3) is the stated SCOPE, not a kill — reporting a score as register
when it is register is the honest outcome, not a failure.

I commit to reporting whichever way it lands.

## Results (recorded 2026-05-28, AFTER the build — reported as-landed)

**Thesis SURVIVED.** Kill-gate held on all three axes. 6 new gate tests
(`tests/test_attestation.py`, 29/29 attestation total; full scoped suite 1174
passed, 8 skipped — +6 over baseline, zero regressions). Live receipt:
`scripts/dogfood/cognometric_attestation_self_2026_05_28.json`.

- **P1 — determinism: HELD (K1 clear).** Two vitals-attestations over an
  identical (prompt, report, substrate) produce the same `digest` AND the same
  embedded `vitals.scores` (text-heuristic tier is a pure function of text;
  scores rounded to 12 decimals for cross-platform stability).
  (`K1_determinism_ok: true`; test `test_K1_vitals_determinism`.)

- **P2 — reproduction + tamper-evidence: HELD (K2, the decisive bar).** `verify`
  re-derives every score from the recorded (prompt, response) and matches. The
  headline: flipping an embedded score AND re-sealing the digest leaves
  `digest_ok: true` (the re-seal fooled the hash) but `vitals_ok: false` — the
  recomputation from the substrate text catches it, so overall `ok: false`. The
  same trust-the-substrate-not-the-agent property the factual claims have, now
  on the instrument scores. (`K2_caught: true`; tests
  `test_K2_resealed_vitals_tamper_is_caught`,
  `test_K2_editing_response_text_diverges_scores`.)

- **P3 — honest boundary: HELD as stated scope (not a kill).** The artifact
  carries a machine-readable `measures` field ("register … NOT ground-truth
  honesty") and per-axis caveats. The live dogfood is the proof of why this
  matters: on a benign, truthful self-report the **deception axis scored 0.9956**
  — the reference-less construct-ceiling artifact, exactly as the embedded
  caveat warns ("REFERENCE-LESS: saturates on benign text; not a deception
  finding without a grounding reference"). The attestation reproduces and is
  tamper-evident AND documents its own scope, so the high score cannot be
  honestly over-read as a deception finding. This re-affirms styxx's validated
  construct ceiling; it does NOT re-litigate it.

- **P4 — chain vitals trend: HELD.** Vitals flow through `attest_chain` (3-tuple
  `(report, ref, prompt)` items); each link's vitals re-derive against its
  recorded pair, and a per-link re-sealed score tamper makes the chain not-ok
  even with intact Merkle links — composition does not weaken the guarantee.
  (test `test_K3_chain_vitals_reproduction_preserved`.)

**What this is, without inflation:** a verifiable, re-derivable, tamper-evident
record of an agent self-report's cognometric REGISTER, chainable into a trend.
It is NOT a proof of honesty — the artifact says so in a field a machine can
read, and the deception=0.9956-on-honest-text result is the honest demonstration
of exactly that boundary.
