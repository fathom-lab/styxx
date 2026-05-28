# Pre-registration — Commit-Pinned Verifiable Attestation

**Stated 2026-05-28, BEFORE the ref-pinning code was written.**
Author: styxx coding agent. Methodology: recursive-discipline (pre-register →
kill-gate → build → run → report whichever way it lands).

## Context

`styxx.attestation` (shipped today, commit `ef897fa`) produces a
content-addressed artifact a third party re-verifies against a git repo
WITHOUT trusting the agent. Its substrate is the **working tree at the moment
of attestation**. The P5 self-dogfood exposed the consequence: the claim
`version is 7.7.10` was true at attest time, then flipped to FAIL once the repo
moved to 7.7.11. The verdict correctly tracked the moving substrate — but the
attestation could not say *"this was true AS OF a fixed point in history."*

## The thesis (why this is a new capability, not a tweak)

A **commit-pinned attestation** binds its substrate to a specific commit SHA.
`attest(report, repo, ref=SHA)` materializes the tree at `SHA` (read-only, via
`git archive` into a throwaway temp dir — never mutating the working tree or
`.git`), runs the existing checkers against that historical tree, and records
the resolved SHA in the artifact. `verify_attestation` re-materializes THAT
SAME commit from the repo and reproduces the verdicts against it.

The result is **immutable temporal provenance**: a claim attested true at a
commit stays verifiably true at that commit forever, regardless of how the repo
evolves afterward. This is the shape a regulator's "as-of-date" conformity
evidence requires, and it generalizes the accidental P5 finding into an
intentional, verifiable capability.

## Pre-registered predictions

- **P1 — determinism under pinning.** Two `attest(ref=SHA)` runs on the same
  repo produce a byte-identical canonical payload and the same digest. The ref
  pins the entire substrate; only `generated_at` (outside the hash) varies.

- **P2 — historical isolation (the headline).** `attest("version is 7.7.10",
  ref=<7.7.10 commit>)` returns **PASS** even though the working tree HEAD is
  7.7.11, and a HEAD/working-tree attestation of the same claim returns
  **FAIL**. The current working tree must NOT leak into the pinned verdict.
  `verify_attestation` reproduces the pinned PASS after re-materializing the
  commit. If the working tree leaks in, pinning is a no-op.

- **P3 — false-at-ref is caught.** A claim that was FALSE at the pinned commit
  (e.g. `version is 7.7.11` pinned to the 7.7.10 commit) returns FAIL — the
  current (true) state does not retroactively rescue it.

- **P4 — read-only.** A pinned attest + verify cycle leaves `git status
  --porcelain` and the `.git` worktree registry byte-for-byte unchanged. No
  checkout, no stash, no worktree registration.

## Kill-gate (stated before the build)

The commit-pinned thesis is **KILLED** (reported negative, not reframed) if ANY of:

- **K1** — non-determinism: two pinned runs differ in the hashed payload.
- **K2** (decisive) — leakage: the pinned verdict reflects the working tree
  rather than the pinned commit (P2 fails), i.e. pinning to the 7.7.10 commit
  does not yield PASS while HEAD is 7.7.11, OR verify cannot reproduce it.
- **K3** — false-at-ref rescued: a claim false at the pinned commit returns
  PASS because current state leaked in (P3 fails).
- **K4** — mutation: the pin cycle alters the working tree or `.git` (P4 fails).

K2 is the decisive bar: the entire value ("as-of-date provenance") rests on the
pinned tree being isolated from the live working tree. I commit to reporting
whichever way it lands.

## Results (run 2026-05-28, reported as-landed) — thesis SURVIVED

All four kill conditions held; 7 gate tests added (17/17 total in
`tests/test_attestation.py`).

- **K1 / P1 — determinism under pinning: HELD.** Two `attest(ref=SHA)` runs
  share a digest.
- **K2 / P2 — historical isolation (decisive): HELD.** Demonstrated live on
  styxx's OWN history. Report = two claims, `version is 7.7.10` and
  `version is 7.7.11`. Pinned to commit `6f9e38c` (the 7.7.10 release):
  `7.7.10` → **PASS**, `7.7.11` → **FAIL**. The live working tree (HEAD =
  7.7.11) inverts exactly: `7.7.10` → **FAIL**, `7.7.11` → **PASS**. The
  current working tree did not leak into the pinned verdict.
  `verify_attestation` re-materialized commit `6f9e38c` and reproduced both
  pinned verdicts (digest_ok, 0 mismatches). Frozen receipt:
  `scripts/dogfood/attestation_pinned_6f9e38c_2026_05_28.json`.
- **K3 / P3 — false-at-ref caught: HELD.** `version is 7.7.11` pinned to the
  7.7.10 commit returns FAIL; current truth does not retroactively rescue it.
- **K4 / P4 — read-only: HELD.** `git status --porcelain` and the `.git`
  worktree registry are byte-for-byte unchanged across a pin cycle (gate test
  `test_K4_pin_cycle_is_read_only`).

A commit-pinned attestation is immutable as-of-date provenance: the same claim
is verifiably true at one commit and false at another, and a third party
reproduces either verdict against the pinned commit without trusting the agent.
