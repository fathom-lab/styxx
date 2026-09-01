# SPEC — the worklog v0.1: a record of what an agent actually wrote

Fathom Lab · 2026-08-31 · An engineering spec, frozen before the implementation.
Successor step named by the pressure-test of the claims-manifest inversion
(`PLAN_prior_art_and_the_next_move`, the manifest verdict) and licensed by
`RESULT_collateral_census_2026_08_31.md`.

**This artifact makes no claim and carries no verdict.** It is a record, not an assertion.
Shipping it verdict-free is deliberate: an instrument that gates on a band whose noise floor
it has not measured is the exact mistake this lab made this morning, and paid for by
disabling a shipped feature. The worklog is the infrastructure that makes the measurement
possible later.

## The problem it exists to solve

An agent's account of its own work has, until now, had one author. The summary is written by
the model; the diff is produced by the same session. Checking one against the other catches
only the disagreements a model is careless enough to leave in prose — which this lab
measured at 22.5% coverage and 0.23 accusation precision on an external corpus, and then
switched off.

The missing artifact is a record with a **different author**: not what the agent *said* it
did, and not what the repository *ended up* looking like, but what the agent's tools
actually wrote, recorded by the harness at the moment of writing. That record can be
compared against both of the others, and the comparisons it enables are factual rather than
interpretive.

## The record

A worklog is an append-only sequence of entries, each describing one write performed through
an instrumented surface:

- the path written, normalised
- the tool that performed the write, as the harness names it
- a digest of the content after the write, and of the content before it where a prior
  version existed
- a monotonic sequence number

Plus a header naming the worklog spec version, the session it belongs to, and the harness
that produced it. The whole record is canonicalised with the same RFC 8785 serialization the
attestation module already exposes, and carries a digest over that canonical form, so a
worklog can be sealed inside a capsule and re-checked byte-for-byte.

## What it is not

- **Not signed.** A worklog is only as trustworthy as the harness that wrote it, and this
  version says so rather than implying otherwise with cryptography that authenticates
  nothing about the harness itself.
- **Not complete.** It records writes through the instrumented surface. A write performed by
  a shell command the harness did not wrap does not appear, and that gap is a permanent
  property to be disclosed at every use, never quietly narrowed.
- **Not a verdict.** No band, no status, no pass or fail. A consumer that wants to compare a
  worklog against a diff may do so; this spec deliberately does not.
- **Not evidence of intent.** A file written and later reverted is two entries, not a
  confession.

## Verification, such as it is

`python -m styxx.worklog verify FILE` checks exactly what can be checked from the record
alone: that the canonical digest matches the entries, that sequence numbers are dense and
ordered, that no entry is malformed. It does **not** check the entries against a repository,
because a worklog verified against the tree it produced would be verifying the harness with
the harness.

When a worklog is sealed in a capsule, the capsule renders it as **UNGATED** — present,
hashed, and explicitly carrying no verdict. That badge is not an apology. It is the honest
state of an instrument whose consuming gate has not yet earned the right to exist.

## Out of scope for v0.1, named

Signatures and harness attestation; comparison against a diff (the `undeclared` band);
comparison against a model-authored declaration; any blocking behaviour anywhere; content
storage — the worklog records digests, never file contents.

---

*The last two instruments this lab shipped were measured and found wanting at the point
where prose becomes a claim. This one makes no claim at all. It exists so that the thing
which eventually does can be measured against something with a different author.*
