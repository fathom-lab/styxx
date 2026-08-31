# PREREG — V13: the four repairs EXTERNAL-1 named, and the held-out test that decides them

Fathom Lab · 2026-08-31 · Frozen before implementation and before any held-out data is
read. Successor to `RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md`, which
disabled the path-claim accusation after it scored 0.23 precision against a 0.95 floor.
Governed by `SPLIT_external_corpus_2026_08_31.md`: design on DEVELOPMENT repositories only,
report the headline on HELD-OUT.

> **AMENDED before implementation** — see `CORRECTION_external1_cause_2026_08_31.md`.
> Repair 1 (suffix matching) is STRUCK: the lookup was never broken, and the cases that
> looked like suffix failures were the harness's status-collapse defect, now fixed. The
> remaining three repairs stand, with the verb-object binding confirmed dominant by a
> counterfactual run. G-R2's recovery target now applies to those three only.

## The claim under test

The gate's failure was not conceptual. It was four mechanical defects, each named by the
blind panel's own transcript. If that diagnosis is right, repairing exactly those four —
and nothing else — restores the accusation to publishable precision. If it is wrong, the
approach itself is in question, and this preregistration is how we find out rather than
argue about it.

## The four repairs, specified before they are written

1. **Suffix matching on path boundaries.** A claimed path matches a diff path when they are
   equal, or when the diff path ends with the claimed path at a directory boundary. So a
   claim naming a bare file matches the same file under a fuller directory prefix, and does
   not match a different file whose name merely ends in the same characters.
2. **Verb–object binding.** When the sentence places the path behind a containment
   preposition — removed *from*, deleted *in*, dropped *inside* a file — the claim is about
   content within that file, and the file's own status is "touched," never "deleted."
3. **Prose nouns that are not files.** A path candidate whose only extension evidence is a
   frozen list of framework and runtime names is not a file claim. The list is written into
   the implementation, and every entry is quoted in the RESULT so the closure is auditable.
4. **Negation scoping.** When the sentence negates the change over the claimed path — avoids
   modifying, does not touch, no need to change, without altering — the file's absence from
   the diff is the sentence coming true, and the claim abstains rather than accuses.

## The invariant that makes this safe, and that the gates enforce

**Every repair is accusation-removing only.** Each one can convert an accusation into a
verification or an abstention, and none can create an accusation that today's code does not
already make. This is asserted directly in the test suite, not merely intended: for the
frozen fixture corpus, the set of accusations after the repair must be a **subset** of the
set before it. A repair that produces a new accusation anywhere fails its gate outright,
whatever it does to precision.

## Gates — all four must pass, thresholds committed now

- **G-R1 (invariant) — the accusation set shrinks or holds, never grows,** across the full
  fixture suite and the development bucket. One new accusation fails the cycle.
- **G-R2 (development recovery)**: on the DEVELOPMENT bucket the repairs must reclaim a
  clear majority of the rejected accusations from EXTERNAL-1's taxonomy — at least
  **two thirds** of the false accusations whose category the four repairs address. Below
  that, the diagnosis was wrong and the RESULT says so.
- **G-R3 (held-out precision, primary) — a FRESH blind panel on HELD-OUT accusations must
  reach precision ≥ 0.95.** New sample, new decoys, the same protocol and the same
  reliability gate: fewer than 27 of 30 decoys correct and the measurement voids with no
  headline. **Only G-R3 licenses re-enabling the accusing verdict in shipped code.**
- **G-R4 (invariants)**: full suite green, and the `xfail(strict=True)` markers EXTERNAL-1
  planted on the lost catches must fail — that is, the catches must come back — and be
  removed in the same commit, exactly as those markers were designed to force.

## What failure looks like, and what we will say

If G-R3 lands below 0.95, the accusation stays disabled, this cycle publishes as a second
failure with the number attached, and the diagnosis is retired rather than retried — four
named defects repaired and precision still short would mean the class is not separable by
mechanical rules, which is a finding worth more than another patch. If G-R2 fails while
G-R1 holds, the taxonomy was a story rather than a cause, and that too gets published under
its own name.

## Boundaries

The repairs address only the path-claim class. They do not touch counts, symbols, or
environment legs. They do not make the gate able to read prose it never parsed. And a
recovered precision figure would license accusing on *this* class in *this* corpus — not a
general claim about agent honesty, which remains unmeasured and stays that way until an
instrument earns the right to measure it.

---

*The failure named four causes. This document commits, before a line is written, to what
counts as having fixed them — and to publishing the second failure just as fast as the
first if the diagnosis was wrong.*
