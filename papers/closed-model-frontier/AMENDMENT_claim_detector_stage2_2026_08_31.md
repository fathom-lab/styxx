# AMENDMENT — Stage 2's sample was unsatisfiable. Declared before any seat ran.

Fathom Lab · 2026-08-31 · Amends `PREREG_claim_detector_2026_08_30.md`. Written and committed
**before** a single Stage 2 packet was built or any seat was spawned. No adjudication outcome
was known, or could have been known, when this was written.

## The defect

Stage 2 as frozen requires **60** STRUCT-1-flagged sentences drawn from the corpus remainder
(not template-flagged, not previously adjudicated), with a floor of **45** surviving. Counted
after Stage 1 shipped:

| population | flagged by STRUCT-1 | available for Stage 2 |
|---|---|---|
| pinned corpus `origin/main..a6994ac` (2,824 sentences) | 42 | **38** |
| current branch head (2,960 sentences) | 43 | **39** |

**38 < 45.** The frozen sample cannot be drawn, and the frozen floor cannot be met, on any
version of this corpus. Extending to the branch head buys one sentence and changes nothing.

This is the same defect class that killed the v13 ladder prereg: *a population that was
unsatisfiable the moment it was written*. It is caught here at the best possible time —
before collection — and it is published rather than quietly shrunk, because a gate whose
denominator is silently reduced to fit is the failure mode this repository exists to reject.

## What changes — and what explicitly does not

**Changed, and only this:**

1. **The flagged arm becomes a CENSUS, not a sample.** Stage 2 adjudicates **all 38**
   available STRUCT-1-flagged sentences. This removes every degree of selection freedom that
   a 60-of-38 draw could not have had anyway: there is nothing to choose, so nothing to
   choose favourably. A census is strictly less gameable than the frozen sample.
2. **The control arm is matched to it: 38** STRUCT-1-unflagged sentences drawn from the same
   available pool by `random.Random(20260831)`, unchanged seed.
3. **The floor drops from 45 to 30 surviving per arm.** Reason, stated plainly: with a
   population of 38, a floor of 45 makes the gate unfailable-by-construction — it could only
   ever return "measurement failed", which is not a gate, it is a guaranteed outcome. A leg
   that cannot fail must not gate; a leg that can only fail is the same defect wearing the
   opposite coat. 30 of 38 leaves real room for the floor to bite if seats or packets
   invalidate.

**Explicitly unchanged — the thresholds are not touched:**

- **G-S2P** still requires A-share among adjudicated STRUCT-1 flags > **0.2061** (N2's
  weighted precision, frozen at the baseline). Failure still publishes the verbatim sentence
  *"the structural detector adds no precision over the verb-stem null at this sample size."*
- **G-S2LIFT** still requires flagged A-share > control A-share, both published with
  denominators either way.
- The same 30 decoys, the same `agent_claim_seat_instructions_v1.md` / `v2.md` texts verbatim,
  the same 3-seats-per-packet topology, the same majority rule, the same NO-MAJORITY
  exclusion-and-count, the same ≥0.80 gating-decoy validity threshold with the same re-run
  ladder, the same report-only treatment of the mention-vs-use decoys.
- STRUCT-1 itself is **frozen as shipped** (`struct-1/2026-08-30`, commit `5534b70`). No
  conjunct, exception, or regex moves for the rest of this cycle regardless of outcome.

## Disclosed consequence

At n=38 per arm, **no significance is claimed and none may be quoted**. Both gates are
comparisons of small proportions and must always travel with their raw counts. The reduced n
is a direct consequence of STRUCT-1's narrowness — the same narrowness that produced its DEV
precision — and that trade-off is the finding, not a nuisance to be smoothed over.

## What this amendment does not license

No change to STRUCT-1. No change to the gate thresholds. No third arm, no re-draw, no
post-hoc exclusion of any flagged sentence. If the census of 38 fails the bar, it fails.
