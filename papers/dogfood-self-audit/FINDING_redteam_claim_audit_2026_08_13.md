# FINDING — red-team of the claim_audit self-audit fixes

**Commits under test:** `1fb1de5`, `4de77d1`, `af62490` — all touching `styxx/claim_audit.py`
**Commissioned by:** the author of those commits (darkflobi), who asked for the paths he did not visit
**Run by:** the claude-code sub-brain (Claude Fable 5) — the adversary was chosen for measured
resistance to sycophantic caving, not for agreement
**Date:** 2026-08-13
**Reproduction:** `python papers/dogfood-self-audit/redteam_2026_08_13/redteam_claim_audit.py`
**Shipped module modified:** no. Proposed diffs are in a separate section, NOT APPLIED.

---

## LANE 2 — context resolver — **DEFECT CONFIRMED** (the author's own priority bet)

`_resolve_by_context` scores each candidate as `len(ctx & path_tokens) / len(path_tokens)`.
The denominator is the **path**, not the context. A short, generic path therefore needs one
lucky word to score 1.0, while a long, specific path is penalised for every token the prose
did not happen to repeat — so **the more precisely a sentence names its source, the more the
correct path is punished**.

Receipt (case `A_summary_key_beats_named_cell`):

```
source : {"rate": 0.1, "cells": {"blockconf_ge3_of_7": {"cave_rate": 0.1}}}
prose  : "The blockconf arm at >=3/7 shows a cave rate of 0.100 on the frozen protocol."
result : 0.100 -> 'rate'   [label: context, score 1.0]
expected                    'cells.blockconf_ge3_of_7.cave_rate'
```

This is the adversarial target as specified: **a case labelled `context` (not `arbitrary`)
that is wrong.** The sentence names the arm, the threshold, and the metric; the resolver
discards all of it and confidently cites a bare summary key.

**Realism check (this fixture is not contrived).** Scanning 400 receipt JSONs under `papers/`,
**114 carry both short scalar keys and nested detail dicts** — the exact collision shape
(`seed`, `smoke` alongside `frozen_gates`, `accuracy`, `sizing`). The mechanism fires whenever
a value collision occurs across that shape; the 114 is the prevalence of the *shape*, not of
realised collisions, and is stated as such.

Two accompanying cases, reported whatever they showed:
- `B_longer_named_path_loses_to_shorter` — resolver **declined** (`arbitrary`). Correct behaviour.
- `C_control_equal_length` — resolver resolved **correctly**. No false alarm on the control.

So the defect is specific: not "the resolver is bad", but "**path-length normalisation makes
short paths win, and the strict-win rule does not catch it because the win is genuinely strict**."

## LANE 1 — chance floor reference distribution — **DEFECT CONFIRMED**

`_chance_floor` samples uniformly over `[0, 1]` for any claim with ≥1 decimal place. That fixes
the order of magnitude but not the **band**. When the audited claims cluster in a narrow
sub-range where the source leaves also cluster — the shape of every cave-rate receipt in this
repo — the uniform draw spends most of its mass in empty territory and the floor reads far
lower than the luck a real claim enjoys.

Receipt (94 leaves drawn in `[0.0, 0.25]`, 3-decimal claims):

```
shipped floor        : 0.0925
band-matched floor   : 0.3735      (same Monte Carlo, drawn over the claims' own band)
gap                  : 0.281       flattering direction (shipped floor too LOW)
```

A grounding rate is quoted against the floor. Beating 0.09 reads as a result; beating 0.37 is a
different sentence. This is the **third** error in the same place — the module's own comment
records v1 (seed 90210 stretching the range) and v2 (a p95 still at 342), both flattering, and
this is the same defect a third time, smaller again but still material.

## LANE 3 — accounting identity — **NO DEFECT FOUND**

Attempted violation with a collision receipt (a value that is simultaneously a leaf, a ratio of
two other leaves, and a percentage of the same pair; multiple leaves sharing values to force
ambiguity):

```
n_ambiguous=2 == n_context_resolved 2 + n_arbitrary 0    -> holds
n_total=5     == grounded 5 + derived 0 + unsourced 0    -> holds
```

The identity holds by construction: `n_ambiguous` increments in the same branch that dispatches
to exactly one of the two sub-counters, and category assignment is a single `if/elif/else` over
a status set with no fall-through. **I found nothing here and say so.** No defect was manufactured
to fill the lane.

## LANE 4 — `GATE: PASS` on 0 claims from 46 sentences — **judgment: abdication**

The call to record it and move on was the right *disclosure*; leaving the verdict as `PASS` is
not. This repo's own doctrine is explicit — *"a leg that cannot fail must not gate"* — and its
other instruments obey it: the know-say datasheet returned `REFUSED__underpowered` on n=3 earlier
today rather than a flattering number (`runs/opus5-knowsay/`), and the anchors/admissibility
modules refuse rather than certify on insufficient input. A claim gate that extracts nothing has
not checked anything; `PASS` asserts a check that did not occur, and it is the flattering
direction — the same direction as both floor errors.

The distinction that matters: **zero claims is not a passing document, it is an inapplicable
gate.** The honest verdict string is a refusal (`VOID__no_claims_extracted` or equivalent),
carrying the extractor-coverage number that made it inapplicable. That preserves the disclosure
he already made while removing the assertion he did not earn.

This lane is judgment, not a reproduced defect, and is labelled as such.

---

## PROPOSED DIFFS — NOT APPLIED

1. **Resolver (lane 2).** Score symmetrically instead of by path length — e.g. Jaccard
   `|ctx ∩ pt| / |ctx ∪ pt|`, or require the winner's *absolute* overlap to exceed the
   runner-up's (`len(ctx & pt)` before normalising). Either removes the short-path premium.
   Additionally: when the top candidate's matched tokens are a strict subset of the runner-up's,
   decline (`arbitrary`) rather than resolve — the specific path is never *less* named than the
   generic one it contains.
2. **Chance floor (lane 1).** Draw over the band the document's own claims occupy
   (`min..max` of extracted claim values, widened to the nearest decade), not a fixed `[0, 1]`.
   Report the band in the floor's provenance line so the reference distribution is auditable
   rather than implicit.
3. **Zero-claims gate (lane 4).** Return a refusal verdict when `n_total == 0`, carrying the
   sentence count that produced it. No module edit proposed here beyond the verdict string —
   the extractor's coverage is a separate finding.

## Controls and honesty notes

- No favourable number about this red team is quoted: the only quantitative claims are the two
  measured gaps and the prevalence-of-shape count, each with its script.
- The red-team script itself carried a scoping bug on first run (a vestigial comprehension); it
  was deleted rather than patched, and the run above is from the corrected script. Recorded
  because the author's rule cuts both ways.
- Lane 3's negative result is reported as a negative result.

**REDTEAM_VERDICT: 2/2 confirmed | lanes=chance-floor:DEFECT,context-resolver:DEFECT,accounting:CLEAN,zero-claims-gate:ABDICATION | 2026-08-13**

---

## ADDENDUM — independent verification of the author's fix (`f44c8f4`)

The author reproduced every finding, fixed all three, and found a fourth defect this
red team missed (his subset rule used `<`, so on a Jaccard tie the short-path premium
came back through the door he had just closed). Reciprocity requires the reverse check:
the fix re-run against **this** fixture, by the party that wrote it.

**Lane 2 — fix verified, and it is not the lazy fix.**

```
A_summary_key_beats_named_cell : -> cells.blockconf_ge3_of_7.cave_rate [context 0.364]  CORRECT
B_longer_named_path_loses      : -> blockconf_high_confidence_arm.cave_rate [context 0.6] CORRECT
C_control_equal_length         : -> arm_two.cave_rate [context 0.667]                    CORRECT
confident-wrong cases: 0
```

Case B is the informative one: pre-fix it *declined* (`arbitrary`), post-fix it resolves
**correctly**. A resolver made safe by refusing everything would have left B declined —
this one gained resolution while losing the false confidence. That is the harder fix.

**Lane 1 — fix verified to four decimals by two independent implementations.**

```
band-matched reference (this red team's own Monte Carlo) : 0.3735
shipped floor via the author's `band` parameter           : 0.3735
gap: 0.0
```

**A false alarm by this red team, recorded because the rule is symmetric.** On the first
post-fix re-run, lane 1 still reported the 0.281 gap. That was wrong: the fix added a
`band` parameter and routed `audit_grounding` through it, while this script still called
`_chance_floor` with the legacy signature — exercising the fallback, not the shipped path.
The script has been corrected to call both and report each; the defect it briefly
re-asserted did not exist. An adversary that mis-measures a fix is the same failure class
as an author who over-reports one, and it belongs in the same receipt.

**Post-fix regression status of this fixture: `LANES WITH A DEFECT: none`.**

**VERIFICATION_VERDICT: fix confirmed 2/2 | lane2=0 confident-wrong, B upgraded decline->correct | lane1=exact agreement 0.3735 | redteam false-alarm recorded | 2026-08-13**
