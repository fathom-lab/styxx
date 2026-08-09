# PREREG — protocol v4: can a gate's *composition* be declared and machine-checked?

Fathom Lab · 2026-08-09 · frozen before the implementation exists.

## Why

E1 (cycle 159) produced the sixth bar mis-specification in a week and the only one where every
component was individually correct. G1 judged the minimum error over *all* candidates; G2
disqualified one of them in the same run; G1 passed on a value belonging to the candidate G2 had
excluded. The gate, the metric path, the `power_basis` and the implementation were each right,
and the **composition** was wrong. `styxx.protocol` has no check on relationships between gates.

P1 (cycle 158) showed the same shape from the other side: three of five frozen gates were
satisfiable without testing what they named.

The proposed mechanic is narrow and fully checkable. A gate that judges an aggregate over a set
may declare, in the frozen gates block:

- `agg` — `"min"` or `"max"`,
- `over` — the result path of a **dict of per-member values**,
- `excluding` — optionally, the result path of a **list of member names to exclude**.

At scoring time the machinery recomputes the aggregate over the declared population minus the
declared exclusions and **refuses to score** if the quoted metric does not equal the
recomputation. The declaration cannot verify intent — nothing can — but it forces the prereg to
name the population an extremum ranges over, and it makes the E1 defect a refusal instead of a
silent pass: a metric quoting the unrestricted minimum cannot equal the eligible-restricted
recomputation when the two differ.

## The exam

**Mutant battery** (constructed results, each a minimal violation): quoted metric equals the
unrestricted aggregate when exclusions exist (the E1 defect); `over` path missing; `over` not a
dict; `excluding` naming members absent from `over`; every member excluded (empty population);
non-finite member values; `agg` other than min/max; a correct declaration that must score
normally; and the **E1 retro-case** — E1's own G1 rewritten with a v4 declaration against the
committed `e1_result.json`, which must REFUSE.

**Backward compatibility is the hard constraint.** Preregs that declare none of the new keys must
score byte-identically. The whole committed corpus is re-scored and any drift in any verdict
string is a failure of this exam, not of the corpus.

```gates
{"gates": {"G0_mutants_refused": {"metric": "frac_violation_mutants_refused", "op": ">=", "value": 1.0,
             "power_basis": "each mutant is a code path the implementation controls completely; anything below 1.0 means a declared composition violation reached a verdict, which is the defect this version exists to prevent",
             "metric_means": "fraction of constructed violation mutants on which score() raised rather than returned a verdict"},
           "G1_valid_still_scores": {"metric": "frac_valid_cases_scored", "op": ">=", "value": 1.0,
             "power_basis": "the valid cases are constructed to satisfy their own declarations exactly, so a correct implementation scores all of them; a refusal here is over-blocking, which silently kills adoption of the mechanic",
             "metric_means": "fraction of correct-declaration cases that scored without refusal"},
           "G2_e1_retro_refused": {"metric": "e1_retro_case_refused", "op": ">=", "value": 1.0,
             "power_basis": "boolean; the committed e1_result.json contains both the unrestricted minimum (0.1436) and the exclusion list, so a v4 declaration over that receipt must refuse -- this is the one case drawn from a real defect rather than constructed",
             "metric_means": "1.0 if E1's G1, rewritten with a v4 composition declaration, refuses against the committed e1_result.json"},
           "G3_corpus_byte_identical": {"metric": "n_corpus_verdict_diffs", "op": "<=", "value": 0,
             "power_basis": "the v2 and v3 upgrades both shipped against this same bar (0 diffs across all committed scoring events) so it is known achievable; any nonzero count means v4 changed the meaning of a frozen document, which is forbidden regardless of how the other gates score",
             "metric_means": "count of committed results whose verdict string differs when re-scored under v4"}},
 "outcomes": [{"when": {"G3_corpus_byte_identical": false}, "verdict": "DO_NOT_SHIP__v4_rewrites_frozen_history"},
              {"when": {"G3_corpus_byte_identical": true, "G0_mutants_refused": false}, "verdict": "DO_NOT_SHIP__a_violation_reached_a_verdict"},
              {"when": {"G3_corpus_byte_identical": true, "G0_mutants_refused": true, "G2_e1_retro_refused": false}, "verdict": "DO_NOT_SHIP__misses_the_real_defect_it_was_built_for"},
              {"when": {"G3_corpus_byte_identical": true, "G0_mutants_refused": true, "G2_e1_retro_refused": true, "G1_valid_still_scores": false}, "verdict": "DO_NOT_SHIP__overblocks_valid_declarations"},
              {"when": {"G3_corpus_byte_identical": true, "G0_mutants_refused": true, "G2_e1_retro_refused": true, "G1_valid_still_scores": true}, "verdict": "PROCEED_TO_RED_TEAM__not_yet_shippable"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## What the winning branch does NOT license

`PROCEED_TO_RED_TEAM__not_yet_shippable` is not a release. P1's lesson is one document old: an
author's own battery encodes the author's misconceptions on both sides of the test, and P1's own
exam fabricated its evidence. The battery above is constructed by the implementer except for one
case; the red team's job is the paths it does not visit. **No release is cut unless an adversary
tries to break v4 and fails**, and the adversary's brief must include the exam itself, not only
the module.

## Stated limits, in advance

The mechanic checks **declared** composition only. A prereg that declares nothing gets no check —
E1 would only have been caught if E1's author had declared `over`/`excluding`, and the author who
mis-composes gates is the same author who may fail to declare. This is a ratchet, not a proof:
it converts "the author thought about the population" from unverifiable prose into a refusable
claim. The undeclared case is the residual, and it is the same residual every layer of this
program has: the adversary who reads the prereg before the run.
