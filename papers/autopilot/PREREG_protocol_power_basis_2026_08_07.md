# PREREG — protocol v2: gates must declare their power basis, and the machinery records when they do not

Fathom Lab · 2026-08-07 · frozen before implementation. `styxx.protocol` is load-bearing for
every finding in this repo and is treated like `certify`: no change without a frozen spec and a
regression battery.

## Why

Three gate bars this week were set without checking whether any instrument could clear them —
b37 G2 (a noise-passable floor), b48 G2 (a max-of-45 statistic judged against a single-draw
bar), C5 G1 (80% of individual pairs at effective sample sizes of 6.9–45.8). Each was written
up, each named the failure mode, and the b48 finding states outright that *naming a failure mode
does not immunize you against it*. It did not. Every other failure class in this program is
refused by machinery; this one still runs on intent, and intent is 0-for-3.

## The change (frozen)

1. A gate MAY carry `"power_basis"`: a string stating how its bar was derived, or the literal
   `"none — exploratory"`.
2. `Experiment.score()` records `power_basis` per gate on the `Verdict` and exposes
   `undeclared_power_gates` — the names of gates with no declaration. **Verdict strings are
   unchanged**, so the 33 existing sealed findings keep verifying byte-identically.
3. `Experiment(..., require_power_basis=True)` raises `GateSpecError` naming every undeclared
   gate. Default is `False` for backward compatibility; **this lab adopts `True` for all new
   preregs**, which is a process commitment, not a code default.
4. `undeclared_power_gates(path)` is exposed as a module function so the whole corpus can be
   audited for which preregs predate the rule.

## Gates

```gates
{"gates": {"G1_backward_compatible": {"metric": "existing_preregs_still_score", "op": ">=", "value": 33,
             "power_basis": "count of gates-bearing preregs in the repo at freeze time; the bar IS the census, so achievable by construction and falsifiable by any regression"},
           "G2_strict_mode_refuses": {"metric": "strict_mode_refusals", "op": ">=", "value": 1,
             "power_basis": "a single constructed undeclared prereg must raise; one instance is sufficient to demonstrate the mechanism, more would not add information"},
           "G3_declared_passes_strict": {"metric": "declared_prereg_scores_under_strict", "op": ">=", "value": 1,
             "power_basis": "this document declares power_basis on every gate, so it is its own positive control"},
           "G4_verdict_strings_unchanged": {"metric": "verdict_string_diffs", "op": "<=", "value": 0,
             "power_basis": "exact string comparison across all existing preregs; zero is the only acceptable value because any diff breaks a committed seal"}},
 "outcomes": [{"when": {"G1_backward_compatible": false}, "verdict": "INVALID__breaks_existing_preregs"},
              {"when": {"G1_backward_compatible": true, "G4_verdict_strings_unchanged": false}, "verdict": "INVALID__changes_committed_verdicts"},
              {"when": {"G1_backward_compatible": true, "G4_verdict_strings_unchanged": true, "G2_strict_mode_refuses": false}, "verdict": "INVALID__strict_mode_does_not_refuse"},
              {"when": {"G1_backward_compatible": true, "G4_verdict_strings_unchanged": true, "G2_strict_mode_refuses": true, "G3_declared_passes_strict": false}, "verdict": "INVALID__strict_mode_refuses_a_declared_prereg"},
              {"when": {"G1_backward_compatible": true, "G2_strict_mode_refuses": true, "G3_declared_passes_strict": true, "G4_verdict_strings_unchanged": true}, "verdict": "POWER_BASIS_LANDED__machinery_now_records_what_intent_could_not"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

## Stated before the run

This does not make bars correct. It makes an undeclared bar **visible and countable**, and makes
the declaration a thing a reader can check against the finding. A gate declaring
`"none — exploratory"` is honest; a gate declaring nothing is now a fact in the receipt.
Red-teamed before release, per the standing rule.
