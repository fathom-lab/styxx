# PREREG — OATH v0.14: range-sanity reports instead of accusing — 2026-09-02

**FROZEN before the A/B is run.** Flag: `V14_RANGE_SANITY_REPORT` in `styxx/certify.py`, default
OFF; every committed certificate is unchanged by this cycle. Runner: `range_sanity_report_ab.py`.
Follows `RESULT_handedness_v3_header_handed_2026_09_02.md`: on the external blind panel, thirteen
of thirteen accusations whose only obligation was the v0.3 range-sanity rule were judged not
claims.

## The rule, and what the flag changes

Range-sanity (v0.3) says a value directly after a bounded-quantity word cannot leave that word's
range — *an AUC of 4.0 cannot be* — and so it FORCES an accusation on such a token whatever leaf
it matches, and creates an obligation the token would not otherwise carry. It was written for
this lab's own prose, where *precision 4.0* is a typo, and it killed a coincidence-verification
class there. On foreign READMEs the same words precede version numbers, counts and configuration
values, and the panel says the rule accuses non-claims.

Under the flag the out-of-range condition is still computed and is REPORTED on the ledger entry
(`range_flag: "out-of-range"`, a new field present only under the flag), and the ladder proceeds
as if the rule had not spoken: a matching leaf verifies, a vocabulary obligation with no match
accuses, an unobligated token abstains. The rule stops being an accuser and becomes a reporter,
which is the pattern the v0.13 UNCOVERED band set.

## Gates

```gates
{"gates": {"G_P_external": {"metric": "external_unresolved_repos", "op": "<=", "value": 0,
                            "power_basis": "the rebuilt corpus is byte-pinned and every repository re-certified today; a repository that cannot be certified under both settings is plumbing, and zero is the only honest allowance"},
           "G_REG_internal": {"metric": "internal_held_to_failed", "op": "<=", "value": 0,
                              "power_basis": "the arc's standing ship gate (PREREG_obligate1 G-O1REG): no committed document may move from HELD to FAILED under a verifier change; one is a kill"},
           "G_K_removed_false": {"metric": "removed_genuine_share", "op": "<=", "value": 0.25,
                                 "power_basis": "the flag exists to stop accusing non-claims; if more than a quarter of the accusations it removes were genuine by the panel, it hides real claims and dies; the prior from thirteen of thirteen is 0.0 and is declared as the reason for the bar's existence, not its level"},
           "G_GAIN": {"metric": "false_accusation_rate_gain", "op": ">=", "value": 0.01,
                      "power_basis": "the external false-accusation rate is 0.2596 as an upper bound; a change that does not move it by a point is not worth a verifier version"}},
 "outcomes": [{"when": {"G_P_external": false}, "verdict": "INVALID__external_rebuild_incomplete"},
              {"when": {"G_P_external": true, "G_REG_internal": false}, "verdict": "KILL__internal_regression"},
              {"when": {"G_P_external": true, "G_REG_internal": true, "G_K_removed_false": false}, "verdict": "KILL__hides_genuine_claims"},
              {"when": {"G_P_external": true, "G_REG_internal": true, "G_K_removed_false": true, "G_GAIN": true}, "verdict": "SHIP__range_sanity_reports"},
              {"when": {"G_P_external": true, "G_REG_internal": true, "G_K_removed_false": true, "G_GAIN": false}, "verdict": "NO_GAIN__leave_the_rule"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

`false_accusation_rate_gain` is the 2026-08-27 rate (95 of 366) minus the panel-judged
false-accusation share among the accusations that REMAIN under the flag. Reported, not gated:
every internal token whose status moves and the direction it moves, every internal document
whose verdict class moves in either direction (a FAILED→HELD is adjudicated by hand in the
RESULT), and the status each removed external accusation moves to.

## What SHIP does and does not do

`SHIP` licenses the RESULT to recommend flipping the default and re-issuing the corpus; it does
not flip it in this cycle. The default stays OFF until the operator's release cycle, because a
verifier change that moves any internal token is a corpus re-issue with its own blast radius,
and that is a separate decision. `KILL` and `NO_GAIN` leave the flag in the tree, OFF, as the
record of a rule that was tried.

## Disclosed prior

Strong, and contaminated: the author read the thirteen-of-thirteen before writing this. The bars
are frozen anyway, and the internal regression gate is the one that can still surprise — the
rule was written for this lab's prose, and this lab's prose is where it may earn its keep.

## Discipline

Committed before the runner is run for data. Smoke (`--smoke`: a dozen internal documents, eight
repositories) is INVALID-only. Result → `range_sanity_report_ab_result.json`, scored through
`styxx.protocol`, RESULT sworn to the receipt. No bar moves. No committed certificate changes.
