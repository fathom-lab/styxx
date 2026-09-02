# PREREG — token-level h on the blind panel: does who handed the verifier its target predict whether the accusation was true? — 2026-09-02

**FROZEN before the sources are computed.** Runner: `handedness_accusations.py`. No new
adjudication: the 366 panel-judged accusations of `RESULT_oath_external_corpus_2026_08_27.md`
(`oath_adjudication_result.json`, three seats, decoy-gated) are joined to the obligation source
the verifier's clause order assigns each token, re-derived from the corpus ledger's line context
and its recorded trigger words. This is the token-level test the handedness declaration
(`DECLARATION_h_mapping_2026_09_01.md`) said must be kept apart from the instrument-level one.

## Hypothesis (formulated here; the commissioning brief's own wording was not reachable)

**H_A1, as formulated here:** among panel-adjudicated accusations, those whose obligation came from
the *object's form* (the token's own printed precision) are more often genuine claims than those
whose obligation came from the *object's text* (a trigger word on the line, `n=` glued to the
token, a correlation word with a bounded value, or a bounded-quantity word before an out-of-range
value). Rationale: a number printed at seven or more fractional digits was copied out of a
computation; a trigger word on a line fires on configuration, file names and prose about other
things. The direction is frozen: object_form higher.

## The re-derivation, exactly

For each accused token, in the verifier's clause order (`styxx/certify.py`, first-writer):

1. **vocabulary** — the ledger's `obligating_words` is non-empty (the harness recorded
   `_TRIGGERS` matches on the full line).
2. **n-glued** — `\bn\s*=\s*$` matches the eighteen characters before the token in the recorded
   context (the token located by its recorded column, adjusted for the context's stripped
   leading whitespace, falling back to first occurrence; the fallback count is reported).
3. **range-correlation** — `_TRIGGERS_CORR` matches the context, the token has at least one
   fractional digit, and its value lies in [−1, 1].
4. **precision** — the token prints at least `V07_PRECISION_DIGITS` (7) fractional digits.
5. **range-sanity** — a bounded-quantity keyword directly precedes the token and its value leaves
   the keyword's range, with the v0.10 slash-pair guard (the verifier's own regexes, copied
   verbatim and cited by line).

A token none of these reach is **unknown** and counts against the plumbing gate: an accusation
requires an obligation, so an unclassifiable accusation means the re-derivation diverged from the
verifier. Sources map to classes by `h_mapping.json`: precision → object_form; the other four →
object_text. Panel verdicts: CLAIM is a genuine accusation, NOT_A_CLAIM a false one; the receipt
holds no UNSURE majorities.

## Gates

```gates
{"gates": {"G_P_plumbing": {"metric": "unknown_or_ambiguous_share", "op": "<=", "value": 0.05,
                            "power_basis": "the re-derivation must reach 95% of accused tokens by a named clause; the verifier's own drop rules (dates, versions, shas) are applied before extraction, so the residue should be near zero and 5% is the allowance for context truncation at 200 characters"},
           "G_N_cells": {"metric": "min_cell_n", "op": ">=", "value": 20,
                         "power_basis": "the join RESULT's smallest published cell was n=23; below 20 a proportion's standard error exceeds 0.11 and no difference the bar names can be read"},
           "G_H1_form_truer": {"metric": "delta_form_minus_text", "op": ">=", "value": 0.15,
                               "power_basis": "the join RESULT read 0.78 vs 0.37 as a finding; a third of that gap, above one standard error at the cell floor, is the smallest difference this lane has treated as a direction rather than noise; no significance is claimed at any n"},
           "G_H1_text_truer": {"metric": "delta_text_minus_form", "op": ">=", "value": 0.15,
                               "power_basis": "the same bar in the opposite direction, so a reversal is a named outcome and not an absence"}},
 "outcomes": [{"when": {"G_P_plumbing": false}, "verdict": "INVALID__rederivation_diverged"},
              {"when": {"G_P_plumbing": true, "G_N_cells": false}, "verdict": "INVALID__underpowered_cell"},
              {"when": {"G_P_plumbing": true, "G_N_cells": true, "G_H1_form_truer": true}, "verdict": "HANDED_BY_FORM_ACCUSES_TRUER"},
              {"when": {"G_P_plumbing": true, "G_N_cells": true, "G_H1_form_truer": false, "G_H1_text_truer": true}, "verdict": "HANDED_BY_TEXT_ACCUSES_TRUER"},
              {"when": {"G_P_plumbing": true, "G_N_cells": true, "G_H1_form_truer": false, "G_H1_text_truer": false}, "verdict": "NO_SEPARATION_BY_SOURCE"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Reported, not gated: the genuine-claim share per source (all five), Wilson 95% intervals for the
two classes, the fallback-location count, the per-repository concentration of each cell, and the
same split over the 23 obligated VERIFIED tokens of the join receipt.

## Disclosed limits

One panel, three seats of one model family, correlated error as the ceiling (as both source
documents disclose). The context is the harness's 200-character stripped line, not the document,
so the previous-line rule for tokens at line start cannot be applied and is counted as a fallback.
Cells will be unequal: vocabulary dominates obligation everywhere in this corpus, so object_form
is the small cell and the power gate is where this study most likely fails. If it does, that is
the finding: the panel data cannot see the object_form cell, and a panel drawn by source is owed.

## Discipline

Committed before the runner is executed. Result → `handedness_accusations_result.json`, scored
through `styxx.protocol`, RESULT sworn to the receipt. No bar moves. No token is re-judged.
