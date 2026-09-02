# PREREG — token-level h, v2: is an accusation handed by a table header truer than one handed by the line? — 2026-09-02

**FROZEN before the join is computed.** Runner: `handedness_accusations_v2.py`, over
`oath_external_epistemics_ledger.jsonl`, which `oath_external_recertify.py` writes from the pinned
corpus re-certified at the current verifier — the verifier's own `obligation_source` per token, no
re-derivation. Supersedes `PREREG_handedness_accusations_2026_09_02.md`, whose re-derivation
diverged on half the accusations (`RESULT_handedness_accusations_INVALID_2026_09_02.md`).

## What v1 could not test, and what this corpus can

v1's hypothesis compared object_form (printed precision) with object_text. On this corpus no
accused token printed seven fractional digits: the object_form cell is empty and no panel drawn
from these accusations can fill it. The question the corpus *can* hold is inside object_text.
The verifier obligates a markdown table cell through its column header (v0.3 `binding_context`),
so an accusation can be handed to it by a header word — a structured, named column such as
*Accuracy* — or by a trigger word in the cell's own line of prose. Both are the object's text;
one is a label the author chose for a column of numbers, the other is a word that happened to be
on the line.

## Hypothesis

**H_T1:** among panel-adjudicated accusations obligated by vocabulary, those obligated through a
table header (the line itself carries no trigger word) are genuine claims more often than those
obligated by the line's own trigger word. A column header names what the numbers under it are; a
trigger word in prose is the co-occurrence M1 warns about. Direction frozen: header higher.

**Disclosed, contaminated prior.** The INVALID v1 run showed, exploratorily, a genuine share of
0.95 on the tokens its re-derivation could not name and 0.64 on line-vocabulary tokens. The author
has seen those numbers; they are the reason this hypothesis is the one being frozen, and the bar
below is frozen anyway. A gate cannot be un-seen; it can be declared.

## The join, exactly

Each of the 366 accusations in `oath_adjudication_result.json` is joined to the re-certified
ledger by `(repo, line, token)`. A token that joins to no row, or to more than one, is unresolved
(plumbing). The token's cell is read from the verifier's own record: `obligation_source ==
"vocabulary"` and `header_bound` (binding context carries a trigger, the line does not) is the
HEADER cell; `obligation_source == "vocabulary"` and not `header_bound` is the LINE cell; every
other source is reported outside the test. The current verifier's status for the token is
recorded but not used: the panel judged the 2026-08-27 accusation, and a token the current
verifier now rescues is still the accusation the panel saw.

## Gates

```gates
{"gates": {"G_P_join": {"metric": "unresolved_share", "op": "<=", "value": 0.05,
                        "power_basis": "the documents are byte-pinned and the verifier is deterministic, so a token the panel saw must exist in the re-certified ledger at the same line and token; 5% is the allowance for the verifier's own extraction having moved since 2026-08-27"},
           "G_N_cells": {"metric": "min_cell_n", "op": ">=", "value": 20,
                         "power_basis": "the join RESULT's smallest published cell was n=23; below 20 a proportion's standard error exceeds 0.11"},
           "G_T1_header_truer": {"metric": "delta_header_minus_line", "op": ">=", "value": 0.15,
                                 "power_basis": "a third of the join RESULT's 0.41 gap, above one standard error at the cell floor; direction, not digits; no significance claimed at any n"},
           "G_T1_line_truer": {"metric": "delta_line_minus_header", "op": ">=", "value": 0.15,
                               "power_basis": "the same bar reversed, so a reversal is a named outcome"}},
 "outcomes": [{"when": {"G_P_join": false}, "verdict": "INVALID__join_incomplete"},
              {"when": {"G_P_join": true, "G_N_cells": false}, "verdict": "INVALID__underpowered_cell"},
              {"when": {"G_P_join": true, "G_N_cells": true, "G_T1_header_truer": true}, "verdict": "HEADER_HANDED_ACCUSES_TRUER"},
              {"when": {"G_P_join": true, "G_N_cells": true, "G_T1_header_truer": false, "G_T1_line_truer": true}, "verdict": "LINE_HANDED_ACCUSES_TRUER"},
              {"when": {"G_P_join": true, "G_N_cells": true, "G_T1_header_truer": false, "G_T1_line_truer": false}, "verdict": "NO_SEPARATION_HEADER_VS_LINE"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Reported, not gated: the genuine share and Wilson interval of every source (`n-glued`,
`range-correlation`, `precision`, `range-sanity`), the share of accusations the current verifier no
longer accuses, per-repository concentration of each cell (one repository holds 194 of the 366
accusations and its weight is printed, not hidden), and the same split over the join receipt's
obligated VERIFIED tokens.

## Discipline

Committed before `handedness_accusations_v2.py` is run for data. The re-certification script is
plumbing and may run first; it computes no cell. Result → `handedness_v2_result.json`, scored
through `styxx.protocol`, RESULT sworn to the receipt. No bar moves. No token is re-judged.
