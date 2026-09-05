# DESIGN — the provenance law: who handed the verifier its target bounds what a verdict can mean, and a test that can kill it

Fathom Lab · 2026-09-02 · **A design, not a preregistration.** Leg 4 of
`papers/PLAN_the_next_level_2026_09_02.md`. It carries a proposed gates block in the
`styxx.protocol` grammar so that signing it is a rename, not a rewrite; until the operator signs
the bars it licenses nothing and scores nothing. Its contaminated prior is committed and cited by
content address, which is the only way this lane is allowed to cite a number it computed after
reading the result it now wants to test.

## The statement

For a verifier adjudicating targets in an author's output, the odds that a handed target is a
claim at all — the extraction term — factor into a FORM term (what the token looks like) and a
PROVENANCE term that is monotone down a ladder: a target the author bound at write time in a form
it cannot disown (sworn output) ≥ a label the author committed to (a table header, an `n=`
register) ≥ a word that co-occurred on the author's line ≥ a rule the verifier wrote outside the
author's idiom. **The provenance term does not depend on form.** If that is true, the extraction
ceiling of any free-prose claim verifier is predictable from its target-provenance mix before a
panel is convened, and cannot be raised by a better adjudicator — only by moving the target up the
ladder, which is what a format the author commits to does.

## The receipts it stands on, and the one it must escape

On the blind panel of late August, <sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/header/genuine_share" k="numeric">header-handed accusations were genuine at 0.9515</sworn>
<sworn r="path:papers/closed-model-frontier/handedness_v3_result.json#/cells/line/genuine_share" k="numeric">against 0.6391 for line-handed ones</sworn>,
one family of seats, one corpus. The exploratory re-cut by token kind
(`EXPLORATORY_handedness_by_kind_2026_09_02.md`, receipt blob
`a5975a92dcb14cc7e858d37a701f08b7ec8c31e1d0e3cf5b2b42f810b6b50b7b`) is the contaminated prior:
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/header~1decimal/genuine_share" k="numeric">decimals handed by a header were genuine at 1.0</sworn>
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/line~1decimal/genuine_share" k="numeric">and by a line at 0.9605</sworn>,
so form carries almost all of the claimhood there;
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/header~1integer/genuine_share" k="numeric">integers handed by a header were genuine at 0.6522</sworn>
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/line~1integer/genuine_share" k="numeric">and by a line at 0.3763</sworn>,
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/odds_ratio_header_over_line/integer_stratum" k="numeric">an odds ratio of 3.1071</sworn>
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/cells/header~1integer/n" k="numeric">on a header cell of 23 rows</sworn>,
<sworn r="path:papers/closed-model-frontier/handedness_v3_by_kind_result.json#/top_repository/rows" k="numeric">with one repository supplying 184 rows</sworn>
of the join. That is the shape an additive-log-odds law predicts: a provenance factor invisible at
the decimal ceiling and decisive where form carries no information. It is also exactly the shape
a table-row leak, one repository and one model family would produce. The design below is built so
those two readings come apart.

## What the experiment does

**Corpus — fresh, and disjoint by construction.** GitHub READMEs and model cards selected by the
query families and caps of `PROTOCOL_oath_external_corpus_2026_08_27.md`, re-run with a new
per-query cap sized to reach about 200 repositories, with every repository named in
`oath_external_corpus.json` or in `oath_external_recertify_summary.json` excluded before any
document is read. Each file is pinned by commit sha and sha256 in a manifest committed BEFORE the
verifier runs (the `oath_external_recertify.py` cache-and-verify pattern). Buckets follow
`SPLIT_external_corpus_2026_08_31.md`. Third-party bytes are cached outside the tree and never
committed.

**Verifier — pinned.** `styxx.certify` at the release the preregistration names, with the
range-sanity flag at its committed default, so every accused token carries the verifier's own
`obligation_source` and `header_bound`. Nothing is re-derived from a ledger.

**Population — every accusation.** Every UNGROUNDED token, no sampling, plus abstained and
verified decoys salted at the 2026-08-27 rule, plus 30 sealed reliability decoys built two-sided
(15 checkable claims, 15 not) from the DEVELOPMENT bucket by declared frames, hashed and
committed before any seat runs.

**Panel — two families, each gated alone.** Family A: Claude through the `claude -p` clean-config
subscription transport (`run_b23_fable.py`). Family B: a local open-weight instruct model on CPU
(Qwen2.5-7B-Instruct at bf16; 3B if RAM refuses; both are in this machine's cache). Three seats
per packet per family, majority within family, the frozen 2026-08-27 seat question verbatim, seats
shown the token and its line only — never the cell, never the verifier's verdict, never this
hypothesis. Key sealed outside the repository, salted digest committed first. Disclosed leak,
unchanged from 2026-08-27: a table row is visibly a table row even without its header. A
follow-up that strips table formatting from the presented line is owed, not designed here.

**Cells.** header × {integer, decimal}, line × {integer, decimal}, rule-handed (report-only,
Wilson), `n=`-glued (report-only; empty in v3 and expected sparse). Every cell reported with and
without its largest repository.

**The three tests.** (1) Within integers, the odds ratio of header over line, one-sided Fisher.
(2) Within decimals, no reversal. (3) Additivity: apply the integer-stratum odds ratio to the
decimal line rate and ask whether the observed decimal header non-claim count falls inside the
predictive interval — the check that separates *provenance is a form-independent factor* from
*structure matters only for integers*.

## Proposed gates — frozen only when signed and renamed PREREG

```gates
{"gates": {"G_F_fresh": {"metric": "manifest_hash_verified_share", "op": ">=", "value": 1.0,
                         "power_basis": "every fetched file must hash-match the committed manifest; one mismatch is a corpus that was not frozen"},
           "G_F_disjoint": {"metric": "repos_overlapping_prior_corpora", "op": "<=", "value": 0,
                            "power_basis": "the 140 and 82 repositories already judged are the contaminated prior's own corpus; one overlap is a re-test, not a test"},
           "G_R_a": {"metric": "decoys_family_a_min_side", "op": ">=", "value": 9,
                     "power_basis": "the extraction-v2 two-sided gate: a family stuck on one label scores 15/0 and fails the other side"},
           "G_R_b": {"metric": "decoys_family_b_min_side", "op": ">=", "value": 9,
                     "power_basis": "as G_R_a; the second family is the load-bearing novelty of this panel and is gated alone"},
           "G_R_overall": {"metric": "decoys_min_family_overall", "op": ">=", "value": 27,
                           "power_basis": "the 2026-08-31 reliability bar, applied to the weaker family so the panel is only as reliable as its weakest member"},
           "G_N_cells": {"metric": "min_integer_cell_n", "op": ">=", "value": 30,
                         "power_basis": "the v3 header-integer cell was 23 rows; at 30 a proportion's standard error is about 0.09 and an odds ratio of 2 is distinguishable from 1 at the one-sided 0.05 level about half the time, which is the floor this lane accepts for a law's first test"},
           "G_L1_or": {"metric": "integer_or_header_over_line", "op": ">=", "value": 2.0,
                       "power_basis": "the contaminated prior is 3.11 on 23 rows; 2.0 is the value at which a form-free provenance factor would still halve a verifier's false-claim odds, and is the smallest effect this lab would act on"},
           "G_L1_p": {"metric": "integer_fisher_p_one_sided", "op": "<", "value": 0.05,
                      "power_basis": "one-sided because the direction was declared here; no correction across the three tests, which are three named questions and not a family"},
           "G_L1_refuted": {"metric": "integer_or_header_over_line", "op": "<=", "value": 1.2,
                            "power_basis": "an odds ratio under 1.2 is indistinguishable from the table-row leak alone; the law as stated is dead at that value"},
           "G_L2_no_reversal": {"metric": "decimal_or_header_over_line", "op": ">=", "value": 1.0,
                                "power_basis": "a form-independent factor cannot point the other way in the other stratum; a reversal refutes the law as stated"},
           "G_L3_additive": {"metric": "decimal_header_nonclaims_in_predictive_interval", "op": ">=", "value": 1,
                             "power_basis": "1 when the observed decimal header non-claim count lies inside the 95% predictive interval implied by the integer odds ratio and the decimal line rate, else 0; outside in the direction of an interaction is the weaker form-conditional law, named as its own outcome"},
           "G_C_concentration": {"metric": "verdict_flips_without_top_repository", "op": "<=", "value": 0,
                                 "power_basis": "the v3 join had one repository in more than half its rows; a verdict that depends on one repository is a verdict about that repository"}},
 "outcomes": [{"when": {"G_F_fresh": false}, "verdict": "INVALID__corpus_not_frozen"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": false}, "verdict": "INVALID__corpus_not_fresh"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_a": false, "G_R_b": false}, "verdict": "INVALID__panel_unreliable"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": false}, "verdict": "INVALID__panel_unreliable"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_a": false}, "verdict": "UNLICENSED__one_family_cleared"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_b": false}, "verdict": "UNLICENSED__one_family_cleared"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_a": true, "G_R_b": true, "G_N_cells": false}, "verdict": "INVALID__underpowered_cell"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_a": true, "G_R_b": true, "G_N_cells": true, "G_C_concentration": false}, "verdict": "INVALID__one_repository"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_a": true, "G_R_b": true, "G_N_cells": true, "G_C_concentration": true, "G_L1_refuted": true}, "verdict": "LAW_REFUTED__grain_was_form"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_a": true, "G_R_b": true, "G_N_cells": true, "G_C_concentration": true, "G_L1_refuted": false, "G_L2_no_reversal": false}, "verdict": "LAW_REFUTED__form_reverses_provenance"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_a": true, "G_R_b": true, "G_N_cells": true, "G_C_concentration": true, "G_L1_refuted": false, "G_L2_no_reversal": true, "G_L1_or": true, "G_L1_p": true, "G_L3_additive": true}, "verdict": "PROVENANCE_HOLDS_WITHIN_FORM"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_a": true, "G_R_b": true, "G_N_cells": true, "G_C_concentration": true, "G_L1_refuted": false, "G_L2_no_reversal": true, "G_L1_or": true, "G_L1_p": true, "G_L3_additive": false}, "verdict": "LAW_AMENDED__interaction_form_conditional"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_a": true, "G_R_b": true, "G_N_cells": true, "G_C_concentration": true, "G_L1_refuted": false, "G_L2_no_reversal": true, "G_L1_or": false}, "verdict": "INDETERMINATE__between_bars"},
              {"when": {"G_F_fresh": true, "G_F_disjoint": true, "G_R_overall": true, "G_R_a": true, "G_R_b": true, "G_N_cells": true, "G_C_concentration": true, "G_L1_refuted": false, "G_L2_no_reversal": true, "G_L1_or": true, "G_L1_p": false}, "verdict": "INDETERMINATE__direction_without_significance"}],
 "smoke_verdict": "INVALID__smoke_plumbing_only"}
```

Reported, never gated: every cell's genuine share with its Wilson interval, with and without its
largest repository; cross-family agreement on non-decoy items (the reliability figure, since two
families of one kind are still two families of one kind); the rule-handed and `n=`-glued cells;
and the share of accusations the pinned verifier no longer makes.

## What it would mean, and what it would not

If `PROVENANCE_HOLDS_WITHIN_FORM`: a measured, falsifiable regularity about measurement itself —
the provenance of the target, not the skill of the judge, bounds the extraction term of a
free-prose verifier — tested on a corpus the lab did not choose, by two families of seats, with
the table-row leak disclosed and the dominant repository removed. The claim that follows is
narrow: it holds for numeric tokens, one verifier, README and model-card prose, and two LLM
families. Whether it extends past numerals, past `styxx.certify`, or past that idiom is owed and
named. If `LAW_REFUTED`: M2's grain loses its mechanism in public, the grain synthesis and the
h-mapping's `grain` column gain a back-pointer naming the killer, and the lab publishes that under
its own title. Either is a result about verification; neither licenses an accusing verdict
anywhere, and the range-sanity, path-claim and evidence branches stay exactly where their own
RESULTs left them.

## What blocks it

A signature on the bars. A corpus fetch (network; one afternoon; the collector exists). The panel
machinery leg E of the plan is building. Nothing needs credits.

---

*The synthesis said structure hands a claim and co-occurrence hands a coin, and the same day the
digit under it turned out to be mostly the kind of token. This is the experiment that decides
whether anything is left once the kind is held fixed — and it is written so that "nothing" is a
verdict the table can return.*
