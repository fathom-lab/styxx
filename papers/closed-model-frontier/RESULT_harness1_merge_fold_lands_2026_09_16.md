# RESULT — HARNESS-1 lands: the corpus reconstruction stops folding merge traffic; 16 accusations and 247 removed-name readings were the harness's

Fathom Lab · 2026-09-16 · Prereg: `PREREG_harness1_merge_fold_2026_09_16.md` (frozen on #122
before this run). Receipts: `external6_harness.py` (the re-fold), `external6_gate_summary.json`,
`external6_summary.json` (the census on the re-folded ledger), `harness1_gates.py`,
`harness1_gates.json`. Instrument: `styxx/diffgate.py` sha256 `397624d5…`, untouched. Counts only;
no PR named.

## The fold, in numbers

The shelf now carries commit identity. Across the corpus's 1,775,765 file rows there are 220,858
commits; **13,916** of them have a message beginning `Merge `, and **1,262** sit at the dataset's
300-row per-commit cap. Of the 71,016 eligible PRs, **7,443** carry a merge commit (12 carry
nothing else and keep their rows, as the prereg said), **953** are capped, and the re-fold drops
612,406 rows, changing the reconstructed file set on **6,697** PRs. Eligibility is unchanged:
the same 71,016 PRs, the same three exclusions, the same 17,939 covered.

## The gates

| gate | result |
|---|---|
| G-H1 the 85 live PRs: ≥14 more file-count items agree with the live count; none stop agreeing | 70 items: **28 → 42** agree, **14 gained, 0 lost** — PASS, at the threshold exactly (upheld: 28 kept, 6 gained, 18 still apart; overturned: 8 gained, 10 still apart; the 13 `only_touches` items now all match the live verdict, 12 did before) |
| G-H2 accusations can only fall where the fold could only inflate | `only_touches` CONTRADICTED **14 → 13** (0 gained); compat readings naming a removed public definition **540 → 293** (252 stopped, 5 started); `files_changed_count` CONTRADICTED **74 → 60** keys, 0 gained, 14 lost (10 now VERIFIED, 4 UNCHECKABLE by the cap); by direction, corpus above the claim 64 → 51, below 11 → 10 — PASS |
| G-H3 BC-2's invariants | by-construction counters **0 / 0 / 0**; on the 63,272 PRs with neither a merge commit nor a cap, 207 `tests_added` / `symbol_added` verdicts checked, **0** changed — PASS |
| G-H4 the numbers that get re-quoted | below, before and after |
| G-H5 what is not claimed | no precision number; **27 of the 66** live PRs with a file-count item still disagree with the live count after the re-fold |

Every claim list is identical on all 71,016 PRs (the claims are the descriptions'); only verdicts
and reasons moved, and every movement is listed in `harness1_gates.json`: `files_changed_count`
10 CONTRADICTED → VERIFIED and 4 CONTRADICTED → UNCHECKABLE (the cap); `only_touches` 1
CONTRADICTED → VERIFIED; `tests_added` 1 CONTRADICTED → UNCHECKABLE (the diff has no Python
without the merge traffic, as EXTERNAL-5 found at the source); `file_touched` 30, `file_deleted`
6 and `file_created` 2 VERIFIED → UNCHECKABLE (a file the description names that only the merge
traffic touched — the instrument withholds, V14); `file_deleted` 1 UNCHECKABLE → VERIFIED. Not one
accusation was gained.

## G-H4 — the published numbers, corrected in place

| figure | as published | after the re-fold |
|---|---|---|
| EXTERNAL-2 / BC-2 accusations on 71,016 PRs (census, claims) | **96** — `files_changed_count` 75, `only_touches` 14, `tests_added` 7 | **80** — 61, 13, 6 |
| PRs carrying a contradiction | 91 | 75 |
| COMPAT-1: PRs whose diff drops a public name (of 8,467 compat claims, all UNCHECKABLE) | **540** — js/ts 243, python 164, go 79, java 40, rust 34 | **293** — js/ts 111, python 85, go 50, java 32, rust 21 |
| claims by verdict | VERIFIED 13,017 · UNCHECKABLE 22,806 · CONTRADICTED 96 | 12,991 · 22,848 · 80 |
| coverage | 17,939 of 71,016 | unchanged |

The COMPAT-1 figure is the one that moves most: **247 of the 540** PRs the compat reading listed as
dropping a public definition were dropping it in merge traffic, not in the PR. The reading was
never an accusation (COMPAT-1's one verdict is UNCHECKABLE), but the number was quoted, and the
corrected number is 293.

## On the live set, by cause

Of the 18 overturned file counts EXTERNAL-5 found, **8 now agree** with the live PR and their
corpus verdicts flip to VERIFIED; **3 sit on capped commits** and read UNCHECKABLE with the cap's
reason; **7 still disagree** — the PR is a different object today than in the dataset, or a merge
the message rule does not see. Of the 52 upheld file counts, 6 more now agree with the live count
(one of the 28 that already agreed sits on a capped commit and now reads UNCHECKABLE), and 2
flip to VERIFIED on the corpus while the live PR still contradicts the description: the description was true at the dataset's snapshot and false after a later push —
the case no diff can tell from a lie, which WH-1 has to name.

## Deviations

None from the prereg. G-H1 passed at its threshold, not above it: the correction's own check had
counted 6 and 8 flips and the gate asked for 14, and 14 is what the re-fold delivered. The
`external6_summary.json` the prereg named is the census's output (`external2_census.py --ledger
external6`); the harness's own summary is `external6_gate_summary.json`, EXTERNAL-3's convention.

---

*The corpus said 340 files where the pull request had 3, and the instrument accused on the
corpus's word. The harness has stopped saying it; the sixteen accusations and the two hundred and
forty-seven removed names it had manufactured are withdrawn here, with the two numbers side by
side, and the thirty-eight files it had wrongly verified are withheld with them.*
