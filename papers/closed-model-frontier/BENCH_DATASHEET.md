# Datasheet — the styxx PR-claim dataset

**Both benchmarks that produced this data were declared INVALID by their own audit gates.** Read
that first; it is the most important thing on this page. `RESULT_bench1_INVALID_2026_09_17.md` and
`RESULT_bench2_INVALID_2026_09_17.md` say exactly how and why. What follows is data, not a
validated benchmark, and the ground-truth labels carry a known and measured error rate.

## What this is

604 claims extracted from 568 agent-authored pull requests, drawn from the 691 PRs in the AIDev
corpus (HuggingFace `hao-li/AIDev`, CC-BY-4.0, Zenodo 10.5281/zenodo.16919272) that carry at least
one mechanically decisive claim. One row per claim:

| field | what it is |
|---|---|
| `url`, `pr_id`, `agent` | the pull request and which coding agent authored it |
| `kind` | `files_changed_count`, `only_touches`, `symbol_added`, `tests_added` |
| `claim_text`, `claim_detail` | the sentence, and what was parsed out of it |
| `diff_sha256`, `diff_bytes` | the live diff GitHub served, so you can verify you have what we read |
| `truth`, `truth_facts` | `bench2_oracle.py`'s label and the facts behind it |

The diffs are not redistributed. `python bench_reproduce.py --fetch` re-fetches every one and
checks it against the published `diff_sha256`; a mismatch means the PR changed after we read it,
or we are wrong, and either is worth knowing.

## What is deliberately absent

**styxx's own verdict on each pull request.** We do not publish accusations against third parties'
pull requests, and there is a second reason here: hand adjudication of every accusation the
instrument made on this corpus found **9 of 11 were wrong**. Ours are not worth taking on trust.

`python bench_reproduce.py --score styxx` regenerates them locally in one command, and
`--score yourmodule:yourfunction` scores anyone else's checker on the same rows. We would rather
you did that than believe us.

The preregistration promised the oracle's label, not the instrument's verdict, so nothing was
withheld that was promised. `bench1_scores.json`, `bench2_scores.json` and `bench2_audit.json`
carry the aggregate numbers, including all the ones that make us look bad.

## Known limits, stated rather than buried

- **The oracle is wrong at a measured rate.** BENCH-2's audit found 3 of the first 7 non-trivial
  sampled items misjudged. BENCH-1's found 30 of 50. Treat `truth` as a strong prior, not a fact.
- **`tests_added` is labelled `EXCLUDED`.** Deciding what counts as "a test" is a judgement and an
  oracle must not carry one. The rows are present; they are not scored.
- **`only_touches` is largely undecidable by this oracle** — 282 of 299 items. DECIDE-1
  (`RESULT_decide1_decidable_fraction_2026_09_17.md`) found by hand that about **52%** of those
  claims are in fact decidable, so most of that 282 is our extraction failing, not the corpus being
  ambiguous.
- **`files_changed_count` agrees with the instrument by construction.** Both count `diff --git`
  headers. A high score there measures agreement between two implementations of one rule.
- **107 of the 691 PRs were unreachable** (renamed or moved repositories), 14 served empty, 2 were
  404. Reach 568/691 = 82.2%, counted and listed rather than dropped.

## Licence and attribution

CC-BY-4.0, inherited from AIDev. If you use this, cite AIDev (Zenodo 10.5281/zenodo.16919272).
The MSR '26 artifact that measured this problem released 974 human-annotated PRs in an unlicensed
repository; closing that gap was half the point of building this, and it is the half that survived.

## Citing the failure

If you cite the numbers, cite the invalidations with them. A benchmark that voided itself twice
and then published the reasons is a different object from one that reported a score.
