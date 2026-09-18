# RESULT — BENCH-1 is INVALID: its own audit gate caught the benchmark, not the instrument

Fathom Lab · 2026-09-17 · Prereg: `PREREG_bench1_pr_claim_benchmark_2026_09_17.md`. Receipts:
`bench1_oracle.py`, `bench1_score.py`, `bench1_scores.json`, `bench1_dataset.jsonl`,
`bench1_population.json`. The instrument was read-only throughout and is unchanged.

**G-B1-1 failed. 30 of the 50 hand-audited items disagree with the oracle; the gate allowed 2.**
The benchmark is void as preregistered. BENCH-2 is re-frozen below it.

## What went wrong

The oracle took the claim's stated prefix at face value. It never asked whether the thing the
sentence named was a path at all.

The corpus is full of sentences where "only" is followed by an ordinary word:

| the sentence | prefix the oracle used | what it then "found" |
|---|---|---|
| "This is documentation-only changes **with** no code logic modifications" | `with` | 1 path outside `with/` |
| "The only changes **are** the action version bumps described above" | `are` | 2 paths outside `are/` |
| "This PR only changes **one** algorithm file" | `one` | 1 path outside `one/` |
| "Only modified **3** core files in the compiler package" | `3` | 5 paths outside `3/` |
| "it only changes **the** internal evaluation logic" | `the` | 64 paths outside `the/` |

Every one of those was labelled CONTRADICTED. Across the whole class, **279 of 299 decidable
`only_touches` items — 93.3% — carried a prefix that is not a path**, and the oracle called
essentially all of them false. Had this shipped, the benchmark would have announced that 98% of
"only touches" claims in agent PRs are lies. That is not a finding. It is a parser bug with a
percentage sign after it.

## The instrument was right and the benchmark was wrong

On those same 279 items the instrument **abstained on 276**, returned no reading on 2, and accused
on exactly 1. Its `only_touches` reading applies the path-shape test BC-2 added for issue #110 —
"only modifies the footer" is not a path — and withholds when the prefix fails it.

So a naive deterministic oracle, written this afternoon with no such gate, produced a false
discovery rate of roughly 93% on this claim kind. The instrument, carrying one repair made under
preregistration three days ago, produced 1 accusation where the naive rule produced 279.

That is the withholding thesis demonstrated rather than asserted, and it was demonstrated against
our own new code. It is worth more than the passing benchmark would have been.

## What survives

`files_changed_count` is sound and is carried into BENCH-2 unchanged. All 116 decidable items name
a file count in words ("115 files changed, 6,341 insertions(+)"); 17 of them were in the audit
sample and none disagreed; the stated integers are genuine file-count claims. On that kind the
instrument scored precision 1.00, recall 1.00, specificity 1.00 against the oracle, and a
similarity baseline given a threshold chosen *after* seeing the labels reached F1 0.544 with
precision 0.387.

That contrast is the one real measurement BENCH-1 produced, and even it must be read carefully:
the oracle counts `diff --git` headers and the instrument registers one file per header, so the
two agree by construction on well-formed diffs. A perfect score there is evidence that two
implementations of the same rule agree, not evidence of discrimination. The prereg said so before
the run and it is repeated here rather than quietly dropped.

## Reach, as preregistered

G-B1-2 passes: **568 of 691 PRs reachable (82.2%)**, against a floor of 80%. Excluded and listed
by reason: 107 HTTP 403 (repositories renamed or moved since the corpus snapshot, the same class
EXTERNAL-5 hit), 14 served empty, 2 HTTP 404. None silently dropped.

## Deviations

None from the protocol; the protocol is what caught this. The audit gate existed because BIN-1
failed the same way three days ago, and it did its job on the first cycle where it mattered. The
sample seed (20260918), the 50 items and the 30 disagreements are in `bench1_scores.json` and
reproducible from the dataset.

---

*We set out to build the benchmark the field was told it needed, and the first thing it measured
was our own oracle calling two hundred and seventy-nine honest pull requests liars. The
instrument declined all but one of them. The benchmark is void; the reason it is void is the
result.*
