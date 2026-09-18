# PREREG — DECIDE-1: what fraction of agent-PR claims can the diff actually settle?

Fathom Lab · 2026-09-17 · Frozen before a single item is adjudicated. The instrument is not
involved in this measurement at all and is not changed by it (`styxx/diffgate.py`, sha256
`4ba947a8…`). Read with `RESULT_bench1_INVALID_2026_09_17.md` and
`RESULT_bench2_INVALID_2026_09_17.md`, which are why this exists.

## Why this exists, and why it must not use an oracle

Two mechanical oracles were written a day apart and both failed their own blocking audits. BENCH-1
read `the` as a directory. BENCH-2, after two admissibility repairs, still mislabelled 9 of the 11
items it was most confident about. The residual errors were sentences where no surface feature
separates a file-scope claim from a statement about program behaviour.

The obvious next question is not "how do we build a third oracle". It is: **how much of this
corpus is decidable from the artifact at all?** That question is worth more than the benchmark
was, and nobody has published a number for it.

It also cannot be answered by an oracle, because deciding whether a claim is decidable is exactly
the judgement that broke both of them. **DECIDE-1 therefore uses no oracle.** Ground truth is
hand adjudication against the live diff, one item at a time, under the rubric below. This is slow
and it caps the sample size, and that is the correct trade.

## Population and sample

The frozen BENCH-1 population: 691 PRs drawn from the 71,016 eligible AIDev PRs (HuggingFace
`hao-li/AIDev`, CC-BY-4.0, Zenodo 10.5281/zenodo.16919272). 568 PRs reached, carrying 604 claims:
`only_touches` 299, `tests_added` 151, `files_changed_count` 116, `symbol_added` 38. The same
fetched bytes are reused, with their recorded sha256s — nothing is re-fetched, so the population
cannot drift.

**Stratified random sample, 25 per kind, 100 items**, seed **20260920**. `symbol_added` has only 38
items, so 25 is a 66% census of it and its interval will be narrow for that reason rather than
because the estimate is better. Sampling within a kind is uniform.

25 per kind is what one careful adjudication pass affords. It yields roughly ±16 percentage points
at 95% per kind and about ±8 pooled. **Those intervals are wide, they will be reported as wide,
and no headline number is to be quoted without one.** A larger sample is the obvious follow-up and
is not attempted here.

## The rubric, fixed now

Each item is adjudicated by reading the claim sentence and the live diff and answering one
question: *could a competent engineer, given only this sentence and this diff, establish whether
the sentence is true or false — without guessing at what the author meant?*

**DECIDABLE** requires all three:

1. the sentence makes an assertion about **this pull request's changes**, not about what the
   resulting program does at runtime, not about a feature being described, and not about some
   other change;
2. the thing it names is the kind of thing the diff contains — a path, a file count, a definition —
   and is identifiable in the diff without inference about intent;
3. the diff is complete enough to settle it (not truncated, not a pointer to a submodule or binary
   whose contents are absent).

**NOT-DECIDABLE** otherwise, with one of these reasons recorded: `runtime_behaviour`,
`prose_or_documentation`, `named_thing_is_not_in_the_diff_vocabulary`, `ambiguous_scope`,
`typo_or_nonexistent_referent`, `diff_incomplete`, `other` (free text).

Decidability is **not** the same as truth. "Only modifies CHANGELOG.md" on a PR that also touches
three workflows is DECIDABLE and false. Both are recorded; the headline is the decidable fraction.

## Adjudication discipline

- Every item gets a written one-line reason. An item with no reason is not adjudicated.
- The adjudication is recorded **before** looking at what the instrument said. The instrument's
  verdict is joined afterwards, for a separate reported statistic, and never consulted while
  deciding.
- Items are adjudicated in a shuffled order that does not group by kind, so a run of similar
  sentences cannot set a rhythm.
- Every adjudication is published in full, per item, with the PR URL, so anyone can disagree with
  any specific call. This is the only defence against the obvious criticism, which is that the
  party with an interest in the answer is also the judge. That criticism is legitimate and is
  stated here rather than rebutted.

## Gates — committed now

- **G-D1-1 (every item reasoned).** 100 of 100 carry a recorded reason and a reason-code. No
  silent calls.
- **G-D1-2 (intervals, always).** Every proportion is published with a 95% Wilson interval. A
  point estimate appearing anywhere without its interval is a violation.
- **G-D1-3 (blind to the instrument).** Adjudications are written and frozen before the
  instrument's verdicts are joined. The join is a separate, later step and the frozen file is
  published.
- **G-D1-4 (the conflict of interest is published).** The result states plainly that Fathom Lab
  adjudicated claims bearing on Fathom Lab's own thesis, publishes every per-item call, and invites
  re-adjudication.
- **G-D1-5 (what is not claimed).** No claim that the instrument is good. No comparison to any
  competitor. No restatement of a decidable-fraction number as an accuracy or recall number for
  anything. This measures the corpus, not the tool.

## What would embarrass the thesis

The thesis is that most of this claim space is undecidable from the artifact, and that recall-led
tooling therefore reports on something other than what it says it does. It is embarrassed if the
decidable fraction comes back high — say above 50% pooled — because then a recall-optimised tool
has plenty of real signal to find and abstention buys much less than we have been saying. That
result gets published in the same place and the same font as any other.

---

*Two oracles failed here, so the third attempt is not an oracle. One hundred claims, read one at a
time by hand, against the diff GitHub actually serves, with a written reason for every call and a
confidence interval on every number. If the answer is inconvenient it still gets published, and
every individual call is exposed so that anyone who thinks we got one wrong can point at it.*
