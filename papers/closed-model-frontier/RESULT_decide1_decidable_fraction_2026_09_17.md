# RESULT — 71% of agent-PR claims are decidable from the diff. The instrument abstains on 94% of them.

Fathom Lab · 2026-09-17 · Prereg `PREREG_decide1_decidable_fraction_2026_09_17.md`, sha256
`e6b3c7dd…`, frozen before a single item was adjudicated. Receipts: `decide1_adjudication.json`
(all 100 items, every call, every reason). Read with `RESULT_bench1_INVALID_2026_09_17.md` and
`RESULT_bench2_INVALID_2026_09_17.md`. The instrument was not used to produce any number here and
is unchanged at sha256 `4ba947a8…`.

**All five gates pass.** The finding contradicts what we said yesterday and most of what we said
this morning.

## The number

100 claims, stratified 25 per kind, seed 20260920, adjudicated by hand against the live diff under
the rubric frozen in the prereg. Decidable means a competent engineer could establish the
sentence's truth from the sentence and the diff without guessing at intent. It does **not** mean
the sentence is true.

| claim kind | decidable | 95% Wilson |
|---|---|---|
| `files_changed_count` | 24/25 — 96.0% | [80.5%, 99.3%] |
| `tests_added` | 23/25 — 92.0% | [75.0%, 97.8%] |
| `symbol_added` | 16/25 — 64.0% | [44.5%, 79.8%] |
| `only_touches` | 13/25 — 52.0% | [33.5%, 70.0%] |
| **pooled (unweighted)** | **76/100 — 76.0%** | **[66.8%, 83.3%]** |

Weighted to the corpus's actual mix of claim kinds (`only_touches` 299, `tests_added` 151,
`files_changed_count` 116, `symbol_added` 38): **71.2% decidable**.

Reasons for the 24 not-decidable items: `ambiguous_scope` 10, `runtime_behaviour` 6,
`prose_or_documentation` 4, `named_thing_not_in_diff_vocabulary` 3, `refers_to_a_different_change` 1.

## What this overturns

Yesterday's framing — ours — was that this claim space is largely undecidable from the artifact,
that the MSR '26 κ ceiling of 0.892 reflects irreducible ambiguity, and that a tool which abstains
is therefore operating at the edge of what can be known. We went further in public this morning
and said under 3% of `only_touches` claims could be settled from the diff.

That 3% was **our own instrument's admission rate**, not a property of the corpus. Hand
adjudication puts the real figure at 52% [33.5%, 70.0%].

This is the third instance in two days of the same error: treating our machinery's output as
ground truth about the world. BENCH-1 did it with the prefix extractor. BENCH-2 did it with the
admissibility test. The public statement did it with the oracle's acceptance rate. The error is
not a bug in any one component; it is a habit, and naming it is the most useful thing in this
document.

## What it means for the instrument

Three measurements, side by side, on the same corpus:

- **71%** of claims are decidable from the diff.
- The instrument returns a verdict on **17 of 299** `only_touches` claims — it abstains on 94%.
- Of the 11 accusations it does make, **9 are false** (`RESULT_bench2_INVALID_2026_09_17.md`) —
  precision 0.18.

So the instrument is silent on most of the signal that is genuinely there, and wrong when it
speaks. Those are two different failures and until today we had measured neither. The abstention we
have been presenting as principled restraint is, in the majority of cases, an extraction failure
wearing restraint's clothes.

The precision-first thesis survives in its narrow form: given a choice between a false accusation
and silence, silence is better, and that remains the design. What does not survive is the claim
that the silence is forced by the domain. The domain is mostly decidable. **The bottleneck is
extraction — pulling the assertion out of the sentence — not judgement.** That is a different
research programme from the one we have been running, and a harder one.

## Conflict of interest, stated plainly (G-D1-4)

Fathom Lab adjudicated 100 claims bearing directly on Fathom Lab's own thesis, and the result
happens to be unflattering, which is not evidence of impartiality. The only real defence is that
every one of the 100 calls is published in `decide1_adjudication.json` with its reason code and a
one-line justification, against a PR URL. Anyone who thinks a call is wrong can point at it by id.

Two calls are flagged CONTESTABLE in the data: items 12 and 82, where a PR body quotes an agent
instruction ("ONLY MODIFY the README.md file at the root of the repository") and we read that as an
implied compliance claim. Counting both as not-decidable moves the pooled figure from 76.0% to
74.0% and `only_touches` from 52.0% to 44.0%. No conclusion here depends on them.

## Limits

n=25 per kind. The intervals are wide and are printed everywhere (G-D1-2); `only_touches` spans
33.5% to 70.0% and no reader should treat 52% as precise. `symbol_added` has only 38 items in the
reachable population, so its 25 are a 66% census and its interval is narrow for that reason rather
than because the estimate is better.

`tests_added` is the least confident of the four. Its adjudication turns on whether a stated count
has exactly one natural unit in the relevant framework — `@Test` functions, `it()` blocks,
`[InlineData]` rows, `AT_SETUP` blocks. Where two units compete and neither matches the stated
number (item 4: 3 `[Fact]` against 37 `[InlineData]` for a claim of 22), we called it
not-decidable. A reasonable person could set that line elsewhere.

Adjudication was blind to the instrument's verdicts, which were joined only after the calls were
frozen (G-D1-3).

## What is not claimed (G-D1-5)

No claim that the instrument is good; this document is evidence it is not, in two distinct ways.
No comparison to any competitor. No restatement of a decidable fraction as an accuracy or recall
figure for anything. This measures the corpus.

---

*We spent two days building benchmarks to find out whether our tool was right, and the thing worth
knowing turned out to be a question we never asked: how much of this is knowable at all. The answer
is most of it. Our tool says nothing about most of it. We had been calling that restraint.*
