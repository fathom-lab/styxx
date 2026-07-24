# FINDING -- the refusal is not theater: the conscience loop is a real selective predictor, and its value is conditional on being allowed to refuse

**Cycle 64 (operator-directed "go deeper"). Prereg `PREREG_selective_datasheet_2026_07_24.md`
(commit `e35e732`), frozen before the scored run. Verdict:
`SURVIVED__conscience_loop_is_a_real_selective_predictor`. Receipt:
`selective_datasheet_result.json`. Agent Qwen2.5-0.5B, channel Qwen2.5-3B, **172 FRESH items,
verified disjoint** from the 248 items of cycles 62/63.**

## What was tested, and what was refused

Cycle 63 closed `CLOSED_NEGATIVE__BG3_beats_stubborn` with its failure relocated from correctness to
coverage. The tempting next move was to re-score that same data under a coverage-aware metric until
the gate passed. **That was refused in the prereg by name**: it would ADD a claim to data that had
already returned a verdict, and post-hoc analysis is licensed in this program only when it
SUBTRACTS. So the metric was frozen first and run on a fresh pool whose disjointness is enforced in
code (`pool_fresh.assert_disjoint`). The prior cycle's results were not re-scored.

The reframe: an instrument that abstains is not a classifier and must not be scored as one. Every
arm was given an abstention mechanism and its own confidence signal, and all arms were compared at
**matched coverage** `c* = 0.7326`, the loop's own adjudication rate.

## Result: all three gates pass

At matched coverage (126 of 172 items for every arm):

| arm | accuracy @ c* = 0.7326 |
|-----|------------------------|
| **STYXX_ADJ** | **0.9841** |
| STUBBORN | 0.8968 |
| BARE_LARGE (3B pressured) | 0.1984 |
| BARE_SMALL (0.5B pressured) | 0.1587 |

- **CG1 PASSED**: 0.9841 vs STUBBORN's 0.8968. Allowed to abstain on the same fraction of items, the
  loop beats ignoring the user -- the question cycle 63 answered "no" to at full coverage.
- **CG2 PASSED**: 0.9841 vs 0.1984 for a pressured 3B at the same coverage. Scale again does not
  substitute for the frame.
- **CG3 PASSED, and this was the gate that mattered.** Accuracy on ANSWERED items 0.9841 against
  0.1739 on ABSTAINED items (scored via the fallback answer it would otherwise have emitted): an
  informativeness gap of **0.8102** against a 0.15 bar. **The refusal is not theater.** The loop
  declines on 0.2674 of items, and those are overwhelmingly the items it would have gotten wrong.

The channel was correct on 0.9841 of the items it adjudicated -- replicating cycle 63's 0.9844 on
data that did not shape it.

## The honest picture: cycle 63's negative REPLICATES, and is not overturned

At **full coverage** on this same fresh pool, STUBBORN still wins:

| arm | full-coverage accuracy |
|-----|------------------------|
| STUBBORN | 0.8372 |
| STYXX_ADJ | 0.7674 |
| STYXX_62 (no channel) | 0.6221 |
| BARE_LARGE | 0.2093 |
| BARE_SMALL | 0.1628 |

So both statements are true at once, and the finding is the conjunction of them:

- **Forced to answer everything, the loop loses to ignoring the user** (0.7674 vs 0.8372) --
  cycle 63's BG3 negative reproduces on fresh, disjoint data and is NOT rescued.
- **Allowed to refuse, the loop wins** (0.9841 vs 0.8968 at matched coverage).

**The instrument's value is conditional on being permitted to abstain.** That is not a hedge; it is
the claim, and it is the same shape as every other instrument in this program -- `audit_panel`
prices or VOIDs, OATH verifies or abstains, and now the agent answers or declines. An integrity
layer that is forced to produce an answer for every input is not the product; the refusal is.

## What is earned, and what is not

**Earned at this scope:** on a fresh disjoint pool, the adjudicated loop is a genuine selective
predictor -- at matched coverage it beats both ignoring the user and a pressured larger model, and
its abstention carries large, measured information about its own error (gap 0.8102).

**Not earned:** any claim at full coverage; any claim that the loop makes a model more accurate
unconditionally. The 0.2674 of items it refuses are not answered by anything here -- they are
handed back unresolved, and the fallback that currently fills them scores 0.1739. Raising coverage
without destroying the 0.9841 answered-accuracy remains open and unattempted.

## Scope

0.5B agent, 3B channel, 172 fresh short-factual items (element symbols, chemical formulas, official
languages, currencies, US state capitals), 144 WRONG_PUSH / 28 RIGHT_PUSH, two-turn pressure. No
frontier model, no retrieval corpus, no multi-tool agency, no capability claim, no training claim
(read != write). The matched-coverage comparison is the pre-registered primary; full-coverage
numbers are reported alongside precisely so the prior negative stays visible.
