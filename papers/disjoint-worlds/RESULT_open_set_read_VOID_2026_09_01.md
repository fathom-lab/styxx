# RESULT — the open-set read is VOID: the null mapper separates too

Fathom Lab · 2026-09-01 · Scored against `PREREG_open_set_read_2026_09_01.md`, frozen and
pushed at `45d7ae5` before this ran. Receipt: `open_set_read_result.json`. Runner:
`run_open_set_read.py`. 756 seconds, CPU-only, from the committed `.npz` banks. No model was
loaded and nothing was collected.

## VERDICT: `VOID__null_mapper_separates`

**G-O2 failed and the prereg's consequence is taken, not argued with.** No open-set capability
is claimed for any model, in either direction.

---

## What passed, and it is worth having on its own

**G-O1 — closed-set reconciliation. PASS, exactly, on all three targets.**

| target | this harness | committed `b34v3_result.json` | reconciles |
|---|---|---|---|
| `llama_1b` | 0.6857 | 0.6857 | yes |
| `gemma_2b` | 0.5714 | 0.5714 | yes |
| `qwen_1p5b` | 0.1429 | 0.1429 | yes |

This is the gate that makes the study legitimate and it is the one clean asset produced today.
`run_open_set_read.py` replicates b34v3's seed, its rng **consumption order**, its
`TransferMap.fit` -> `assignment_from_map` -> `fit_mlp` pipeline, and re-derives all three
published read figures to four decimals. **The open-set harness is provably the committed
apparatus plus a reject option, not a different measurement.** Any successor design can build on
it without re-litigating whether it is measuring the same thing.

---

## What failed

**G-O2 — the null must fail. FAILED.** A random-orthogonal mapper, fit on shuffled targets and
pushed through the identical pipeline, was required to score `AUROC(margin) <= 0.55`. Observed:

| target | null-mapper AUROC | bar |
|---|---|---|
| `llama_1b` | **0.5861** | <= 0.55 |
| `gemma_2b` | **0.5584** | <= 0.55 |
| `qwen_1p5b` | 0.5012 | <= 0.55 |

Two of three exceed it. **The top-1/top-2 margin separates present-target from absent-target
trials even when the map carries no correspondence at all**, so on this statistic the separation
is not evidence about transported content.

### The mechanism, named plainly

An IN query is a concept whose own target vector is one of the 35 candidates, and the query is
that same concept read out of the other model. Whatever survives a random map — norm structure,
the anisotropy of the bank, the fact that a point's own image sits in the array — is enough to
depress `d1` relative to `d2` without any correspondence being transported. Absence of a target
is therefore partly detectable from **geometry alone**. That is a property of the banks and the
statistic, not of the reader.

`b48` in this same arc died on a mis-specified null and that failure is committed. This gate was
written because of it, and it did its job.

---

## The numbers that are NOT licensed, printed anyway

Under a void run these decide nothing. They are recorded because suppressing an observation
because its gate failed is how a corpus becomes a highlight reel.

| target | AUROC(margin) IN vs OOV | null | top-1 within C on IN trials |
|---|---|---|---|
| `llama_1b` | 0.7665 | 0.5861 | 0.8286 |
| `gemma_2b` | 0.6033 | 0.5584 | 0.7429 |
| `qwen_1p5b` | 0.5061 | 0.5012 | 0.2571 |

**`llama_1b` 0.7665 would have cleared the 0.75 OPEN-SET SIGNAL bar. It is void and must not be
quoted as a result**, here or anywhere downstream. The real-minus-null gaps (0.180, 0.045,
0.005) are likewise not a repaired statistic — subtracting a null after seeing it is exactly the
post-hoc move preregistration exists to prevent. A successor must freeze its statistic first.

---

## What this does and does not change

**The E = 1 finding stands and is untouched.** `read_top1` is still an index-matched `argmin`
over an array containing the truth (`run_b31v2.py:90-93`, `run_b34v3.py:46-49`), so every
published read figure in this arc is still an A-term. This run failed to *measure* the missing
term; it did not find the term to be absent.

**No correction is owed to the arc's published numbers.** G-O3 was never reached, so
`TELEPATHY_READ_BAR_CLEARED__labelfree_pairing_reads_crossfamily` is neither upheld nor
impeached by this run. The obligation to describe those numbers as closed-set-conditional comes
from the source reading, not from this experiment.

**A void is not a null result.** We did not learn that the reader cannot decline. We learned that
this statistic cannot tell us, because it answers partly from geometry.

---

## The repair, for a successor preregistration

The statistic must be invariant to what the null exploits. Candidates, to be chosen and frozen
**before** the next run rather than after inspecting these numbers:

1. **Per-query standardisation** — z-score the margin against the query's own distance
   distribution over the candidate array, so bank anisotropy divides out.
2. **A rank statistic** — the normalised rank gap, which discards distance scale entirely.
3. **Matched-null calibration as part of the design** — declare in advance that the reported
   quantity is the null-adjusted separation, with the null fit under the same seed and its bar
   set on development concepts.
4. **Harder OOV probes** — draw absent-target probes whose nearest candidate distance is matched
   to the IN distribution, so presence cannot be inferred from proximity alone.

Whichever is chosen must clear the same G-O2 null before any G-O3 verdict is read.

---

## Disclosure

This is the second void of 2026-09-01. The first
(`../closed-model-frontier/ADDENDUM_extraction_ceiling_gate_unsatisfiable_2026_09_01.md`) voided
because its reliability gate could not be **built** from the packet it named. This one is
different in kind and better: the gate was built, it **ran**, and it **failed**. That is a
completed experiment with a negative outcome, not an unexecutable document.

Three preregistrations were frozen today and two runs are void. A lab that writes more
preregistrations than it completes measurements should be told so by its own receipts, and this
paragraph is that receipt.
