# RESULT — representational reliability: real, replicated, and redundant

**Verdict by the preregistered PRIMARY gate: NOT SUPPORTED.**
**Verdict on the confound gate: REPLICATED, on fresh disjoint items.**

Prereg: `PREREG_rdm_reliability_confirmatory_2026_08_21.md` (frozen and pushed
before the run). Raw: `out_rdm_reliability_confirmatory_last.json`.

---

## the numbers

Qwen2.5-1.5B-Instruct, layer 21/28, 500 PopQA items **disjoint from the
exploratory set** (disjointness asserted in code), accuracy 0.110.

| | exploratory (INVALID run) | **confirmatory (fresh items)** |
|---|---:|---:|
| rho(reliability, correct) | −0.1621 | **−0.1609** |
| partial rho (controls: length, log popularity) | −0.1321 | **−0.1341** |
| one-sided p (partial) | 0.003 | **0.0013** |
| reliability vs log popularity | +0.027 | **−0.006** |

The correlation replicated to three decimal places on items the exploratory
analysis never touched. It is not a length artifact (rho(length, correct) =
−0.010) and it is **orthogonal to popularity**, the known difficulty driver
(rho(pop, correct) = +0.159).

**And it does not matter.**

| model | AUC |
|---|---:|
| token confidence alone (logprob + entropy + margin) | **0.8679** |
| reliability alone | 0.6415 |
| both | 0.8733 |

**G1: delta-AUC +0.0053, 95% CI [−0.0057, +0.0148] — includes zero.**

## what this means

Representational reliability carries genuine, replicable, confound-surviving
information about whether a 1.5B model's answer is correct — AUC 0.64 alone, far
above chance. **It adds essentially nothing to the token-confidence signal styxx
already ships**, which reaches 0.868 on its own.

The primary gate was written to ask exactly this: not *"is there signal?"* but
*"is there signal **beyond the baseline we already have**?"* The answer is no.

A real phenomenon that does not earn deployment. Had the primary gate been
"does reliability beat chance", this would have been reported as a success, and
it would have been true, and it would have been useless.

## the direction is inverted, and unexplained

Higher representational reliability predicts a **less** likely correct answer —
the opposite of the original hypothesis, and it replicated. We do not know why,
and this design cannot say. It is not popularity (orthogonal) and not length
(null). Recorded as an open question rather than decorated with a story.

## what the process cost, and what it bought

Four runs on one local GPU, ~9 minutes total, no API spend.

- Attempts 1 and 2: `INVALID` on **G4, a gate that was measuring the wrong
  property** — absolute spread (IQR) for a rank-based statistic. The run it
  rejected had 497 distinct values in 500 items. Corrected to a ties/distinct
  criterion, the confirmatory run scored **100% distinct**.
- The exploratory signal was found post-hoc in a rejected run, disclosed as
  such, and tested on data it had never seen.
- The confirmatory run then split the verdict: the effect is real, and the
  primary gate still says no.

Reporting the exploratory −0.16 alone would have been a finding. It would also
have been misleading, because the number nobody would have quoted is
**AUC 0.868 for the baseline**.

## honest limits

One model, one scale, one layer, one task, ~55 positives per run. A wide CI on a
small positive class is a power statement; here the point estimate (+0.005) is
small enough that power is not the story — the baseline being strong is.

Nothing here generalizes to other models, layers or tasks, and nothing here is
causal.
