# FINDING — cross-family competence cliff: pre-reg metric UNDERPOWERED; exploratory signal shows the hallucination cliff is cross-family invariant, the refusal cliff is not

**Run 2026-06-22, pre-registered in `PREREG_crossfamily_local_cliff_2026_06_22.md` (frozen before
data, commit `ae48acf`).** Local open-weights, no API key. Three families — Qwen2.5-3B-Instruct,
Llama-3.2-3B-Instruct, gemma-2-2b-it — over the same 790-item TruthfulQA set (answer-key SHA
`07ea5d2e…` verified), N=10 stateless samples/question, NLI bidirectional/asymmetric same-answer
judge (the only apparatus difference from the committed gpt-4o-mini run; see prereg apparatus log).
K_precondition PASSED for all three (modal-belief rate 0.749 / 0.805 / 0.835).

## The pre-registered metric came out UNDERPOWERED — reported as inconclusive, not as a result

**L1 (committed_precision cross-family rank invariance) printed mean Spearman 0.997 "SURVIVED" —
but this is a thin-data artifact and is NOT claimed.** The pre-registered thin-domain guard
(committed_n ≥ 5 in BOTH compared models) left only **3–7 usable domains per pair**, because small
open models refuse heavily under the belief-coherence gate (per-domain refusal 0.6–1.0), so the
committed subset is tiny. A Spearman of 1.000 on 3 points is not evidence. **Honest verdict on the
pre-registered metric: INCONCLUSIVE / underpowered.** The committed-precision cliff is not
measurable cross-family at this model scale with this 790-item set — the models commit on too few
items per domain. (L2 worst-domain persistence likewise FAILED on sparse data.)

This is the discipline catching a trap: a single-pass "go big" run would have trumpeted
"cross-family cliff invariance SURVIVED at 0.997." It is false.

## Exploratory (NOT pre-registered): the well-powered per-domain signals

`refusal_rate`, `ungated_hallucination_rate`, and `useful_answer_rate` are computed over all `n`
items per domain (not the committed subset), so all 37 domains are populated. Pairwise Spearman
over 37 domains, mean across the 3 families:

| per-domain signal | mean pairwise Spearman | dispersion |
|---|---|---|
| **ungated hallucination rate** (where the model errs) | **+0.770** | 0.72 – 0.84 |
| useful-answer rate | +0.536 | 0.51 – 0.56 |
| **refusal rate** (how the model gates itself) | **+0.426** | 0.32 – 0.54 |

**The hallucination cliff is strongly cross-family invariant; the refusal cliff is markedly less
so.** Where three independently-trained families (different architectures, tokenizers, corpora) get
answers *wrong* is substantially shared (0.77). Where they choose to *abstain* is much more
model-specific (0.43).

These are **exploratory** — they were not the pre-registered bars (which were committed_precision).
They are reported as a clearly-labelled diagnostic that motivates a pre-registerable follow-up, not
as a passed kill-gate.

## Why this is interesting — it lands on the program's central asymmetry

The ancient-question capstone established: *minds converge on WHAT they represent, not on HOW they
process/control it* — universality lives in representation, not mechanism. The exploratory split
here is the same shape at the reliability level:

- **WHERE knowledge fails (hallucination) = a property of the represented knowledge → cross-family
  invariant (0.77).**
- **HOW a model gates its own uncertainty (refusal) = a control/mechanism property → substrate-
  specific (0.43).**

Representation shared, control not — a fourth independent instance, on the predicted side. Held as
*suggestive*, not proven, pending a pre-registered confirmatory run.

## Exploratory open↔closed (cross-vendor): tested and REJECTED — confounded by the judge

The tempting next step was to extend the hallucination-cliff invariance across the open/closed
divide by comparing the three open families to the committed gpt-4o-mini cliff. **It does not hold,
and the way it fails shows why:**

| comparison | hallucination cliff (mean Spearman) | refusal cliff (mean Spearman) |
|---|---|---|
| open ↔ open (all NLI-judged) | **+0.77** | +0.43 |
| open ↔ closed gpt-4o-mini | **+0.23** | +0.52 |

The ordering **inverts**: open↔open has hallucination ≫ refusal (the rep/mechanism split); open↔closed
has refusal > hallucination. That inversion is **consistent with an apparatus confound** — the open
families were scored by the NLI judge and gpt-4o-mini by an LLM judge, and *hallucination rate
(correctness)* is exactly the quantity most sensitive to the judge, while the *refusal rate
(stability/clustering)* is less so. **But this is a HYPOTHESIS, not a demonstrated confound** (audit
issue C): the matched-judge run (`run_xvendor_matched.py`) that would prove it is built but **has not
yet executed**, so I cannot rule out that the open↔closed divergence is partly real. Either way, **the
cross-vendor extension is NOT claimed.** The only invariance claimed is the within-apparatus open↔open
0.77.

**The clean fix needs no API key** and is built + queued (`run_xvendor_matched.py`): gpt-4o-mini's
per-item resamples are already on disk (`truthfulqa_benchmark_result.json`), so they can be
re-judged with the *identical* NLI judge and gpt-4o-mini's cliff recomputed under the matched
apparatus. Then open↔closed is judge-confound-free. It runs once the GPU frees from the Rung 2 read
sweep. (This is logged here because the extension was *tested and rejected*, not silently dropped.)

## Honest bounds / confounds

- **Underpowered on the pre-registered metric.** The headline pre-reg result is inconclusive; do
  not cite the 0.997.
- **Adversarial-dataset confound on the hallucination invariance.** TruthfulQA is *constructed* to
  target misconceptions humans (and models) commonly hold; some cross-family hallucination overlap
  may reflect the benchmark's design (shared targeted-hard domains) rather than a deep property. A
  non-adversarial benchmark (e.g. a balanced QA set) is the clean follow-up.
- **Refusal-cliff 0.426 is not a clean estimate (audit issue D).** Llama-3B refuses 100% in 11/37
  domains (≥0.95 in 19/37); tied maxima mechanically depress the Spearman via range-restriction. The
  *direction* survives (on the non-saturated subset, refusal invariance 0.34 ≪ hallucination 0.755),
  but do not cite 0.426 as a clean figure — cite the direction + the range-restriction caveat.
- **Apparatus difference from the shipped cliff.** The NLI judge is not the LLM judge of the
  committed gpt-4o-mini run, so these local maps are not numerically cross-comparable to the shipped
  `competence_cliff()` artifact — the comparison here is strictly *across the three local families
  under one shared apparatus*, as pre-stated.
- Small models (1–3 B), single run, single 790-item set, in-silico. No consciousness claim.

## What does NOT change

The shipped `styxx.compliance.competence_cliff()` (7.18.0, gpt-4o-mini) is unaffected — nothing is
wired from this run, because the pre-registered metric was inconclusive. The committed artifact
stands as-is.

## The pre-registerable follow-ups

1. **Hallucination-cliff cross-family invariance** as a *primary* pre-registered metric (it is
   well-powered: all 37 domains, n items each), on TruthfulQA AND a non-adversarial benchmark to
   kill the dataset confound.
2. Larger open models (7B+) that commit more, to make the committed-precision cliff measurable
   cross-family.

## Receipts

- Prereg: `PREREG_crossfamily_local_cliff_2026_06_22.md` (`ae48acf`). Runner: `run_local_cliff.py`.
- Per-family receipts: `crossfamily_gate_{Qwen2_5_3B_Instruct,Llama_3_2_3B_Instruct,gemma_2_2b_it}.json`.
- Aggregate: `crossfamily_cliff_result.json`. Exploratory analysis reproducible from the three gate
  JSONs (per-domain refusal_rate / ungated_hallucination_rate Spearman).
