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

## Exploratory open↔closed (cross-vendor): judge confound DEMONSTRATED; invariance PARTIAL after matching

The tempting next step was to extend the hallucination-cliff invariance across the open/closed divide
by comparing the three open families to the committed gpt-4o-mini cliff. The naive comparison fails,
and a follow-up run — `run_xvendor_matched.py`, **now executed** (2026-06-23) — shows *why* it failed
and how much of the failure was apparatus, not signal. gpt-4o-mini's per-item resamples were already
on disk (`truthfulqa_benchmark_result.json`, no API key), so they were re-judged with the **identical
NLI judge** used for the open families and gpt-4o-mini's cliff recomputed under the matched apparatus:

| comparison | hallucination cliff (mean Spearman) | refusal cliff (mean Spearman) |
|---|---|---|
| open ↔ open (all NLI-judged) | **+0.770** | +0.426 |
| open ↔ closed gpt-4o-mini — OLD (LLM judge) | +0.231 | +0.518 |
| open ↔ closed gpt-4o-mini — **MATCHED (NLI re-judge)** | **+0.473** | +0.416 |

**The judge confound is demonstrated, and it was large.** Two signatures, both predicted:
1. **The inverted ordering flips back.** OLD open↔closed had refusal (0.52) **>** hallucination (0.23)
   — the opposite of the rep/mechanism split. Under the matched judge the ordering **restores** to
   hallucination (0.47) **>** refusal (0.42), matching the open↔open shape.
2. **The hallucination invariance roughly doubles** (0.231 → 0.473) and the per-family estimates
   **tighten** (matched: Qwen +0.486 / Llama +0.443 / gemma +0.489 — a 0.05 spread, vs the OLD's wide
   scatter). Refusal stays low and model-specific throughout (0.28–0.60, mean ≈ 0.42) — control is
   substrate-specific across the open/closed divide too, exactly as the asymmetry predicts.

**But — honest bound — cross-vendor invariance is PARTIAL, not full.** Even with the judge matched,
open↔closed hallucination (0.473) is still well below open↔open (0.770). A real residual gap remains,
and it is **not cleanly separable here**: this run matches only the *judge*, not the *generation
apparatus* (gpt-4o-mini was sampled via the OpenAI API under a different decoding pipeline than the
locally-sampled open families). So the residual 0.47-vs-0.77 could be true open/closed vendor
divergence **or** the remaining generation-apparatus mismatch — this run cannot tell them apart.

**What is now claimed:** (a) the OLD 0.23 *understated* cross-vendor hallucination-cliff invariance —
roughly half of the open↔closed attenuation was a judge artifact; (b) the rep/mechanism ordering
(hallucination shared ≫ refusal shared) **survives across the open/closed divide** once the judge is
matched; (c) cross-vendor hallucination-cliff invariance is **real but partial (≈0.47)**, with a
residual gap that needs a generation-matched run to resolve. Nothing is wired from this; the shipped
`competence_cliff()` (gpt-4o-mini, LLM-judged) is untouched. Receipts:
`xvendor_gpt4omini_nli_gate.json`, `xvendor_matched_invariance_result.json`,
`analyze_xvendor_matched.py`.

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
3. **Generation-matched open↔closed** to resolve the residual 0.47-vs-0.77 gap: regenerate the open
   families' samples under gpt-4o-mini's decoding pipeline (or vice versa) so that *both* generation
   and judge are matched. Only then can the residual cross-vendor gap be attributed to true vendor
   divergence rather than apparatus.

## Receipts

- Prereg: `PREREG_crossfamily_local_cliff_2026_06_22.md` (`ae48acf`). Runner: `run_local_cliff.py`.
- Per-family receipts: `crossfamily_gate_{Qwen2_5_3B_Instruct,Llama_3_2_3B_Instruct,gemma_2_2b_it}.json`.
- Aggregate: `crossfamily_cliff_result.json`. Exploratory analysis reproducible from the three gate
  JSONs (per-domain refusal_rate / ungated_hallucination_rate Spearman).
