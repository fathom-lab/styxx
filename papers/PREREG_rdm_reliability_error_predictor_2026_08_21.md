# PREREGISTRATION — is representational reliability a validity channel?

**Frozen 2026-08-21, before any result was computed.** Written to be losable.

---

## why this experiment exists

The 2026-06-03 geometry post-mortem killed geometry-as-manipulation-detector
three ways and named the surviving residue:

> **RDM-reliability as confidence/quality signal — untested as error predictor,
> real next expt for confidence-router.**

Untouched since. Yesterday's synthesis says why it is the interesting one: the
whole 74-defect arc was about measurements that could not say whether they had
measured anything, and the cure was a **validity channel** carried beside the
value. This asks whether a model's own representation carries such a channel
about its own answer.

Concretely: **when a model's internal representation of a question is unstable,
is its answer more likely to be wrong?**

## H1 and its null

- **H1.** Per-item representational reliability predicts answer correctness
  **beyond** the token-confidence signal styxx already ships.
- **H0.** It does not: once baseline confidence is in the model, reliability adds
  nothing distinguishable from zero.

H0 is the expected outcome by this lab's base rate (62 of 163 cycles ended in a
loss, null, retraction or INVALID) and will be published the same way if it wins.

## the measurement

**Model.** `Qwen2.5-1.5B-Instruct`, local, fp16, single GPU. No API, no network.
Chosen because the geometry lane's own work used 1B–1.5B models, so this is the
regime the residue was named in.

**Data.** PopQA (`akariasai/popqa`), N = 500 items sampled with a fixed seed
across the subject-popularity range. Short factual answers with alias lists, so
grading is exact-match and needs no judge — deliberately: **an LLM judge is a
measurement that can fail silently, which is the class this program studies.**

**Answer + correctness.** Greedy decode, max 24 new tokens. Correct iff a
normalized alias appears in the normalized generation.

**Baseline confidence (the control that matters).** From the generated answer's
own token distribution: mean token logprob, mean entropy, mean top-2 margin.
This is what styxx already has, and it is the bar the new signal must clear.

**Representational reliability, per item.**
1. Hidden states at layer L = ⌊0.75 · n_layers⌋, mean-pooled over prompt tokens
   → one vector per item.
2. Randomly split the feature dimensions into two disjoint halves, A and B.
3. Build item×item cosine-dissimilarity matrices RDM_A and RDM_B.
4. `r_i` = Spearman correlation between row *i* of RDM_A and row *i* of RDM_B —
   how stably item *i* sits relative to every other item across two independent
   halves of the representation.
5. Average `r_i` over **20 random splits** (fixed seed) so a single split's noise
   is not the finding.

`r_i` is batch-relative by construction. That is a stated property, not a
discovered flaw: it measures stability of an item's position in *this* cohort's
geometry.

## kill gates — frozen thresholds, decided now

**G1 — PRIMARY.** ΔAUC = AUC(baseline + reliability) − AUC(baseline alone),
logistic regression, 5-fold cross-validated, 95% bootstrap CI over 2000
resamples.
→ **If the CI includes 0, H1 is NOT SUPPORTED.** No reframing, no
"trending", no subgroup rescue.

**G2 — CONFOUND.** This lab has repeatedly found guardrails riding on length
(3 of 4 in the 2026-06-25 audit). Partial Spearman of `r_i` with correctness,
controlling for **prompt token length** and **log subject popularity**, must
keep the same sign and remain nonzero at p < 0.05.
→ **If reliability is explained by length or popularity, it is a proxy, not a
mechanism**, and G1 passing does not rescue it.

**G3 — VALIDITY.** If overall accuracy is below 10% or above 90%, the outcome
has too little variance for an AUC to mean anything.
→ **Verdict INVALID, not a null.** An underpowered cell reported as a negative
is the same lie as an unmeasured value reported as a pass.

**G4 — SANITY.** Reliability must not be near-constant: if the interquartile
range of `r_i` is < 0.02, the measure is a constant dressed as a variable
(this program has shipped one of those before — `memory_integrity`, 24/24
identical) → **INVALID**.

## what would make this INVALID rather than negative

- G3 or G4 trip.
- The model degenerates (empty or repeated generations on >20% of items).
- Any post-hoc change to layer L, split count, or N after seeing an outcome.
  The values above are frozen. If a different layer is explored later it is a
  **new, separately reported** experiment, not this one.

## what a positive result would and would not license

**Would:** that a model's representational stability carries information about
its own error, *on this model, this task, this layer*, beyond token confidence.

**Would not:** that this generalizes across models, tasks, or layers; that it is
causal; that it is deployable. The geometry lane already learned that lesson the
expensive way — an AUC of 1.00 that meant nothing because a control was missing.

## artifacts

Script `scripts/rdm_reliability_prereg.py`, raw output
`papers/out_rdm_reliability_2026_08_21.json`. Every number in the writeup comes
from that JSON; nothing is typed by hand.
