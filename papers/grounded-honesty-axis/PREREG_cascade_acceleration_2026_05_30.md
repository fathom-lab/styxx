# PREREG — ACCELERATION: the load-bearing detector as a COMPUTE ROUTER (a cheap-gate → expensive-resampling cascade)

**Registered 2026-05-30, before computing the cascade metric.** The cascade score has never been
computed on this data — it is a new metric — so there is no data-peek, though the underlying
per-item signals were collected by prior detection-locus runs (not for this test). The input rows are
SHA-256'd and printed before scoring.

## The question

The honesty layer's strong signal (N=10 resampling instability) costs 10 forward passes/item. The
cheap single-pass entropy gate costs 1 and TIES resampling on derivation but is WEAKER on factual
recall (`SYNTHESIS_detection_locus`). Can the cheap gate **route** compute — clear the confident
majority in one pass, escalate only the uncertain fraction to expensive resampling — and retain
~full detection at a fraction of the cost? This operationalizes the honesty-knob principle (*the
detector is load-bearing*) one step further: **the detector is also the compute router.**

## Protocol (offline, over already-collected detection-locus per-item receipts)

Pool every detection-locus receipt whose rows carry BOTH signals on the same item — `clean_entropy`
(cheap, 1 pass), `instability` (expensive, N=10), `group` (label). White-box open models
(Qwen / Llama / Gemma) across arithmetic / code / logic / factual.

- **Cascade** with tier-1 threshold `tau1` on `clean_entropy`: escalate iff `clean_entropy >= tau1`;
  `cascade_score = (tau1 + instability)` if escalated else `clean_entropy` (escalated items rank
  above non-escalated — escalation is itself a positive signal — refined within each region by its
  own signal: a valid monotone combined ranking).
- **Compute** (forward passes / item) `= 1 + N * escalation_rate`. Baselines: resample-all `= N`,
  cheap-only `= 1`.
- Sweep `tau1`; the operating point = the MIN-compute `tau1` whose `AUC_cascade >= AUC_full - 0.02`.

## Bars (fixed)

| Bar | Statement | Threshold |
|---|---|---|
| **A1** | the cascade retains ~full detection at a fraction of full compute | ∃ operating point with `AUC_cascade ≥ AUC_full − 0.02` **AND** `compute ≤ 0.40 · N` (pooled) |
| **A2** *(descriptive)* | the saving is regime-structured | larger compute saving where the cheap gate is strong (derivation) than where it is weak (factual) |

**RESULT = SURVIVED iff A1.** A1 is the falsifiable acceleration claim: if the cheap gate cannot
route compute without losing >0.02 AUC, or needs >40% of full compute to do so, the cascade does not
accelerate and this REPORTS_AS_LANDED with the honest curve.

## Scope (stated before the run)

Compute counted in forward passes (cheap gate = 1, each resample = 1); real wall-clock also depends
on batching. This ACCELERATES detection (routes compute) — it does not change WHAT is detected and
corrects nothing (correction is the closed negative). Single-token closed-model confabulation is out
of scope (the first-token cheap gate fails there; closed/span regimes are excluded). Offline,
feasibility-grade, white-box. The deliverable, if SURVIVED, is a deployable cascade: *run the 1-pass
gate inline, escalate the uncertain fraction* — the honesty layer at near-cheap-gate cost.
