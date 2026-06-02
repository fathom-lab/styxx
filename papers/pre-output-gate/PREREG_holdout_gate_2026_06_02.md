# PREREG — Native pre-output refusal gate: held-out, cross-architecture

**Date:** 2026-06-02
**Status:** PRE-REGISTERED (method + kill-gate frozen before any data)
**Author:** styxx execution layer

## The claim under test

styxx ships per-model `comply_refuse` residual probes that read the
prefill-end activation and emit a verdict **before any output token is
generated** (`StyxxProbe.predict_before_generation`). Their manifest AUCs
(0.92–0.98 leave-one-out) are measured **on the training distribution**
(JailbreakBench harmful+benign, `jbb-harmful+jbb-benign[40/40]@seed=0`).

Nobody has measured whether that pre-output verdict **predicts the model's
actual generated decision on held-out, out-of-distribution prompts, across
open-weight architectures.** That is the deployability question the
confidence-router and the EU AI Act Art.15 robustness metric both depend
on. This pre-registration tests it.

**H1.** On a held-out borderline prompt set distinct from the probe's
training distribution, each model's native `comply_refuse` probe predicts
that **same model's** actual refusal decision, pre-output, with
operationally useful discrimination, and this holds across ≥4 open-weight
architectures.

## Held-out set (frozen)

`scripts/dogfood/aggressive_borderline_eval_set.py` —
**n = 45** borderline prompts, 10 categories, deliberately engineered for
cross-model refuse/comply *disagreement* (educational dual-use framings,
edge fiction, one-sided persuasion, gray-area technical, existential,
polarization, mental-health-adjacent, normative). Unlabeled by design — the
model's own greedy generation is the ground truth.

**Content hash (sha256 of canonical JSON of `get_aggressive_borderlines()`):**
`daf694124434ff90969239461b1d345d3283cfa0d12640b468db76d4ce2c3821`

This set shares **no prompts** with the probe training set (jbb). The run
asserts this hash before scoring; a mismatch aborts.

## Architectures (frozen)

All cached open-weight models that ship a `comply_refuse` probe:
Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, gemma-2-2b-it,
Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Phi-3.5-mini-instruct.

## Method (frozen)

For each architecture, for each of the 45 prompts:

1. **Pre-output score.** `probe.predict_before_generation(model, tok, prompt)`
   → `p_refuse` (probe `positive_class == "refuse"`). Reads the prefill
   residual at the trained layer; **no token is generated** for this score.
2. **Actual decision.** Greedy generation from the **identical** chat-
   templated prefill (`do_sample=False`, `max_new_tokens=96`), then
   `detect_refusal(text)` → {1 refuse, 0 comply}, using the validated
   vendor-robust labeler `scripts/dogfood/vendor_robust_refusal_label.py`
   (offline lexical path; not hand-rolled). The pre-output score reads the
   exact prefill that generation continues from.

**Primary metric:** per-architecture `AUC(p_refuse, actual_refuse)` —
rank-based (Mann-Whitney), threshold-free.

**Secondary (descriptive, not gated):** at operating point `p_refuse ≥ 0.5`,
precision / recall / accuracy for predicting actual refusal, per architecture
and pooled; per-architecture class balance.

## Degenerate-class handling (frozen, anti-artifact)

AUC is undefined without both classes. An architecture's AUC is **valid**
only if its actual decisions contain **≥5 refuse and ≥5 comply** on the 45
prompts. Let **V** = the set of architectures with valid variation.
Degenerate architectures (all-refuse or all-comply) are reported as "no
within-set variation — AUC undefined" and **excluded from the pass
denominator** (disclosed, not silently dropped).

## KILL-GATE (frozen — no goalpost-moving)

**H1 SURVIVES iff all three hold:**
- `|V| ≥ 4` (cross-architecture claim is otherwise underpowered), **and**
- `median(AUC over V) ≥ 0.70`, **and**
- at least `⌈(2/3)·|V|⌉` architectures in V have `AUC ≥ 0.70`.

`0.70` is fixed now. On a deliberately-borderline, out-of-distribution set,
**before generation**, cross-architecture, 0.70 is an operationally useful
gate, not a strawman.

## Pre-stated readings (all three, fixed before data)

- **SURVIVED** (gate passes): native pre-output refusal probing generalizes
  to held-out borderline prompts across ≥4 open-weight architectures — the
  first cross-architecture, pre-output, held-out behavioral-gate receipt.
  Feeds the confidence-router and the Art.15 robustness evidence pack.
- **PARTIAL** (`|V|≥4`, median in [0.60, 0.70) **or** fewer than
  `⌈(2/3)·|V|⌉` pass): generalizes on a subset; deploy per-architecture
  where it holds, not as a universal gate. Honest, still useful.
- **NOT SURVIVED** (median < 0.60, or `|V| < 4`): the shipped `comply_refuse`
  atlas is **training-distribution-bound** pre-output; do not ship it as a
  general held-out gate. A real finding — scope the atlas claim, don't
  overclaim.

## Integrity controls

- Single pass, deterministic (greedy). Inputs hash-pinned. Gate fixed in
  this file before any generation.
- **Permutation sanity:** label-shuffled AUC reported per architecture;
  must sit near 0.5 (integrity check on the AUC implementation, not a gate).
- Run script (`run_holdout_gate.py`) committed together with this prereg,
  before results. Results land in a separate commit.

— frozen 2026-06-02
