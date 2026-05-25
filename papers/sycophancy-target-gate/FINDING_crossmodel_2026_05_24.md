# Finding · cross-model replication — PASS (+ a newly surfaced separate FP mode)

**Date:** 2026-05-24 · **Verdict:** the pre-registered kill-gate (P1–P4)
**REPLICATES** cross-model. Run once on hashed holdout `065ea439…` (prereg
`88c81d4`, lock `45156e5`). Generators: `gpt-4o` + `gpt-3.5-turbo`, both distinct
from the `gpt-4o-mini` the v0 detector was trained on.

## Result (τ = 0.30)

| metric | C0 | C1 | **C2** | bar | per-model C2 (4o / 3.5) |
|---|---|---|---|---|---|
| FPR apology | 0.20 | 0.16 | **0.08** | P1 ≤0.20 ✓ | 0.04 / 0.12 |
| — adversarial 2nd-person | 0.06 | 0.00 | **0.00** | — | 0.00 / 0.00 |
| recall flattery | 1.00 | 1.00 | **1.00** | P2 ≥0.90 ✓ | 1.00 / 1.00 |
| AUC flat vs restrained | 0.9925 | 0.9935 | **0.9935** | P3 ≥C0−0.03 ✓ | 1.0 / 1.0 |
| AUC flat vs apology | 0.990 | 0.992 | **0.996** | P4 ≥0.85 ✓ | 0.997 / 1.0 |
| mean apology | 0.168 | 0.138 | **0.098** | — | 0.073 / 0.124 |

95% bootstrap CI (C2 pooled): FPR apology [0.02, 0.16]; AUC flat-vs-apol
[0.987, 1.0]. **Both models individually clear P1** (gpt-4o 0.04, gpt-3.5-turbo
0.12). The gate never misfired on flattery on either model (recall 1.00).

## Honest reading

- **The apology-FP fix replicates across models.** The effect is largest exactly
  where the harm is worst: gpt-3.5-turbo's apologies trip the v0 detector at 0.32
  baseline, and the gate cuts that to 0.12. gpt-4o apologies are already cleaner
  (0.08 → 0.04). The grammatical (attachment) mechanism is not specific to the
  training model — as expected.
- **Scope still bounded.** This is cross-MODEL **within-vendor** (OpenAI). No
  `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` is script-usable in this environment, so
  **cross-VENDOR remains untested** and blocked on an operator-supplied key. A
  different vendor's RLHF register could differ; this run does not speak to that.

## A SEPARATE false-positive mode this work does NOT fix (reported, not hidden)

`fpr_restrained@τ` = **0.30 pooled, 0.60 for gpt-3.5-turbo** — i.e. a large share
of *measured, evidence-first technical answers* also cross the 0.30 gate, and the
self-vs-other gate **does not change this** (restrained answers are outward/neutral,
not self-directed apology, so they are correctly left untouched). This is a
**distinct** v0 over-firing population from the self-apology one this bet targeted:
gpt-3.5-turbo's terse "Yes, it's true that…" answers fire the agreement / counter-
absence features without being sycophantic.

It does not break this result — P1–P4 are all about apology-FP and flattery-recall,
which pass — and the gate does not make it worse (restrained is unchanged across
C0/C1/C2). But it is the honest headline limit: **this work fixes the self-directed-
apology false positive; it does not fix the broader "terse honest agreement reads as
sycophancy" ceiling**, which is most visible on gpt-3.5-turbo. That is a candidate
for the next pre-registered bet (it likely needs the same counter-absence /
agreement-register rethink, possibly the documented NLI stance feature), not a
lexical patch.

## Ship status

With cross-model replication passing, the documented ship-gate is met. The guard
is integrated as `cognometrics._cogn_needs_revision(..., response=...)` via
`styxx.guardrail.self_directed_gate` (suppress-only `min(raw, gated)`; touches no
published instrument weights/fingerprint), with tests. Cross-VENDOR replication
remains the open follow-up before claiming model-general transfer.
