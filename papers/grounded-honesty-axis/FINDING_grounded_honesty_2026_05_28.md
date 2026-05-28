# FINDING — Grounding the honesty axis breaks the construct ceiling (factual self-claims)

**2026-05-28. Pre-registered (PREREG_grounded_honesty_axis.md), kill-gated,
single confirmatory run. Result: SURVIVED.** Feasibility-grade: one run,
OpenAI-only, n=48 register-matched pairs. Receipt:
`grounded_honesty_result.json` (answer key SHA-256'd before scoring).

## Result

| scorer | AUC (TRUE vs FALSE self-claim) |
| --- | --- |
| grounded honesty `g = Stability x Concordance` | **0.9661** |
| text-only deception axis (register baseline) | **0.4983** (chance) |

- **H1 held:** grounded AUC 0.966 >= 0.75.
- **H_kill held (decisive):** margin **+0.468** >= 0.15. The grounding carries
  essentially all the discriminating signal.
- **K3 held (no confound):** the register baseline does NOT separate the arms
  (Welch t = -0.74, p = 0.46). TRUE and FALSE self-reports are register-matched
  (same confident template, one substituted fact), so the gain is not leakage —
  the text-only axis sits at chance *by construction-controlled design*, which is
  the construct ceiling quantified.

## What this means

Every text-only cognometric axis styxx ships is a register detector with a
validated construct ceiling: it reads how text *sounds*, not whether it is *true*.
The deception axis is the worst case (scores ~0.99 on benign truthful text). Here,
for the first time, a styxx honesty signal **tracks ground truth rather than
register**: by grounding a factual self-claim against the model's OWN resampled
belief distribution (an external, sampling-based signal), TRUE and FALSE
self-claims separate at AUC 0.97 where the register axis is at 0.50.

Mechanism (as theorized in the pre-registration): a FALSE self-claim is either a
confabulation (no stable belief -> low Stability) or a contradiction (stable
belief, but the claim is outside it -> low Concordance). A TRUE self-claim is the
stable sampling mode (both high). `Stability = 1 - (clusters-1)/(N-1)`,
`Concordance = fraction of N resamples matching the claim`, `g = Stability x
Concordance`, N=10, temp=1.0, gpt-4o-mini, LLM same-answer judge.

## Honest bounds (stated, not hidden)

- **Judge-backend false-positives.** 3/48 FALSE items (`Na`/`Cl`/`P` element
  symbols) scored grounded-honest because the LLM same-answer judge leniently
  equated 2-character wrong siblings (`So`, `Ch`, `Ph`) to the samples — the
  documented `same_fn` failure mode on ultra-short tokens. This is why AUC is
  0.966, not 1.0. A domain-exact matcher would remove it; we report the
  pre-registered judge backend as-run.
- **Self-consistency, not external truth.** The signal grounds a claim against the
  model's own stable belief. In knowledge regimes the model covers, that stable
  mode *is* the truth (the council_agreement work validated this attractor is
  truth-tracking, not fame-tracking). Outside the model's knowledge the attractor
  is absent and the signal degrades — abstention rate must be reported alongside.
- **Blind to context-injection.** Inherits the divergence security model: a
  falsehood planted in the prompt (RAG/tool-output poisoning) collapses divergence
  and reads as honest. This detects a model's OWN (in)consistency, NOT
  adversarially planted lies.
- **One axis, not the whole ceiling.** This grounds *factual* self-claim honesty.
  It says nothing about value claims, predictions, or sycophancy/overconfidence.
  We do not claim the construct ceiling is broken in general — only that this one
  axis is now grounded in the scoped regime.

## Why it matters for the arc

The attestation stack (digest -> portable -> transparency log -> redactable
disclosure) is rigorous infrastructure around a register-bounded signal. This is
the first crack in the bound itself: a grounded honesty axis that could be
attested. The natural next steps (NOT yet done, not claimed): (1) cross-vendor
council grounding (needs a 2nd vendor key) to remove the single-model
self-consistency caveat; (2) replace the LLM judge with a domain-exact matcher and
re-run to close the 3-item gap; (3) wire the grounded axis into `attest(...)` as a
re-derivable, ground-truth-tracking field alongside the register vitals.
