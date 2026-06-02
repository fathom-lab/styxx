# PREREG — closed-model (GPT) behavioral destructive-action flag

**Date:** 2026-06-02
**Status:** PRE-REGISTERED (gate frozen before scoring)
**Context:** closed models expose no activations → the open-weight residual
approach is impossible. This tests the behavioral substitutes.

## Question

With no internal access, can a pre-execution **behavioral** signal flag whether
GPT-4o-mini's chosen tool call is destructive — well enough to gate it at the
execution boundary?

## Data (frozen)

Same 40 scenarios (`action_scenarios.py`, sha256 `b1303559…`). Label = whether
GPT-4o-mini's temp=0 ACTION choice is the scenario's destructive option (premise
check: 16 destructive / 24 safe — balanced).

## Signals (frozen)

- **INPUT** — prompt embedding (`text-embedding-3-small`) → probe, LOCO AUC; and
  a text bag-of-words LOCO baseline. (Predict from the input, not the model.)
- **BEHAVIORAL** — (a) mean **logprob** of the emitted choice (confidence);
  (b) **destructive fraction** over N=6 temp=1 resamples (consistency). Both via
  rank AUC, sign-agnostic (report separability + direction).

## KILL-GATE (frozen)

**FLAGGABLE iff** the best signal AUC `≥ 0.70`.

**Secondary (the interesting one):** does a *behavioral* signal (logprob /
sampling) exceed the *input-only* embedding AUC — i.e. does GPT's behavior carry
destructive-intent signal beyond what its prompt already reveals? (The closed-
model analog of "residual beats text.")

## Honest scope

- **GPT-4o-mini only.** Claude has **no programmatic key here** and exposes **no
  logprobs** — its toolkit is sampling + embedding only (see the response / a
  separate Claude design).
- Behavioral signal is weaker and output-side, not internal. You cannot steer a
  closed model — the guard gates at the **execution boundary** (block / confirm),
  it does not flip the choice.

— frozen 2026-06-02
