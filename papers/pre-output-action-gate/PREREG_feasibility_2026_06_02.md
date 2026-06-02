# PREREG — Pre-output action gate, FEASIBILITY probe

**Date:** 2026-06-02
**Status:** PRE-REGISTERED (method + kill-gate frozen before any data)
**Brief:** `.styxx/STYXX_PREOUTPUT_ACTION_GATE_2026_06_02.md`
**Lineage:** extends the SURVIVED pre-output refusal gate
(`papers/pre-output-gate/`) from text refusal to agent actions.

## The one question

Before committing days to a full agent-trajectory study, answer the cheap
prerequisite: **does an open-weight model's end-of-prefill residual (before any
token is generated) encode the destructive-vs-safe distinction of a requested
action at all?** If not, the pre-output action gate is far harder than hoped
and we pivot. If yes, the necessary condition holds and we build the real
study.

This is a NECESSARY-condition feasibility test, NOT the product claim. It does
not yet test prediction of *emitted* actions on real agent trajectories.

## Data (frozen)

`papers/pre-output-action-gate/action_intent_set.py` — **48 prompts**, 24
matched destructive/safe pairs across **24 action domains** (db, filesystem,
cloud, git, finance, access, email, k8s, backups, secrets, …). Each pair shares
domain + target resource and differs only in destructiveness/irreversibility.
**sha256:** `3266ecc75b3e3d402a2fec20fbfcbb283e06215b559f172e4d1add527d40bab7`
(run asserts this before scoring).

Honest limitation, stated up front: on this clean controlled set, separation
can be partly **lexical** (the destructive verb is in the prompt). That is
acceptable for a feasibility GO/NO-GO. The leave-one-domain-out (LODO) metric
is the first check of whether the signal is **abstract** rather than per-domain
verb memorization.

## Models (frozen)

Cached open-weight instruct models: Qwen2.5-1.5B, Qwen2.5-3B,
Llama-3.2-1B, Llama-3.2-3B.

## Method (frozen)

For each model, each prompt: read the residual at the **last prefill token**
for **every layer** (`output_hidden_states`, no generation). Then:
- **LOO AUC per layer** (leave-one-out logistic regression, standardized,
  L2 C=0.5): primary "does the signal exist" estimate.
- **LODO AUC** (leave-one-domain-out) at the LOO-best layer and at a fixed ~65%
  proportional-depth layer: "is it abstract" — train on 23 action domains,
  test on the held-out 24th.

## KILL-GATE (frozen — no goalpost-moving)

**Feasibility SURVIVES iff** best-layer **LOO AUC ≥ 0.80 on ≥ 2 models**
(of the models with valid runs, `|ok| ≥ 2`).

`0.80` is set now. On a clean controlled set, a real pre-emission signal should
clear it comfortably; failing to is itself informative (the residual does not
linearly encode destructiveness pre-emission).

## Pre-stated readings (fixed before data)

- **SURVIVED (abstract):** gate passes AND median LODO (of passers, best layer)
  ≥ 0.70 → the pre-emission residual encodes destructiveness AND it generalizes
  to unseen action domains. Strong green light for the full study.
- **SURVIVED-WEAK:** gate passes but median LODO < 0.70 → the signal exists but
  may be largely per-domain/lexical; proceed, but the full study must control
  for lexical confounds (paraphrase, novel verbs).
- **NOT SURVIVED:** fewer than 2 models clear LOO AUC 0.80 → the pre-emission
  residual does not cleanly encode destructiveness even on a clean set. Do NOT
  build the full action-gate study on this mechanism; publish the boundary and
  pivot (runner-up: hallucination/abstain).

## Integrity controls

- Single deterministic pass, inputs hash-pinned, gate fixed in this file before
  any residual is read.
- LOO and LODO are honest generalization estimators (no sample trains on
  itself; LODO holds out whole action types).
- Runner (`run_feasibility.py`) committed with this prereg, before results.

— frozen 2026-06-02
