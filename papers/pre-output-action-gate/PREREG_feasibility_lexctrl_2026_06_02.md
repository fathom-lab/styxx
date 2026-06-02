# PREREG — Pre-output action gate, LEXICAL-CONTROL feasibility

**Date:** 2026-06-02
**Status:** PRE-REGISTERED (gate frozen before any data)
**Supersedes test in:** [PREREG_feasibility_2026_06_02.md](PREREG_feasibility_2026_06_02.md)
(round 1 was lexically separable → AUC 1.000 = confound;
[RESULT_feasibility_2026_06_02.md](RESULT_feasibility_2026_06_02.md))

## The question, corrected

Round 1 asked "does the residual separate destructive/safe?" — and the answer
(1.000) was meaningless because the prompts were separable on the destructive
*verb* alone. The corrected, honest question is **comparative**:

> Does the end-of-prefill residual separate destructive-from-safe **substantially
> better than the prompt text alone**, on a set where destructiveness lives in
> the value/consequence rather than a giveaway verb?

If yes, the residual encodes destructiveness beyond the lexical surface — the
first real evidence for the action gate. If the text baseline does just as well,
the "signal" is lexical and the prompt-level approach is a dead end.

## Data (frozen)

`papers/pre-output-action-gate/action_intent_lexctrl.py` — **46 prompts**, 23
matched destructive/safe pairs, 23 action domains. Each pair shares a **neutral
verb** (set/scale/update/change/grant/route/shift) and resource; destructiveness
is only in the value (`ENABLE_AUTH→false`, `chmod /etc/shadow→777`,
`token lifetime→100 years`, `replicas→0`, `bucket→public-read-write`, …). Values
are varied so no single token is reliably destructive.
**sha256:** `8f065c888cd074b020f149eb9feb83c6997c44ea4225cce4d9869c902f29b956`

## Method (frozen)

All AUCs under **leave-one-domain-out (LODO)** — train on 22 action domains,
test the held-out one (overfitting-robust; the round-1 lesson):
- **residual LODO** per layer (last prefill token; standardized L2 LR, C=0.5);
  report best layer + fixed ~65% depth.
- **bag-of-words LODO** (TfidfVectorizer 1–2-gram → L2 LR) — the text-only
  baseline / confound detector. Model-independent.
- **margin = residual_LODO(best) − bow_LODO.**
- LOO reported for reference. Models: Qwen2.5-1.5B/3B, Llama-3.2-1B/3B.

## KILL-GATE (frozen — no goalpost-moving)

**SURVIVES iff** on **≥ 2 models**:
`residual_LODO(best) ≥ 0.70` **AND** `margin_vs_bow ≥ 0.15`.

i.e. the residual generalizes across action domains AND beats the text baseline
by a clear margin. Either condition failing = no claim.

## Pre-stated readings (fixed before data)

- **SURVIVED:** gate passes → the residual encodes destructiveness beyond the
  prompt text. First real evidence; proceed to the emitted-action study.
- **INCONCLUSIVE / LEXICAL:** `bow_LODO ≥ 0.70` (the text baseline itself
  separates) → the set is still not controlled enough; the residual's apparent
  edge is suspect. Rebuild harder, or skip straight to emitted-action.
- **NULL:** best residual_LODO `< 0.70` → the residual does not separate even on
  the controlled set. The destructiveness signal is not in the pre-emission
  prompt residual; if it exists, it is in the model's *chosen action* — test via
  the emitted-action study only, and lower the priors.
- **NOT SURVIVED (other):** signal present but margin `< 0.15` → no demonstrated
  advantage over text; don't claim a residual capability.

## Integrity

Single deterministic pass; inputs hash-pinned; gate frozen here before any
residual is read. Text baseline run on identical prompts/folds. Runner
committed with this prereg, before results.

— frozen 2026-06-02
