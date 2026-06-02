# PREREG — open / native tool-calling replication

**Date:** 2026-06-02
**Status:** PRE-REGISTERED (gate frozen before scoring)
**Tests:** the #1 validity caveat of [RESULT_emitted_action](RESULT_emitted_action_2026_06_02.md)
— that result used a presented menu + "ACTION: <name>" text format.

## The question

Does the pre-emission residual still predict the model's chosen destructive
action when the model acts through its **native function-calling interface**
(real `tools=` schemas, structured `tool_call` output) instead of our presented
menu? If yes, the menu was not doing the work and the result is robust to the
realistic elicitation. If no, the format mattered — an honest bound.

## Method (frozen)

Same 40 scenarios (`action_scenarios.py`, sha256 `b1303559…`), same label (the
**emitted** tool's destructiveness), same residual read (last prefill token, all
layers) and same LOCO + text-baseline scoring as the parent study. The ONLY
change: tools are presented via the native `apply_chat_template(tools=…)`
function-calling interface and the emitted `tool_call` is parsed from the
model's structured output. Models with native tool support: Qwen2.5-1.5B/3B,
Llama-3.2-3B. (Smoke: Qwen-1.5B 15/24, Llama-3B 11/29 — balanced, clean parse.)

## KILL-GATE (frozen)

**REPLICATES iff** on **≥ 2 class-balanced models** (≥8 each):
`residual_LOCO(best) ≥ 0.70` **AND** `margin_vs_bow ≥ 0.15`.

## Readings (fixed)

- **REPLICATED:** ≥2 models pass → the signal survives native tool-calling; the
  presented-menu format was not the cause. Materially strengthens the claim and
  the ActionGuard primitive.
- **UNDERPOWERED:** <2 balanced models under native calling → inconclusive.
- **FORMAT-DEPENDENT:** balanced but <2 pass → the earlier result leaned on the
  menu format; the claim is bounded to that harness until redesigned.

Single deterministic pass, inputs hash-pinned, gate frozen here. Runner
committed before results.

— frozen 2026-06-02
