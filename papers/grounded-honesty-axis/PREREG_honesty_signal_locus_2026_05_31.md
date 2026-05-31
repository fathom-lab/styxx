# PREREG — the honesty SIGNAL-LOCUS law: does span's advantage over first-token GROW with capability?

**REGISTERED 2026-05-31, before the span signals were measured on any rung.** Same battery as the
first-token arm, reused by import (SHA-256 `c7b43dd38e11b5893b6d56ee720e4fe3e2c126e0bb472cd8c353887f17fa0cfa`).
**SIGN-OFF:** Flobi — *"go harder"* (2026-05-31).

## Why this bet (what the falsified arm surfaced)

`FINDING_honesty_scaling_law_2026_05_31.md` killed the first-token scaling law (REPORT_AS_LANDED,
Spearman +0.13, p 0.80; Llama inverts). The detection-locus arc separately showed first-token **fails**
on strong closed models where **span aggregation recovers** detection (gpt-4o-mini 0.76 → 0.99). Put
together, the hypothesis is not "calibration scales" but **the working honesty signal escalates**: as a
model gets stronger, legibility moves from the first token to the whole span. Operationally: span's
*advantage over first-token* should grow with capability.

## The bet, in one sentence

The difficulty-controlled self-knowledge **gain from reading the whole answer span instead of just the
first token** increases with model capability — `sep_ctrl_span − sep_ctrl_firsttoken` rises with
accuracy.

## Design (paired, same items)

Same hashed battery (168 items, 7 operand-size bins) and same ladder. Per item: one greedy answer
(exact-integer label), then read the realized answer span in ONE forward pass and compute, on the SAME
item: first-token entropy, span max-entropy, span min-margin. Difficulty-controlled `sep_ctrl` (within
operand-size bin, ≥3 wrong & ≥3 right, sample-weighted AUC wrong>right) for each signal. Span primary
= **min-margin** (the detection-locus winner on numeric answers); span max-entropy reported alongside.
Pairing on identical items makes `span − first-token` a clean within-item contrast, not a cross-run one.

## Ladder

The 7 first-token rungs (Qwen2.5 0.5/1.5/3B, Llama-3.2 1/3B, gemma-2-2b, gemma-3-1b) **plus** a strong
top rung **Qwen2.5-7B-Instruct (4-bit)** if it loads in ~7 GB — widening the capability range the
first-token arm lacked (it topped out at 46% accuracy). If 4-bit is unavailable, n=7 and the
narrow-range limit is disclosed.

## Bars (FIXED)

| Bar | Statement | Threshold |
|---|---|---|
| **LOCUS** *(key)* | span out-grows first-token as capability rises | Spearman(accuracy, `sep_ctrl_span_minmargin − sep_ctrl_firsttoken`) **≥ +0.60** |
| **SCALE-span** *(secondary)* | span calibration itself scales | Spearman(accuracy, `sep_ctrl_span_minmargin`) — reported |

**RESULT = SURVIVED iff LOCUS ≥ +0.60.** Exact-permutation p reported; n ≤ 8 underpowers strict
significance (disclosed) — the bar is the effect size. Powered ≥ 30 wrong + 30 right per rung.

## Scope and honest boundary

Same as the first-token arm: white-box open families only, multiplication only, single run,
feasibility-grade, capability axis = battery accuracy (proxy). Span requires multi-token answers
(numeric answers qualify). Detects/abstains; corrects nothing. NOT a universal oracle, NOT
cross-vendor (both CLOSED NEG). A NULL here (span also flat / no growing advantage) is the harder,
equally-publishable claim: white-box honesty calibration does not scale by signal locus either.

## One line

Does reading the whole answer instead of just its first token buy *more* self-knowledge as models get
stronger — the instrument-escalation law — measured paired on the same hashed battery, with an exact
kill-number.
