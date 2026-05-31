# FINDING — the honesty SCALING LAW (white-box arm) is FALSIFIED: difficulty-controlled single-pass self-knowledge does NOT scale with capability (0.5–3B); the apparent scaling was a difficulty confound

**REPORT_AS_LANDED.** PREREG `PREREG_honesty_scaling_law_2026_05_31.md` (registered + signed before
data; battery SHA-256 `c7b43dd3…0cfa`). 7-rung white-box ladder, fresh hashed multiplication battery
(168 items, 7 operand-size bins), greedy single pass, exact-integer labels (no judge), local RTX 4070,
feasibility-grade. Scorer `score_honesty_scaling.py`; receipts `honesty_scaling_result_*.json`.

## The bet, and the kill

Holding difficulty fixed, does a model's clean first-token entropy separate the answers it gets WRONG
from the ones it gets RIGHT (`sep_ctrl`), and does that **sharpen with capability**?

| model | params B | accuracy | `sep_raw` (confounded) | **`sep_ctrl`** |
| --- | --- | --- | --- | --- |
| gemma-3-1b-it | 1.0 | 0.232 | 0.837 | 0.496 |
| Llama-3.2-1B | 1.24 | 0.321 | 0.976 | **0.941** |
| Qwen2.5-0.5B | 0.5 | 0.351 | 0.669 | 0.724 |
| Llama-3.2-3B | 3.21 | 0.405 | 0.921 | **0.687** |
| gemma-2-2b | 2.6 | 0.405 | 0.889 | 0.703 |
| Qwen2.5-1.5B | 1.5 | 0.423 | 0.878 | 0.582 |
| Qwen2.5-3B | 3.0 | 0.458 | 0.838 | 0.764 |

- **SCALE:** Spearman(accuracy, `sep_ctrl`) = **+0.126**, exact-permutation p = **0.798** (bar ≥ +0.60,
  p < 0.05) → **FALSE.** Not a weak positive — no relationship (p=0.80).
- **MONOTONE:** Qwen 0.5B→3B Δ = **+0.040** (< 0.05) → FALSE; Llama 1B→3B Δ = **−0.255** → FALSE
  (the *bigger* model is *worse*). Both eligible families fail → **FALSE.**
- Robustness (all-item, bin-standardized `sep_ctrl_z`): Spearman = **−0.234** — agrees, flat-to-inverted.

**RESULT = REPORT_AS_LANDED.** Difficulty-controlled single-pass self-knowledge does not scale with
capability across this 0.5–3B ladder.

## The claims that land

1. **The scaling law is false in this regime.** Flat (ρ=+0.13, p=0.80). The single decisive point:
   **Llama-3.2-1B — the weakest model — reads its own errors best (`sep_ctrl` 0.94), and Llama-3.2-3B
   reads them worse (0.69).** One clean inversion makes any positive scaling unreachable.

2. **The confound was doing the work — the load-bearing methodological result.** `sep_raw`
   (difficulty-uncontrolled) is 0.67–0.98 across the board — exactly the kind of number one might
   publish as "models know when they're wrong." It is mostly *"hard problems carry higher entropy,"*
   not self-knowledge: holding difficulty fixed collapses it (e.g. Qwen-1.5B 0.878 → 0.582), and even
   `sep_raw` itself does not scale (ρ=0.05). **Apparent single-pass self-knowledge signals can be
   difficulty artifacts; control difficulty or the number is not what it looks like.**

3. **Mechanism (EXPLORATORY, `mine_locus_2026_05_31.py`): global confidence-sharpening, not
   calibration.** As capability rises, raw first-token entropy on wrong answers falls hard
   (ρ=−0.775) — but it falls on *right* answers too; the difficulty-controlled, wrong-specific signal
   is flat (z-entropy-on-wrong ρ=+0.04). More capable small models are not more *calibrated*; they are
   more *confident across the board*, which compresses the right/wrong entropy gap. The tempting
   "confident confabulation rises with capability" story is **not** supported at n=7 here — reported as
   it is, not as hoped.

## What it bounds (the honest boundary it draws)

Thesis claim 3 — *honesty has a model-strength gradient* — was measured **Claude (frontier) vs weak**,
a far larger capability gap, in **stated confidence** (Brier ~0.10). This run shows that gradient does
**not** exist for **white-box first-token entropy** in **small models (0.5–3B)**. So the gradient is
bounded to the **frontier and/or stated-confidence** regime; it is not a property of white-box
single-pass uncertainty at small scale. A boundary drawn by a killed bet — load-bearing.

## What it opens (the next bet this surfaces)

The null + the detection-locus arc together suggest the real law is not *"calibration scales"* but
*"the working honesty SIGNAL escalates with capability"*:
- **first-token entropy** — works for weak white-box models, but **flat/saturating** with capability
  (this run);
- **span aggregation** — recovers detection where first-token fails on **strong closed** models
  (`FINDING_detection_locus_gpt_span_2026_05_30`, gpt-4o-mini AUC 0.99);
- **stated confidence / retrieval** — the **frontier** instrument (Claude Brier 0.10; the retrieval
  door).

Pre-registerable next: the **span arm** on this same hashed battery + a strong rung — does
*difficulty-controlled span calibration* scale where first-token did not? That isolates whether the
null is the **signal** (first-token) or the **models**, and assembles the three arms onto one capability
axis as the honest "instrument-escalates-with-capability" map.

## Honest scope

n=7, multiplication only, 0.5–3B (all sub-50% accuracy — no strong rung), first-token only, single
run, instrument-author-set battery (hashed pre-run). Detects/abstains; corrects nothing. Capability
axis = battery accuracy (proxy). NOT a universal oracle, NOT cross-vendor (both CLOSED NEG). The
mechanism analysis is EXPLORATORY (post-hoc), not pre-registered.

## One line

White-box single-pass self-knowledge does **not** scale with capability across 0.5–3B (Spearman
+0.13, p=0.80; Llama inverts) — the apparent scaling was a difficulty confound, more capable small
models just sharpen globally — which bounds the honesty gradient to the frontier/stated-confidence
regime and surfaces the real bet: the working honesty signal *escalates* (first-token → span →
stated/retrieval) with capability.
