# FINDING — the honesty SIGNAL-LOCUS law SURVIVED: span's advantage over first-token GROWS with capability (instrument escalation)

**SURVIVED.** PREREG `PREREG_honesty_signal_locus_2026_05_31.md` (signed *"go harder"*). Same hashed
battery `c7b43dd3…0cfa`. 8-rung white-box ladder, paired first-token vs span on identical items; stats
+ matching via the unit-tested `_evallib`. Scorer `score_locus.py`; receipts `honesty_span_result_*.json`,
`honesty_locus_summary.json`.

## Result

| model | params B | accuracy | first-token | span (min-margin) | **span − first-token** |
|---|---|---|---|---|---|
| gemma-3-1b | 1.0 | 0.232 | 0.496 | 0.556 | +0.059 |
| Llama-3.2-1B | 1.24 | 0.321 | 0.941 | 0.918 | −0.023 |
| Qwen2.5-0.5B | 0.5 | 0.351 | 0.724 | 0.827 | +0.103 |
| Llama-3.2-3B | 3.21 | 0.405 | 0.687 | 0.610 | −0.077 |
| gemma-2-2b | 2.6 | 0.405 | 0.703 | 1.000 | +0.297 |
| Qwen2.5-1.5B | 1.5 | 0.423 | 0.582 | 0.788 | +0.206 |
| Qwen2.5-3B | 3.0 | 0.458 | 0.764 | 0.977 | +0.214 |
| **Qwen2.5-7B** (4-bit) | 7.0 | **0.512** | 0.628 | 0.986 | **+0.358** |

- **LOCUS** (key): Spearman(accuracy, span−first-token) = **+0.695** ≥ +0.60 → **SURVIVED.**
  perm_p = 0.064 (n=8 underpowers strict significance — the registered bar is the effect size; the
  p-value is *suggestive, not conclusive*, disclosed).
- **SCALE-span** (secondary): Spearman(accuracy, span) = +0.527 — span calibration itself trends up, weaker.
- max-entropy advantage: ρ = **0.766** (the same law, stronger).

## The claims that land

1. **Instrument escalation is real.** The gain from reading the whole answer span instead of just the
   first token **grows with capability** (ρ 0.70). Decisive point: **Qwen-7B, the most capable rung, has
   the largest span advantage (+0.358)** — the law's prediction, met at the top of the ladder.
2. **It completes the arc the first-token null opened.** First-token calibration is **flat** with
   capability (ρ = −0.04 here, confirming `FINDING_honesty_scaling_law`); span calibration rises relative
   to it. "Honesty doesn't scale" was the wrong frame — **the legible locus moves outward**: first-token
   → span.
3. **Deployable:** span beats first-token **on average** (0.833 vs 0.691), and the margin compounds with
   model strength — gate on the span, not the first token. Consistent with the detection-locus result
   that span recovers closed-model confab where first-token fails.

## The unified law (across arms)

As capability rises, the honesty signal that *works* migrates outward: **first-token entropy** (weak
white-box) → **span aggregation** (stronger white-box, and closed models) → **stated confidence**
(frontier). This run is the white-box leg, pre-registered and SURVIVED; the closed-model span leg is
`FINDING_detection_locus_gpt_*`; the frontier stated-confidence leg is the self-audit.

## Honest scope

Feasibility-grade, n=8, multiplication only, white-box, single run. **perm_p 0.064** — the effect size
clears the bar but significance is suggestive; more rungs are needed to harden it. Qwen-7B ran 4-bit
(nf4). Span requires multi-token answers. Detects/abstains; corrects nothing. NOT a universal oracle,
NOT cross-vendor.

## One line

Span's advantage over first-token climbs with capability (ρ 0.70, SURVIVED) — the honesty signal's
legible locus moves outward as models get stronger, exactly the law the falsified first-token scaling
pointed to.
