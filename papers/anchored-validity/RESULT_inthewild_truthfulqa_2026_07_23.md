# RESULT — In-the-wild: gold anchors on a real heterogeneous panel (TruthfulQA) — PARTIAL, and a mechanism — 2026-07-23

Frozen by `PREREG_inthewild_truthfulqa_2026_07_23.md` (commit bf0848e; red-teamed GO_WITH_FIXES).
Receipt: `inthewild_truthfulqa_result.json`. Runner: `run_inthewild_truthfulqa.py`. **Verdict:
PARTIAL** under the frozen forks — the strict KILL bar (failure at >=2 of 3 prevalences) was met at
only one prevalence, so no kill is claimed; the measured pattern is reported verbatim.

## One line

On a real heterogeneous judge panel (Gemini-2.5-flash-lite + Qwen2.5-3B + Qwen2.5-1.5B) auditing
TruthfulQA correctness, blatant gold anchors license correct numbers **when the failure is rare** and
miss it by up to **0.249** **as the failure becomes common** — while honest ladder anchors track the
truth **perfectly at every prevalence (45/45)**. Gold-anchor validity is prevalence-dependent; the
honest ladder is not.

## Preconditions (all passed, one judge voided honestly)

- **Parse-rate:** gemini-2.5-flash **0.9304 -> VOIDED** (< 0.95). Disclosure: this is a *harness*
  limitation, not a judge-quality result — gemini-2.5-flash is a thinking model whose reasoning
  occasionally consumed the whole output budget and returned no parseable verdict; it is not evidence
  about its judging. flash-lite / qwen-3b / qwen-1.5b = 1.0. **Audited panel = the 3 valid judges.**
- **Sensitivity (positive control) fired:** every judge flagged the blatant-absurd gold-positives
  INCORRECT at >= 0.95 (flash 1.0, flash-lite 1.0, qwen-3b 1.0, qwen-1.5b 0.95). The probe is not deaf.
- **Diversity:** organic cross-judge disagreement 0.69 (>= 0.15) -> the panel is not rank-1; the audit
  can run (this is exactly the fatal the red-team caught for a single-model persona panel).

## The measurement (15 replicates x 3 prevalence points)

| true prevalence | GOLD covers / miss / void | GOLD median miss | LADDER covers |
|---|---|---|---|
| 0.30 | 14 / 1 / 0 | 0.127 | 15 / 15 |
| 0.50 | 8 / 7 / 0 | 0.1545 | 15 / 15 |
| 0.70 | 1 / 14 / 0 | 0.2493 | 15 / 15 |

Frozen forks: KILL at one of the three prevalences (only p=0.70 clears gold-miss>=13 &
median-miss>=0.15 & ladder-covers>=10), REFUTED at one of the three (only p=0.30 has gold-covers>=10),
DEGENERATE at none. Neither reaches the two-of-three threshold -> **PARTIAL**, reported verbatim. No bar
was moved.

## The mechanism (why gold fails with prevalence)

The judges' measured false-negative rate on imitative falsehoods (missing a plausible-but-false answer)
is **flash-lite 0.08, qwen-3b 0.10, qwen-1.5b 0.72** — two decent judges and one weak one. Blatant gold
anchors (obvious true / absurd false) put every judge at ~0 error, so the audit infers "the panel is
near-perfect" and reports an over-confident calibration. When the true prevalence of incorrect answers
is low, few items are exposed to the panel's real (weak-judge-driven) miss rate, so the gold-anchored
estimate stays near truth. As prevalence rises, the un-modeled false-negatives dominate and the
gold-anchored estimate drifts — to a 0.249 median miss at p=0.70. **Ladder anchors** — real held-out
`Best Incorrect Answer`s drawn from the same distribution — measure that false-negative rate directly,
so the ladder-anchored audit covers the truth at every prevalence. This is the paper's thesis with a
new, quantitative twist: gold's licence is prevalence-conditional; the honest ladder's is not.

## Reading

This corroborates and sharpens the paper rather than replaying its dramatic kill. The paper located the
danger zone in the "weak-to-mid judge tier"; here, a real panel with two decent frontier/local judges
does **not** collapse — the weak judge (qwen-1.5b, FN 0.72) carries the failure, and gold hides it only
as the measured quantity becomes common. The frozen fork returning PARTIAL, not KILL, is the discipline
working: on a real panel the failure is real but conditional, and we report the condition instead of a
headline. The instrument's own repair — ladder anchors — worked in the wild, covering 45/45.

## Scope / disclosure

Single API vendor (OpenAI key dead; disclosed), gemini-2.5-flash dropped on a thinking-model parse
limitation (disclosed), one dataset (TruthfulQA; ground truth = its labels, with the usual label-noise
caveat), the failure mode is TruthfulQA's own imitative falsehoods. Positive class = "answer is
incorrect." The method can only void a judge, never bless one; PARTIAL means "the failure is real but
did not clear the frozen kill bar on a real panel," not "judges are fine." OATH-certified.
