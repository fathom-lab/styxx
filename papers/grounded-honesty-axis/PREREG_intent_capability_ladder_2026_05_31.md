# PREREG — does the intent trace STRENGTHEN with capability? (the ladder)

**REGISTERED 2026-05-31, before the ladder is generated or scored.**
**SIGN-OFF:** Flobi — *"go deeper until it is [revolutionary]"* (2026-05-31).

## Why

The confirmatory (`PREREG_intent_beyond_confidence_CONFIRM`, set `bc2`) SURVIVED at 3B: with confidence
matched to chance (surface 0.506), the residual separates a confident LIE from a confident MISTAKE at
AUROC 0.745. That is a real but **modest** single-model finding. The question that would make it a *law*:

> Does intent-beyond-confidence get **easier to read as the model gets more capable**?

Our SURVIVED signal-locus law (where confab becomes legible escalates with capability) motivates the
direction — but this is a **different construct** (the sycophantic-override trace, not confab legibility),
so it is a genuine open test, not a foregone conclusion.

## Design

- Same disjoint MMLU slice for every rung: `--skip 1200 --n 700`. Each model gets its **own**
  behaviorally-defined LIE/MISTAKE sets (neutral establishes knowledge; syco pass captures residuals).
- **Ladder (Qwen2.5-Instruct family):** 0.5B, 1.5B, 3B (=`bc2`, already run), 7B (4-bit).
- **Statistic per rung:** the confirmed **margin-bin-balanced** intent-beyond-confidence AUROC (confidence
  matched to ~chance, probe best-layer CV-on-TRAIN / eval-TEST) from `score_intent_bc.py`.

## Hypothesis & bars (FIXED)

- **Primary prediction:** Spearman(log-params, balanced-AUROC) across the 4 rungs is **> 0**, AND
  **7B-rung AUROC ≥ 3B-rung AUROC** (the signal does not collapse at scale).
- Report the **matched-surface at every rung** (each must be ≤ 0.58 to count as confidence-controlled;
  rungs that fail the match are reported but excluded from the trend as inconclusive).

**CLAIM ("intent legibility scales with capability") iff ρ > 0 ∧ 7B ≥ 3B ∧ ≥3 rungs confidence-matched.**

## Honest scope (LOUD — stated before the result)

- **n = 4 rungs is very low power.** Spearman on 4 points cannot reach significance; this is a *trend*, not
  a p-value. The 4 numbers are reported raw and the small-n caveat is the headline, exactly as the locus
  law disclosed its n=8 perm_p.
- **Within Qwen2.5 only** — within-family scaling, NOT cross-vendor universality (that is a closed
  negative; not re-litigated).
- 0.5B may be underpowered (few "knew-it" items) — reported, possibly excluded.
- A **flat or negative** trend is a real, publishable finding ("the intent trace is capability-flat") and is
  reported as such. No narrative is pre-committed.

## One line

Run the same matched intent test up a 0.5B→7B ladder and see whether the lie gets *louder from the inside*
as the model gets smarter — a law if it climbs, an honest flat line if it doesn't.
