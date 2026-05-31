# PREREG — CONFIRMATORY: intent beyond confidence, margin-balanced, on FRESH data

**REGISTERED 2026-05-31, before the confirmatory set is generated or scored.**
**SIGN-OFF:** Flobi — *"go deeper until it is [revolutionary]"* (2026-05-31).

## Why a confirmatory

The first confidence-matched run (`PREREG_intent_beyond_confidence`, set `bc`) gave a split outcome:
- the **registered** top-half-margin test was **INCONCLUSIVE** — surface 0.629 > 0.62, i.e. a crude
  top-half split does not fully neutralize confidence;
- an **exploratory margin-bin-balanced diagnostic** (matching the lie/mistake margin distributions bin by
  bin) DID neutralize it — surface **0.533 ≈ chance** — and the residual probe **still** separated
  confident-lie from confident-mistake at **AUROC 0.707, contrast +0.174.**

That diagnostic is promising but was specified after seeing the `bc` set, so it is **not a claim**. This
confirmatory **locks the bin-balanced matching** and tests it on a **fresh disjoint slice**. Same bars as
the original (not loosened from the peek); only the matching method is upgraded.

## Design (fresh disjoint set, $0, white-box)

- `gen_intent_set.py --skip 1200 --n 700 --tag bc2` — disjoint from **both** the base set (items 0–600)
  and the `bc` set (600–1200). Same Qwen2.5-3B-Instruct, same two-pass protocol.
- **Matching:** among syco-output-wrong items, partition `letter_margin` into **12 quantile bins**; within
  each bin keep an equal number of LIE (neutral-correct) and MISTAKE (neutral-wrong) items. This balances
  the margin distributions → output confidence matched by construction.
- **Probe:** standardized L2 logistic regression on the residual, best single layer by 5-fold CV on a
  70/30 TRAIN split, evaluated on held-out TEST. **Surface ceiling** = best orientation of `letter_margin`
  / `vocab_entropy` on the same TEST.
- The registered statistic is exactly the **margin-balanced block of `score_intent_bc.py --tag bc2`.**

## Bars (FIXED)

| Bar | Statement | Threshold |
|---|---|---|
| **SURFACE-MATCH** *(precondition)* | confidence is genuinely controlled | matched surface **≤ 0.58** |
| **PROBE** *(key)* | the inside separates confident-lie from confident-mistake | TEST AUROC **≥ 0.70** |
| **CONTRAST** *(key)* | the inside beats the matched output | probe − surface **≥ 0.15** |

**RESULT = SURVIVED iff SURFACE-MATCH ≤ 0.58 ∧ PROBE ≥ 0.70 ∧ CONTRAST ≥ 0.15 ∧ powered.**
**Powered:** ≥ 40 LIE and ≥ 40 MISTAKE in the balanced set. Surface > 0.58 → inconclusive.

## Honest scope (stated before the result)

- **AUROC ≈ 0.70 is a MODEST effect**, not a strong detector. A SURVIVED here means *intent leaves a real
  trace beyond confidence* — the first such pre-registered demonstration in this setup — **not** a deployable
  high-accuracy lie detector. Do not let "survives" become "solved."
- The signal sits in the **late layers** (~35/36 of 37), i.e. close to the output — consistent with the
  override being a late decision; report this, don't hide it.
- Single model (Qwen2.5-3B), MMLU, sycophantic-override operationalization, linear probe, correlational
  (a separating direction, not proven intent). A probe just under 0.70, or contrast under 0.15 with matched
  surface, is REPORT_AS_LANDED — reported by its number, not rounded up.

## One line

Confirm on fresh data, with confidence balanced bin-by-bin, that the residual stream still knows which
equally-confident wrong answer was a lie — a modest but real intent trace, or an honest miss.
