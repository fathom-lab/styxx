# Finding · The Council — reference-free fabrication detection works (PASS); "truth" not yet earned

**2026-05-25.** Prereg `preregistration_council_2026_05_25.md`. Council of 4 OpenAI
models (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`, `gpt-4.1-mini`). **Verdict: PASS**
(C1 ∧ C2) — strong — with a sharp boundary drawn on what it does and does not establish.

## Result (agreement = largest cross-model equivalence cluster ÷ council size)

| tier | mean agreement (judge) | mean agreement (cosine) |
|---|---|---|
| real-common | 1.00 | 0.88 |
| real-obscure | 1.00 | 0.78 |
| fake | 0.28 | 0.31 |

| bar | judge | cosine |
|---|---|---|
| **C1 real vs fake AUC** | **1.00** | 0.99 |
| **C2 obscure-real vs fake AUC** | **1.00** | 0.98 |

PASS (C1 ≥ 0.75 ∧ C2 ≥ 0.70). On all 8 obscure-real questions the four models converged
on the same core answer; on fakes they diverged or abstained (agreement 0.25–0.50).

## Two real positives

1. **Reference-free separation of real from fabricated, AUC 1.0** — no reference, no
   ground-truth lookup; the *council itself* is the grounding. This is a swing at the
   reference-less-validity problem the synthesis filed as closed for text-only methods.
2. **Correlated confabulation did NOT appear** — the deepest pre-registered risk. On
   fakes the models invented *different* fabrications (or abstained); they did not share
   a lie. The "a fabrication has no attractor" hypothesis held on this set.

## The line I am drawing (what is NOT shown)

**AUC 1.0 is exactly where this session has been wrong before, so:** the "obscure-real"
tier (1938 Nobel laureate, *French Connection* director, Calypso Deep) is
**consensus-known** — every frontier model has these in training. So the Council here
demonstrates **reference-free CONSENSUS-vs-FABRICATION, not reference-free TRUTH.** The
mechanism it rides on is "do independent models share this in training," which equals
*truth* only when the fact is widely enough documented to be a shared attractor.

The untested case — and the difference between "detects fabrication" (shown) and
"detects truth" (not shown) — is the **fame↔truth seam**: facts that are *genuinely rare*
(real, but known to only one or two models). There, the truth-attractor hypothesis
predicts convergence (the models that know it agree) while the fame hypothesis predicts
collapse (most don't know it → divergence → false "fake"). This set does not stress that
seam. **Do not cite this as a truth detector.**

## Honest scope & the deepest caveat

- 4 models, **all OpenAI** — shared lineage/training. Same-vendor consensus is the weak
  form; **correlated confabulation and shared blind spots are far more likely within one
  vendor.** A cross-vendor council (key-blocked) is the strong test, and the one that
  would most threaten C1/C2.
- n = 6 / 8 / 8, single run, agreement quantized to {0.25, 0.50, 0.75, 1.0} with a
  4-model council (coarse).
- Fakes here are moderately plausible; *more* plausible fakes (real-adjacent names) are
  the correlated-confabulation stress test.

## Next (the decisive follow-ups, pre-registerable)

1. **Ultra-rare-real tier** — verified-true facts known to ~1–2 of the models — to split
   *fame* from *truth*. This is the experiment that earns or kills the word "truth."
   (Requires careful, reliably-sourced rare facts — mislabel risk is the design hazard.)
2. **Cross-vendor council** — the consensus signal's real threat (blocked on a non-OpenAI
   key).
3. **Plausible-fake stress** — does correlated confabulation appear when the fakes are
   more real-sounding?

## Place in the arc

Intra-sample inconsistency (Tier-3 detector) → inter-*model* agreement (this) are the
same idea on two axes: **fabrication is divergent, fact is convergent.** Across samples it
catches a model's own confabulation; across models it catches fabrication reference-free —
but only up to the consensus boundary. A genuinely new grounding substrate (the council),
honestly scoped: it detects *fabrication vs consensus knowledge*, and whether it reaches
*truth* is the next, harder question.
