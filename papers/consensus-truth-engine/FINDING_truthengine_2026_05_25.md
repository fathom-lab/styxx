# Finding · The Truth Engine — accuracy-boost FAILS, calibrated abstention WORKS (PASS=FALSE, useful half)

**2026-05-25.** Prereg `preregistration_consensus_2026_05_25.md`. Cross-vendor consensus
as a reference-free answer layer. **Verdict: PASS = FALSE** — T1 fails, T2 passes — and
the split is exactly the honest prior, not a surprise.

## Result (TriviaQA-150 holdout, gpt-4o-mini + Qwen2.5-3B + gemma-2-2b-it)

| metric | value |
|---|---|
| per-model accuracy | gpt-4o-mini **0.887** · gemma-2-2b 0.653 · qwen-3B 0.527 |
| coverage (answered, agreement ≥ 0.66) | 0.79 |
| **consensus accuracy on answered** | 0.890 |
| **best-member accuracy on answered** | **0.941** |
| best-member accuracy on **abstained** | 0.688 |
| **T1** consensus beats best member (+0.05) | **FAIL** (−0.051) |
| **T2** abstention calibrated (Δ ≥ 0.20) | **PASS** (0.253) |

## What failed (T1) — and why it's not a surprise

Consensus did **not** out-answer its best member; it was 5 points *worse*. Cause, exactly
as pre-registered: **two weak local voices (0.53, 0.65) drag a strong one (0.89)** —
majority vote across mismatched strength regresses toward the weaker members. On
near-ceiling TriviaQA there's no headroom for consensus to recover. **The accuracy-boost
claim is false for a strength-mismatched council.** An accuracy lift would need
*comparable-strength* cross-vendor models (strong non-OpenAI = the API-key frontier);
this is not evidence it can't work there, only that it doesn't here.

## What worked (T2) — the real, useful capability

**The abstention is calibrated, and strongly.** When the council fractures (no ≥2/3
agreement → abstain, 21% of items), those are disproportionately the questions even the
*best* model gets wrong: best-member accuracy is **0.941 on answered vs 0.688 on
abstained** — a 25-point gap. So the fracture is a real, reference-free signal of "this is
hard / likely wrong." `styxx.consensus()` is therefore honestly a **reference-free
calibrated *abstention* layer** — "know when to refuse" — **not** an accuracy-booster.
That is genuinely useful (a closed-model agent gets a no-ground-truth "I'm not sure here"
that tracks real difficulty) and it is novel in the reference-free, cross-vendor form.

## Place in the Epistemic Immune System (VISION doc)

Reflex #1 (Consensus) resolves to: **abstention — validated; answer-improvement — not (on
mismatched-strength councils).** Combined with the prior validated reflexes
(confabulation flag, cross-vendor truth-tracking), the immune system is now: **2.5 of 4
reflexes** with receipts — calibrated reference-free abstention added, the accuracy-lift
honestly parked behind the comparable-strength-vendor requirement.

## Honest scope

n=150, single run, τ=0.66, one strong + two weak (2–3B) members. The result is a clean
statement of *where consensus helps and where it doesn't*: it doesn't make a strong model
more accurate by averaging in weak ones (obvious in hindsight, now measured); it does
give a reference-free, calibrated abstention signal. Report stands; no spin. If
`styxx.consensus()` ships, it ships as an **abstention/uncertainty** primitive with this
finding as its scope.
