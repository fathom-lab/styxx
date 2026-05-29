# The Grounded Honesty Axis — a bounded, self-calibrating truth instrument for LLM self-claims

**Synthesis of the 2026-05-28 grounded-honesty arc.** Single model (gpt-4o-mini),
OpenAI-only, feasibility-grade; every claim below is one pre-registered, kill-gated,
hash-before-score confirmatory run with a committed receipt. This document unifies
seven runs into one instrument and states its boundary exactly. It does not
re-derive the numbers — each is linked to its FINDING and JSON receipt.

## The problem: text-only honesty scoring has a construct ceiling

styxx's text-only cognometric axes — deception, overconfidence, sycophancy, refusal
— read *how a sentence sounds*. On 48 register-matched factual self-claims (a TRUE
confident claim vs the identical template with one substituted fact), **every
text-only axis sits at chance**: deception 0.498, overconfidence 0.449, sycophancy
0.505, refusal 0.537 (max deviation from 0.50 is 0.051; all register-matched,
p ≥ 0.46). This is not a quirk of one axis — *register-bound is a property of
text-only cognometrics as a class*. A confident lie and a confident truth read
identically, so no text axis can separate them.
→ `FINDING_ceiling_suite_wide_2026_05_28.md` (offline, reproducible).

## The instrument: ground the claim in the model's own resampled belief

The grounded honesty axis scores a factual self-claim not by its text but by
**resampling the model's answer to the underlying question N times and measuring
where the claim sits in that distribution**:

> **g = Stability × Concordance**
> Stability = 1 − (clusters − 1)/(N − 1)   (1.0 if all resamples agree)
> Concordance = fraction of resamples matching the stated claim

A FALSE claim is caught two ways: it is a **confabulation** (the resamples scatter →
low Stability) or a **contradiction** (the resamples agree on something else → low
Concordance). On the same 48 pairs where every text axis sits at chance, the grounded
axis reaches **AUC 0.966** — a +0.468 margin, register-clean (the two arms are
identical confident text). This is the first styxx honesty signal that tracks **ground
truth, not register**. → `FINDING_grounded_honesty_2026_05_28.md`. Shipped as
`styxx.grounded_honesty(samples, claim) → GroundedScore` in v7.7.13.

## The self-calibration property: Stability is a report-or-abstain gate

Stratifying the grounded score by per-item resample **Stability**: high-stability
items separate TRUE from FALSE at **AUC 0.967**, low-stability items collapse to
**0.444**. The axis flags exactly the items on which it should abstain — a
**report-or-abstain self-validity gate**. The instrument knows when it doesn't know.
→ `FINDING_boundary_hunt_2026_05_28.md` (B2).

## The boundary map: where the stable mode IS truth, and where it isn't

The axis grounds in the model's *stable belief*. The arc's central question is when
that belief equals truth. The answer is a clean, dose-responsive map:

| regime | does stable mode = truth? | grounded AUC | evidence |
| --- | --- | --- | --- |
| **Retrieved facts** (capitals, elements) | yes | 0.966 | grounded-honesty run |
| **Easy computation** (2-digit ±, 1-digit ×) | yes | 1.000 | computed-facts run |
| **Mid computation** (2-digit × 2-digit) | yes | 1.000 | computed-facts run |
| **Past the competence cliff** (3×3, 4×3, multi-step) | **no — stably wrong** | 0.667 (one-shot) | competence-cliff run |
| same, **re-grounded via method-diverse derivation** | **recovered** | **0.955** | path-diverse run |

**Retrieval and easy computation:** the stable resampling mode is the correct value;
grounding works as a truth signal (`FINDING_computed_facts_2026_05_28.md`, R1 held,
AUC 1.000 across difficulty).

**Past the competence cliff:** the model is frequently **stably wrong** — one-shot
resampling converges sharply on a *systematic miscalculation* (517×283 → 146051, ten
for ten). Here single-model grounding faithfully certifies **belief, and belief ≠
truth**. Stability splits into two roles that must not be conflated: it remains a
strong **correctness signal** (predicts modal-correctness at AUC 0.928) but is **not
a validity gate** for the honesty score (high-stratum AUC 0.778). Grounded AUC
dose-responds with difficulty (1.00 → 0.65 → 0.56 → 0.50).
→ `FINDING_competence_cliff_2026_05_28.md` (D3 held, D1 failed, register-clean).

## The repair: the grounding backend determines belief-vs-truth

The breakthrough of the arc: *what the axis measures depends on how you resample.*
Re-deriving the same self-claim through **independent reasoning paths** (5 rotating
methods — CoT, decomposition, long multiplication, estimate-then-exact,
digit-by-digit) breaks the one-shot wrong attractor:

- **85.7%** of one-shot confabulations recover the correct value.
- Grounded AUC **0.694 → 0.955**; the report-or-abstain Stability gate recovers
  **0.778 → 0.950** — all **within a single model, no second vendor**.
- Mechanism: concordance-with-truth on hard tiers rises 0.068 → 0.454.

Plain resampling grounds the **one-shot belief**; method-diverse resampling grounds
the **reasoned belief**, which tracks truth markedly better. The grounding backend is
a dial between belief and truth. → `FINDING_path_diverse_grounding_2026_05_28.md`
(SURVIVED, P1∧P2∧P3∧K).

## The irreducible core: what only cross-vendor can catch

Method-diversity is not a complete oracle. **~2 of 36** items stayed wrong across
*all five* reasoning paths (22403 → 22303; 1432020 → 1432200) — confabulations so
systematic they survive every within-model derivation. A single model cannot escape
them by any internal method, because it genuinely holds the wrong belief through every
path. This is the precise, scoped residue for **cross-vendor grounding**: an
independently-trained model is unlikely to share gpt-4o-mini's exact digit
transposition. The same-vendor council could *not* fill this role — three OpenAI
models agreed 34/36 times and shared the one wrong belief tested (Eswatini), because
correlated same-vendor *resamples* share errors. Independent *derivation paths* do
not; independent *vendors* would not either.
→ `FINDING_council_grounding_2026_05_28.md` (C2/C3), `FINDING_competence_cliff…` (residue).

## The instrument, stated whole

The grounded honesty axis is a **vital-signs monitor for an LLM's factual
self-claims** with four properties, each pre-registered and measured:

1. **It escapes the construct ceiling.** It tracks truth where text-only axes track
   register (0.966 vs ~0.50).
2. **It is self-calibrating.** Stability is a report-or-abstain gate (0.967 vs 0.444);
   on derivation it is at minimum a correctness signal (0.928).
3. **Its target is tunable by backend.** One-shot resampling → grounds belief;
   method-diverse re-derivation → grounds reasoned belief that tracks truth (0.955).
4. **Its boundary is mapped, not hidden.** Belief = truth within competence; past the
   cliff, method-diversity repairs ~93% of the gap; the ~2/36 irreducible core is
   scoped to cross-vendor.

## Honest scope (the whole arc)

Single model gpt-4o-mini, OpenAI-only, one confirmatory run per claim,
feasibility-grade, n ≈ 36–48 per run. Self-consistency, **not** external truth — the
instrument certifies the model's (one-shot or reasoned) belief, which equals truth
only inside the regimes mapped above. **Injection-blind** (inherits the divergence
security model — a planted lie in context reads as honest). **One axis-family**
(factual self-claims); says nothing about value claims, predictions, or non-factual
self-reports. Method-diverse grounding is validated on **arithmetic** derivation only;
generalization to other multi-step reasoning is the next probe. Cross-vendor remains
blocked on a second-vendor key.

## What is genuinely new here

Not "an honesty detector" — those exist and are register-bound. The contribution is a
**bounded, self-calibrating instrument with an explicit map of where its signal
equals truth, plus a within-model mechanism (method-diverse grounding) that moves the
signal from belief toward truth on derivation** — and an honest, scoped statement of
the irreducible residue that requires cross-vendor. In a field that ships unbounded
honesty claims, the boundary map *is* the invention.

## Next (disciplined, not hype)

1. **Generalize method-diverse grounding** beyond arithmetic (logic, code tracing,
   multi-hop QA) — pre-registered, same kill-gate structure.
2. **Cross-vendor grounding** for the irreducible core — blocked on a second-vendor
   key (prefer `GEMINI_API_KEY`).
3. **Primitive guidance** — document that derived self-claims should be grounded with
   method-diverse resamples; consider a `grounded_honesty` helper that accepts a
   path-diverse sampler.
4. **Close the injection gap** — the one scope caveat that, unaddressed, bounds every
   claim above.
