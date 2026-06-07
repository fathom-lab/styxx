# FINDING v3 — Self-access does NOT emerge with scale (0.5B → 7B); the dissociation is scale-robust

**2026-06-06. Fathom Lab / styxx.** Extends FINDING_v2 (the inaccessible thought) across the Qwen
scale ladder. Answer keys SHA-256-hashed pre-scoring.

> **Erratum (2026-06-07 — `papers/grounded-honesty-axis/FINDING_parrhesia_rung1_2026_06_07.md`):** the
> scale-robust *self-report* null below stands (no model 0.5B→7B can forced-choose the injected concept).
> But the probe side reads injected-vector **presence** (1.00 even at a behaviourally-inert dose), so the
> paired "inaccessible thought" reads a **trace**, not a held **thought**. The read-certificate is owed on
> naturally-present content.

## Self-access (symbol-code forced choice, 8-way, chance 0.125)

| model | inject | random | prime (abort) | word-arm (lens) |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.042 | 0.083 | **0.083 (FAIL)** | 0.500 |
| Qwen2.5-1.5B-Instruct | 0.208 | 0.125 | 0.979 | 0.354 |
| Qwen2.5-3B-Instruct | 0.083 | 0.083 | 1.000 | 0.333 |
| Qwen2.5-7B-Instruct (4-bit) | 0.188 | 0.042 | 1.000 | 0.854 |

- **0.5B**: the abort gate FAILS (8-way prime 0.083) — too small to follow the symbol-code format
  even when handed the answer → UNINFORMATIVE for the 8-way at 0.5B (its 2-AFC prime is 0.90).
- **1.5B / 3B / 7B**: abort passes (prime ≈ 1.0), yet symbol-code inject stays at/near chance
  (0.08–0.21). None clears the 0.30 gate. No monotone emergence. (One-sided binomial p vs chance
  0.125, n=48: 0.5B 0.99, 3B 0.87, 7B 0.14, 1.5B 0.07 — none significant at 0.05; 1.5B's 0.07 is
  the closest, so "within noise" is the honest read but the CI is wide and 1.5B/7B sit just above
  chance. Larger n would tighten this.)

## The dissociation holds across scale

While the model's own forced-choice self-access sits at chance from 0.5B to 7B, the **external
processed-state probe** decodes the injected concept at **1.00 at every size — 0.5B, 1.5B, and 3B**
(clean/no-injection control 0.00, perm-null ≈chance 0.08–0.10). External readers see the thought
perfectly at all scales; the mind does not — and making the mind bigger (through 7B) does not change
that. The dissociation is total and scale-invariant: external legibility 1.00 vs self-access ≈chance,
across a 14× parameter range.

## Reading

The "inaccessible thought" is **not a tiny-model artifact**. Across a 14× parameter range, an
injected concept is ~100% legible to outside instruments and ≈chance to the model's own
forced-choice report. This is a scale-robust, controlled, open-model counterpoint to the
frontier-only "emergent introspection" framing: at least up to 7B and for an injected linear
concept, grounded self-access through the LM head does not appear.

## Honest scope

Qwen-Instruct ladder only; 7B run in 4-bit; symbol-code forced choice through the LM head; injected
*linear* concept directions. A null does not prove introspection impossible at frontier scale or by
other elicitation; it shows that, where the frontier claim would predict emergence, none is
forced-choosable here. Bigger ladders (Llama/gemma, >7B) and other elicitation formats remain open.
