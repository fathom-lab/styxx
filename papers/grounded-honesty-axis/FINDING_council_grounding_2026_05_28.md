# FINDING — Cross-model council grounding: same-vendor diversity is too low to add a gate

**2026-05-28. Pre-registered (PREREG_council_grounding.md), one confirmatory run.
Feasibility-grade: three OpenAI models, n=36 hard pairs.**
Receipt: `council_grounding_result.json` (answer key SHA-256'd before scoring).

Council = `gpt-4o-mini` + `gpt-4o` + `gpt-3.5-turbo`, N=10 resamples each,
pooled (30 samples/item) and judged against each claim.

## Headline (the split: C1 held, C2 and C3 failed — and the failures are the point)

| prediction | result |
| --- | --- |
| **C1 — diversity doesn't destroy the signal** (AUC ≥ 0.90) | **HELD: 0.945** |
| **C2 — cross-model agreement is a 2nd validity gate** | **FAILED** |
| **C3 — council corrects the Eswatini inversion** | **FAILED (did not correct)** |

## C1 held: same-vendor council grounding is viable

Pooling three models' resamples and grounding against the pooled belief gives
**AUC 0.945** on the hard set — statistically indistinguishable from the
single-model gpt-4o-mini 0.952. Adding a weak member (gpt-3.5-turbo) did **not**
dilute the signal. The pooling mechanism is sound; the grounded axis survives the
move from one model to a council.

## C2 failed — because the council almost never disagrees

I predicted cross-model agreement would be a second self-validity gate (high
agreement → trust, low agreement → abstain). It is not established here, for a
**structural** reason: **34 of 36 items had full 3/3 council agreement.** There is
essentially no low-agreement stratum to gate on (n_low = 2). With only two
disagreement items the "gate" AUC is 0.75 — failing the pre-registered `< 0.75`
collapse bar, and on a sample too small to mean anything.

The honest read: **three same-vendor models are too correlated to populate a
disagreement signal.** They share training lineage, so they tend to agree —
*including when they are wrong* (see C3). Cross-model agreement among same-vendor
models therefore adds little beyond the within-model Stability gate (B2) we already
have. This is a direct, quantified argument for cross-**vendor** diversity rather
than cross-model.

## C3 failed — the council shares the wrong belief (exactly the pre-registered prior)

On the lone genuine wrong-belief item, **Eswatini**, the council did **not** correct
the single-model inversion. Modal answers: `[Mbabane, Mbabane, Lobamba]` — gpt-4o-mini
**and** gpt-4o both say Mbabane (the administrative capital); only gpt-3.5-turbo
emits Lobamba (the royal/legislative capital). Pooled g_true (Lobamba) = 0.29 <
g_false (Mbabane) = 0.65: **still inverted.**

The dissent did narrow the inversion (single-model was 0.09 vs 0.80 → council 0.29
vs 0.65), but it did not flip the verdict. This is precisely the honest prior I
pre-registered: same-vendor models share wrong beliefs, so a same-vendor council
cannot escape a shared confabulation. **Removing the confidently-wrong-belief
caveat requires cross-VENDOR grounding — a genuinely independent signal — not more
of the same vendor.** That step remains blocked on a second-vendor key.

## Honest bounds (stated, not hidden)

- **Single run, OpenAI-only, three same-vendor models, n=36.** A same-vendor
  council is a *partial* external signal at best.
- **C2 is under-powered, not refuted.** The agreement gate could still hold on a
  set engineered to produce disagreement; this set didn't, because same-vendor
  models agree on 94% of items. We report C2 as not established here, not as a
  closed negative on the idea of an agreement gate in general.
- **Astana/Nur-Sultan** is the same city renamed (modes split `Nur-Sultan` /
  `Astana` / `Nur-Sultan`); the judge correctly treats both as near-equivalent, a
  minor label artifact, not a real disagreement.
- Inherits all prior scope: self-consistency not external truth, injection-blind,
  one axis.

## Net

The grounded honesty axis carries cleanly from one model to a council (C1, AUC
0.945). But the two things I hoped the council would buy — a cross-model
disagreement gate (C2) and a fix for the confidently-wrong-belief inversion (C3) —
**both fail for the same single reason: same-vendor models are too correlated.**
They agree 34/36 times and they share the Eswatini error. This is the most
rigorous possible motivation for the next step: **the remaining caveat is removable
only by a cross-VENDOR signal**, where the second model's beliefs are genuinely
independent. The within-model Stability gate (B2) remains the one demonstrated
self-validity signal; cross-model agreement, within a vendor, adds nothing — and
proving *that* is itself the result.
