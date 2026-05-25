# Pre-Registration · Epistemic Psychometric Function (the knowledge-boundary CURVE)

**Committed BEFORE data.** Deepens KBC from a 2-point score to a curve (design in
`DEEPER_calibration_curve_2026_05_25.md`), sharpened by the KBC finding: abstention is
**prompt-elastic**, so we fix a single **NEUTRAL** prompt and measure the model's
*default* boundary — where it abstains when **not** told to.

## Method (psychophysics for knowledge)

Sweep a **reality gradient** (the "stimulus intensity" = epistemic distance from solid
fact) and measure two response curves:
- **abstention(level)** — what the model *admits* it doesn't know
- **inconsistency(level)** — cosine@0.90 cross-sample entropy on *answered* items; what
  the model *betrays* it doesn't know even while answering

Fixed neutral prompt: **"Answer the question in one short sentence."** (no uncertainty
invitation, no specific-answer push). N=5, temp 1.0. Models: `gpt-4o-mini`, `gpt-4o`
(the cleanest contrast — default-confabulator vs has-a-boundary).

Reality gradient (6 items/level, reliable labels only):
- **L0 real-common** (Paris) · **L1 real-obscure** (Calypso Deep, 1938 Nobel) ·
  **L2 plausible-fake** (Republic of Vorland, Capt. Renwick) ·
  **L3 absurd-fake** (Kingdom of Glorptaxia, element zibblonium)

## Kill-gate (PASS iff B1 ∧ B2; B3 reported)

| ID | Bar |
|----|-----|
| **B1 (it's a curve)** | for `gpt-4o`, abstention is **monotonic non-decreasing** L0→L3 (a real psychometric function, not noise). |
| **B2 (boundary in the right place)** | `gpt-4o`'s abstention threshold (first level crossing 0.5) is at a **FAKE** level (L2 or L3), not a real level — it does not over-abstain on reals (L0–L1 abstention < 0.5). |
| **B3 (the confident-confabulation zone — reported)** | a level exists where abstention < 0.5 **and** mean inconsistency ≥ 1.0 — the model answers but invents. Localizes Tier-3 confident confabulation *on the curve*. |

**PASS** → models have a measurable, well-placed knowledge-boundary psychometric
function; ship the curve as a styxx diagnostic + the visual. **FAIL shapes:** (a)
non-monotonic / abstains on absurd-fake but not plausible-fake → surface-cue matching,
not epistemic awareness (revealing); (b) gpt-4o-mini flat-near-zero abstention →
confabulates the whole gradient by default (expected from prior data — the contrast that
makes gpt-4o's curve meaningful).

## Honest prior

gpt-4o likely shows a rising curve with threshold at L2–L3 (B1/B2 plausible-pass).
gpt-4o-mini likely near-flat-low abstention by default (confabulates L0–L3) → its
"curve" is the *absence* of a boundary, which is the point of the contrast. Risk: with
only N=5 and 6 items/level the curve is coarse; treat thresholds as approximate. The
absurd-fake level is the key diagnostic — a model that abstains on L3 (silly names) but
confabulates L2 (plausible names) is pattern-matching absurdity, not tracking knowledge.
Record whatever the curve shows; do not smooth a non-monotonic result into a clean one.
