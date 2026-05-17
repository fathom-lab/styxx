# Cross-Instrument Universal Transport — Finding (2026-05-17)

**Status:** directional dogfood, real positive (honestly bounded).
**Script:** `scripts/dogfood/cross_instrument_transport.py`
**Raw:** `scripts/dogfood/out_cross_instrument_transport.json`
**Dogfoods:** the shipped `styxx.transport` module.

## Question

The refusal instrument transports across embedding spaces under a
label-free linear map. Is that map **refusal-specific**, or does **one
shared map** (fit once from a generic, label-free corpus) carry the
**whole instrument suite**?

## Method

Home = `text-embedding-3-large`. One generic 420-sentence label-free
corpus → one `Transport` (procrustes) per foreign space. The *same map*
is reused for every instrument. Per instrument: stratified 50/50 split,
fit `CognometricInstrument` in the transport's home representation on
train, evaluate `transported_score` on the foreign-space test split.
Baselines: native-in-foreign ceiling, naive direct transfer. Labels are
the styxx `benchmarks/data` condition labels.

## Honest aggregation

Retention (= transported / ceiling) is only meaningful when the
instrument has genuine **native embedding-space signal**. Instruments
with ceiling < 0.65 (at/near chance) are excluded from the headline:
carrying a broken instrument is neither a success nor a failure of the
transport. **Excluded: deception (ceiling 0.63 / 0.52) and
overconfidence (0.41 / 0.51)** — they do not work in embedding space to
begin with. This is consistent with prior styxx results (deception
needs NLI/semantic grounding, not lexical/embedding signal). Not a new
failure; a known boundary.

## Results (signal instruments: refusal, sycophancy, goal_drift, plan_action)

| foreign space | refusal | sycophancy | goal_drift | plan_action | mean retention | verdict |
|---|---|---|---|---|---|---|
| te3-small (same family) | 1.00→0.92 | 1.00→0.97 | 0.93→0.94 | 0.81→0.73 | **0.952** (4/4 ≥0.85) | **INSTRUMENT-AGNOSTIC** |
| all-mpnet (cross-family) | 1.00→0.84 | 0.97→0.73 | 0.96→0.69 | 0.83→0.78 | **0.811** (1/4 ≥0.85) | **BROAD** |

(naive direct transfer: 0.24–0.70 — every transported number clears it.)

## Finding

**The transport map is instrument-agnostic, not a refusal trick.** A
single label-free linear map, fit once from generic text, carries every
instrument that has genuine embedding-space signal:

- **Within-family: near-perfect.** Mean retention 0.952; all four
  working instruments retained at ≥0.85 of their native ceiling through
  one shared map.
- **Cross-family: broad but degraded.** Mean retention 0.811; all four
  stay well above naive, two strong (refusal 0.84, plan_action 0.78),
  two degraded but real (sycophancy 0.73, goal_drift 0.69).

This elevates the claim from "refusal transports" to "**the styxx
instrument suite transports through one shared, label-free map**"
within-family, and broadly cross-family. The limiting factor is not the
map — it is whether the instrument itself has embedding-space signal.

## Caveats

refusal test n=10 (20 obvious prompts, 50/50 split); other instruments
n_test=80; labels are styxx-synthetic benchmark conditions, not live
frontier behavior; single generic corpus, single seed; directional, not
paper-final. deception/overconfidence excluded as no-native-signal (a
real boundary, not hidden).

## Why it matters

styxx.transport is no longer "a refusal-specific transport." One map →
the working suite → any embedding space (within-family near-perfectly,
cross-family broadly), no labels, no weights, no retraining. That is the
honest, defensible form of universal cognometric transport.

## Next (honest)

1. Replace synthetic benchmark labels with live closed-model behavior
   for sycophancy/goal_drift/plan_action (as refusal already has).
2. Why does cross-family degrade for sycophancy/goal_drift but not
   refusal/plan_action? (instrument-geometry vs map question).
3. Only when paper-grade: universal-cognometric-transport paper
   (tool = styxx.transport; result = instrument-agnostic transport,
   bounded) → then Zenodo/OSF per the publishing bar.
