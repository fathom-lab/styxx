# FINDING — Cross-model demarcation map: the inaccessible-thought dissociation holds on exactly ONE model

> **Erratum (2026-06-07 — `papers/grounded-honesty-axis/FINDING_parrhesia_rung1_2026_06_07.md`):** even the
> one-model dissociation below is **trace-vs-thought-conflated** — on Qwen-1.5B the probe reads the injected
> concept at 1.00 *regardless of behavioural liveness* (1.00 at steering +0.007), so the read certifies
> injected-vector **presence**, not a held **thought**. The cross-model *steering*-validation here stands
> (only Qwen-1.5B is live); but "dissociation" overstates the read side. Self-report blindness stands; the
> read-certificate is owed on naturally-present content.

**2026-06-07. Fathom Lab / styxx.** The owed council arm that resolves the scope correction the
dogfood forced on the "1.00 across three families" claim. Verdict: **the full steering-validated
dissociation holds on Qwen2.5-1.5B-Instruct only; everywhere else the probe-read is a decode of a
behaviourally inert injection — not the dissociation.**

## Why this run

The flagship dissociation (external probe recovers an injected concept the model's own forced-choice
self-report cannot) requires the injection to be a *live thought* — i.e. **steering-validated**: the
injected direction must actually move generation toward the concept. The probe accuracy (1.00) and the
self-report null replicate cross-family, but steering-validation was measured only on Qwen-1.5B and
**never recorded per model**. Without it, a cross-family probe-read at 1.00 could be the trivial
"a linear decoder finds the linear vector you added" — an inert vector, not an inaccessible thought.

## Method

Per council model: measure `steering_gain` (MiniLM-cos shift of generation toward the concept under
injection vs clean, coherence-gated) at the *same* layer/alpha the forced-choice self-report run used;
combine with the existing on-disk forced-choice self-report (Reader A) and external probe (Reader B)
results. Frozen thresholds: injection LIVE iff gain ≥ 0.15 and coherence ≥ 0.80; self-report BLIND iff
8-way inject acc < 0.30 (chance 0.125); probe SEES iff acc ≥ 0.80; abort gate iff prime-2AFC ≥ 0.75.

## The map

| model | steer gain | self-report (FC) | prime | probe | verdict |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | +0.000 | 0.04 | 0.90 | 1.00 | PROBE-READ-ONLY (inert vector) |
| **Qwen2.5-1.5B** | **+0.158** | **0.21** | 1.00 | 1.00 | **DISSOCIATION (live thought; self-report blind; probe recovers)** |
| Qwen2.5-3B | +0.079 | 0.08 | 1.00 | 1.00 | PROBE-READ-ONLY |
| Qwen2.5-7B | −0.036 | 0.19 | 1.00 | n/a | INCONCLUSIVE (no probe arm) |
| Llama-3.2-3B | +0.032 | 0.17 | 1.00 | 1.00 | PROBE-READ-ONLY |
| gemma-2-2b | −0.071 | 0.10 | 1.00 | 1.00 | PROBE-READ-ONLY |

## Honest read

- **The dissociation is a one-model result here.** Only Qwen-1.5B's injection is behaviourally live
  (+0.158), and it *barely* clears the 0.15 bar — while being the **least-blind** self-reporter (0.21,
  the highest on the council). The dissociation is sharpest exactly where it is hardest to establish.
- **Self-report is blind everywhere** (0.04–0.21, all below 0.30) and the abort gate passes everywhere
  (≥0.90) — those parts of the claim *do* generalize. What does **not** generalize is the *live
  injection*: the diff-of-means concept direction at 0.60-depth/α=10 does not steer Qwen-0.5B/3B/7B,
  Llama-3B, or gemma-2B. On those models the probe's 1.00 is decoding an inert vector, so it is **not**
  evidence of an inaccessible thought.
- **This fully vindicates the scope correction.** "The dissociation holds across three families" was
  the trivial probe-read everywhere but Qwen-1.5B. The defensible claim is: *self-report is blind +
  abort gate valid cross-family; the full steering-validated dissociation is established on Qwen-1.5B.*

## What this does NOT settle / owed

Steering-validation here is *method-specific* (diff-of-means, 0.60-depth, α=10, MiniLM metric). A
stronger steering method, a per-model dose/layer search, or a 2AFC self-report (vs 8-way) could change
which models qualify — this maps the dissociation **for this apparatus**, not for all possible
injections. A model passing only the probe-read is not a falsification of the dissociation; it is
**out of scope** (inert injection). Next: a per-model steering dose/layer sweep to find each model's
live regime before re-testing self-report there. Receipts: `council_demarcation_map.json`,
`steer_valid_{tag}.json`, `run_council_map.py`.
