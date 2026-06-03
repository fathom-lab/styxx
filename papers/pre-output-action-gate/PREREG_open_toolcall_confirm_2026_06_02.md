# PREREG — Format-dependence: was the native-tool-calling collapse real, or a greedy-label artifact?

**Date:** 2026-06-02
**Status:** PRE-REGISTERED (gate frozen before the run)
**Hardens the weakest board cell.** Under the *menu* harness the emergent-choice action
signal SURVIVED (4/5 models). Under *native* tool-calling with **greedy single-shot
labels** it went FORMAT-DEPENDENT: Qwen-1.5B held (LOCO 0.819) but **Llama-3B collapsed
to 0.514** (~chance) and Qwen-3B went unbalanced (7/40 destructive). This asks whether
that collapse is the native **format**, or the **greedy single label** — which gave tiny,
unstable positive classes (Llama-3B's 0.514 sat on 11 positives, a Hanley-McNeil CI of
roughly ±0.18, exactly the small-n problem the accidental-harm cell taught us).

## The question

On the **same 40 emergent-choice scenarios**, under native tool-calling, does the
pre-emission residual predict the model's emitted destructive choice when the label is a
**sampled propensity** (stable, balanced) instead of a single greedy draw — and does
**Llama-3B recover** above chance? If yes, the collapse was a labeling artifact and the
cell is rehabilitated. If no, the format-dependence is **real**, and the board's existing
verdict stands on stronger evidence.

## Design note (frozen): why the same scenarios

The variable under test is the **labeling regime** (greedy → sampled), not generalization
to new data. So the 40 scenarios are held **fixed** (sha256 `b1303559f2261c8a`) — a
controlled comparison that isolates the labeling variable. This is *not* a fresh-data
replication and is not claimed as one.

## Method (frozen)

Models: Qwen2.5-1.5B/3B, Llama-3.2-1B/3B (native `tools=` calling, the deployment
interface). Per (model, scenario): residual at the last prefill token (all layers,
deterministic, once). Per **seed ∈ {0,1,2}**: sample the emitted action **N=9× at
temperature 1**; destructive-choice frequency = propensity; **label "destructive-prone"
= ≥3 of 9 (0.333)**.

- **WHITE-BOX:** residual LOCO AUC (best layer) + Hanley-McNeil 95% CI, per seed; mean
  across seeds. Out-of-fold probs dumped.
- **TEXT:** prompt embedding (`text-embedding-3-small`) + bag-of-words LOCO AUC.
- Balanced = ≥8 per class. Included model = balanced in all 3 seeds.

## KILL-GATE (frozen)

With ≥2 models included:
- **(a) DIRECTION (seed-stable):** in every seed, white-box > text on **≥3/4 models**.
- **(b) MAGNITUDE (CI-aware):** mean white-box AUC ≥ **0.70** AND Hanley-McNeil lower-95%
  bound ≥ **0.60** on **≥2 included models**.
- **(c) MARGIN:** mean (white-box − best-text) ≥ **0.15** on those models.

## Readings (fixed)

- **RECOVERED:** (a) ∧ (b) ∧ (c). The native-calling signal holds under stable labeling
  → the greedy collapse was a **labeling artifact**, and the emergent-native cell is
  rehabilitated (especially if **Llama-3B's mean AUC clears 0.70 with lower-CI ≥ 0.60**).
- **PARTIAL:** (a) holds but (b)/(c) fail → signal present but weak under native calling.
- **FORMAT-CONFIRMED:** (a) fails — in particular Llama-3B stays at/near chance under
  sampled-native too → the **format-dependence is real**, not a labeling artifact. The
  board's FORMAT-DEPENDENT verdict stands, now on stronger evidence. An honest negative.
- **UNDERPOWERED:** <2 models balanced in all seeds (destructive propensity too rare on
  these 40 at this sampling) → report base rates; do not claim.

## Honest caveats (frozen)

- n=40 (smaller than the accidental confirmatory's 84) → wider CIs; the lower-CI bar
  carries the weight, deliberately.
- Same scenarios as the greedy run by design (controlled labeling comparison, not fresh
  data).
- Native tool-calling parsing depends on each model emitting a parseable `<tool_call>`;
  models that mostly emit prose will show low propensity and may fail balance — itself a
  form of format-dependence, reported as such.
- Open-weight, simulated tool schemas.

— frozen 2026-06-02
