# PRE-REGISTRATION — single-layer causal localization sweep: does the install peak sit where the geometry said?

**Written 2026-05-29, BEFORE any code for this test is written.** Two open threads converge
here:
1. Tonight's second-model replication (`FINDING_second_model_replication_2026_05_29.md`) left
   the I1 *causal localization* leg UNTESTABLE on Llama-3.2-1B: a 3-layer band knockdown is
   destructive anywhere on a 16-layer net (control saturates, late not separable from early).
   The minimal-footprint fix is a SINGLE-layer intervention.
2. The whole arc has a descriptive line (suppression-rhythm: the install's median *flip layer*)
   and a causal line (disinhibition: the band *causally* installs the answer) that have never
   been linked in one measurement. A per-layer causal sweep can ask: **is the causal install
   peak the same layer the descriptive geometry flagged?**

Models: **Qwen2.5-1.5B-Instruct** (28 layers — where the 5-layer band test already localized;
serves as the method validation) AND **meta-llama/Llama-3.2-1B-Instruct** (16 layers — the
actually-open question). Same n=36 arithmetic items, SAE-free full-vocab logit-lens, in-code
arithmetic ground truth SHA-256'd pre-scoring, exact-integer correctness (no judge),
greedy/deterministic. Run once per model.

## Method

For each one-shot confabulation with a clean teacher-forced baseline (γ=1 argmax == realized
token) and an alignable divergence position `pos`: for EACH decoder layer `i ∈ [0, N−1]`,
attenuate ONLY layer `i`'s residual write at `pos` to zero (`h_out → h_in`, i.e. γ=0 single
layer) and read the next-token argmax. `removed_i = (argmax ≠ realized token)`. Aggregate the
per-layer removal rate `r(i) = mean over confabs of removed_i`. Coherence at the target layer =
numeric-argmax rate (single-layer knockdown is far less destructive than a band, so blunting is
not expected; reported regardless).

Reference layers (PRE-COMMITTED, NOT hand-picked):
- **install center** `c = round(median flip_layer hidden-idx) − 1` (decoder index), from THIS
  model's own suppression-rhythm geometry. (Qwen: median 25 → c=24. Llama: median 11.5 → c=11.)
- **early control center** `cc = round(8·N/28)` (Qwen → 8; Llama → 5) — the proportional early
  position, same rule as the band run.
- **decoder late threshold** `Lt = ceil(2·(N−1)/3)` in decoder-index terms (Qwen 18; Llama 10).

## Predictions / bars

- **L1 — the removal curve peaks late (corroborator):** `argmax_i r(i) ≥ Lt`.
- **L2 — single-layer localization (core):** `r(c) − r(cc) ≥ 0.30` AND `r(c) ≥ 0.50` AND a
  per-confab discordant sign-test (knock-c-only-removes vs knock-cc-only-removes) p < 0.05.
- **L3 — causal peak = geometric flip (the unification, core):** `|argmax_i r(i) − c| ≤ 2`,
  i.e. the layer whose single-layer knockdown most often removes the commitment sits within ±2
  of the layer the descriptive flip-geometry independently flagged.

**SURVIVED (per model) iff L2 ∧ L3.** L1 is a reported corroborator. Powering: ≥ 12 usable
confabs (else report unpowered). A per-model SURVIVED on Llama specifically closes the I1
localization leg that the band method could not test on a shallow net. A null on L2 (no single
layer dominates) or L3 (causal peak ≠ geometric flip) is reported as such: it would mean the
install is distributed across layers (no single bottleneck) or that the descriptive flip layer
is not the causal install point.

## Honest scope (pre-committed)

Single-position, teacher-forced, single-layer γ=0 knockdown read at the divergence position
only — not multi-token regeneration. SAE-free logit-lens. Two small open models, one confirmatory
run each, feasibility-grade n=36, arithmetic only. Reference layers are fixed by rule from each
model's own measured flip-geometry and the shared proportional control rule — not tuned to the
verdict. This tests *causal localization of the install* and its *agreement with the descriptive
geometry*; it does NOT touch the standing correctness bound (every internal lever moves
confidence/abstention; only re-derivation moves correctness) and makes no claim about truth
recovery (single-layer knockdown, like the band, is expected to yield uncertainty, not truth).
