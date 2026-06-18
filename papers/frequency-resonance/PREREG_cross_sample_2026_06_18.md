# PREREG — cross-sample dynamical consensus: do K resamples' DYNAMICS agree?

**2026-06-18. FROZEN before generation/feature code or data.** The switching object is alive but
sub-threshold on a single trajectory (`RESULT_nonstationarity_2026_06_18.md`: 0.672, fails 0.70,
+0.21 over the dead spectrum). Its own stop rule names the fix: stop reading ONE trajectory's
dynamics; resample K generations and read whether their **dynamics agree**. This pre-registers that
test and — critically — the confound that decides whether it is novel.

## Thesis
For a prompt the model has a stable internal answer to, K resampled generations should traverse
**similar dynamics** (high dynamical consensus). When the model must improvise (no stable answer →
the confabulation regime), the K generations' dynamics should **diverge**. This is the dynamical
analogue of styxx's proven edge — sampling divergence broke the grounded-honesty ceiling (AUC 0.966
vs 0.498 text, `project_grounded_honesty_ceiling_break`) — but read on the **dynamics** (Koopman
operator / non-stationarity signature per sample), not on answer text.

## Why it could beat the existing divergence tools (the novelty bet)
Semantic-entropy / answer-agreement divergence misses **lucky agreement**: K samples that land on the
same answer via different improvised internal routes. Dynamical consensus can catch that — same
destination, different dynamics. If it does NOT add over semantic entropy, it is divergence
repackaged and we say so.

## Method
- Model `EleutherAI/pythia-410m` (GPU if available). Two prompt sets, label by construction:
  **answerable/stable** (simple facts the model completes consistently; label 0) vs
  **underspecified/improvise** (open continuations forcing improvisation; label 1).
- For each prompt: generate **K=6** continuations, temperature 0.8, ~32 new tokens.
- Per generation: deep-layer (L18) (T×24) PCA trajectory → its `nonstationarity_features` vector
  (the alive object) AND the trajectory itself.
- **Cross-sample dynamical-consensus features (per prompt):** variance across the K samples of each
  non-stationarity feature; mean pairwise distance of the K samples' dynamical signatures.
- **Semantic-entropy baseline (per prompt):** spread of the K samples' last-token hidden states
  (mean pairwise cosine distance) — the cheap "do the answers diverge" signal.

## Kill-gates (pre-stated; in order; stop at first fail)
- **K1-CS — signal exists.** Cross-sample dynamical-consensus features separate answerable vs
  improvise at **5-fold CV AUROC ≥ 0.70**.
- **K-confound — novelty (the real bet).** The dynamical-consensus classifier must beat the
  semantic-entropy baseline by **≥ 0.05 AUROC** AND survive partialling the semantic baseline out.
  If dynamical ≤ semantic ⇒ divergence repackaged, not novel — report honestly. **This gate most
  likely needs the real honest/confabulated dataset (answers-agree-but-grounding-differs), not this
  cheap answerable/improvise proxy** (where answer entropy separates by construction). The cheap
  proxy tests K1 only; K-confound here is indicative, the GPU/dataset run is decisive.

## Honest priors
- P(K1-CS ≥ 0.70 on the cheap proxy) ≈ **0.6** — answerable vs improvise is a strong contrast and the
  single-trajectory object already hit 0.67; K samples should push it over.
- P(K-confound passes, i.e. genuinely beats semantic entropy) ≈ **0.3** — the honest hard part; the
  cheap proxy is rigged toward semantic entropy separating, so a clean novelty verdict needs the real
  dataset.

## Stop rule (binding)
- **K1-CS fails** ⇒ dynamical integrity (single AND cross-sample) is closed on cheap proxies; the
  dynamical line is done — do not flog. styxx keeps its existing (text/semantic) divergence tools.
- **K1-CS passes, K-confound indicative-positive** ⇒ the genuine arc opens: pre-register the
  real-dataset K-confound (answers-agree/grounding-differs) + the obfuscation-robustness test. THIS
  is the "something" — but only if the data says so.
- **K1-CS passes, K-confound indicative-negative** ⇒ alive but not novel; fold the dynamical signature
  into styxx's divergence stack as a secondary feature, do not over-sell it as new.
