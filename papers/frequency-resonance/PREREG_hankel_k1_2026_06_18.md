# PREREG — resurrect-or-bury K1 with time-delay (Hankel) Koopman extraction

**2026-06-18. FROZEN before any data is collected.** This pre-registers a single, isolating
re-test of the Spectral Integrity Probe's kill-gate K1, using the one extraction fix the
shelving note prescribed but never tried.

## Where this stands (no spin)
The Spectral Integrity Probe (`INSTRUMENT_spectral_integrity_2026_06_04.md`) is the #1-leverage
frontier the oscillation machinery opens: a white-box deception signal read from the **spectrum**
of an agent's residual-stream trajectory, hypothesized to survive obfuscation where static linear
probes collapse to chance (arXiv 2505.27958). It was **SHELVED at K1**: the DMD-spectrum could not
separate even *coherent vs word-shuffled* text (the grossest cognitive-state proxy):
- v1, distilgpt2 L4, single-layer: **AUROC 0.434**
- v2, pythia-410m, multi-layer [6,12,18], 96-token trajectories: **AUROC 0.562**

Both under the pre-registered **0.70** bar. The shelving note's own verdict: what died is *this
specific crude extraction* (single-step exact DMD on a ~90-sample trajectory), **not** dynamical
integrity signals in general, and it prescribed: *"fix extraction (deeper/multiple layers, larger
model, longer trajectories, richer features), re-pass K1 on the coherence proxy, then spend GPU."*

v2 tried bigger-model + multi-layer + longer-trajectory and still failed. **What no version tried
is the principled estimator for this exact regime.** Single-step exact DMD fits `h_{t+1} ≈ A h_t`
directly on the observed coordinates; on ~90 noisy, partially-observed samples its eigenvalues are
variance-limited and it cannot resolve oscillatory modes whose period is a non-trivial fraction of
the window. The standard fix (Takens embedding / Hankel-DMD / HAVOK; Brunton et al., Arbabi–Mezić)
is to **delay-embed** the trajectory before the fit, reconstructing the latent attractor so a linear
operator can capture modes the raw coordinates cannot.

## Hypothesis
K1 failed because of estimator variance on short trajectories, not because the spectral premise is
empty. Delay-embedding before DMD should recover oscillatory structure single-step DMD misses; if
the premise has *any* gross-cognitive-state content, K1 clears 0.70 under proper extraction. If it
still fails with the right estimator, the premise is robustly dead.

## Method — ONE change vs v2, everything else identical
- **Identical:** model `EleutherAI/pythia-410m`; layers `[6, 12, 18]`; data = the `PASSAGES`
  coherent-vs-shuffled set from `coherence_spectral_test.py` (same `shuffle_words`, same `auroc`);
  per-layer trajectory = mean-centered hidden states → SVD → top-24 PCs; the same 5 spectral
  features per layer `{dominant_freq, weighted_freq, spectral_entropy, high_band_frac,
  weighted_decay}`; `StandardScaler` → `LogisticRegression(C=0.5)`; **5-fold CV** AUROC.
- **The only change:** before `dmd_modes`, each (T×24) layer trajectory is **delay-embedded** with
  `q` delays into a Hankel state `[h_t; h_{t+1}; …; h_{t+q-1}]`, then DMD is fit on the embedded
  trajectory. `q` is the sole new hyperparameter.

## Kill-gate (PRIMARY — the verdict; one number)
- **K1-Hankel.** Condition A = the pure extraction swap at the **pre-registered primary `q = 12`**
  (a ~12-token embedding window — roughly one clause — leaving ~80 snapshots for a stable fit),
  identical features/model/data/classifier. **Verdict gate: 5-fold CV AUROC ≥ 0.70.**

## Secondary (TRANSPARENCY ONLY — cannot rescue a failed primary)
- `q`-sweep `{4, 8, 16, 24}` reported in full (delay sensitivity).
- Per-layer single-feature AUROC (where the signal, if any, lives).
- A richer-feature variant (PSD spectral-flatness + #significant modes that delay-embedding makes
  meaningful), clearly labelled exploratory.

## Honest priors
- P(primary A clears 0.70) ≈ **0.30–0.40.** Delay embedding genuinely helps short-trajectory mode
  recovery, so this is a real and non-trivial shot — but two prior failures and the fact that
  coherence≠deception keep it a coin-flip leaning *fail*. I do not expect a miracle; I expect a
  *clean* answer.

## Stop rule (anti-goalpost-moving, binding)
- The verdict is the PRIMARY number at `q = 12`. No secondary result rescues a failed primary.
- **If A clears 0.70:** K1 recovers via principled extraction → the obfuscation-robustness gate K2
  (the actual value proposition) becomes the next pre-registered GPU step.
- **If A fails:** the Spectral Integrity Probe is **closed** — robustly dead under the correct
  estimator, not re-shelved-pending-more-rework. One real nail, then stop flogging it. The
  oscillation arc's honest contribution stands as the methods/repro result it already is.
