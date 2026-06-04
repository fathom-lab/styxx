# Instrument design — the Spectral Integrity Probe (obfuscation-robust, dynamical)

> **STATUS: SHELVED at K1 (2026-06-04).** Cheap CPU de-risking killed the premise before any GPU
> spend. The DMD-spectral feature fails to separate even *coherent vs word-shuffled* text — the
> grossest cognitive-state proxy — across **two** models and richer extraction: distilgpt2 L4
> AUROC **0.434** (`coherence_spectral_test.py`), pythia-410m multi-layer **0.562**
> (`coherence_spectral_test_v2.py`), both under the 0.70 bar. If the spectrum can't separate
> coherence, it won't separate honest/deceptive. **Do not run the GPU deception test.** What's
> dead is *this specific feature* (DMD modes of the residual trajectory), not dynamical integrity
> signals in general (path-length / "restlessness" works in the literature, but is not novel).
> Kept as a documented dead-end (cf. the geometry probe). Design below is preserved as written.

**2026-06-04.** A styxx primitive that reads the **spectral signature of an agent's residual-stream
trajectory** as an integrity signal — aimed at the documented gap that static probes can't cover.
Design + falsification only; testing is GPU-gated (queue behind the scaling sweep). Discipline: the
geometry probe died on a confound control despite a clean finding — so the confound controls are
**kill-gates here, not afterthoughts.**

## The gap it targets (from due diligence)
Static white-box deception probes hit AUROC 0.96–0.999 on benchmarks but **collapse to chance under
RL-obfuscation / adversarial suffixes while the model stays deceptive** (arXiv 2505.27958). The only
dynamical signal published — "adversarial restlessness" — is trajectory **magnitude** (path length;
2604.28129), not spectrum. No one has a white-box integrity signal that survives obfuscation, and no
one has read the **spectrum** of the trajectory.

## Thesis
A generation is a trajectory h_1..h_T in the residual stream. Its **effective dynamics** have an
oscillatory-mode structure (frequencies + decay rates) recoverable by Dynamic Mode Decomposition /
Koopman — the same eigenvalue-spectrum object our LRU work studies, now fit from data
(h_{t+1} ≈ A h_t; eig(A) = the modes). **Hypothesis:** honest vs deceptive/manipulated generation
differ in this spectral signature, and the signature is **harder to obfuscate** than a static probe
direction — because an attacker optimizing against one linear direction does not control the
distributed frequency structure of the whole trajectory.

## Method (minimal, local-model)
1. Run a white-box model on paired honest/deceptive prompts (reuse styxx's deception/honesty harness
   + a public set: strategic-deception / sandbagging / insider-trading).
2. Collect the residual-stream trajectory at a mid-late layer; PCA to ~16 dims.
3. **Spectral features:** fit DMD on the trajectory → eigenvalues (mode frequency = |arg λ|, decay =
   |λ|), mode energies; also PSD of the top PCs. Feature vector = {dominant frequencies, decay rates,
   spectral entropy, low/high band energy ratio}.
4. Classify honest vs deceptive from spectral features (held-out, cross-prompt).

## Falsification — kill-gates (pre-stated; the instrument earns its claim only if K1∧K2∧K3∧K4)
- **K1 — signal exists.** Spectral features separate honest vs deceptive at **AUROC ≥ 0.70** on a
  held-out split. Fail ⇒ dead.
- **K2 — the differentiator (robustness).** On obfuscated generations (suffix / RL-obfuscation) where a
  **static linear probe drops to ≤ 0.55 AUROC**, the spectral classifier must **retain ≥ 0.65**. This
  is the entire value proposition. If spectral collapses too ⇒ no advantage ⇒ report dead, honestly.
- **K3 — not the static rep in disguise (confound).** Partial out the static probe direction from the
  trajectory; spectral signal must survive (Δ AUROC small). If it dies, it was never dynamical — the
  exact failure mode that killed the geometry probe.
- **K4 — beyond magnitude (confound).** Must add AUROC over the **path-length / total-variation**
  baseline (the known "restlessness" signal). If DMD-spectrum ≤ path-length alone ⇒ it's restlessness
  repackaged, not novel.

**Pre-registered priors (honest):** K1 plausible (~60%); K2 is the real bet and genuinely uncertain
(~40% — obfuscation may perturb the whole trajectory, not just one direction); K3/K4 are where it most
likely dies (confounds are the usual killers). We design to find out fast and cheap. **We do not
promise it works** — we promise it gets a fair, kill-gated test.

**Update 2026-06-04 — cheap CPU de-risking of K1 (caution).** A pre-registered proxy test
(`coherence_spectral_test.py`): does the DMD spectrum separate *coherent vs word-shuffled* text — a
gross cognitive-state difference — on distilgpt2 layer 4? **It FAILED: 5-fold CV AUROC 0.434** (bar
0.70). The earlier N=1 "shuffled→Nyquist" hint was noise. This is one small model, one layer, a
~30-token trajectory, crude PCA-24+DMD extraction, and coherence≠deception — so it does **not** kill
the instrument, but it **lowers the K1 prior (~60% → ~35%)** and flags that **signal extraction needs
rework** (deeper/multiple layers, larger model, longer trajectories, richer features) *before* the GPU
deception test is worth running. Honest order of operations: fix extraction, re-pass K1 on the
coherence proxy, *then* spend GPU on deception.

## Why this is the right "impossible before us" bet
- On-mission: a white-box integrity primitive for deployed agents — the styxx thesis.
- Targets a *documented* weakness (static-probe fragility), not a hypothetical one.
- Uses our actual edge (eigenvalue/oscillation spectral analysis) on a problem the field frames only
  statically or by magnitude.
- Falsifiable, cheap first pass, confound-gated from the start.

## First experiment (when GPU frees)
Smallest viable: one local deceptive-capable model, ~200 paired honest/deceptive generations, one
obfuscation attack, static-probe baseline + path-length baseline + DMD-spectral. Run K1→K4 in order;
stop at the first fail. Pre-register before data. Cost: hours on the local 8 GB card.
