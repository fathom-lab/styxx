# Frontier due diligence — where can this push to a level that wasn't reachable before?

**2026-06-04.** Honest novelty audit of the oscillation/resonance arc against the literature,
then the one place it opens a genuine frontier. Discipline: separate *what we discovered* from
*what we cleanly reproduced*, and aim the ambition at the real gap.

## 1. What we built today
Pre-registered, in-silico: oscillation ≈ doubles ordered-memory capacity (phase-clamp ablation);
capacity is **resonant** in frequency, not monotonic (peak band, collapse at Nyquist); a phase-coding
theory that explains it; a running scaling test (θ*·W ≈ const?). Rigorous, falsified its own
monotonic bet, bounded its own over-claims.

## 2. Due diligence — what is already known (so we don't overclaim)
- **The resonance phenomenon is established in neuroscience, not new.** Frequency-dependent working
  memory with an optimum is causal and known: 4-Hz tACS *enhances* WM capacity while 7-Hz *reduces*
  it; parietal theta *speed* drives capacity (PMC5868840). Higher-frequency-hurts is not our
  discovery.
- **θ·W ≈ const is essentially theta–gamma nesting.** WM capacity ≈ gamma cycles per theta cycle
  (the 7±2 account) is a ratio-of-timescales law of the same form as our scaling hypothesis. If our
  experiment confirms θ*·W ≈ const, it is a clean computational *instantiation*, not a new law.
- **Oscillatory SSMs are an active ML area** (Oscillatory State-Space Models, arXiv 2505.12171;
  complex-eigenvalue recurrent nets). Our LRU rig is inside known territory.
- **Honest verdict on the research arc:** its value is *methodological* — a controlled, pre-registered,
  mechanism-isolating in-silico demonstration with the discipline most of this literature lacks. Real,
  but not "impossible before us." Good for a methods/repro contribution; do not sell it as a discovery.

## 3. The genuine frontier — an obfuscation-robust spectral integrity signal
The due diligence on the *instrument* side found a real, documented gap:
- Static white-box deception/honesty probes are strong on benchmarks (AUROC 0.96–0.999 on Llama-70B
  strategic deception; arXiv 2502.03407) **but fragile**: adversarial suffixes drop recall to ~0,
  distractor behaviors decouple deception from the probe direction, and **RL-Obfuscation pushes probe
  scores to chance while the model stays deceptive** (Pressure-Testing Deception Probes, 2505.27958).
- The only *dynamical* signal in the literature — "adversarial restlessness," an elevated activation
  **path length** over a multi-turn attack (Latent Adversarial Detection, 2604.28129) — uses
  trajectory *magnitude* only, not its *spectral content*.

**Nobody has a white-box integrity signal that survives obfuscation, and nobody has read the
*spectrum* of the activation trajectory.** That is the level that wasn't reachable before — and it is
exactly where our oscillation machinery applies. Hypothesis: the **spectral signature of an agent's
residual-stream trajectory** carries an integrity signal that is *harder to obfuscate* than a static
probe direction, because an attacker optimizing against one linear direction does not control the
distributed frequency structure of the whole trajectory. Design: `INSTRUMENT_spectral_integrity_2026_06_04.md`.

## 4. The second frontier (GPU-gated) — toy → real
Everything here is a toy LRU. The reach move: test whether the resonance / θ*·W law appears in the
recurrent state of **real** oscillatory SSMs (Mamba-class with complex modes) and whether a real LLM's
trajectory spectrum behaves as the instrument predicts. This validates toy→real and feeds the
instrument. Needs the GPU (currently held by the scaling sweep); queue behind it.

## 5. Honest ranking of leverage
1. **Spectral integrity instrument** (the documented gap; on-mission; "impossible before us"). Design now,
   test when GPU frees. **Highest leverage.**
2. **Finish the scaling sweep cleanly** (capstone of the research arc; saturation caveat noted).
3. **Toy→real resonance** (validates and feeds #1; GPU-gated).
4. Resist: framing the resonance itself as a novel discovery. It is a clean reproduction; sell the
   *method* and the *instrument*, not the phenomenon.

Sources: PMC5868840 (theta speed drives WM capacity); arXiv 2505.12171 (oscillatory SSMs);
arXiv 2502.03407 (linear deception probes 0.96–0.999); arXiv 2505.27958 (probes collapse under
obfuscation); arXiv 2604.28129 (adversarial restlessness / path length).
