# POSITIONING — ENTRAIN-OSS vs the 2024–26 oscillatory-SSM frontier — 2026-07-23

Scan date 2026-07-23 (web). Purpose: place the ENTRAIN-OSS invention precisely against published
work so the contribution is a genuinely open sub-angle, not a re-invention. Novelty overclaims have
burned this arc before (`project_spectral_integrity_closed_2026_06_18`), so this doc freezes a
**novelty gate** and names the one paper that could sink it.

## The two tiers (never conflate)

- **Woo tier** (Rife / 528 Hz / "raise your vibration"): zero validated tools. Not ours; STYXX
  filed it under metaphysics (`SURVEY_frequency_vibration_2500yr_2026_06_04.md`).
- **Real tier** — oscillation as a *computational primitive*: a hot, mainstream frontier. This is
  where "the frequency" was actually harnessed, and where STYXX competes.

## Prior art on the real tier

| Work | What it does | The gap ENTRAIN-OSS targets |
|---|---|---|
| **LinOSS** — Oscillatory State-Space Models (Rusch & Rus, MIT, ICLR 2025, arXiv 2410.03943) | Linear SSM built as forced harmonic oscillators; ~2× Mamba, ~2.5× LRU on a 50k-length task. Frequencies **static** (learned-fixed at inference). | Reports **no** within-architecture ablation isolating oscillation from a decay baseline; frequencies never adapt to the input online. |
| **D-LinOSS** — "Learning to Dissipate Energy…" | Decouples frequency ω from damping \|λ\|; per-channel **static**; cell-swap ablation vs S5/LRU/Mamba. | Cell-swap changes the whole cell (confounded); still static frequency; no single-knob phase-clamp. |
| **TIDES** (arXiv 2605.09742) — **the dangerous neighbor** | Selective SSM; deliberately puts input-dependence on **step-size/decay**, *not* the imaginary eigenvalue. States input-dependent imaginary eigenvalues **hurt** and per-token frequency remap "has no clean interpretation." | Its negative is about **per-token** (fast) frequency modulation. It does **not** test a **slow, error-integrating entrainment loop** (small κ) that preserves modal-basis coherence — precisely ENTRAIN-OSS's design and escape hatch. Must be confronted head-on. |
| **Tuning Frequency Bias of SSMs** (ICLR 2025, arXiv 2410.02035) | Tunes innate frequency bias via **init scaling** / Sobolev filter (train-time). | Static bias, not runtime input-adaptive. This *is* prior art for the "resonance-informed init" idea → we demote init to a free side-benefit, not a claim. |
| **AOSNET / Adaptive Oscillatory-State Alignment** (arXiv 2606.06010) | Hilbert analytic-signal descriptors align local states to a **learnable global oscillatory prior** via a gate; non-rigid periodicity in forecasting. | Aligns to a *fixed global prior* in descriptor space; **not** an online frequency-locked *recurrence*. Same motivation, different mechanism (we entrain the eigenvalues online + give causal phase-clamp evidence). |
| **AKOrN** — Kuramoto oscillatory neurons (ICLR 2025, arXiv 2410.13821) | Oscillator neurons **synchronize phases** for binding/reasoning at fixed natural frequencies. | Phase synchrony for grouping, not frequency **adaptation to input timescale** for memory/timing. Shows the field accepts oscillator-coupling layers. |
| **Task-Level Insights from Eigenvalues** (NeurIPS 2025, arXiv 2510.09379) | **Correlational**: eigenvalue magnitude ↔ memory vs selectivity; unifies attention + SSM by spectra. | Correlational magnitude analysis; no **causal** phase intervention. This is the frame the STYXX "resonance profiler" tool extends to the phase axis, causally. |
| **Physics-informed OSSM** (arXiv 2606.02623) | Swaps temporal cell as an ablation for PDE solving. | Cell-swap again, not a single-knob phase-clamp. |

## The open sub-angle STYXX owns

> A **slow, closed-loop frequency entrainment** of recurrent eigenvalues — a differentiable PLL that
> nudges a small bank of oscillator frequencies toward the input's local dominant period on a
> timescale ≫ the token rate (coherence-preserving) — evaluated in the regime static banks cannot
> cover: **within-sequence drifting / starved-budget multi-timescale** structure. This is (i) not
> LinOSS/D-LinOSS (static), (ii) not TIDES (which moved input-dependence onto decay *because*
> per-token frequency failed — the slow-loop limit is untested), (iii) not AOSNET (descriptor-gated
> alignment, not entrained recurrence), (iv) not Tuning-Frequency-Bias (init-time). It is the exact
> door STYXX's own arc logged as open ("jittered/drifting periods, long horizons").

Plus a second, orthogonal contribution that ships regardless of the layer's fate: the **causal
phase-clamp / entrainment ablation** (clamp θ→0 or κ→0, re-eval, ΔAcc) — the within-architecture
oscillation-vs-decay isolation LinOSS/Mamba never ran, the causal complement to 2510.09379's
correlational analysis.

## NOVELTY GATE — verdict

**PASS.** No published SSM implements *slow closed-loop frequency entrainment of recurrent
eigenvalues evaluated on drifting-timescale sequence tasks*. The nearest neighbor, TIDES, reports a
**negative** on the naive *per-token* variant — which strengthens rather than blocks the slow-loop
framing, and which ENTRAIN-OSS must cite and out-perform (or concede to) honestly.

**Residual before any PUBLIC claim (not required before the internal kill-gate):** read TIDES
(2605.09742) and D-LinOSS in full text to confirm neither runs a slow-tracking frequency loop on
drifting periods. If either does, **PIVOT** the flagship to the nested cross-frequency (theta-gamma)
coupling layer — which this scan found **no** SSM implements (all theta-gamma hits are neuroscience).

Sources: arXiv 2410.03943, 2605.09742, 2410.02035, 2606.06010, 2410.13821, 2510.09379, 2606.02623.
