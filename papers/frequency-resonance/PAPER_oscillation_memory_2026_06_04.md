# Oscillation and the structure of ordered memory: a resonance, not a ladder

**styxx / fathom-lab · draft 2026-06-04 · in-silico, pre-registered**

> Status: experiments 1–2 complete and committed; experiment 3 (scaling law) running — its section
> is a stub to be filled on completion. Author drafts; operator owns submission (arXiv/Zenodo).

## Abstract

A recurring intuition holds that mind runs on frequency, and that *higher* frequency means a
higher state. Put inside a computational mind, this becomes a sharp, falsifiable prediction:
ordered-memory capacity should rise monotonically with the frequency of a network's oscillatory
dynamics. We test it in a complex-diagonal linear recurrent unit (LRU) trained on ordered copy,
with every prediction frozen before data. Three results. **(1)** Oscillation is a
capacity-extending mechanism: a phase-clamp ablation roughly *doubles* capacity when the
eigenvalue phase is free to rotate versus pinned to zero (kcap 6.0 vs 2.67), and the free network
*keeps* its oscillation rather than abandoning it. **(2)** Capacity is **resonant** in frequency,
not monotonic: it peaks at an interior band (~0.06–0.5π rad/step) and *collapses* at the Nyquist
limit to at-or-below the no-rhythm baseline — the monotonic "higher is better" hypothesis is
falsified (Spearman 0.16), and the resonance survives re-derivation across capacity thresholds and
seeds. **(3, running)** We test whether the optimum scales inversely with the retention window,
θ\*·W ≈ const. A single-mode phase-coding theory explains (1)–(2) and shows (3) is *diagnostic*
between two mechanisms. The honest landing: frequency is fundamental to memory — as a band to
tune, not a ladder to climb. Past the optimum, higher frequency is the computational analogue of a
seizure, not enlightenment.

## 1. Introduction

That rhythm matters to cognition is old and well-supported: theta–gamma coupling is credited with
holding ordered items in working memory (Lisman & Jensen 2013), and the "magic number" capacity of
working memory (Miller 1956) has long been linked to nested oscillations. A stronger, folk version
of the idea — *raise your frequency, raise your mind* — makes a prediction the careful literature
does not: that more frequency is monotonically better. Readable in-silico models let us test the
strong version directly and cleanly, with the prediction registered in advance. We do so in an LRU
(Orvieto et al. 2023), the minimal recurrent substrate whose eigenvalues expose oscillation as a
single knob: the phase θ of a complex eigenvalue λ = r·e^{iθ}.

## 2. Theory: a phase code over the retention window

A single complex mode is a *clock with a fade*: an item written at time τ contributes
r^{t−τ}·e^{iθ(t−τ)} at readout t — a decaying magnitude (a recency gradient) and a phase that tags
*when* it was written. Ordered recall needs the phase channel: K items written one step apart carry
phases spaced by θ. Two failure modes bound the useful range — at θ→0 the items share phase ~0 (no
order code), and at θ→π (Nyquist) the mode flips sign every step so adjacent items are maximally
confusable. Between them lies an optimum where the held items tile the phase circle ~once. **The
resonance is therefore the generic signature of a phase code** (full derivation:
`THEORY_phase_coding_2026_06_04.md`; Figure `phase_coding_clock.png` makes it visible — K items as
phasors that tile the clock at the optimum and collapse onto each other at Nyquist). The same model
shows the optimum is set by one of two competing budgets — the *number* of items (write-span;
θ\*~c/K) or the *duration* of the hold (window; θ\*~c/W) — which experiment 3 adjudicates.

**Scope of the theory (an explicit negative).** This phase-coding account is *qualitative*. We
tried to make it quantitative — a linear code-geometry model predicting the optimum in closed form
(Gram conditioning; singular spectrum above a noise floor) — and it **fails to reproduce the
resonance** (`NEGATIVE_analytic_shortcut_2026_06_04.md`): the decay-magnitude structure dominates
the linear spectrum, and the phase-separability benefit materializes only through the *trained
nonlinear readout*. The mechanism explains the shape; it does not yet predict the optimum from
first principles, and the empirical sweeps remain the arbiter.

## 3. Experiment 1 — oscillation doubles ordered-memory capacity (phase-clamp ablation)

Two LRUs, identical in every weight init except the eigenvalue phase: FREE (θ learnable → can
rotate/oscillate) vs CLAMPED (θ≡0 → real eigenvalues, pure decay). Matched state, parameters, and
RNG draws; only rotation differs. Ordered copy, 3 seeds.

| K | 1 | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FREE acc | 1.00 | 0.99 | 0.99 | 0.97 | 0.90 | 0.73 | 0.64 |
| CLAMPED acc | 1.00 | 0.96 | 0.82 | 0.72 | 0.59 | 0.46 | 0.39 |

Capacity (kcap, acc≥0.80): **FREE 6.0 vs CLAMPED 2.67** — oscillation ≈ doubles ordered-memory
capacity; the gap widens with load (a capacity-specific benefit, not a constant offset); and the
free net keeps its rotation (osc_use 0.62). Oscillation is a powerful, capacity-extending
mechanism — used when available, only partially substitutable (by multi-timescale decay) when not.

## 4. Experiment 2 — capacity is resonant in frequency, not monotonic

Freezing θ at fixed values swept from 0 to Nyquist (one network per frequency), 3 seeds:

| θ/π | 0.0 | 0.0625 | 0.125 | 0.1875 | 0.25 | **0.375** | 0.5 | 0.6875 | 0.875 | 0.97 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kcap | 2.67 | 4.00 | 3.67 | 3.67 | 4.00 | **5.33** | 4.00 | 4.00 | 4.00 | **2.00** |

Interior peak at θ\*≈0.375π; collapse to 2.00 at Nyquist — below the no-rhythm baseline at the
registered 0.80 threshold. Monotonic "higher is better" is falsified (Spearman 0.16; the highest
frequency is the global *minimum*). **Robustness:** the RESONANT verdict and the falsification hold
across capacity thresholds 0.70–0.90 and all 3 seeds (interior peak everywhere). Two sub-claims are
bounded: the optimum is a *band* (~0.06–0.5π), not a precise point; and "below baseline" is
threshold-specific ("at-or-below, never above"). Figure: `frequency_resonance_curve.png`.

## 5. Experiment 3 — does the optimum scale as 1/window? *(running — stub)*

Pre-registered (`PREREG_scaling_law_2026_06_04.md`): insert a delay of D pad tokens between write
and recall, locate θ\*(D) at the capacity edge, test θ\*·W ≈ const (W = D + kcap\*). Theory makes it
diagnostic: NULL ⇒ θ\* is item-count-bound (relative phases preserved under uniform rotation);
SCALING ⇒ θ\* is window-bound. Predicted constant if scaling holds: ~0.66 cycle over the window
(θ\*·W ≈ 1.3π rad). *[Result + verdict + `scaling_law_curve.png` to be inserted on completion;
note a possible kcap-saturation limit on θ\* resolution, to be checked with a continuous capacity
measure.]*

## 6. Discussion

Frequency is fundamental to ordered memory, but as a **resonance, not a ladder**. The mechanism is
phase-coding over the retention window: too slow and items do not separate; too fast and the phase
wraps within the window, aliasing the stored items. The collapse at maximum frequency is the
computational signature of the same principle by which hypersynchrony in tissue is pathology, not
insight: a seizure, not enlightenment. The phase-clamp result gives controlled in-silico support to
the theta–gamma account (Lisman & Jensen 2013) while the partial rescue by multi-timescale decay,
and the fact that attention-based models do ordered memory with no rhythm at all, locate rhythm as
a mechanism for recurrent *efficiency*, not a requirement for the function. Finally, a learned
*spectrum* of frequencies (the free net) outperforms any single tone, consistent with a Fourier-like
positional code that tiles multiple scales at once.

## 7. Limitations

In-silico; one task (ordered copy), one architecture (LRU), one hidden size, 3 seeds, capacity at a
0.80 threshold. The theory is an order-of-magnitude phase-coding argument, not an exact capacity
theorem. We make no claim about human consciousness — the "frequency/consciousness" framing is the
*intuition tested*, and the data refine it into a statement about computational memory. Every claim
above was registered before its data; sub-claims that did not survive re-derivation are bounded in
text, not removed.

## 8. Methods

Complex-diagonal LRU (256 modes), |λ|∈[0.9,0.999], complex input projection (matched 2D real state
across arms), 2-layer GELU readout. Ordered-copy: K symbols (vocab 12) then K GO tokens, recall in
order; experiment 3 inserts D pad tokens before the GO block. Adam, lr 2e-3, 4000 steps, batch 64,
grad-clip 1.0. Each result is the mean over 3 seeds; capacity = largest K with mean acc ≥ 0.80.
Pre-registrations, runners, results, and figures are committed under `papers/frequency-resonance/`.

## References (to format)

Miller 1956 (magic number 7±2) · Lisman & Jensen 2013 (theta–gamma) · Sussillo & Barak 2013
(substrate flexibility of recurrent dynamics) · Orvieto et al. 2023 (LRU) · Gu & Dao 2023 (Mamba,
real-eigenvalue SSM) · Vaswani et al. 2017 (attention; rhythm-free memory).
