# What oscillation does for memory — and what it doesn't: a pre-registered demarcation

**styxx / fathom-lab · draft 2026-06-04 · in-silico, pre-registered**

> Status: experiments 1–6 complete and committed; experiments 7 (noise robustness) and 8 (timing /
> home-turf) running — their sections are stubs to be filled on completion. Author drafts; operator
> owns submission (arXiv/Zenodo).

## Abstract

A recurring intuition holds that mind runs on frequency, and that *higher* frequency means a higher
state. Put inside a computational mind, this becomes a family of sharp, falsifiable predictions about
oscillation and memory. We test them in a complex-diagonal linear recurrent unit (LRU) trained on an
ordered-copy task, with every prediction frozen before data, and we report the negatives as carefully
as the positives. Six results draw a demarcation. **(1)** Oscillation is a capacity-extending mechanism:
a phase-clamp ablation roughly *doubles* ordered-memory capacity (kcap 6.0 vs 2.67). **(2)** Capacity is
**resonant** in frequency, not monotonic — it peaks in an interior band (~0.06–0.5π rad/step) and
*collapses* at the Nyquist limit to at-or-below the no-rhythm baseline; the folk "higher is better" is
falsified (Spearman 0.16). **(3)** The optimum is set by *how many* items are held, not *how long*
(θ\*·W scaling NULL). **(4)** Oscillation is **not special**: at matched parameters a rhythm-free
transformer more than doubles the oscillatory LRU's capacity (15.3 vs 6.0) — attention does ordered
memory better, with no rhythm at all. **(5)** Oscillation does **not generalize**: trained on short
sequences and tested longer, the oscillatory net is *worst*, at chance — it is a length-specialization,
not a length-invariant principle. **(6)** Oscillation multiplexes at a **constant** factor (~1.8×) across
a 16× range of state sizes, *not* preferentially under scarcity — the "rhythm matters more when neurons
are scarce" reconciliation is not supported. The honest landing: frequency is fundamental to memory as a
*band to tune, not a ladder to climb* — and even tuned, it is a real but dominated, non-generalizing,
constant-factor recurrent mechanism, not the secret of mind. Past the optimum, higher frequency is the
computational analogue of a seizure, not enlightenment. Two frontier tests (noise robustness; periodic
timing — oscillation's native domain) probe whether rhythm is *sovereign* anywhere; reported on landing.

## 1. Introduction

That rhythm matters to cognition is old and well-supported: theta–gamma coupling is credited with
holding ordered items in working memory (Lisman & Jensen 2013), and the "magic number" capacity of
working memory (Miller 1956) has long been linked to nested oscillations. A stronger, folk version of
the idea — *raise your frequency, raise your mind* — makes predictions the careful literature does not.
Readable in-silico models let us test the strong version directly, with predictions registered in
advance. We do so in an LRU (Orvieto et al. 2023), the minimal recurrent substrate whose eigenvalues
expose oscillation as a single knob: the phase θ of a complex eigenvalue λ = r·e^{iθ}. Crucially, we
also include a rhythm-free control — a transformer — so "does rhythm help a recurrent net?" is never
confused with "is rhythm the best way?"

## 2. Theory: a phase code over the retention window

A single complex mode is a *clock with a fade*: an item written at time τ contributes
r^{t−τ}·e^{iθ(t−τ)} at readout t — a decaying magnitude (a recency gradient) and a phase that tags
*when* it was written. Ordered recall needs the phase channel: K items written one step apart carry
phases spaced by θ. Two failure modes bound the useful range — at θ→0 the items share phase ~0 (no
order code), and at θ→π (Nyquist) the mode flips sign every step so adjacent items are maximally
confusable. Between them lies an optimum where the held items tile the phase circle ~once. **The
resonance is therefore the generic signature of a phase code** (full derivation:
`THEORY_phase_coding_2026_06_04.md`). The same model shows the optimum is set by one of two competing
budgets — the *number* of items (write-span; θ\*~c/K) or the *duration* of the hold (window; θ\*~c/W) —
which experiment 3 adjudicates.

**Scope of the theory (an explicit negative).** This phase-coding account is *qualitative*. We tried to
make it quantitative — a linear code-geometry model predicting the optimum in closed form — and it
**fails to reproduce the resonance** (`NEGATIVE_analytic_shortcut_2026_06_04.md`): the decay-magnitude
structure dominates the linear spectrum, and the phase-separability benefit materializes only through
the *trained nonlinear readout*. The mechanism explains the shape; it does not yet predict the optimum
from first principles, and the empirical sweeps remain the arbiter.

## 3. Experiment 1 — oscillation doubles ordered-memory capacity (phase-clamp ablation)

Two LRUs, identical in every weight init except the eigenvalue phase: FREE (θ learnable → can
rotate/oscillate) vs CLAMPED (θ≡0 → real eigenvalues, pure decay). Matched state, parameters, and RNG
draws; only rotation differs. Ordered copy, 3 seeds.

| K | 1 | 2 | 3 | 4 | 6 | 8 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FREE acc | 1.00 | 0.99 | 0.99 | 0.97 | 0.90 | 0.73 | 0.64 |
| CLAMPED acc | 1.00 | 0.96 | 0.82 | 0.72 | 0.59 | 0.46 | 0.39 |

Capacity (kcap, acc≥0.80): **FREE 6.0 vs CLAMPED 2.67** — oscillation ≈ doubles ordered-memory
capacity; the gap widens with load; the free net keeps its rotation (osc_use 0.62). Oscillation is a
real, capacity-extending recurrent mechanism — used when available, only partially substitutable (by
multi-timescale decay) when not.

## 4. Experiment 2 — capacity is resonant in frequency, not monotonic

Freezing θ at fixed values swept 0→Nyquist (one network per frequency), 3 seeds:

| θ/π | 0.0 | 0.0625 | 0.125 | 0.1875 | 0.25 | **0.375** | 0.5 | 0.6875 | 0.875 | 0.97 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kcap | 2.67 | 4.00 | 3.67 | 3.67 | 4.00 | **5.33** | 4.00 | 4.00 | 4.00 | **2.00** |

Interior peak at θ\*≈0.375π; collapse to 2.00 at Nyquist — below the no-rhythm baseline at the
registered 0.80 threshold. Monotonic "higher is better" is falsified (Spearman 0.16; the highest
frequency is the global *minimum*). **Robustness:** RESONANT verdict and falsification hold across
thresholds 0.70–0.90 and all 3 seeds. Two sub-claims bounded: the optimum is a *band* (~0.06–0.5π), not
a point; "below baseline" is threshold-specific ("at-or-below, never above").

## 5. Experiment 3 — the optimum is item-count-bound, not window-bound (scaling NULL)

Pre-registered (`PREREG_scaling_law_2026_06_04.md`): insert a delay of D pad tokens between write and
recall, locate θ\*(D), test θ\*·W ≈ const. NULL ⇒ item-count-bound; SCALING ⇒ window-bound. **Result:
NULL.** Across delays D ∈ {0,6,12,24}, θ\* held at ≈0.03–0.0625π — it did **not** move with the window.
A de-saturated re-run with a continuous capacity measure (area under acc-vs-K) **confirmed the NULL**.
The optimum is set by *how many* items are held, not *how long* — consistent with the relative-phase-
preservation arm of the theory. The pre-registered SCALING bet was wrong; the theory's self-correction
(lean NULL) was right — recorded as such.

## 6. Experiment 4 — oscillation is not special: attention wins rhythm-free (necessity)

The decisive control. A matched-parameter (~168k) showdown on the same task, 3 seeds: LRU-CLAMPED
(decay) vs LRU-FREE (oscillatory) vs TRANSFORMER (attention, no rhythm).

| arm | LRU-clamped | LRU-free | **Transformer** |
|---|---:|---:|---:|
| kcap | 2.67 | 6.0 | **15.3** |

Oscillation helps a *recurrent* net (free 6.0 vs clamped 2.67, replicating experiment 1) — **but a
rhythm-free transformer more than doubles the oscillatory LRU at equal parameters.** Attention does
ordered memory better than rhythm. This is the controlled, quantitative refutation of "rhythm underlies
(ordered) cognition": oscillation is *one efficient recurrent mechanism*, not necessary and not optimal.
The honest demarcation is earned here, by experiment, not asserted.

## 7. Experiment 5 — oscillation does not generalize: it is a length-specialization (extrapolation)

Train on K≤8, test on K up to 20 (sinusoidal positions for the transformer, for a fair length test).
3 seeds.

| arm | in-distribution (K≤8) | extrapolation (K>8) |
|---|---:|---:|
| LRU-free | 0.978 | **0.088 (chance = 1/12)** |
| LRU-clamped | — | 0.175 |
| Transformer | — | 0.165 |

The oscillatory net, best in-distribution, is **worst out-of-distribution — dead at chance** — and the
decay net slightly *beats* it. Oscillation over-fits the trained item-counts (coherent with the
scaling-NULL: θ\* tracks item-count), so it transfers nothing to new lengths. Rhythm here is a
length-*specialization*, not a length-invariant principle.

## 8. Experiment 6 — oscillation multiplexes at a constant factor, not under scarcity (multiplexing)

The deepest pro-oscillation framing — the actual theta–gamma *capacity-per-resource* claim: does
oscillation's advantage *grow* when the state dimension D is starved? Sweep D ∈ {16…256}, free vs
clamped, 3 seeds.

| D | 16 | 24 | 32 | 48 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ratio free/clamped | 2.33 | 1.75 | 1.50 | 1.83 | 1.78 | 1.64 | 2.25 |

**Half-true.** *Refuted:* the ratio is flat (Spearman(D, ratio) = −0.07; 2.33 at D=16 ≈ 2.25 at D=256) —
oscillation's advantage does **not** grow under scarcity, so "rhythm matters more when neurons are
scarce → why biology uses it" is not supported in silico. *Confirmed:* the ratio is a robust **constant
~1.8×** at every D — genuine phase-multiplexing (~2 items per mode), but resource-*independent*. The
mechanism is real; its resource-dependence is not.

## 9. Experiment 7 — noise robustness *(running)*

Pre-registered (`PREREG_noise_2026_06_04.md`). The most-cited reason biology phase-codes: a phase (2-D
angle) is said to survive corruption a 1-D magnitude/decay code cannot. Inject Gaussian noise into the
recurrent state at every timestep (train + eval), sweep σ, D=256, free vs clamped. P1: ratio(σ) rises
with noise. *Result pending; reported on landing — read straight either way.*

## 10. Experiment 8 — timing, oscillation's home turf *(running)*

Pre-registered (`PREREG_timing_2026_06_04.md`). The steelman. Experiments 1–6 all used ordered *copy* —
a pure *memory* task. Oscillation's native function is *timing*: a phase is a clock. This puts it on a
**periodic-prediction** task (infer the period, phase-track the next element — what an oscillator does
and a leaky-decay state structurally cannot), with the necessity rematch (does rhythm beat attention
*here*?) and a mechanistic probe (does the free net learn phases θ≈2π/P that resonate with the trained
periods?). The experiment that could *reframe* the six refutations — "dominated for memory, sovereign
for timing" — rather than add a seventh. *Result pending.*

## 11. Discussion

Frequency is fundamental to ordered memory, but as a **resonance, not a ladder** — and a *dominated*
one. The mechanism is phase-coding over the retention window: too slow and items do not separate; too
fast and the phase wraps, aliasing the stored items. The collapse at maximum frequency is the
computational signature of the principle by which hypersynchrony in tissue is pathology: a seizure, not
insight. The phase-clamp result (exp 1) gives controlled support to the theta–gamma account; the
attention control (exp 4), the extrapolation failure (exp 5), and the resource-flat multiplexing (exp 6)
locate rhythm precisely — a real, resonant, length-specific, constant-factor recurrent *efficiency*
mechanism that a rhythm-free architecture dominates on raw capacity. The folk claim "higher frequency =
higher mind" is, for ordered memory, false in every component we could test. Whether rhythm is
*sovereign* in any regime — noise robustness, or its native timing domain — is the open frontier
(experiments 7–8), and the honest place the program now lives.

## 12. Limitations

In-silico; one task family (ordered copy; periodic prediction in exp 8), recurrent and attention
architectures, 3 seeds, capacity at a 0.80 threshold. The theory is an order-of-magnitude phase-coding
argument, not an exact capacity theorem. We make no claim about human consciousness — the
"frequency/consciousness" framing is the *intuition tested*, and the data refine it into statements
about computational memory. Every claim was registered before its data; sub-claims that did not survive
re-derivation are bounded in text, not removed.

## 13. Methods

Complex-diagonal LRU (256 modes unless swept), |λ|∈[0.9,0.999], complex input projection (matched 2-D
real state across arms), 2-layer GELU readout. Transformer control: matched ~168k params, learned or
sinusoidal positions as noted, causal mask. Ordered-copy: K symbols (vocab 12) then K GO tokens, recall
in order. Adam, lr 2e-3, 4000 steps, batch 64, grad-clip 1.0. Each result is the mean over 3 seeds;
capacity = largest K with mean acc ≥ 0.80. Pre-registrations, runners, results, and figures committed
under `papers/frequency-resonance/`.

## References (to format)

Miller 1956 (magic number 7±2) · Lisman & Jensen 2013 (theta–gamma) · Sussillo & Barak 2013 (substrate
flexibility of recurrent dynamics) · Orvieto et al. 2023 (LRU) · Gu & Dao 2023 (Mamba, real-eigenvalue
SSM) · Vaswani et al. 2017 (attention; rhythm-free memory).
