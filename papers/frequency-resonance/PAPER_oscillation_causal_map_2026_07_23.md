# What Oscillation Buys a State-Space Model: Seven Controlled Causal Ablations and a Reusable Profiler

> **Banner, added 2026-10-23, with `NOTE_prior_art_full_text_reads_2026_10_23.md` and
> `RESULT_pmnist_robustness_2026_10_23.md`.** The NOTE records full-text reads of four prior papers, each read
> twice: the LRU paper (Orvieto et al., arXiv 2303.06349), S4D (Gu, Gupta, Goel and Ré, arXiv 2206.11893), LinOSS
> (Rusch and Rus, arXiv 2410.03943) and D-LinOSS (Boyer, Rusch and Rus, arXiv 2505.12171); its section 5 files
> this draft's sentences. The RESULT re-ran the permuted-MNIST headline under a sealed preregistration and reads
> `BUDGET_CONFOUND__gap_does_not_survive_3x_steps` (`pmnist_robustness_result.json#/verdict`), worded so: On
> permutation 1234, at 12,000 steps under a cosine schedule re-stretched to that budget, the gap is 0.1097 (two
> FREE seeds against three CLAMPED seeds); at 4,000 steps on three new seeds it is 0.3182. The 4000-step gap does
> not survive the longer budget on this permutation. Every sentence below is left as written. Each line here names
> its sentence by section and a quoted fragment, not by line number.
>
> The lab no longer stands behind these:
>
> - **Priority-shaped, not supportable as written:** §3's closing sentence (it begins "Across all of these,"), and
>   the two ordinals of §1, "becomes, for the [ordinal] time, a *controlled experiment*" and "the [ordinal]
>   within-architecture map" (ordinal elided in both). S4D's Table 3(a) (2022) is a controlled real-versus-complex
>   comparison inside one model, and the LRU paper's PathX phase-range runs (2023) are controlled phase
>   experiments.
> - **Contradicted in part as general sentences** (true of LinOSS and D-LinOSS, not of the field): the Abstract's
>   "Yet the field evaluates them by comparing whole architectures", and §1's "is untested, because the standard
>   comparison is between whole models" and "We supply the missing causal instrument". S4D's Table 3(a) sets a
>   real initialization (S4D-Real) against a complex one (S4D-Inv) inside one diagonal SSM at matched size.
> - **Known wrong:** "S5/LinOSS-class" in §2 ("the oscillatory (S5/LinOSS-class) model") and in §4.1 ("A deep
>   S5/LinOSS-class classifier"). The model is a three-block complex-diagonal linear recurrence, LRU-style (`CLRU`
>   in `run_smnist_ablation.py` and `run_pmnist_ablation.py`); the S5 paper was not read.
> - **Known wrong:** Appendix A's equivalence, in its title ("the phase-clamp is LinOSS's oscillation ablation"),
>   in "a statement about LinOSS itself, not a proxy" and in "is identical to clamping the restoring force". In
>   LinOSS-IM magnitude and phase are tied (`|λ| = √S = cos θ` per mode, derived in the NOTE from LinOSS's
>   Proposition 3.1), so θ → 0 forces `|λ|` → 1, the integrator the appendix itself describes at A = 0, while
>   CLAMPED keeps a learnable magnitude below 1. The CLAMPED arm is not reachable inside LinOSS. The appendix's
>   algebra holds; the equivalence does not follow from it.
> - **Exceeds every read:** "the oscillation *is* the long-range mechanism" (Abstract) and "it *is* the long-range
>   mechanism; a decay model of equal budget cannot substitute for it" (§4.1). The measurement is one
>   architecture, one permutation, two seeds and 4000 steps, and none of the four papers supports the general
>   sentence.
> - **Does not carry at the RESULT's verdict:** the permuted-MNIST headline of the Abstract and §4.1 (+31.2
>   points, 92.0% against 60.7%, "widens 7.6×"). Its numbers stand as what was measured at 4000 steps on two seeds
>   and permutation 1234 (`pmnist_ablation_result.json`).
>
> These stand, with what they owe:
>
> - **True as written, needs the S4D pointer:** §4.1's "This is the controlled single-knob within-architecture
>   ablation the whole-model benchmarks cannot run." What it implies, that no within-architecture
>   real-versus-complex comparison exists, does not hold: S4D's Table 3(a).
> - **A paraphrase of LinOSS's abstract, needs its table:** §1's "beats Mamba ~2× on 50k-length sequences".
>   LinOSS's Table 2 (PPG-DaLiA, length 49,920) gives 1.66× against Mamba (test MSE 6.4 against 10.65, ×1e-2).
> - **Approximate:** §4.1's "matched-param". FREE registers θ as a parameter and CLAMPED as a buffer, so FREE
>   trains 192 more parameters (64 modes × 3 blocks): 101,002 against 100,810 in both MNIST receipts.
> - **Omitted from §3:** the LRU paper, whose magnitude parameterization (`nu = log(−log r)`) and ring
>   initialization `CLRU` uses, and S4D.
> - **Stands, with version pins:** §3's "LinOSS (2410.03943) and D-LinOSS use *static* (fixed-after-training)
>   frequencies and report no within-architecture oscillation-vs-decay ablation" (LinOSS v1-v3, D-LinOSS v1-v2).
>   Appendix A's closing "the within-architecture ablation the LinOSS paper does not report" is literally true for
>   the same reason, though the clamp is not that ablation (above).
> - **Outside every read:** the sequential-MNIST sentences of the Abstract and §4.1 ("causally load-bearing"); no
>   read paper has sequential MNIST.
>
> The reads cover four papers. TIDES, the S5 paper and the rest of §3's related work were not read, so nothing
> here is a statement about the literature, and no priority claim is made. The text below is unchanged.

*STYXX / fathom-lab · draft 2026-07-23*

## Abstract

Oscillatory state-space models (SSMs) — LinOSS, S5, and their kin — are a leading approach to
long-sequence modeling, built on complex eigenvalues whose imaginary parts make the recurrent state
*rotate*. Yet the field evaluates them by comparing whole architectures (e.g. LinOSS vs Mamba), which
cannot isolate what the oscillation *itself* contributes: two architectures differ in dozens of ways at
once. We introduce a **single-knob phase-clamp**: hold an entire model fixed and flip only whether the
eigenvalue phase θ is learnable (rotation → oscillation) or clamped to zero (real eigenvalues → pure
decay), with all other initialization RNG-matched so the two arms are bit-for-bit identical when the
knob is off. Using it we run seven controlled causal ablations. We find: (1) on a real long-range
benchmark (sequential MNIST) oscillation is **causally load-bearing** — an oscillatory model reaches
98.4% and beats a bit-for-bit-identical decay model at 94.3% (+4.1 pts, both seeds), and clamping the
oscillation out of the *trained* model collapses it to chance (9.8%); on **permuted** MNIST, where
locality cannot substitute, the gap **widens 7.6× to +31.2 pts** (92.0% vs 60.7%) — the oscillation
*is* the long-range mechanism; (2) *adaptive* frequency —
tuning each oscillator to the input online — genuinely helps, but its benefit is **non-monotone in
detector complexity**, peaking at a simple windowed estimator and collapsing when the estimator is made
deeper, clearing a strong bar only under mode-scarcity and only on an easy task — a clean starved-budget
win that does *not* survive a harder, longer drifting task; (3) *nested* cross-frequency (theta-gamma)
coupling is **redundant** — a flat learnable oscillatory bank already sits at the phase-multiplexing
ceiling. We release the phase-clamp as a **resonance profiler**: point it at any trained complex-
eigenvalue SSM and read off, causally, what its oscillation and its adaptation each buy. Every result
is pre-registered with a frozen decision gate and a firing positive control; three of the swings
produced encouraging one-seed signals that the frozen multi-seed gates overturned, which we report.

## 1. Motivation

The 2,500-year intuition that frequency and rhythm are fundamental to mind (Pythagoras → Fourier →
Berger's EEG) becomes, for the first time, a *controlled experiment* when the "mind" is a fully-readable
computational one. In machine learning the intuition has already been cashed: LinOSS (Rusch & Rus, ICLR
2025) builds a bank of forced harmonic oscillators and beats Mamba ~2× on 50k-length sequences; S4/S5
and their complex-eigenvalue variants underpin the state-space revolution. But *why* they work — whether
the oscillation is causally essential or an incidental feature of a good architecture — is untested,
because the standard comparison is between whole models. We supply the missing causal instrument and use
it to draw the first within-architecture map.

## 2. Method: the single-knob phase-clamp

A complex-diagonal SSM mode evolves `h(t) = λ·h(t−1) + u(t)` with `λ = r·e^{iθ}`. The magnitude `r`
controls memory; the **phase θ controls oscillation** (θ=0 → real eigenvalue → pure decay; θ≠0 →
rotation). Our knob is θ:

- **FREE**: θ learnable → the oscillatory (S5/LinOSS-class) model.
- **CLAMPED**: θ≡0 → a real-eigenvalue decay SSM, everything else identical.

We draw θ's initializer in *both* arms so every non-θ parameter is bit-for-bit identical under a seed;
setting the knob off recovers the baseline exactly (verified `max|Δ|=0`). Adaptive-frequency variants
add a detector that drives θ over time through a slow phase-locked loop with gain κ; `κ=0` recovers the
static model bit-for-bit, so the *adaptation* is isolated by a second knob. All recurrences are computed
with a parallel associative scan (`O(log T)`), verified equal to the `O(T)` sequential reference to
machine precision — enabling the full study on a single 8 GB GPU.

**Discipline.** Every experiment freezes a pre-registration with a numeric decision gate and a *positive
control* (an oracle that has the effect by construction) before data; a null is meaningless if the
positive control does not fire. No gate bar is moved post-hoc; no favorable operating point is promoted
to a headline. Reported numbers are grounded against their receipts by an automatic certifier.

## 3. Related work

LinOSS (2410.03943) and D-LinOSS use *static* (fixed-after-training) frequencies and report no
within-architecture oscillation-vs-decay ablation. TIDES (2605.09742) finds *per-token* input-dependent
imaginary eigenvalues *hurt* and moves input-dependence onto the decay/step-size instead — our
slow-loop entrainment is the untested escape hatch, and we quantify exactly where it helps. "Tuning
Frequency Bias of SSMs" (2410.02035) tunes an innate bias at initialization, not online. AOSNET
(2606.06010) aligns states to a fixed global oscillatory prior in descriptor space, not an entrained
recurrence. AKOrN (2410.13821) synchronizes Kuramoto oscillators at fixed frequencies for binding.
"Task-Level Insights from Eigenvalues" (2510.09379) analyzes eigenvalue *magnitude* correlationally;
our contribution is the *causal, phase-specific* intervention. Across all of these, the move no one
makes — clamp the oscillation off, hold everything else fixed, measure Δ — is ours.

## 4. Results

### 4.1 Is oscillation load-bearing on real long-range data? (sequential MNIST) — YES

A deep S5/LinOSS-class classifier (784-length, `H=64`, 3 blocks), FREE (θ learnable) vs CLAMPED (θ≡0),
matched-param and RNG-matched, 2 seeds, 4000 steps:

| arm | mean test acc |
|---|---|
| FREE (oscillatory) | **0.9837** (98.2 / 98.6) |
| CLAMPED (decay) | **0.9428** (94.0 / 94.6) |

The **architectural gap is +4.1 points** (both seeds positive), and the **within-model reliance is
+88.5 points** — clamping the oscillation out of the *trained* model in place drops it to 9.8%, chance.
On real long-range data the oscillation is causally load-bearing: it is the source of the oscillatory
model's advantage, and a trained oscillatory SSM routes almost all of its computation through the
rotation. This is the controlled single-knob within-architecture ablation the whole-model benchmarks
cannot run.

**The gap widens 7.6× on genuinely long-range data.** Sequential MNIST has local pixel structure, so
the decay model reaches 94% and the gap is a modest +4.1. Removing that crutch with a fixed pixel
permutation (permuted MNIST) is decisive:

| task | FREE (osc) | CLAMPED (decay) | gap |
|---|---|---|---|
| sequential MNIST (locality) | 0.9837 | 0.9428 | +0.0408 |
| **permuted MNIST (no locality)** | **0.9195** | **0.6073** | **+0.3122** |

The oscillation gap **widens 7.6×** — the decay SSM collapses to 60.7% while the otherwise-identical
oscillatory one still reaches 92.0%. Unlike the adaptive-frequency smokes, the effect *grew* from smoke
(+0.13) to full training (+0.31), because the decay model plateaus while the oscillatory one keeps
improving. On genuinely long-range data the oscillation is not merely helpful — it *is* the long-range
mechanism; a decay model of equal budget cannot substitute for it.

### 4.2 Adaptive frequency: a non-monotone detector dose-response

On a within-sequence drifting-period prediction task, an oracle that locks a diverse bank to the *true*
drifting period beats a static bank by +0.17 — the adaptation prize is real. A learned model must
estimate that period online. The advantage over static, by detector, at a comfortable budget (D=8):

| detector | extra params | advantage | capture of oracle | verdict |
|---|---|---|---|---|
| single projection | ~+130 | +0.040 | 23% | KILL |
| **windowed conv (1-layer)** | ~+3.6k | **+0.085** | **50%** | **WEAK (peak)** |
| deep conv + harmonic spread | ~+14.6k | +0.002 | 1% | KILL (collapse) |

The dose-response is **non-monotone**: it peaks at a simple windowed detector and *collapses* when the
detector is made deeper and given the oracle's own inductive bias (8× the parameters buy nothing). Under
mode-scarcity (D=4) the windowed detector captures the *entire* prize and beats the oracle (+0.129), a
clean win. Adaptive frequency therefore works, is captured by a *simple* estimator, and wins cleanly
only when the mode budget is starved — it cannot be bought with detector complexity.

**But that clean win does not scale.** On a harder, longer drifting task (L=192, 4 segments, periods
[3,20]) the same windowed detector is *worse than static at every budget* (RICH−STATIC = −0.04 to
−0.14, the deficit growing with D), even though the oracle prize is real and grows to +0.24 at D=12.
Online frequency estimation over longer sequences with a wider, faster-drifting period range is harder,
and the detector that won on the easy task actively hurts here. The adaptive-frequency effect is thus
bounded on every side: it is real (the oracle proves the prize exists and grows), captured only by a
*simple* estimator, present only in a *narrow easy* regime (short sequences, narrow period band, starved
budget), and it neither scales to harder drifting tasks nor buys a clean win at a comfortable budget.
The learnable estimator cannot reach the prize outside that corner.

### 4.3 Nested cross-frequency (theta-gamma) coupling is redundant

We implement the brain's theta-gamma memory code as an SSM primitive — a slow clock gating fast modes
into ordered phase-slots — on an ordered-recall capacity task. A capacity-headroom positive control (a
bank with twice the modes) fires at +0.18, but explicit nesting captures only +0.009 (5%) of it. A flat
learnable oscillatory bank already realizes the phase-multiplexing; **capacity comes from more modes,
not cleverer coupling.** This answers why no SSM implements cross-frequency coupling: it is redundant.

### 4.4 The resonance profiler

The phase-clamp generalizes to a **profiler** for any trained complex-eigenvalue SSM: clamp θ→0 (and,
for adaptive models, the drive→0) in place and re-evaluate, decomposing accuracy into a decay floor plus
the causal contributions of static oscillation and of adaptation. On a trained adaptive-frequency model
(D=8) it reads: decay floor 0.027; +0.385 from static oscillation; +0.160 from adaptation; total
oscillation reliance +0.545. This is the within-architecture causal complement to correlational
eigenvalue-spectrum analyses, and it ships as the durable artifact regardless of any layer's verdict.

## 5. Discipline as a result

Across the adaptive-frequency swings, three encouraging one-seed / short-training signals
(+0.120, +0.110, −0.037) were **overturned** by the frozen multi-seed gates (to +0.085, +0.002,
+0.040). No gate was moved; the deepest detector was pre-committed as the last swing; the collapses and
near-misses were shipped straight. The trustworthiness of a causal map depends on exactly this refusal
to let an exciting smoke overstate the effect.

## 6. Limitations and future work

Toy-to-modest scale (SSM widths ≤128, synthetic drifting tasks plus real MNIST, ≤2 seeds on the
flagship). Two hoped-for levers for adaptive frequency are now falsified — a *bigger detector* (collapses)
and a *harder/longer task* (the starved-budget win does not scale) — which sharply bounds where adaptive
frequency is useful. The remaining open work: sharper real-task rungs for the oscillation ablation
(permuted MNIST, Long Range Arena, where locality cannot substitute for oscillation), a param-matched
wider-static efficiency control, and running the resonance profiler on a published LinOSS/Mamba
checkpoint (a Torch bridge for the JAX LinOSS release). The map we draw is causal, controlled, and bounded
on every axis we probed; the real-task oscillation result and the profiler are the durable contributions.

## Appendix A: the phase-clamp is LinOSS's oscillation ablation (equivalence)

Our ablation clamps the phase of a complex-diagonal eigenvalue; we show this is exactly the
oscillation-off ablation for the LinOSS forced-harmonic-oscillator layer, so the flagship result is a
statement about LinOSS itself, not a proxy. LinOSS integrates `x'' = −A x + B u` (diagonal `A ≥ 0`).
The LinOSS-IM (implicit) discretization gives, per mode, a first-order recurrence on the state
`s_k = (x_k, y_k)` with `y = x'`:

```
S = 1 / (1 + Δt²·A)
M = [[ 1 − Δt²·A·S ,  Δt·S ],
     [   −Δt·A·S   ,   S   ]]          s_k = M · s_{k−1} + f_k
```

`M` is a real 2×2 matrix. Its eigenvalues are `λ = (tr ± √(tr²−4·det))/2` with `tr = 1 + S − Δt²AS`,
`det = S`. For `A > 0` the discriminant is negative → **complex-conjugate eigenvalues**: the mode
*rotates* (oscillation), with magnitude `√det = √S < 1` providing damping. For `A = 0`, `S = 1` and
`M = [[1, Δt],[0,1]]` → a repeated real eigenvalue 1: a pure **integrator, no rotation**. Hence LinOSS
is, up to a per-mode change of basis, a **complex-diagonal SSM `λ = r·e^{iθ}`** with `r = √S` and
`θ = arg(λ)` set by `A`; the oscillation lives entirely in `θ` (equivalently, in `A`). Clamping the
phase `θ → 0` (our FREE→CLAMPED knob) is identical to clamping the restoring force `A → 0` — the
removal of the oscillation from a LinOSS layer. The seq-MNIST result therefore isolates, for the
LinOSS class, that the oscillation is causally load-bearing — the within-architecture ablation the
LinOSS paper does not report.

## Receipts

`PREREG_/RESULT_/CERT_` for `entrainment`, `entrain_rich`, `entrain_harm`, `nested_crossfreq`,
`smnist_ablation`, `pmnist_ablation`, `scarcity_scale`; `SYNTHESIS_frequency_adaptation`;
`POSITIONING_entrainment_lit`; `resonance_profiler.py` + `INSTRUMENT_resonance_profiler`;
`fig_adaptation_map.png`, `fig_flagship_sharpening.png`. All runs on one RTX 4070 Laptop (8 GB), no
external API.
