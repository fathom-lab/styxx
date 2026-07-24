# PREREG — ENTRAIN-OSS kill-gate (drifting-period entrainment) — 2026-07-23

**PRE-DATA AMENDMENT (2026-07-23, before any confirmatory run; smoke/plumbing only so far):** the
ENTRAIN PLL update drops the `[0,π]` clamp on θ (i.e. `θ(t)=(1−κ)θ(t−1)+κ·g·ω̂(t)`, no clamp). Reason:
`e^{iθ}` is periodic, so unbounded θ is a valid eigenvalue phase, and an unclamped θ gives ENTRAIN
*more* freedom — it can only **strengthen** the invention, never flatter STATIC, so it does not
weaken the KILL. This also makes the θ recurrence linear, enabling a parallel associative scan
(`lin_scan`, O(log T)) that replaces the O(T) loop and is red-team-verified equal to the sequential
reference (`scan==seq`, max|Δ|≈3e-7, complex & real). ORACLE keeps its `[0,π]` clamp (instant, no
recurrence). `κ=0` still reduces ENTRAIN to STATIC bit-for-bit (max|Δ|=0.0). Gate numbers unchanged.

**FROZEN before any confirmatory data.** Runner: `run_entrain_timing.py`. Compute: 1× RTX 4070
Laptop (8 GB), torch 2.5.1+cu121. This prereg governs the Phase-1 decisive falsification in
`~/.claude/plans/sunny-toasting-toast.md`.

## Question

The prior arc (`RESULT_timing_2026_06_04.md`) showed the free oscillatory net wins periodic-timing
tasks via a **broad static phase bank** — it does *not* tune θ→2π/P (enrichment near target ≈ 1.0×,
i.e. chance). So a static bank already covers a fixed period. The invention (ENTRAIN-OSS) can only
matter in the regime that finding leaves open: when the input period **drifts within a sequence**
and the mode budget **D is starved**, so no fixed bank can cover the range at once. Decisive
question, frozen:

> Does a layer that **tunes its own oscillation frequency to the input online** (a slow learned
> phase-locked loop over the recurrent eigenvalues) beat the **best static-frequency bank of equal
> mode budget**, in a drifting-period regime where an oracle proves adaptation is rewarded?

## Task (frozen)

Within-sequence **drifting-period** next-symbol prediction. Each sequence = `S=3` segments of
`SEG_LEN=32` (total `L=96`). Each segment draws an integer period from `[PMIN,PMAX]=[3,12]`;
adjacent segments are forced to differ (guaranteed drift, 2 change-points). The stream is
quasi-periodic (`x(t) = motif[(t−seg_start) mod P_seg]`). A target position is **scored only when
≥ 2 local periods have elapsed since the segment start** (`local ≥ 2·P_seg`) — the local period is
then inferable; the un-inferable warmup is masked (`-100`). Metric: mean next-symbol accuracy over
scored positions, averaged over 3 seeds `[0,1,2]`, `EVAL_N=1024`.

## Arms (frozen)

- **STATIC** — θ learned but **constant** in time (the strong LinOSS-style competitor; *this*, not
  decay, is the baseline to beat).
- **ENTRAIN** (the invention) — θ_j(t) evolves by a slow PLL:
  `θ_j(t)=clamp(θ_j(t−1)+κ_j·(g_j·ω̂(t)−θ_j(t−1)), 0, π)`, with online input-frequency estimate
  `ω̂(t)=angle(z(t)·conj(z(t−1)))`, `z(t)=a·x(t)+i·b·x(t)`, learned `κ_j∈(0, κ_max=0.1]`, learned
  harmonic ratio `g_j`, learned init `θ_j(0)`.
- **ORACLE** (positive control, PC1) — a **diverse** bank locked to the **true** fundamental:
  `θ_j(t)=clamp(spread_j·2π/P(t), 0, π)`, `spread=geomspace(0.5,2.0,D)` fixed, instant lock. Proves
  whether re-pointing frequency to the input's true timescale is rewarded *at all*.
- **CLAMPED** — θ≡0, pure decay (floor/sanity; primary D only).
- **TRANSFORMER** — attention, learned positions (context only; **not** in the gate).

**Single-knob discipline:** `κ≡0` reduces ENTRAIN to STATIC **bit-for-bit** (red-team asserted
`max|Δ|=0.0` before the run; entrain-only params drawn after the read head so shared params
RNG-match STATIC). STATIC and ENTRAIN use the **same sequential recurrence** ⇒ **matched-compute by
construction** (the FLOP-fairness confound only bites the transformer, which is not gated).
Matched-param: STATIC vs ENTRAIN differ only by `a,b,κ,g` (≈ `2·d_in+2D` params); exact counts
reported.

## Frozen kill-gate

Config: `D_SWEEP=[4,8,16]`, **primary D=8**, **fallback D=4**, `STEPS=1500`. Define
`adv = ENTRAIN−STATIC`, `orc = ORACLE−STATIC` on the drift task (seed-averaged).

Decision tree (evaluated at primary D=8; if the positive control does not fire there, re-evaluate
once at the pre-registered fallback D=4 — this fallback is gated **only** on the positive control,
never on ENTRAIN's performance, so it is not a garden-of-forking-paths pick):

- **ABSTAIN** iff `orc < 0.10` at **both** D=8 and D=4 — the task does not reward adaptation at the
  tested mode budgets, the instrument cannot detect the effect, **no conclusion** about the
  invention. (Direct antidote to the arc's "almost falsified Plato" failure: no verdict when the
  positive control is silent.) → redesign (wider period range / smaller D) and re-freeze.
- **GREENLIGHT** iff `orc ≥ 0.10` **and** `adv ≥ 0.10` **and** `adv ≥ 0.5·orc` (captures ≥50 % of
  the oracle's available adaptation gap, tying the win to tracking). → proceed to Phase 2.
- **KILL** iff `orc ≥ 0.10` **and** `adv < 0.05` (adaptation is available and rewarded, but the
  learnable loop fails to capture it). → ship the honest negative; pivot to nested cross-frequency
  or the profiler tool.
- **WEAK** otherwise (`0.05 ≤ adv < 0.10`, or capture < 50 %) — real but sub-threshold; **not** a
  greenlight; do not scale on it alone.

**NO-HARM secondary gate:** on the fixed-period task at the used D, `ENTRAIN−STATIC ≥ −0.03`
(entrainment must not break the stationary case).

**Corroboration (reported, non-gating):** the ENTRAIN advantage should be **non-increasing in D**
(the invention is designed to help most under mode scarcity; a rising trend is a red flag the win
is an artifact).

## Discipline commitments

- Gate frozen here, before data. No post-hoc D selection beyond the positive-control-gated fallback.
- Positive control is load-bearing: a null is meaningless if ORACLE does not fire.
- The noise band is ≈ ±0.05 (arc precedent on integer-symbol accuracy at 3 seeds); the +0.10 bar
  sits above it.
- No re-scoring of motivating data; ENTRAIN gets its best fair shot (learned κ, g, detector), so a
  KILL is decisive.
- Result written to `entrainment_result.json` + `RESULT_entrainment_2026_07_23.md`, numbers grounded
  by `python -m styxx.certify` (expect OATH-HELD, 0 UNGROUNDED).
