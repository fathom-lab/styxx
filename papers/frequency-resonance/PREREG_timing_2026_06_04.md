# PREREG — oscillation on its HOME TURF: does rhythm win at periodic timing? (the steelman)

**Date (frozen before data):** 2026-06-04. The deepest, fairest test of the whole arc. Every prior
experiment used ordered *copy* — a pure **memory** task, where the honest objection is "rhythm isn't the
point of that task." But oscillation's *native* function is **timing**: a complex eigenvalue at frequency
θ=2π/P is literally a clock that resonates with a period-P signal; its phase encodes position-in-cycle.
The fair test is to put oscillation on its home turf — **predicting periodic temporal structure** — and
ask whether it finally wins, and whether it finally beats attention.

## Task (frozen) — periodic next-symbol prediction
- Vocab V=12. Per sequence: sample period P, sample a random motif m[0..P-1], emit x[t]=m[t mod P] for
  t=0..L-1 (L=48). The net reads x[0..t] and predicts x[t+1] at every step. **Score only t ≥ 2P** (after
  two full periods are visible — the period is *inferable*; this rewards phase-tracking, not luck).
- This needs the net to (a) infer the period and (b) phase-track to predict the next element — exactly
  what an oscillator does natively and a decay (leaky-average) state cannot represent.

## Arms (frozen) — the controlled knob + the necessity rematch
- **LRU-CLAMPED** (θ≡0, decay only) · **LRU-FREE** (θ learnable, oscillatory) · **TRANSFORMER** (attention,
  **learned** positions). Matched params (~168k, the necessity-run budget). 3 seeds.
  *Position scheme matters here:* sinusoidal positional encoding is itself a bank of oscillators at many
  frequencies — using it would inject rhythm into the "rhythm-free" arm and confound P2. Learned absolute
  positions are the genuinely rhythm-free baseline; and because the period P varies per sequence, learned
  positions do NOT trivially solve the task (the net must still infer P from content and attend back P).
- Metric: per-period prediction accuracy acc(P) on the scored positions, swept over **P ∈ {2,3,4,5,6,8,10,12}**.

## Hypotheses & predictions (frozen)
- **P1 — periodic timing IS oscillation's native domain:** FREE's advantage over CLAMPED GROWS with period
  P — Spearman(P, acc_free − acc_clamped) ≥ **+0.4**, AND at the longest periods (P ≥ 8) acc_free −
  acc_clamped ≥ **+0.15**. Mechanism: phase encodes cycle position; decay cannot represent "P steps ago."
- **P2 — the necessity rematch on home turf:** does rhythm beat attention *here*, where it should? Compare
  acc_free vs acc_transformer at P ≥ 8. **If acc_free ≥ acc_transformer − 0.05**, oscillation finally has a
  domain it is not dominated in (the genuine niche). **If acc_transformer > acc_free + 0.05**, rhythm is
  dominated even on its native task — the demarcation is complete.

## Decision rule (frozen)
- **RHYTHM'S NICHE IS TIMING** iff P1 holds — oscillation's edge grows with period. This is the positive
  reframe the honest arc has been circling: rhythm is not special for *capacity* (refuted 6×) but IS the
  mechanism for *temporal structure* — which is what modern neuroscience increasingly says brain rhythms
  are *for* (timing & coordination, not storage). Then P2 says whether it's special or merely sufficient.
- **RHYTHM NOT SPECIAL EVEN AT TIMING** iff P1 fails — if oscillation does not even win at periodic
  prediction, the "frequency underlies cognition" claim is bounded as far as controlled silico can bound it.
- Report acc(P) for all three arms regardless.

## Caveats (frozen)
- One periodicity family (exact repetition; not jittered/noisy periods), one architecture family per arm,
  matched params. A clean P1 here is the strongest *and most honest* pro-oscillation result possible —
  oscillation winning on the task it is mechanistically built for — and it would REFRAME the six prior
  refutations rather than overturn them (capacity ≠ timing). This is the experiment that could turn the
  whole arc from "rhythm is dominated" to "rhythm is dominated *for memory* but sovereign *for timing*."
