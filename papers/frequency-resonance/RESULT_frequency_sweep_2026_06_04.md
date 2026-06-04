# RESULT — Ordered-memory capacity is RESONANT in oscillation frequency, not monotonic

**Date:** 2026-06-04 · **Verdict (frozen rule): RESONANT.** Gate:
`PREREG_frequency_sweep_2026_06_04.md`. Sweep of frozen oscillation frequency θ (rad/step) in
the validated rhythm-rescue LRU rig, ordered-copy task, 3 seeds. Identical config except θ.

## The numbers — kcap(θ)

| θ / π | 0.0 | 0.0625 | 0.125 | 0.1875 | 0.25 | **0.375** | 0.5 | 0.6875 | 0.875 | 0.97 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **kcap** | 2.67 | 4.00 | 3.67 | 3.67 | 4.00 | **5.33** | 4.00 | 4.00 | 4.00 | **2.00** |

- **Interior peak θ\* = 0.375π**, kcap 5.33. **Collapses to 2.00 at Nyquist (θ=0.97π)** — *below*
  the no-rhythm baseline (θ=0 → 2.67). Spearman(θ, kcap) = **0.16** (no monotone trend).
- Reference: **FREE** (learned per-mode spectrum) kcap **6.0** (all 3 seeds), keeping real
  oscillation (osc_use ≈ 0.62 — did not collapse to zero).

## Scorecard vs the three frozen hypotheses

1. **H_mono — "higher frequency, greater capacity" (operator): FALSIFIED.** Not monotone
   (ρ=0.16, needed ≥0.90); argmax is interior, not at Nyquist. Worse: the highest frequency is
   the **global minimum** (2.0), beneath no-rhythm. *Higher is not better — past the optimum,
   higher is actively destructive.*
2. **H_res — resonant interior optimum, θ\*/π ∈ [0.1, 0.4] (pre-registered prior): CONFIRMED.**
   θ\*=0.375π is interior and inside the pre-stated band; capacity falls 3.3 items from peak to
   Nyquist (rule required ≥2). Called before the data.
3. **H_spectrum — learned spectrum beats best single tone by ≥1 item (prior): FAILED its own
   bar.** FREE 6.0 vs best single 5.33 = +0.67, short of the +1.0 threshold. *Directionally a
   spread helps, but it did not clear the registered bar — not claimed.*

**Net: the operator's monotonic claim is falsified; my resonance call is confirmed; my spectrum
sub-call missed its own gate.** Both sides took a hit where the data said so — that is what makes
the resonance result trustworthy.

## What it means (honest)

**Frequency is fundamental to memory — but as a resonance, not a ladder.** The mechanism is
clean: too slow (θ→0) under-uses phase coding and items don't separate; too fast (θ→Nyquist)
wraps phase *within* the retention window so items alias and collide — capacity drops below
having no rhythm at all. There is a **best band** (~0.375π here) that tiles the window with
distinguishable phases. The free net independently parks itself in the productive region rather
than maximizing frequency.

This is the grounded refinement of the "vibration underlies mind" intuition: it does — and the
honest law is **tune to the resonant band, don't maximize frequency.** It even echoes real
tissue, where runaway high-frequency synchrony is not enlightenment but a **seizure**; health is
the right band and the right coupling. We reproduced that principle in silico, from a frozen
prediction.

## How it fits the program

Deepens rhythm-rescue (presence/absence → frequency-dependence). Rhythm-rescue: oscillation
≈ doubles capacity vs none. This: that benefit is **band-limited and non-monotone** — a specific
optimum, with a cliff beyond it. Together: *oscillation is a capacity-extending mechanism with a
resonant optimum, not a "more is better" knob.*

## Caveats (frozen)

- One task (ordered copy), one architecture (LRU), in-silico, n=3 seeds, capacity at a 0.80
  threshold, single hidden size. Read the **shape** (resonant, cliff at Nyquist), not the exact
  θ\* — the optimum's location depends on retention-window length, |λ|, and d.
- Single-fixed-θ shares one frequency across all 256 modes by design (the H_spectrum contrast).
- Productizing into a styxx instrument is a **separate** per-feature validation (cf. the geometry
  probe, which died on a confound control despite a clean research finding). Not claimed here.
