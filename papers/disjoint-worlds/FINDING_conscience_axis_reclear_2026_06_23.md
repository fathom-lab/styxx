# FINDING — value-axis re-clear (A0): generic behavioral axes share a recoverable subspace, but the honesty axis escapes it

**Pre-registered in `PREREG_conscience_axis_reclear_2026_06_22.md` (frozen `503c087`, before any
axis-clear data). Runner `run_g0clear_axes.py` (`c136b3b`). Llama-3.2-3B-Instruct, local, no API key.**

## Why this run exists

Rung 3 (cross-mind conscience-as-control) came back **VOID/ALIGNER_LIMITED**: the G0 instrument,
cleared on *noun-concepts*, does not host the honesty **axis** (a diffuse value direction) — the
positive control G0′ was 0.294 ≪ 0.80, so the conscience-transfer result was uninterpretable, not a
read≠write confirmation. The prereg's fix: rebuild the cleared subspace from *value/behavioral axes*
instead of noun-concepts, and test whether the held-out honesty axis is hosted (pc_cos ≥ 0.80 = A0
pass → conscience-transfer becomes validly testable; < 0.80 = INSTRUMENT-CEILING, itself a geometry
finding). The prereg's pre-stated prior: **A0 is the real risk and may fail; possibly only clears at
large bank sizes.**

## Method (as pre-registered)

Each value axis = mean(positive-primed answer-token residual) − mean(negative-primed), over a shared
40-question neutral set, L2-normalized, at candidate layers {0.40, 0.50, 0.60}·n_layers = {11, 14, 17}.
PCA-clear over the axis bank with the SELECTION/FINAL anti-overfit seal: **the honesty axis is held out
entirely**; split the remaining axes into a train cloud (PCA) and a SEL set; sweep (layer, k) to
maximize pc_cos on SEL; lock; then test pc_cos(held-out honesty axis, locked subspace) on FINAL.
**A0 passes iff the held-out honesty axis is hosted at pc_cos ≥ 0.80.** (k collapses to n_train−1
because the cloud has fewer directions than the k-grid; the effective test is honesty vs the full
≤(n_train−1)-dim PCA subspace of the other axes.)

## Result — first pass, 24-axis bank (`g0clear_result_axisllama3b.json`)

| quantity | value |
|---|---|
| bank size | 24 (23 non-honesty + honesty held out) |
| train / sel | 18 / 5 |
| locked | layer 11, k 17 |
| **SEL value axes hosted** (held-out *generic* axes) | **pc_cos 0.819** |
| **honesty axis hosted** (FINAL) | **pc_cos 0.633** |
| verdict | **INSTRUMENT-CEILING** (A0 fail, < 0.80) |

**The informative split: it is not that value directions fail to share a subspace — they largely do.**
The held-out *generic* value axes (certainty, formality, politeness, … the SEL set) are hosted by the
cleared subspace at **0.819 ≥ 0.80**. It is the **honesty axis specifically that escapes it (0.633)**.
Honesty is not a low-dimensional combination of style/affect/social behavioral directions — it is more
idiosyncratic than the rest of the value bank.

**Caveat that motivated the decisive run:** the prereg targeted **≥ 40** axes and predicted A0 "may
only clear at large bank sizes." 24 is below that target, and with 18 training axes the subspace dim
was capped at 17. A fail at 24 is therefore the *predicted-but-weak* outcome — it could be a bank-size
artifact rather than a property of honesty. So the bank was expanded to 40 distinct value axes (no
honesty-proxies, to avoid leakage into the held-out direction) and A0 re-run.

## Result — decisive, 40-axis bank (`g0clear_result_axisn40.json`)

| quantity | 24-axis bank | **40-axis bank (prereg target)** |
|---|---|---|
| bank size (non-honesty + held-out honesty) | 23 + 1 | **40 + 1** |
| train / sel | 18 / 5 | 30 / 10 |
| locked | layer 11, k 17 | layer 11, k 29 |
| **SEL generic value axes hosted** | 0.819 | **0.876** |
| **honesty axis hosted** (FINAL) | 0.633 | **0.707** |
| verdict | INSTRUMENT-CEILING | **INSTRUMENT-CEILING (A0 fail, < 0.80)** |

**Confirmed at the prereg's target bank size. The split is now clean and strong.** Expanding the bank
from 24 → 40 distinct value axes (no honesty-proxies) moved honesty 0.633 → 0.707 — a real rise, so
the 24-axis fail *was* partly bank-size-limited, exactly as flagged. But +17 diverse axes bought only
+0.074, and honesty remains **0.17 short of the 0.80 bar**, while the generic SEL value axes clear
*even better* with the larger bank (0.819 → 0.876). Honesty's pc_cos is rising with N but plateauing
below threshold; the generic value-axis manifold is recovered cleanly and honesty sits measurably
outside it (0.71 vs 0.88).

**A0 FAILS → Rung 3 is NOT run** (pre-registered gate: conscience-control is tested *only if* A0
passes; the G0′ positive control would fail and the transfer would be uninterpretable, as it did in
the VOID run that motivated this). I did **not** keep expanding the bank hunting for a pass — the
prereg fixed the bar at ≥40, it was met, honesty failed; a much larger *curated* bank is banked as a
future pre-registered extension, not chased now (axis-quality / leakage risk grows with bank size).

## Interpretation

The honesty direction sits measurably outside the span of 40 other value axes — while those axes span
each other cleanly (0.876) — so honesty is **not reducible to a blend of style, affect, and social
stance**; it is more idiosyncratic / higher-dimensional than the generic behavioral manifold. The
mission consequence is concrete: cross-mind conscience-**control** cannot be built on a *linear
value-axis correspondence map*, because the very direction one would transfer (honesty) is the one
that escapes the recoverable subspace — so the G0′ positive control is structurally bound to fail.
This converges with the program's standing asymmetry — cross-mind **representation** (read) transfers
where **control** (write) does not — now one level deeper: even *within a single model*, the honesty
axis resists placement in a shared low-dim value basis that generic axes occupy. **READ-ONLY
conscience remains the natural primitive** (cf. `project_conscience_mount_2026_06_12`,
`project_crossmind_write_channel_2026_06_20`); the conscience-as-governor extension is not reachable
by this route.

## Honest bounds

- Single model (Llama-3.2-3B), single 40-question prime set, one seed, in-silico. No consciousness claim.
- Axis quality is itself a source of error: each axis is a prompt-contrast direction, not audited for a
  guaranteed native steering effect before inclusion (the prereg flagged this). A noisy axis pollutes
  the cloud; the result is a property of *this* bank construction.
- A0 fail = the honesty axis is not hosted by *this* linear PCA subspace at *this* bank size; it does
  not prove no value-axis representation hosts honesty (a non-linear or much larger/curated bank is not
  ruled out). Reported as INSTRUMENT-CEILING with the max pc_cos and bank size stated, per prereg.
- Nothing is wired from this run. The shipped `styxx.mount` / `styxx.Conscience` (read-only cooperative
  monitor) is unaffected — this tests the *control* extension, which remains not-established.

## Receipts

- Prereg: `PREREG_conscience_axis_reclear_2026_06_22.md` (`503c087`). Runner: `run_g0clear_axes.py`.
- 24-axis: `g0clear_result_axisllama3b.json`. 40-axis: `g0clear_result_axisn40.json` (`G0_pc_cos_FINAL` 0.7066, `G0_pass` false).
- Predecessor (VOID Rung 3 that motivated this): `FINDING_rung3_steerlayer_2026_06_22.md`.
