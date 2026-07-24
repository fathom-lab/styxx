# RESULT — ENTRAIN-HARM: KILL — the dose-response is non-monotone; detector complexity has an optimum — 2026-07-23

Frozen by `PREREG_entrain_harm_2026_07_23.md`. Receipt: `entrain_harm_result.json`. Runner:
`run_entrain_harm.py` (parallel scan; red-team `scan==seq`, `κ=0==STATIC` bit-for-bit). 3 seeds, 1500
steps, RTX 4070. **Verdict: KILL** at the pre-registered primary D=8.

## One line

Giving the detector the oracle's own inductive bias — a deeper 2-layer conv estimating a single
fundamental (band-constrained off Nyquist) × a learned per-mode spread — **did not** cross the flagship
bar. It captured **1%** of the oracle gap (+0.002), *worse* than the simple one-layer windowed conv
(+0.085, 50%), despite **8× the parameters** (16,661 vs 2,052). The adaptive-frequency dose-response is
**non-monotone**: it peaks at a simple windowed detector and collapses with more complexity.

## Results (drifting-period accuracy, seed-averaged)

| D | STATIC | HARM (deep conv + spread) | ORACLE | HARM−STATIC |
|---|---|---|---|---|
| 4 | 0.247 | 0.265 | 0.351 | +0.0181 |
| **8** (primary) | 0.460 | 0.463 | 0.631 | **+0.0022** |

`oracle−static = +0.1703` (positive control fires), `capture = 0.0132`, `adv < 0.05` → **KILL**.
No-harm (fixed period): +0.0208.

## The dose-response (the actual finding)

Adaptive-frequency advantage over static at D=8, by detector:

| detector | params (extra) | HARM/RICH−STATIC | capture of oracle | verdict |
|---|---|---|---|---|
| single projection | ~+130 | +0.040 | 23% | KILL |
| **windowed conv (1-layer)** | ~+3.6k | **+0.085** | **50%** | **WEAK (best)** |
| deep conv + harmonic spread | ~+14.6k | +0.002 | 1% | KILL |

**Detector richness is not monotonically helpful — it has an optimum at the simple windowed conv.** The
deeper, better-inductive-bias detector fails to train usefully at this budget: 8× the params buy
nothing, and it lands at static. A clean flagship win at the comfortable D=8 budget does **not** exist
across the detectors tried; the clean adaptive-frequency win is real but lives only under mode-scarcity
(D=4, where the windowed conv hit +0.129).

## The discipline (the trust spine — three catches now)

Its smoke read **+0.110** (GREENLIGHT); the full 3-seed / 1500-step run reads **+0.002** (KILL) — the
**third** over-optimistic smoke the frozen gate has overturned this session (ENTRAIN −0.037→+0.040,
RICH +0.120→+0.085, HARM +0.110→+0.002). No bar was moved. This was pre-committed as the **last**
detector swing; there is **no detector #4** fished to cross the bar. The 3-point non-monotone
dose-response is the reported finding.

## Reading

The honest headline stands and is now bounded on both sides: **learned online frequency adaptation
gives an SSM a real edge, best captured by a simple windowed detector (~50% of the oracle prize,
WEAK), strongest under mode-scarcity (clean win at D=4) — and it does not reach a clean flagship at a
comfortable budget, with more detector complexity actively hurting.** Where the prize is genuinely
reachable is narrow and specific; STYXX mapped exactly where, and the frozen gates refused to let an
exciting smoke overstate it three times.

Scope: toy (D≤8, L=96, integer symbols, 3 seeds). What remains genuinely open is not "a bigger
detector" (falsified here) but a different regime — the mode-scarcity budget where the D=4 win lives,
at scale — and the resonance profiler as the durable instrument.
