# RESULT — ENTRAIN-RICH: WEAK at D=8, GREENLIGHT at D=4 — the detector IS the bottleneck — 2026-07-23

Frozen by `PREREG_entrain_rich_2026_07_23.md`. Receipt: `entrain_rich_result.json`. Runner:
`run_entrain_rich.py` (parallel scan; red-team `scan==seq`, `κ=0==STATIC` bit-for-bit). 3 seeds, 1500
steps, RTX 4070. **Verdict: WEAK** at the pre-registered primary D=8 — real, mechanism-tied, below the
strict +0.10 bar.

## One line

Replacing ENTRAIN's cheap single-projection frequency detector with a **learned causal conv over a
short window** roughly **doubles** the captured adaptation advantage — from +0.040 (KILL) to **+0.085**
(50% of the oracle ceiling) at D=8, and to **+0.1286 at D=4, where it beats even the diverse-bank
oracle**. Learned online frequency adaptation *works*; the bottleneck was the **detector's
expressiveness**, exactly as the ENTRAIN KILL flagged. It lands just under the strict flagship bar at
the comfortable budget, and clears it under mode-starvation.

## Results (drifting-period accuracy, seed-averaged)

| D | STATIC | RICH (conv) | ORACLE | RICH−STATIC | capture of oracle |
|---|---|---|---|---|---|
| 4 | 0.247 | 0.375 | 0.351 | **+0.1286** | >100% (beats oracle) |
| **8** (primary) | 0.460 | 0.545 | 0.631 | **+0.0847** | **49.7%** |

The **dose-response** across detector richness (same task, same oracle, same gate): single-projection
`ω̂=angle(z·z*)` captured 23% (`RESULT_entrainment`, KILL); the windowed conv (`KW=7`) captures
**50%** (this result, WEAK). No-harm on fixed period: **+0.0858** (well above −0.03 — entrainment does
not break the stationary case; it *helps* it).

## The frozen gate (primary D=8)

`oracle−static = +0.1703 ≥ 0.10` → positive control fires. `rich−static = +0.0847`, capture `0.4974`.
`adv < 0.10` → **WEAK** (real but sub-threshold). Not a GREENLIGHT — the frozen bar is not moved to
rescue it. At D=4 (in the pre-registered sweep, reported not headline): `rich−static = +0.1286`,
GREENLIGHT-level and above the oracle — consistent with the whole arc's pattern that adaptation helps
most under mode scarcity.

## The fairness caveat (stated plainly)

RICH adds params (D=8: 5652 vs STATIC 2052 — the Conv1d(64→8, k=7) is ~3.6k). These are
**mechanism-gated**: at `κ=0` the conv output never reaches θ, so RICH is bit-for-bit STATIC
(red-team `max|Δ|=0.0`). The extra params can affect the output through **exactly one channel** —
learned frequency adaptation — so the +0.085 is a clean causal measurement of *the mechanism*, not of
raw capacity. The separate *efficiency* question — whether those params would buy more as extra modes
(a param-matched wider static bank) — is not settled here and is the honest next control.

## Reading

This is the discipline catching an over-optimistic smoke (1 seed/400 steps read +0.120; the honest 3
seeds/1500 steps read +0.085) — the second such catch this session. But it also **converts the ENTRAIN
KILL into a real, bounded positive**: the mechanism is sound, the prize (oracle +0.17) is genuinely
partly reachable, and the lever is the detector. The honest headline is not "a super-tool that beats
everything" — it is "learned frequency adaptation gives an SSM a real edge on drifting-timescale
sequences, strongest under mode scarcity (clean win at D=4), detector-limited at a comfortable budget."

Scope: toy (D≤8, L=96, integer symbols, 3 seeds). Open: a still-richer detector (does capture keep
climbing toward the oracle past 50%?), a param-matched wider-static efficiency control, and the
mode-scarcity regime where the D=4 win lives.
