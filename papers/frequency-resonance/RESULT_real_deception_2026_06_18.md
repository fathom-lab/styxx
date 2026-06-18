# RESULT — the real instrument test: switching dynamics carry NO real deception signal. Dynamical line CLOSED.

**2026-06-18.** Executed `PREREG_real_deception_2026_06_18.md` (frozen a0b4933, before code). Outcome:
**clean kill with a valid positive control.** The switching object — alive on a coherence toy proxy —
does not survive contact with real instructed deception. The static probe does the entire job.

## What ran
Qwen2.5-1.5B-Instruct (GPU, greedy), 60 factual questions × 3 conditions (honest / deceptive /
deceptive-obfuscated), paired. Token-time residual trajectory at layer 20 (~0.7 depth). Static linear
probe (mean residual → StandardScaler+PCA40+LogReg, positive control) vs switching/non-stationarity
features (treatment). 5-fold CV AUROC. n=60 per condition.

## Results
| contrast | STATIC probe (control) | SWITCHING (treatment) |
|---|---|---|
| honest vs deceptive | **1.000** | **0.501** |
| honest vs obfuscated | 1.000 | 0.622 |

## Gates (binding)
- **K0 — positive control: PASS (static = 1.000 ≥ 0.75).** The setup is valid; a strong deception
  signal is present and the static linear probe separates it perfectly (consistent with the literature's
  0.96–0.999). This is exactly the validity the cross-sample proxy lacked — so the treatment result
  below is interpretable.
- **K1-real — switching carries deception: FAIL (0.501 < 0.70).** Chance. The switching object carries
  **no** real-deception signal; the static probe carries all of it.
- **K2 — moot.** K1 failed, so the obfuscation gate is not evaluated. (Note: the weak "sound natural"
  instruction didn't even drop the static probe — 1.000 → 1.000 — so this model/attack wouldn't have
  tested K2 regardless; the real static-collapse attacks are gradient suffixes / RL-obfuscation, out of
  scope here.)

## Verdict
**The dynamical-integrity line is CLOSED on real data.** The 0.672 that looked alive
(`RESULT_nonstationarity`) was coherence-vs-shuffle structure, not deception-relevant dynamics — it
vanishes (0.501) the moment the contrast is real honest-vs-deceptive generation. Because the positive
control is perfect (1.000), this is a clean negative, not an underpowered one. Per the pre-registered
stop rule: closed, not flogged. No 5th swing.

## What this session established (4 pre-registered swings, all converged)
1. Spectral probe (DMD spectrum) — CLEAN KILL.
2. Single-trajectory switching — alive **on the coherence toy only** (0.672).
3. Cross-sample cheap proxy — INCONCLUSIVE (positive control failed).
4. **Real instructed deception — CLEAN KILL (switching 0.501, static 1.000).**

The honest synthesis: **there is no white-box *dynamical* (spectral or switching) integrity signal for
deception that adds over the static probe.** The static linear direction already separates instructed
deception perfectly; the trajectory's frequency content and its non-stationarity add nothing. styxx's
proven assets stand — the static probe and sampling-divergence — and the exotic dynamical extensions are
falsified by their own gates.

## The honest redirect (the question that stays open)
The **obfuscation-robustness problem is real and unsolved** — static probes do collapse under gradient
suffixes / RL-obfuscation in the literature, and that is the genuine frontier for a deployed integrity
layer. What this session did was **falsify the dynamical answer to it** (spectrum: no; switching: no).
That is a real contribution — it rules out a plausible and much-hyped direction with four pre-registered
gates and a validated positive control — and it says the robustness fight has to be won on the static /
sampling signals under *real* attacks, not on trajectory dynamics. The frequency/dynamical program is
honestly complete: a clean methods/reproduction result on the resonance side, and a fully-falsified
product extension on the integrity side.
