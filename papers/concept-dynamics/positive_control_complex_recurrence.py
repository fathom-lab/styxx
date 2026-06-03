# -*- coding: utf-8 -*-
"""
positive_control_complex_recurrence.py — close the eigenvalue hierarchy.

The rhythm finding rests on a dividing line: complex-valued recurrence eigenvalues ->
oscillation; real-valued -> exponential decay (commitment). We showed the two NEGATIVE
arms empirically (transformer commits, Mamba-1 commits) and structurally (no/real
eigenvalues). This is the POSITIVE control: a minimal linear recurrence

    h_{t+1} = lambda * h_t + drive * noise_t ,   s(t) = Re(h_t)

driven by noise, observed via its real part. With lambda = r*exp(i*omega):
  - COMPLEX (omega != 0): the impulse response ROTATES -> genuine endogenous oscillation
    at frequency omega/2pi (NOT a planted sinusoid -- it emerges from the dynamics).
  - REAL (omega = 0): pure decay -> an AR(1)-like commitment process.

If the VALIDATED instrument (analyze_concept_dynamics) detects oscillation in the complex
arm and NOT the real arm, the dividing line is demonstrated end-to-end with endogenous
dynamics, and the transformer/Mamba-1 BOTH_COMMIT result is fully credible: the instrument
sees rhythm exactly when a recurrent substrate's eigenvalues produce it.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_concept_dynamics import analyze_series


def recurrence(n, lam, rng, drive=0.4):
    h = 0 + 0j
    s = np.empty(n)
    for t in range(n):
        h = lam * h + drive * rng.standard_normal()      # real-valued noise drive
        s[t] = h.real
    return s


def main():
    rng = np.random.default_rng(1)
    n, N = 64, 40
    period = 8
    w = 2 * np.pi / period

    print("POSITIVE CONTROL — does complex recurrence oscillate (and real not)?")
    print(f"  n={n}, trials={N}, complex period={period} tokens; sweep eigenvalue magnitude r\n")

    conds = [("REAL    (omega=0, r=0.40  ~ real models' AR1)", complex(0.40, 0.0)),
             ("REAL    (omega=0, r=0.95  high-AR1, edge-leak)", complex(0.95, 0.0))]
    for rr in (0.92, 0.97):
        conds.append((f"COMPLEX (omega=2pi/8, r={rr})            ", rr * np.exp(1j * w)))

    for name, lam in conds:
        hits, freqs = 0, []
        for _ in range(N):
            s = recurrence(n, lam, rng)
            res = analyze_series(s, rng, n_surr=300)
            if res.get("usable") and res["surr_pass"] and res["F_pass"]:
                hits += 1
                freqs.append(res["peak_freq"])
        det = hits / N
        fstr = f", peak freq ~{np.median(freqs):.3f} cyc/tok (expect ~{1/period:.3f})" if freqs else ""
        print(f"  {name:38s} oscillation detected {hits}/{N} = {det:.2f}{fstr}")

    print("\n  PASS iff: complex arm detection HIGH, real arm detection LOW")
    print("  -> demonstrates: complex eigenvalues -> endogenous oscillation our instrument catches;")
    print("     real eigenvalues -> commitment. The dividing line, end-to-end.")


if __name__ == "__main__":
    main()
