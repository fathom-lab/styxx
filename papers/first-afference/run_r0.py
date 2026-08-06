"""R0 — validate the R1 instrument on synthetic worlds, per PREREG_r0_instrument_validation_2026_08_05.md.

Runs the EXACT R1 machinery (run_r1.measure) on generated worlds with known ground truth.
`--smoke` = tiny n, world C only, INVALID-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_r1 import measure                      # noqa: E402

SMOKE = "--smoke" in sys.argv
N_BINS = 40 if SMOKE else 240
SEEDS = [11] if SMOKE else [11, 12, 13]
N_DAYS = 4


def make_world(seed: int, coupled: bool, clocked: bool, alpha: float = 1.0):
    """240 one-minute bins scattered over 4 simulated days; AR(1) latent + clock + tanh maps."""
    rng = np.random.default_rng(seed)
    mins = np.sort(rng.choice(N_DAYS * 24 * 60, size=N_BINS, replace=False))
    hours = (mins // 60) % 24
    gaps = np.diff(mins, prepend=mins[0])

    def ar1(dim):
        z = np.zeros((N_BINS, dim))
        z[0] = rng.standard_normal(dim)
        for i in range(1, N_BINS):
            rho = 0.9 ** gaps[i]                       # decay with real time gap
            z[i] = rho * z[i - 1] + np.sqrt(max(1 - rho ** 2, 1e-6)) * rng.standard_normal(dim)
        return z

    z_shared = ar1(4)
    z_room = z_shared if coupled else ar1(4)
    z_agent = z_shared if coupled else ar1(4)
    ang = 2 * np.pi * hours / 24
    c = np.stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)], axis=1) \
        if clocked else np.zeros((N_BINS, 4))

    def project(z, dim, w_rng):
        Wz = w_rng.standard_normal((4, dim)) / 2
        Wc = w_rng.standard_normal((4, dim)) / 2
        return (np.tanh(alpha * z @ Wz + c @ Wc)
                + 0.15 * w_rng.standard_normal((N_BINS, dim)))

    XR = project(z_room, 12, np.random.default_rng(seed + 40))
    XA = project(z_agent, 24, np.random.default_rng(seed + 80))
    return XR, XA, hours


def main() -> int:
    results = {"prereg": "PREREG_r0_instrument_validation_2026_08_05.md", "smoke": SMOKE,
               "n_bins": N_BINS, "seeds": SEEDS, "worlds": {}}
    worlds = {"C": (True, True), "K": (False, True), "N": (False, False)}
    if SMOKE:
        worlds = {"C": (True, True)}
    per = {w: [] for w in worlds}
    for s in SEEDS:
        for w, (coupled, clocked) in worlds.items():
            t0 = time.time()
            XR, XA, hours = make_world(s, coupled, clocked)
            m = measure(XR, XA, hours, seed=s)
            per[w].append(m)
            results["worlds"][f"{w}_s{s}"] = m
            print(f">> world {w} seed {s}: disc={m['disc']:.4f} hm={m['hourmatched_null']:.4f} "
                  f"free={m['free_null']:.4f} [{time.time()-t0:.0f}s]", flush=True)

    if not SMOKE:
        med = lambda w, k: round(float(np.median([m[k] for m in per[w]])), 4)  # noqa: E731
        results["c_disc_minus_hourmatched_null"] = med("C", "disc_minus_hourmatched_null")
        results["k_disc_minus_hourmatched_null"] = med("K", "disc_minus_hourmatched_null")
        results["k_disc_minus_free_null"] = med("K", "disc_minus_free_null")
        results["n_disc_over_chance_ratio"] = med("N", "disc_over_chance_ratio")

        results["power_surface"] = {}
        for alpha in (0.25, 0.5, 1.0):
            XR, XA, hours = make_world(SEEDS[0], True, True, alpha=alpha)
            m = measure(XR, XA, hours, seed=SEEDS[0])
            results["power_surface"][str(alpha)] = {
                "disc_minus_hourmatched_null": m["disc_minus_hourmatched_null"],
                "disc": m["disc"]}
            print(f">> power alpha={alpha}: margin={m['disc_minus_hourmatched_null']:.4f}",
                  flush=True)

    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_r0_instrument_validation_2026_08_05.md").score(
            results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"r0_result{'_smoke' if SMOKE else ''}.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
