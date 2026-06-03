# -*- coding: utf-8 -*-
"""
analyze_ssm_contrast.py — run the validated spectral battery (analyze_concept_dynamics)
on the transformer-vs-SSM concept trajectories, with the per-arch random-direction null
as the key control (does the CONCEPT direction oscillate more than random directions in
the SAME model — controlling for generic state dynamics).

Frozen by PREREG_ssm_contrast_2026_06_03.md.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_concept_dynamics import analyze_series

RAND_SAMPLE = 4   # random columns analyzed per trajectory (null), keep runtime sane


def main():
    raw = json.loads((HERE / "ssm_contrast_raw.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    result = {"per_arch": {}}
    for arch, ad in raw["archs"].items():
        if ad.get("status") != "ok":
            result["per_arch"][arch] = {"status": ad.get("status"), "error": ad.get("error")}
            continue
        osc, rho, slope, rand_osc = [], [], [], []
        for tr in ad["traj"]:
            r = analyze_series(tr["sig"], rng)
            if not r.get("usable"):
                continue
            osc.append(bool(r["surr_pass"] and r["F_pass"]))
            rho.append(r["rho"]); slope.append(r["trend_slope"])
            rand = np.array(tr["rand"])                              # [t, N_RAND]
            for k in range(min(RAND_SAMPLE, rand.shape[1])):
                rr = analyze_series(rand[:, k], rng, n_surr=300)
                if rr.get("usable"):
                    rand_osc.append(bool(rr["surr_pass"] and rr["F_pass"]))
        n = len(osc)
        result["per_arch"][arch] = {
            "status": "ok", "model": ad["model"], "n": n,
            "osc_frac": round(float(np.mean(osc)), 3) if n else None,
            "rand_osc_frac": round(float(np.mean(rand_osc)), 3) if rand_osc else None,
            "mean_ar1_rho": round(float(np.mean(rho)), 3) if rho else None,
            "mean_trend_slope": round(float(np.mean(slope)), 4) if slope else None}

    pa = result["per_arch"]
    t, s = pa.get("transformer", {}), pa.get("ssm", {})
    reading = "INCONCLUSIVE (a model arm failed)"
    if t.get("status") == "ok" and s.get("status") == "ok":
        t_osc = t.get("osc_frac") or 0.0; s_osc = s.get("osc_frac") or 0.0
        s_rand = s.get("rand_osc_frac") or 0.0
        if s_osc >= 0.33 and s_osc >= 2 * max(t_osc, 0.05) and s_osc >= 2 * max(s_rand, 0.05):
            reading = (f"SSM_OSCILLATES — Mamba concept signal shows rhythm (osc {s_osc}) far above "
                       f"transformer ({t_osc}) and its own random null ({s_rand}): a recurrence-driven "
                       f"oscillation feedforward attention cannot produce")
        elif t_osc < 0.2 and s_osc < 0.2:
            reading = (f"BOTH_COMMIT — neither architecture oscillates above noise (transformer {t_osc}, "
                       f"ssm {s_osc}); both commit. Consistent with the architecture: a transformer has no "
                       f"recurrence, and Mamba-1's real-valued scan has no complex eigenvalues for rotation. "
                       f"The complex-SSM (Mamba-3) test is the stronger follow-up.")
        else:
            reading = (f"MIXED — transformer osc {t_osc}, ssm osc {s_osc}, ssm-random {s_rand}; "
                       f"ssm shows more structure than transformer but does not clear the oscillation bar")
    result["reading"] = reading
    (HERE / "ssm_contrast_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(reading)
    print("wrote ssm_contrast_result.json")


if __name__ == "__main__":
    main()
