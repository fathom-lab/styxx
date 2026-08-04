"""B38 — the legibility cliff, per PREREG_b38_legibility_cliff_2026_08_04.md.

Noise-dose the target's concept points, watch RSA fall, measure label-free discovery at each
dose; interpolate discovery at qwen's RSA level. Two curves: llama_3b->gemma (decisive),
llama_3b->llama_1b (control). CPU-from-cache. `--smoke` = 3 doses at 40/10, INVALID-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from run_g0clear import CONCEPTS as C          # noqa: E402
import run_disjoint_worlds as R                 # noqa: E402
from styxx_transfer import TransferMap          # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 343
QWEN_RSA = 0.881
DOSES = [0, 0.1, 0.4] if SMOKE else [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]


def load(fname):
    z = np.load(HERE / fname, allow_pickle=True)
    return np.asarray(z["pts"])          # already in CONCEPTS order


def main():
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]
    XA_full = load("_b31v2_ptsA.npz")
    targets = {"gemma": load("_b31v2_pts_gemma_2b.npz"),
               "llama1b": load("_b31v2_pts_llama_1b.npz")}

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(C))
    n_fin = 10 if SMOKE else 70
    n_tr = 40 if SMOKE else (len(C) - n_fin)
    tr_i = idx[n_fin:n_fin + n_tr]
    XA = XA_full[tr_i]
    iu = np.triu_indices(len(tr_i), 1)

    results = {"prereg": "PREREG_b38_legibility_cliff_2026_08_04.md", "seed": SEED,
               "smoke": SMOKE, "doses": DOSES, "qwen_rsa_target": QWEN_RSA, "curves": {}}
    for tname, XT_full in targets.items():
        XT = XT_full[tr_i]
        sigma_unit = float(np.mean((XT - XT.mean(0)).std(0)))
        curve = []
        for di, f in enumerate(DOSES):
            t0 = time.time()
            nz = np.random.default_rng(3801 + di).standard_normal(XT.shape) * (f * sigma_unit)
            XTn = XT + nz
            rsa = float(np.corrcoef(R.distmat(XA)[iu], R.distmat(XTn)[iu])[0, 1])
            perm = rng.permutation(len(XA)); XTs = XTn[perm]; true_col = np.argsort(perm)
            tm = TransferMap.fit(XA, XTs, k=kstar)
            MA = np.stack([tm.transfer_point(x) for x in XA])
            _, col = linear_sum_assignment(np.linalg.norm(MA[:, None, :] - XTs[None, :, :], axis=-1))
            disc = float((col == true_col).mean())
            curve.append({"dose": f, "rsa": round(rsa, 4), "disc": round(disc, 4)})
            print(f">> {tname} dose {f}: rsa={rsa:.4f} disc={disc:.4f} [{time.time()-t0:.0f}s]",
                  flush=True)
        results["curves"][tname] = curve

    g = results["curves"]["gemma"]
    results["gemma_dose0_disc"] = g[0]["disc"]
    results["spearman_disc_vs_dose"] = round(float(
        spearmanr([r["dose"] for r in g], [r["disc"] for r in g]).statistic), 4)
    # interpolate disc at qwen's RSA on the gemma curve
    pts = sorted(g, key=lambda r: r["rsa"])
    dq, note = None, "interpolated"
    for lo, hi in zip(pts[:-1], pts[1:]):
        if lo["rsa"] <= QWEN_RSA <= hi["rsa"]:
            w = (QWEN_RSA - lo["rsa"]) / max(hi["rsa"] - lo["rsa"], 1e-9)
            dq = lo["disc"] + w * (hi["disc"] - lo["disc"])
            break
    if dq is None:
        nearest = min(g, key=lambda r: abs(r["rsa"] - QWEN_RSA))
        dq, note = nearest["disc"], f"nearest_dose_substituted (rsa {nearest['rsa']})"
    results["disc_at_qwen_rsa"] = round(float(dq), 4)
    results["disc_at_qwen_rsa_note"] = note
    # reported, not gated: transition width in RSA units (80% -> 20% of baseline)
    base = results["gemma_dose0_disc"]
    hi_t, lo_t = 0.8 * base, 0.2 * base
    r_hi = max((r["rsa"] for r in g if r["disc"] <= hi_t), default=None)
    r_lo = max((r["rsa"] for r in g if r["disc"] <= lo_t), default=None)
    results["transition_width_rsa"] = (round(abs(r_hi - r_lo), 4)
                                       if r_hi is not None and r_lo is not None else None)
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b38_legibility_cliff_2026_08_04.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b38_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                   encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    print(f"disc@qwenRSA={results['disc_at_qwen_rsa']} ({note}) | "
          f"width={results['transition_width_rsa']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
