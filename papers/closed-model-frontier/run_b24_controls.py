"""B24 controls addendum — persist the red-team-mandated controls to disk.

The adversarial verification (wf_96778321-0b8) established that the pre-registered token-pair GroupKFold
firewall is VACUOUS on this data (108 unique (xid,yid) pairs -> all singleton groups -> item-level CV in
disguise), and that the load-bearing invariance evidence is the NON-vacuous first-CHARACTER firewall plus a
selection-corrected max-permutation null — neither of which was saved to disk. This script computes and
persists them, read-only over the frozen residuals_b24.npz. No verdict logic here; the FINDING carries the
interpretation (REPORT_AS_LANDED).
"""
from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_b24_headtohead import cv_auc  # noqa: E402  (same estimator family)
from run_behavioral_sycophancy import norm  # noqa: E402

N_PERM_MAXNULL = 120
SEED = 0


def main() -> int:
    npz = HERE / "residuals_b24.npz"
    d = np.load(npz, allow_pickle=True)
    A = d["A"].astype(np.float32)
    B = d["B"].astype(np.float32)
    meta = json.loads(str(d["meta"]))
    y = np.array([m["y"] for m in meta])
    L = A.shape[1]
    committed = [m["X"] if m["y"] == 1 else m["Y"] for m in meta]
    g_char = np.array([norm(c)[:1] for c in committed])
    g_len = np.array([str(min(len(norm(c).split()[0]) if norm(c) else 0, 6)) for c in committed])
    lens_only = np.array([len(norm(c)) for c in committed], dtype=float)
    tokpair = np.array([f"{m['xid']}_{m['yid']}" for m in meta])

    out = {
        "addendum_for": "b24_headtohead_result.json",
        "residuals_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
        "n": int(len(y)), "n_held": int(y.sum()), "n_caved": int((1 - y).sum()),
        "tokenpair_groups_are_singletons": bool(len(set(tokpair)) == len(y)),
        "n_first_char_groups": int(len(set(g_char))),
        "length_only_auroc": float(roc_auc_score(y, lens_only)),
    }

    cells = []
    for pos, HSp in (("A", A), ("B", B)):
        for lyr in range(L):
            X = HSp[:, lyr, :]
            a_char, _ = cv_auc(X, y, GroupKFold(5), groups=g_char)
            a_len, _ = cv_auc(X, y, GroupKFold(5), groups=g_len)
            cells.append([pos, lyr, round(a_char, 4), round(a_len, 4)])
    out["charfw_ramp_pos_layer_char_len"] = cells
    best_char = max(cells, key=lambda c: (c[2] if not np.isnan(c[2]) else -1))
    best_len = max(cells, key=lambda c: (c[3] if not np.isnan(c[3]) else -1))
    out["first_char_firewall_best"] = {"cell": best_char[:2], "auroc": best_char[2]}
    out["length_bucket_firewall_best"] = {"cell": best_len[:2], "auroc": best_len[3]}
    print("char-firewall best:", best_char, "| len-firewall best:", best_len, flush=True)

    # selection-corrected max-permutation null: shuffle labels, take the MAX grouped(tokpair) AUROC over
    # ALL 74 cells per shuffle — the honest null for a best-of-74 headline.
    rng = np.random.RandomState(SEED)
    maxnull = []
    for p in range(N_PERM_MAXNULL):
        yp = rng.permutation(y)
        mx = -1.0
        for pos, HSp in (("A", A), ("B", B)):
            for lyr in range(L):
                a, _ = cv_auc(HSp[:, lyr, :], yp, GroupKFold(5), groups=tokpair)
                if not np.isnan(a) and a > mx:
                    mx = a
        maxnull.append(mx)
        if (p + 1) % 20 == 0:
            print(f"maxnull perm {p+1}/{N_PERM_MAXNULL} running p95={np.percentile(maxnull,95):.3f}", flush=True)
    out["selection_corrected_maxnull"] = {
        "n_perm": N_PERM_MAXNULL, "p95": float(np.percentile(maxnull, 95)),
        "p99": float(np.percentile(maxnull, 99)), "max": float(np.max(maxnull)),
        "observed_WB_best": 0.9406,
        "n_perm_ge_observed": int(sum(1 for v in maxnull if v >= 0.9406)),
    }

    (HERE / "b24_controls_addendum.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "charfw_ramp_pos_layer_char_len"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
