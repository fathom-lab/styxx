"""Project real measured concept geometry to 3D for the showcase constellation.

Loads normeq_reps.npz (the norm-equalized geometry from the anatomy-v2 run), takes one mind's
192-concept representation, PCA-projects to 3D, and exports real coordinates + categories + real
cosine-nearest edges to geometry_render.json. This is the data the live signature visualization
renders — actual styxx measurements, not hand-placed particles.

Usage: python papers/showcase-viz/project_geometry.py [MIND]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "papers" / "mind-instrument"))

from styxx import mind  # noqa: E402
from run_telepathy_v0 import PROBES, PROBE_CAT  # noqa: E402

NPZ = REPO / "papers" / "mind-instrument" / "normeq_reps.npz"
WORDS = mind.BATTERY + PROBES
CATS = mind.BATTERY_CATEGORY + PROBE_CAT


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "gemma-2-2b"
    z = np.load(NPZ)
    if which not in z:
        print("minds:", list(z.keys())); return 2
    R = z[which].astype(float)                      # (192, d)
    n = R.shape[0]

    # PCA to 3D (centered SVD)
    X = R - R.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = (X @ Vt[:3].T)                          # (192, 3)
    coords = coords / (np.abs(coords).max(0) + 1e-9)  # normalize each axis to [-1,1]
    var = (S[:3] ** 2 / (S ** 2).sum()).round(4).tolist()

    # real cosine-nearest edges (k=2 per node, dedup)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    sim = Rn @ Rn.T
    np.fill_diagonal(sim, -2)
    edges = set()
    for i in range(n):
        for j in np.argsort(-sim[i])[:2]:
            edges.add((min(i, int(j)), max(i, int(j))))

    out = {
        "source": "normeq_reps.npz (anatomy-v2 norm-equalized geometry)",
        "mind": which, "n_concepts": n, "pca_variance_explained_xyz": var,
        "categories": list(dict.fromkeys(CATS)),
        "nodes": [{"w": WORDS[i], "cat": CATS[i],
                   "x": round(float(coords[i, 0]), 4),
                   "y": round(float(coords[i, 1]), 4),
                   "z": round(float(coords[i, 2]), 4)} for i in range(n)],
        "edges": sorted(edges),
    }
    (HERE / "geometry_render.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"{which}: {n} concepts -> 3D PCA (var xyz={var}), {len(edges)} edges -> geometry_render.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
