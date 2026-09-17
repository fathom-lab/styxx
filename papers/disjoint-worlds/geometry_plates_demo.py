#!/usr/bin/env python3
"""geometry_plates_demo.py — rebuild the four-model geometry plate from the committed banks.

    git clone https://github.com/fathom-lab/styxx && cd styxx/papers/disjoint-worlds
    pip install -e '.[plate]'      # from the repository root
    python geometry_plates_demo.py          # writes geometry_plates_four_models.png + _agreement.json

No GPU, no downloads, no torch: the banks (`_b31v2_pts_*.npz`, 462 concept points per model) are
in the repository. Runs in well under a minute on a laptop.

What it does, exactly:
  1. loads each model's 462 x d bank, centres it, unit-normalises each row;
  2. RDM = 1 - X X^T  (correlation distance between every pair of concepts), one per model;
  3. prints the Pearson correlation of every pair of RDM upper triangles (the measurement);
  4. builds one control: qwen's own RDM with its 462 items shuffled (seed 343);
  5. renders every RDM as a geometry plate with `geoplate.py` (the view) into one image.

The picture cannot make two models agree more than the printed r says they do.
"""
import json, os, sys, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# the styxx that renders is the one in this checkout, not whatever pip installed elsewhere: run by
# path, sys.path[0] is this directory and `styxx` would resolve to a site-packages copy that may
# predate geoplate (it did, on the lab's own second machine, 2026-09-13)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from styxx.geoplate import coefficients, render_grid, coefficients_sha256  # noqa: E402

BANKS = {
    "llama_3b": "_b31v2_ptsA.npz",
    "llama_1b": "_b31v2_pts_llama_1b.npz",
    "gemma_2b": "_b31v2_pts_gemma_2b.npz",
    "qwen_1p5b": "_b31v2_pts_qwen_1p5b.npz",
}
LABELS = {
    "llama_3b": "llama-3.2-3b (meta)",
    "llama_1b": "llama-3.2-1b (meta)",
    "gemma_2b": "gemma-2-2b (google)",
    "qwen_1p5b": "qwen-2.5-1.5b (alibaba)",
}
CONTROL = "control (qwen, items shuffled)"


def rdm(path: str) -> np.ndarray:
    X = np.load(path)["pts"].astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return 1.0 - X @ X.T


def main() -> None:
    R = {k: rdm(os.path.join(HERE, f)) for k, f in BANKS.items()}
    n = next(iter(R.values())).shape[0]
    iu = np.triu_indices(n, 1)
    perm = np.random.default_rng(343).permutation(n)
    R[CONTROL] = R["qwen_1p5b"][np.ix_(perm, perm)]
    names = list(BANKS) + [CONTROL]

    agree = {a: {b: float(np.corrcoef(R[a][iu], R[b][iu])[0, 1]) for b in names} for a in names}
    print(f"{n} concepts; RDM agreement (pearson, upper triangles):")
    for a, b in itertools.combinations(names, 2):
        print(f"  {a:32s} vs {b:32s} r = {agree[a][b]:.3f}")

    items = []
    for k in BANKS:
        others = [o for o in BANKS if o != k]
        sub = "rdm r vs others: " + "  ".join(f"{agree[k][o]:.2f}" for o in others)
        items.append((LABELS[k], sub, coefficients(R[k])))
    items.append(("control: qwen, 462 items shuffled",
                  "rdm r vs the four: " + "  ".join(f"{agree[CONTROL][o]:.2f}" for o in BANKS),
                  coefficients(R[CONTROL])))
    out = render_grid(items, os.path.join(HERE, "geometry_plates_four_models.png"),
                      title="four models, three companies, one concept geometry — "
                            f"{n} concepts, committed banks, disjoint-worlds arc", ncols=3)
    # the cross-machine reproduction target: png bytes depend on the plotting library, these do not
    agree["coefficients_sha256"] = {k: coefficients_sha256(coefficients(R[k])) for k in names}
    json.dump(agree, open(os.path.join(HERE, "geometry_plates_agreement.json"), "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
