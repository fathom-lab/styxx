#!/usr/bin/env python3
"""styxx.geoplate — a Chladni figure for a representation, not a hash.

styxx.plate maps a HASH to a figure and is maximally sensitive: one changed
character scrambles everything (that is what an identity check wants).
geoplate.py maps a MATRIX to a figure and is continuous: two matrices that
agree produce sand that settles on nearly the same lines, and the more they
disagree the more the sand moves (that is what a comparison wants).

Input: a symmetric representational dissimilarity matrix R (n x n) over a
fixed, shared item order. Method, fixed and published here so nobody can tune
it after seeing the pictures:

  1. fill the (always-zero) diagonal with the off-diagonal mean, then center R;
  2. take the 2-D DCT-II of R and keep the low-frequency block i, j < K
     (K = 8), dropping the DC term;
  3. normalise the block to unit Frobenius norm;
  4. the plate field is U(x, y) = sum_ij W_ij cos(i pi x) cos(j pi y) on the
     unit square — the same cosine modes a square plate vibrates in;
  5. sand settles on the nodal set U = 0.

The figure is a VIEW of the matrix, not a measurement of it. The number that
matters is printed with it: the Pearson correlation of the two matrices'
upper triangles. The plate cannot make two geometries agree more than that
number says they do; a permuted-item control (same matrix, items shuffled)
shows what disagreement looks like.
"""
import sys, math, hashlib
import numpy as np


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:  # pragma: no cover
        raise SystemExit("plate rendering needs matplotlib: pip install 'styxx[plate]'")


def _dctn(R):
    try:
        from scipy.fft import dctn
    except ImportError:  # pragma: no cover
        raise SystemExit("geoplate needs scipy: pip install 'styxx[plate]'")
    return dctn(R, type=2, norm="ortho")

K = 8
PLATE = "#0b0b0d"; SAND = "#e8d8b0"; SAND_DIM = "#b8a67e"; INK = "#6f6a5e"


def coefficients(R: np.ndarray, k: int = K) -> np.ndarray:
    R = np.array(R, dtype=np.float64, copy=True)
    # the zero diagonal is a property of every RDM, not of this one: fill it
    # with the off-diagonal mean so it cannot make unrelated matrices look alike
    off = ~np.eye(R.shape[0], dtype=bool)
    R[~off] = R[off].mean()
    R = R - R.mean()
    W = _dctn(R)[:k, :k].copy()
    W[0, 0] = 0.0
    W /= (np.linalg.norm(W) + 1e-12)
    return W


def field(W: np.ndarray, res: int = 900):
    x = np.linspace(0, 1, res)
    X, Y = np.meshgrid(x, x)
    U = np.zeros_like(X)
    k = W.shape[0]
    for i in range(k):
        ci = np.cos(i * np.pi * X)
        for j in range(k):
            if W[i, j] == 0.0:
                continue
            U += W[i, j] * ci * np.cos(j * np.pi * Y)
    return U


def sand(ax, U, seed: int):
    rng = np.random.default_rng(seed)
    N = 260_000
    px = rng.uniform(0, 1, N); py = rng.uniform(0, 1, N)
    ix = (px * (U.shape[1] - 1)).astype(int); iy = (py * (U.shape[0] - 1)).astype(int)
    # distance to the nodal line, first order: |U| / |grad U| (plate units), so the
    # sand band has the same width whether the field crosses zero steeply or gently
    gy, gx = np.gradient(U, 1.0 / (U.shape[0] - 1))
    d = np.abs(U[iy, ix]) / (np.hypot(gx[iy, ix], gy[iy, ix]) + 1e-9)
    keep = rng.uniform(0, 1, N) < np.exp(-(d / 0.010) ** 2)
    ax.scatter(px[keep], py[keep], s=0.35, c=SAND, alpha=0.9, linewidths=0, marker=".")
    keep2 = rng.uniform(0, 1, N) < 0.35 * np.exp(-(d / 0.004) ** 2)
    ax.scatter(px[keep2], py[keep2], s=1.1, c=SAND_DIM, alpha=0.8, linewidths=0, marker=".")
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=INK, lw=0.6, alpha=0.6)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor(PLATE)


def render_grid(items, out: str, title: str | None = None, ncols: int = 3, size: int = 1500):
    """items: list of (label, sublabel, W). Renders a grid of plates."""
    plt = _plt()
    n = len(items); nrows = math.ceil(n / ncols)
    dpi = 200
    fig = plt.figure(figsize=(size / dpi, size / dpi * nrows / ncols + 0.55), dpi=dpi, facecolor=PLATE)
    for idx, (label, sub, W) in enumerate(items):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        U = field(W, res=700)
        seed = int.from_bytes(hashlib.sha256(W.tobytes()).digest()[:8], "big")
        sand(ax, U, seed)
        ax.set_title(label, color=SAND, family="monospace", size=8, pad=4)
        ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha="center", va="top", color=INK,
                family="monospace", size=6)
    if title:
        fig.suptitle(title, color=SAND, family="monospace", size=8, y=0.995)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.05, wspace=0.08, hspace=0.25)
    fig.savefig(out, dpi=dpi, facecolor=PLATE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    raise SystemExit("import styxx.geoplate and call render_grid([...]) — see papers/disjoint-worlds/geometry_plates_demo.py")
