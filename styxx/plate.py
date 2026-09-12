#!/usr/bin/env python3
"""styxx.plate — a Chladni figure for a hash.

    python -m styxx.plate <64-hex sha256> [out.png] [--label "text"]

Deterministic: the same digest always produces the same figure; one changed
byte anywhere in the digest produces a different one. The figure is the
zero set (nodal lines) of a superposition of square-plate standing-wave
modes, which is the same mathematics Chladni drew sand on in 1787. It is a
picture of the number and nothing else: it claims nothing about the
document the number came from.

Where the digest bytes go:
the digest is re-hashed (sha256 of its bytes) so every character reaches
every parameter; of the re-hash:
  bytes  0..7   -> four (m, n) mode pairs, each in 1..8
  bytes  8..11  -> four signed amplitudes in [-1, 1]
  bytes 12..15  -> four mode families (cos / sin) and a plate rotation
  bytes 16..23  -> grain seed for the sand scatter
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

PLATE = "#0b0b0d"
SAND = "#e8d8b0"
SAND_DIM = "#b8a67e"
INK = "#6f6a5e"


def _modes(digest_hex: str):
    try:
        raw = bytes.fromhex(digest_hex)
    except ValueError:
        raw = b""
    if len(raw) != 32:
        raise SystemExit("need a 64-hex sha256 digest")
    # re-hash so that EVERY byte of the digest reaches every parameter:
    # change one hex character anywhere and the whole figure changes
    b = hashlib.sha256(raw).digest()
    pairs = []
    for k in range(4):
        m = 1 + (b[2 * k] % 8)
        n = 1 + (b[2 * k + 1] % 8)
        if m == n:                      # m == n gives a degenerate (empty) figure
            n = 1 + ((n) % 8)
        pairs.append((m, n))
    amps = [((b[8 + k] / 255.0) * 2.0 - 1.0) for k in range(4)]
    amps = [a if abs(a) > 0.15 else (0.15 if a >= 0 else -0.15) for a in amps]
    fam = [(b[12 + k] & 1) for k in range(4)]          # 0: cos family, 1: sin family
    rot = (b[15] / 255.0) * math.pi / 2                 # plate rotation, cosmetic
    seed = int.from_bytes(b[16:24], "big")
    return pairs, amps, fam, rot, seed


def field(digest_hex: str, res: int = 900):
    pairs, amps, fam, rot, seed = _modes(digest_hex)
    x = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(x, x)
    U = np.zeros_like(X)
    for (m, n), a, f in zip(pairs, amps, fam):
        if f == 0:
            U += a * (np.cos(m * np.pi * X) * np.cos(n * np.pi * Y)
                      - np.cos(n * np.pi * X) * np.cos(m * np.pi * Y))
        else:
            U += a * (np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
                      - np.sin(n * np.pi * X) * np.sin(m * np.pi * Y))
    return X, Y, U, (pairs, amps, fam, rot, seed)


def render(digest_hex: str, out: str, label: str | None = None, size: int = 1400):
    plt = _plt()
    digest_hex = digest_hex.strip().lower()
    X, Y, U, (pairs, amps, fam, rot, seed) = field(digest_hex)
    rng = np.random.default_rng(seed)

    dpi = 200
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor=PLATE)
    ax = fig.add_axes([0.06, 0.10, 0.88, 0.88])
    ax.set_facecolor(PLATE)
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal"); ax.axis("off")

    # sand: sample points and keep the ones that sit near a node (|U| small),
    # densest right on the line — how real sand actually settles on a plate
    N = 260_000
    px = rng.uniform(-1, 1, N); py = rng.uniform(-1, 1, N)
    ix = ((px + 1) / 2 * (U.shape[1] - 1)).astype(int)
    iy = ((py + 1) / 2 * (U.shape[0] - 1)).astype(int)
    # distance to the nodal line, first order: |U| / |grad U| (plate units), so the
    # sand band has the same width whether the field crosses zero steeply or gently
    gy, gx = np.gradient(U, 2.0 / (U.shape[0] - 1))
    d = np.abs(U[iy, ix]) / (np.hypot(gx[iy, ix], gy[iy, ix]) + 1e-9)
    keep = rng.uniform(0, 1, N) < np.exp(-(d / 0.012) ** 2)
    c, s = math.cos(rot), math.sin(rot)
    qx, qy = px[keep], py[keep]
    rx, ry = c * qx - s * qy, s * qx + c * qy
    inside = (np.abs(rx) <= 1) & (np.abs(ry) <= 1)
    ax.scatter(rx[inside], ry[inside], s=0.35, c=SAND, alpha=0.9, linewidths=0, marker=".")
    # a faint second layer of coarser grains for texture
    keep2 = rng.uniform(0, 1, N) < 0.35 * np.exp(-(d / 0.005) ** 2)
    qx, qy = px[keep2], py[keep2]
    rx, ry = c * qx - s * qy, s * qx + c * qy
    inside = (np.abs(rx) <= 1) & (np.abs(ry) <= 1)
    ax.scatter(rx[inside], ry[inside], s=1.1, c=SAND_DIM, alpha=0.8, linewidths=0, marker=".")

    # the plate edge
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=INK, lw=0.6, alpha=0.6)

    short = digest_hex[:12]
    fig.text(0.5, 0.045, short, ha="center", va="center", color=SAND, family="monospace", size=11)
    if label:
        fig.text(0.5, 0.018, label, ha="center", va="center", color=INK, family="monospace", size=7)
    fig.savefig(out, dpi=dpi, facecolor=PLATE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    label = None
    if "--label" in sys.argv:
        label = sys.argv[sys.argv.index("--label") + 1]
        args = [a for a in args if a != label]
    if not args:
        raise SystemExit(__doc__)
    digest = args[0]
    out = args[1] if len(args) > 1 else f"plate_{digest[:12]}.png"
    print(render(digest, out, label))
