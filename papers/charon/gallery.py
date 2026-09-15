#!/usr/bin/env python3
"""gallery.py — the ledger as a gallery.

    python papers/charon/gallery.py papers/charon/charon.log.jsonl papers/plates/charon/

One small plate per Charon line, keyed by the line's `entry_id` (its chain hash), in sequence
order, plus the head plate rendered large. Because every entry_id chains to the one before it, a
single changed line changes its own plate AND every plate after it: tampering with the ledger is
visible as sand moving from the tampered line to the end of the sheet.
"""
import sys, os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from styxx.plate import field, render, PLATE, SAND, SAND_DIM, INK  # noqa: E402


def small_sand(ax, digest_hex: str, res: int = 220, n: int = 40_000):
    X, Y, U, (pairs, amps, fam, rot, seed) = field(digest_hex, res=res)
    rng = np.random.default_rng(seed)
    px = rng.uniform(-1, 1, n); py = rng.uniform(-1, 1, n)
    ix = ((px + 1) / 2 * (res - 1)).astype(int); iy = ((py + 1) / 2 * (res - 1)).astype(int)
    gy, gx = np.gradient(U, 2.0 / (res - 1))
    d = np.abs(U[iy, ix]) / (np.hypot(gx[iy, ix], gy[iy, ix]) + 1e-9)
    keep = rng.uniform(0, 1, n) < np.exp(-(d / 0.02) ** 2)
    c, s = math.cos(rot), math.sin(rot)
    rx, ry = c * px[keep] - s * py[keep], s * px[keep] + c * py[keep]
    inside = (np.abs(rx) <= 1) & (np.abs(ry) <= 1)
    ax.scatter(rx[inside], ry[inside], s=0.5, c=SAND, alpha=0.9, linewidths=0, marker=".")
    ax.set_xlim(-1.02, 1.02); ax.set_ylim(-1.02, 1.02); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor(PLATE)


def main(log_path: str, out_dir: str, ncols: int = 18):
    os.makedirs(out_dir, exist_ok=True)
    lines = [json.loads(l) for l in open(log_path)]
    header, entries = lines[0], lines[1:]
    n = len(entries); nrows = math.ceil(n / ncols)
    dpi = 150
    fig = plt.figure(figsize=(ncols * 1.0, nrows * 1.0 + 0.8), dpi=dpi, facecolor=PLATE)
    for i, e in enumerate(entries):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        small_sand(ax, e["entry_id"])
        ax.text(0.5, -0.04, f"{e['seq']}", transform=ax.transAxes, ha="center", va="top",
                color=INK, family="monospace", size=4)
    head = entries[-1]["entry_id"]
    kinds = {}
    for e in entries:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    title = (f"charon — {n} lines, head {head[:12]} — "
             + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    fig.suptitle(title, color=SAND, family="monospace", size=9, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01, wspace=0.05, hspace=0.30)
    sheet = os.path.join(out_dir, "charon_gallery.png")
    fig.savefig(sheet, dpi=dpi, facecolor=PLATE); plt.close(fig)
    head_png = render(head, os.path.join(out_dir, "charon_head.png"),
                      f"charon head, line {entries[-1]['seq']} of {n}")
    print(sheet); print(head_png); print("head", head)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ".")
