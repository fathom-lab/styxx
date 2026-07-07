"""Render THE figure: the Honesty-Knowledge coupling curve from attack_sweep_result.json.

x = full-spectrum re-locked held-out AUROC ; y = held-out knowledge accuracy. The EVADABLE corner
(low AUROC, high knowledge) is shaded. Frozen-L18 points shown faded for contrast. Deterministic.

Usage: python papers/read-neq-write/make_curve.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    res = json.loads((HERE / "attack_sweep_result.json").read_text(encoding="utf-8"))
    pts = res["points"]
    ref = res["ref_knowledge"]

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    # EVADABLE corner: relock AUROC <= 0.60 AND knowledge >= 0.75
    ax.add_patch(Rectangle((0.40, 0.75), 0.60 - 0.40, 1.02 - 0.75,
                           facecolor="#c0392b", alpha=0.10, zorder=0))
    ax.text(0.50, 0.985, "EVADABLE\n(read scrubbed, knowledge kept)", ha="center", va="top",
            fontsize=9, color="#c0392b", weight="bold")

    seeds = sorted({p["seed"] for p in pts})
    markers = {seeds[0]: "o", **({seeds[1]: "s"} if len(seeds) > 1 else {})}
    for p in pts:
        m = markers.get(p["seed"], "o")
        # faded frozen-L18 point
        ax.scatter(p["frozen_l18_heldout_auroc"], p["knowledge_acc"], marker=m,
                   s=36, facecolor="none", edgecolor="#95a5a6", alpha=0.6, zorder=2)
        ax.plot([p["frozen_l18_heldout_auroc"], p["relock_auroc"]],
                [p["knowledge_acc"], p["knowledge_acc"]], color="#bdc3c7", lw=0.8, zorder=1)
        # full-spectrum re-lock point (the load-bearing one), colored by lambda
        sc = ax.scatter(p["relock_auroc"], p["knowledge_acc"], marker=m, s=90,
                        c=[p["lambda"]], cmap="viridis", vmin=0, vmax=10,
                        edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(f"λ={p['lambda']:g}", (p["relock_auroc"], p["knowledge_acc"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.axvline(0.60, color="#c0392b", ls="--", lw=1, alpha=0.7)
    ax.axhline(0.75, color="#c0392b", ls="--", lw=1, alpha=0.7)
    ax.axhline(ref, color="#2c3e50", ls=":", lw=1, alpha=0.6)
    ax.text(0.995, ref + 0.005, f"ref knowledge {ref:.2f}", ha="right", fontsize=8, color="#2c3e50")
    ax.axvline(0.50, color="#7f8c8d", ls=":", lw=0.8, alpha=0.5)
    ax.text(0.50, 0.42, "read = chance", rotation=90, fontsize=7, color="#7f8c8d", va="bottom")

    ax.set_xlabel("full-spectrum re-locked honesty read  (held-out AUROC)", fontsize=10)
    ax.set_ylabel("retained knowledge  (held-out True/False accuracy)", fontsize=10)
    ax.set_title(f"Honesty–Knowledge Coupling  —  {res['verdict']}\n"
                 f"{res['model']}  ·  faded = naive frozen-L18 auditor, filled = full-spectrum re-lock",
                 fontsize=10)
    ax.set_xlim(0.40, 1.0); ax.set_ylim(0.40, 1.02)
    cbar = fig.colorbar(sc, ax=ax); cbar.set_label("attack λ (knowledge-replay weight)", fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out = HERE / "coupling_curve.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out.name}  (verdict {res['verdict']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
