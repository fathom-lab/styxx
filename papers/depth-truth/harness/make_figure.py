"""THE figure (PREREG_v2 §11): confidence-vs-depth scatter, colored by
correctness, per dataset. GPU-free, deterministic, reads results/*.jsonl.

Confidence axis = SE (the §2 primary opponent). Two panels: ID (TriviaQA),
OOD-1 (PopQA-rare). TruthfulQA is omitted (KG3-gated, 242/250 grade_ambiguous).
Writes results/figure_depth_vs_confidence.png.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.normpath(os.path.join(_HERE, "..", "results"))


def _load_cc(fn):
    rows = []
    with open(os.path.join(_RESULTS, fn), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("excluded_flag") is None and isinstance(r.get("correct"), bool):
                rows.append(r)
    return rows


def _panel(ax, rows, title):
    xs_c = [r["depth"] for r in rows if r["correct"]]
    ys_c = [r["SE"] for r in rows if r["correct"]]
    xs_w = [r["depth"] for r in rows if not r["correct"]]
    ys_w = [r["SE"] for r in rows if not r["correct"]]
    ax.scatter(xs_w, ys_w, s=18, c="#c0392b", alpha=0.6, label=f"wrong (n={len(xs_w)})",
               edgecolors="none")
    ax.scatter(xs_c, ys_c, s=18, c="#27ae60", alpha=0.6, label=f"correct (n={len(xs_c)})",
               edgecolors="none")
    ax.set_xlabel("circuit-attribution depth (first content token)")
    ax.set_ylabel("semantic entropy (SE)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.15)


def main():
    id_rows = _load_cc("main_id.jsonl")
    ood1_rows = _load_cc("main_ood1.jsonl")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _panel(axes[0], id_rows,
           f"ID — TriviaQA (n={len(id_rows)})\nH1 AUROC(depth)=0.547 CI[0.474,0.618] NULL")
    _panel(axes[1], ood1_rows,
           f"OOD-1 — PopQA-rare (n={len(ood1_rows)})\nH3 dAUC=-0.052 CI[-0.107,-0.012] (anti)")
    fig.suptitle(
        "depth-truth keystone: circuit-depth does NOT separate correct from wrong "
        "(gemma-2-2b, PREREG_v2) — CLOSED_NEGATIVE",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(_RESULTS, "figure_depth_vs_confidence.png")
    fig.savefig(out, dpi=130)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
