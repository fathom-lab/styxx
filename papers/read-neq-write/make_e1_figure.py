"""THE E1 figure: same attacked weights, two auditors.

For the decisive family (Qwen-1.5B), at each knowledge-preserving attack point, compare the stage-2
auditor (probe calibrated on the ATTACK set the attacker optimized against -> reads chance) against
the E1 auditor (probe calibrated on a private clean set the attacker never saw -> reads honesty).
The gap between the two bars IS the calibration-poisoning effect.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    s2 = json.loads((HERE / "attack_sweep_result.json").read_text(encoding="utf-8"))
    e1 = json.loads((HERE / "e1_result.json").read_text(encoding="utf-8"))
    fam = "Qwen/Qwen2.5-1.5B-Instruct"
    # stage-2 relock (ATTACK-calibrated) keyed by (seed, lambda)
    s2m = {(p["seed"], p["lambda"]): p["relock_auroc"] for p in s2["points"]}
    e1p = [p for p in e1["points"] if p["family"] == fam and p["eval_knowledge"] >= 0.75]
    e1p.sort(key=lambda p: (p["seed"], p["lambda"]))
    labels = [f"seed {p['seed']}\nλ={p['lambda']:g}" for p in e1p]
    attack_cal = [s2m.get((p["seed"], p["lambda"]), np.nan) for p in e1p]
    clean_cal = [p["best_clean_calib_auroc"] for p in e1p]
    know = [p["eval_knowledge"] for p in e1p]

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.axhspan(0.40, 0.60, color="#7f8c8d", alpha=0.10, zorder=0)
    ax.text(len(labels) - 0.5, 0.50, "chance band", ha="right", va="center", fontsize=8, color="#7f8c8d")
    b1 = ax.bar(x - w / 2, attack_cal, w, label="auditor calibrated on the ATTACK set (stage 2)",
                color="#c0392b", edgecolor="k", linewidth=0.5)
    b2 = ax.bar(x + w / 2, clean_cal, w, label="auditor calibrated on a PRIVATE clean set (E1)",
                color="#2e86de", edgecolor="k", linewidth=0.5)
    ax.plot(x, know, "o--", color="#27ae60", lw=1.4, ms=7, label="retained knowledge (held-out)")
    ax.axhline(0.70, color="#2e86de", ls=":", lw=1, alpha=0.7)
    ax.axhline(0.60, color="#c0392b", ls=":", lw=1, alpha=0.7)
    for r in list(b1) + list(b2):
        ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 1), textcoords="offset points")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("honesty read on the attacked weights (EVAL AUROC)", fontsize=10)
    ax.set_ylim(0.40, 1.0)
    ax.set_title("read≠write STANDS — the evasion was the auditor's, not the model's\n"
                 "Qwen2.5-1.5B · same knowledge-preserving attack · the only difference is the auditor's calibration data",
                 fontsize=10.5)
    ax.legend(loc="upper center", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    out = HERE / "e1_two_auditors.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
