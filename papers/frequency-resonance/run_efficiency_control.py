# -*- coding: utf-8 -*-
"""run_efficiency_control.py -- frozen by PREREG_efficiency_control_2026_09_02.

THE CONTROL THE ARC OWED ITSELF. RESULT_entrain_rich named "a param-matched wider static bank" as the
honest next control for the adaptive-frequency advantage. This runner re-uses run_entrain_rich.py's
task, models, training loop, evaluation and red-team checks verbatim (imported, not copied), and adds
one arm: STATIC at the smallest width whose parameter count is at least RICH's. Nothing else changes.

  python run_efficiency_control.py [--smoke]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_entrain_rich as R                      # noqa: E402  (reads --smoke from sys.argv)

ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from styxx.protocol import Experiment             # noqa: E402

PREREG = HERE / "PREREG_efficiency_control_2026_09_02.md"
ANCHORS = {"static": 0.4604, "rich": 0.5451}      # entrain_rich_result.json, D=8, drift


def matched_static_width(d_rich: int) -> int:
    target = R.nparams("rich", d_rich)
    d = d_rich
    while R.nparams("static", d) < target:
        d += 1
    return d


def main() -> int:
    smoke = R.SMOKE
    print(f"device={R.DEV} smoke={smoke} steps={R.STEPS} seeds={R.SEEDS} L={R.L} periods=[{R.PMIN},{R.PMAX}]", flush=True)
    R.redteam()
    widths = {8: {"static": 8, "rich": 8, "oracle": 8, "static_matched": matched_static_width(8)}}
    if not smoke:
        widths[4] = {"static": 4, "rich": 4, "static_matched": matched_static_width(4)}
    res = {"prereg": PREREG.name, "config": {"steps": R.STEPS, "seeds": R.SEEDS, "L": R.L,
                                              "periods": [R.PMIN, R.PMAX], "device": R.DEV,
                                              "torch": torch.__version__, "smoke": smoke},
           "widths": {}, "params": {}, "drift": {}, "fixed": {}, "per_seed": {}}
    for d, arms in widths.items():
        key = str(d)
        res["widths"][key] = arms
        res["params"][key] = {a: R.nparams("static" if a == "static_matched" else a, w) for a, w in arms.items()}
        res["drift"][key], res["fixed"][key], res["per_seed"][key] = {}, {}, {}
        for arm, w in arms.items():
            arch = "static" if arm == "static_matched" else arm
            dacc, facc = [], []
            for s in R.SEEDS:
                m = R.train(arch, w, s)
                dacc.append(R.evaluate(m, drift=True))
                facc.append(R.evaluate(m, drift=False))
                print(f"  D={d} {arm:15s} (width {w:2d}, params {res['params'][key][arm]:5d}) seed {s}: "
                      f"drift {dacc[-1]:.4f} fixed {facc[-1]:.4f}", flush=True)
            res["drift"][key][arm] = float(np.mean(dacc))
            res["fixed"][key][arm] = float(np.mean(facc))
            res["per_seed"][key][arm] = {"drift": dacc, "fixed": facc}
    g = res["drift"]["8"]
    metrics = {
        "anchor_max_abs_dev": round(max(abs(g["static"] - ANCHORS["static"]), abs(g["rich"] - ANCHORS["rich"])), 4),
        "oracle_minus_static_8": round(g["oracle"] - g["static"], 4),
        "static_matched_minus_rich_8": round(g["static_matched"] - g["rich"], 4),
        "rich_minus_static_matched_8": round(g["rich"] - g["static_matched"], 4),
    }
    if "4" in res["drift"]:
        g4 = res["drift"]["4"]
        metrics["reported_static_matched_minus_rich_4"] = round(g4["static_matched"] - g4["rich"], 4)
    res["metrics"] = metrics
    verdict = Experiment(PREREG, repo_root=ROOT).score(metrics, smoke=smoke)
    res["verdict"] = verdict.verdict
    res["gates"] = verdict.gates
    out = HERE / ("efficiency_control_smoke.json" if smoke else "efficiency_control_result.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n  metrics:", json.dumps(metrics))
    print(f"\n===== VERDICT: {res['verdict']} =====\nwrote {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
