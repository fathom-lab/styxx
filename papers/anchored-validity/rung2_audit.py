"""STAGE B rung 2: audit the CLAUDE persona panel (arm's-length subagent judges).

Usage: python rung2_audit.py logician=<file> casual=<file> analyst=<file> checker=<file>
Each <file> is a judge-session transcript; verdict lines are extracted by strict pattern,
deduped by first occurrence, and written verbatim to rung2_verdicts_<persona>.txt receipts.
The audit consumes the matrices label-free; the held-out constructed truth then scores it.
Demonstration-grade per the prereg: ONE draw, conflict disclosed. ASCII only.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
from styxx.anchors import audit_panel

PAT = re.compile(r"\b((?:org|neg|pos)_7001_\d{4})\s+(YES|NO)\b")


def extract(path):
    """Pull verdict lines from a transcript file (json-lines tolerant): walk every string in
    every json object; fall back to raw-text regex. First occurrence per id wins."""
    verdicts = {}
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    texts = []
    for line in raw.splitlines():
        try:
            obj = json.loads(line)
        except Exception:
            texts.append(line); continue

        def walk(o):
            if isinstance(o, str):
                texts.append(o)
            elif isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(obj)
    for t in texts:
        for m in PAT.finditer(t):
            verdicts.setdefault(m.group(1), 1 if m.group(2) == "YES" else 0)
    return verdicts


def main():
    files = dict(a.split("=", 1) for a in sys.argv[1:])
    assert set(files) == {"logician", "casual", "analyst", "checker"}, sorted(files)
    items = json.loads((HERE / "rung2_items_7001.json").read_text(encoding="utf-8"))
    held = json.loads((HERE / "rung2_truth_7001_HELDOUT.json").read_text(encoding="utf-8"))
    truth, roles, org_ids = held["truth"], held["roles"], held["organic_ids"]
    ids = [it["id"] for it in items]

    personas = ["logician", "casual", "analyst", "checker"]
    cols, coverage_notes = {}, {}
    for p in personas:
        v = extract(files[p])
        missing = [i for i in ids if i not in v]
        cols[p] = v
        coverage_notes[p] = {"parsed": len(v), "missing": len(missing),
                             "missing_ids": missing[:10]}
        (HERE / f"rung2_verdicts_{p}.txt").write_text(
            "\n".join(f"{i} {'YES' if v[i] else 'NO'}" for i in ids if i in v) + "\n",
            encoding="utf-8")
    # items with any missing verdict are DROPPED from every judge (panel must be rectangular);
    # the drop count is reported -- a judge that skipped items shrinks n for everyone
    keep_ids = [i for i in ids if all(i in cols[p] for p in personas)]
    org = [i for i in keep_ids if i in set(org_ids)]
    negs = [i for i in keep_ids if roles.get(i) == "neg_anchor"]
    poss = [i for i in keep_ids if roles.get(i) == "pos_anchor"]
    V = np.array([[cols[p][i] for p in personas] for i in org])
    Vn = np.array([[cols[p][i] for p in personas] for i in negs])
    Vp = np.array([[cols[p][i] for p in personas] for i in poss])
    y = np.array([truth[i] for i in org])
    pi_true = float(y.mean())

    audit = audit_panel(V, Vn, Vp, n_boot=300, null_sims=200, seed=7001)
    org_alpha = V[y == 0].mean(0); org_beta = V[y == 1].mean(0)
    mv = float((V.mean(1) > 0.5).mean())
    out = {"prereg": "PREREG_STAGE_B_rung1_2026_07_20.md (rung 2 section)",
           "panel": "four fresh-context Claude persona subagents, arm's-length",
           "n_organic": len(org), "k_neg": len(negs), "k_pos": len(poss),
           "dropped_items": len(ids) - len(keep_ids), "parse": coverage_notes,
           "pi_true_realized": pi_true,
           "audit": {k: audit.get(k) for k in ("verdict", "pi", "ci", "regime", "s", "s_ci",
                                               "activated", "misfit", "kept", "alpha", "beta",
                                               "coverage_note")},
           "audit_covered": (bool(audit["ci"][0] <= pi_true <= audit["ci"][1])
                             if audit.get("verdict") == "ESTIMATED" else None),
           "audit_err": (abs(audit["pi"] - pi_true) if audit.get("pi") is not None else None),
           "mv_pi": mv, "mv_err": abs(mv - pi_true),
           "anchor_alpha": Vn.mean(0).tolist(), "anchor_beta": Vp.mean(0).tolist(),
           "organic_alpha": org_alpha.tolist(), "organic_beta": org_beta.tolist(),
           "delta_alpha_anchor_minus_organic": (Vn.mean(0) - org_alpha).tolist(),
           "delta_beta_anchor_minus_organic": (Vp.mean(0) - org_beta).tolist(),
           "judge_accuracy_vs_constructed_truth": [
               float((V[:, j] == y).mean()) for j in range(4)],
           "error_correlation_offdiag_y0": np.cov(V[y == 0].T)[
               np.triu_indices(4, 1)].tolist()}
    (HERE / "rung2_audit_result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
