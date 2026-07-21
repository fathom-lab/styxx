"""Hardening arc part 1: transfer of the kill (two new task families, blatant anchors) and the
ladder REPAIR (same-generator anchors on the original task). PREREG_hardening_part1_2026_07_20.md
is the frozen contract. Same panel and machinery as rung 1 (imports stage_b_rung1). Crash-safe
per-replicate checkpoint keyed by arm; re-launch skips completed (arm, seed) cells. `--smoke`
writes only *_SMOKE_INVALID*. ASCII only.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import stage_b_rung1 as S
from stage_b_corpus import build_corpus
from styxx.anchors import audit_panel

ARMS = [
    {"arm": "repair_ladder_attr", "family": "attr", "anchor_style": "ladder", "base": 4000},
    {"arm": "numeric_blatant", "family": "numeric", "anchor_style": "blatant", "base": 5000},
    {"arm": "temporal_blatant", "family": "temporal", "anchor_style": "blatant", "base": 6000},
    {"arm": "chain_blatant", "family": "chain", "anchor_style": "blatant", "base": 8000},
    {"arm": "chain_ladder", "family": "chain", "anchor_style": "ladder", "base": 8500},
]
R = 15
N_ORG, K_ANCHOR, PI_TRUE = 240, 80, 0.35
CKPT = HERE / "stage_b_hardening_checkpoint.jsonl"


def one_replicate(arm, seed, tok, model, yes_ids, no_ids, n_org, k_anchor):
    organic, anchors, truth = build_corpus(seed, n_organic=n_org, k_anchor=k_anchor,
                                           pi=PI_TRUE, family=arm["family"],
                                           anchor_style=arm["anchor_style"])
    neg_items = [a for a in anchors if a["role"] == "neg_anchor"]
    pos_items = [a for a in anchors if a["role"] == "pos_anchor"]
    P = list(S.PERSONAS)
    j = lambda items, **kw: np.stack(
        [S.judge(tok, model, yes_ids, no_ids, p, items, **kw) for p in P], axis=1)
    V, Vn, Vp = j(organic), j(neg_items), j(pos_items)
    Vd, Vdn, Vdp = j(organic, redact=True), j(neg_items, redact=True), j(pos_items, redact=True)

    y_true = np.array([truth[it["id"]] for it in organic])
    pi_true = float(y_true.mean())
    audit = audit_panel(V, Vn, Vp, n_boot=300, null_sims=200, seed=seed)
    deaf = audit_panel(Vd, Vdn, Vdp, n_boot=100, null_sims=0, seed=seed)
    mv = float((V.mean(1) > 0.5).mean())
    ds = S.ds_em(V)
    clamp = np.concatenate([np.full(len(V), -1.0), np.zeros(len(Vn)), np.ones(len(Vp))])
    ss = S.ds_em(np.vstack([V, Vn, Vp]), clamp=clamp)
    org_alpha = V[y_true == 0].mean(0); org_beta = V[y_true == 1].mean(0)
    return {"arm": arm["arm"], "seed": seed, "pi_true_realized": pi_true,
            "audit": {k: audit.get(k) for k in ("verdict", "pi", "ci", "regime", "s", "s_ci",
                                                "activated", "misfit", "kept", "alpha", "beta")},
            "audit_covered": (bool(audit["ci"][0] <= pi_true <= audit["ci"][1])
                              if audit.get("verdict") == "ESTIMATED" else None),
            "deaf_verdict": deaf["verdict"],
            "mv_err": abs(mv - pi_true), "ds_err": abs(ds["pi"] - pi_true),
            "ss_ds_err": abs(ss["pi"] - pi_true),
            "audit_err": (abs(audit["pi"] - pi_true) if audit.get("pi") is not None else None),
            "delta_alpha_anchor_minus_organic": (np.asarray(audit["alpha"]) - org_alpha).tolist(),
            "delta_beta_anchor_minus_organic": (np.asarray(audit["beta"]) - org_beta).tolist(),
            "organic_alpha": org_alpha.tolist(), "organic_beta": org_beta.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    done = set()
    if CKPT.exists() and not args.smoke:
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                done.add((d["arm"], d["seed"]))
            except Exception:
                pass
    tok, model, yes_ids, no_ids = S.load_model()
    n_org, k_anchor = (24, 8) if args.smoke else (N_ORG, K_ANCHOR)
    reps = 1 if args.smoke else R
    for arm in ARMS:
        for i in range(1, reps + 1):
            seed = arm["base"] + i + (9000 if args.smoke else 0)
            if (arm["arm"], seed) in done:
                continue
            t0 = time.time()
            rec = one_replicate(arm, seed, tok, model, yes_ids, no_ids, n_org, k_anchor)
            if args.smoke:
                (HERE / "stage_b_hardening_SMOKE_INVALID.json").write_text(
                    json.dumps(rec), encoding="utf-8")
            else:
                with open(CKPT, "a", encoding="utf-8", newline="\n") as f:
                    f.write(json.dumps(rec) + "\n")
            print(f"  {arm['arm']} seed {seed}: {rec['audit']['verdict']} "
                  f"pi={rec['audit'].get('pi')} covered={rec['audit_covered']} "
                  f"deaf={rec['deaf_verdict']} ({time.time() - t0:.0f}s)")
    print("DONE")


if __name__ == "__main__":
    main()
