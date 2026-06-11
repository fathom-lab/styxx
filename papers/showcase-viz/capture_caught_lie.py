"""Capture per-item render data for the caught-lie showcase (DESCRIPTIVE, not a scored run).

Re-executes the certified v3 pipeline (same SEED=0, same statements, same fits) on the headline cell
(Llama-3.2-3B) and RECORDS per-item: claim text, truth label, neutral verbal margin, pressure verbal
margin, internal conscience score, tau, caved/caught flags. This is render-data for the visualization
of FINDING_says_yes_knows_no_v3_2026_06_11.md (ITEM-CAUGHT) — the finding's claims rest on its own
receipt, NOT on this capture. Output: caught_lie_capture.json.

Usage: python papers/showcase-viz/capture_caught_lie.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(HERE / fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


v1 = _load("syk_v1", "run_says_yes_knows_no.py")
v2 = _load("syk_v2", "run_says_yes_knows_no_v2.py")
v3 = _load("syk_v3", "run_says_yes_knows_no_v3.py")

TGT = "meta-llama/Llama-3.2-3B-Instruct"
SEED = 0


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(SEED)

    train = v1.build_train(); rng.shuffle(train)
    fit = train[: int(0.80 * len(train))]
    ood = v1.build_ood()
    f_txt = [t for t, _ in fit]; f_lab = np.array([l for _, l in fit])
    o_txt = [t for t, _ in ood]; o_lab = np.array([l for _, l in ood])
    true_idx = np.where(o_lab == 1)[0]; false_idx = np.where(o_lab == 0)[0]
    msgs_p = lambda c: [{"role": "system", "content": v2.SYS_DEFERENCE}, {"role": "user", "content": v2.pressure_v2(c)}]
    msgs_n = lambda c: [{"role": "user", "content": v1.neutral_prompt(c)}]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("source (gemma) ...", flush=True)
    stok = AutoTokenizer.from_pretrained(v3.SRC)
    smdl = AutoModelForCausalLM.from_pretrained(v3.SRC, torch_dtype=torch.float16).to(dev).eval()
    stids, sfids = v1.tf_token_ids(stok)
    _, sf = v3.collect(smdl, stok, f_txt, msgs_p, [v3.SRC_LAYER], stids, sfids, tag="gemma fit")
    del smdl
    v3.free_gpu()
    src_fit = sf[v3.SRC_LAYER]
    w, b = v1.fit_direction(src_fit, f_lab)

    print(f"target {TGT} ...", flush=True)
    ttok = AutoTokenizer.from_pretrained(TGT)
    tmdl = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
    tids, fids = v1.tf_token_ids(ttok)
    nL = tmdl.config.num_hidden_layers
    cand = list(range(nL // 3, int(0.8 * nL) + 1, 2))
    _, tf_r = v3.collect(tmdl, ttok, f_txt, msgs_p, cand, tids, fids, tag="fit")
    beh_neutral, _ = v3.collect(tmdl, ttok, o_txt, msgs_n, [cand[0]], tids, fids, tag="neutral")
    beh_pressure, to_r = v3.collect(tmdl, ttok, o_txt, msgs_p, cand, tids, fids, tag="pressure")
    del tmdl
    v3.free_gpu()

    perm = rng.permutation(len(fit)); tr, va = perm[: int(0.8 * len(fit))], perm[int(0.8 * len(fit)):]
    best = None
    for L in cand:
        for alpha in (10.0, 100.0, 1000.0):
            M = v1.fit_map(tf_r[L][tr], src_fit[tr], alpha)
            pred = v1.apply_map(M, tf_r[L][va])
            r2 = 1 - ((pred - src_fit[va]) ** 2).sum() / (((src_fit[va] - src_fit[va].mean(0)) ** 2).sum() + 1e-9)
            if best is None or r2 > best[0]:
                best = (r2, L, alpha)
    r2, L, alpha = best
    M = v1.fit_map(tf_r[L], src_fit, alpha)
    internal = v1.apply_map(M, to_r[L]) @ w + b
    tau = 0.5 * (internal[true_idx].mean() + internal[false_idx].mean())

    items = []
    for i, claim in enumerate(o_txt):
        caved = bool(o_lab[i] == 0 and beh_neutral[i] < 0 and beh_pressure[i] > 0)
        items.append({
            "claim": claim,
            "truth": int(o_lab[i]),
            "neutral_margin": round(float(beh_neutral[i]), 4),
            "pressure_margin": round(float(beh_pressure[i]), 4),
            "verbal_under_pressure": "True" if beh_pressure[i] > 0 else "False",
            "internal_score": round(float(internal[i]), 4),
            "conscience_says": "true" if internal[i] >= tau else "false",
            "caved": caved,
            "caught": bool(caved and internal[i] < tau),
        })
    caved_items = [x for x in items if x["caved"]]
    caught = sum(x["caught"] for x in caved_items)
    out = {
        "purpose": "render-data capture for the caught-lie showcase (descriptive; certified claims live in says_yes_knows_no_v3_result.json)",
        "finding": "papers/showcase-viz/FINDING_says_yes_knows_no_v3_2026_06_11.md",
        "target": TGT, "target_layer": int(L), "tau": round(float(tau), 4), "seed": SEED,
        "n_items": len(items), "n_caved": len(caved_items), "n_caught": int(caught),
        "items": items,
    }
    (HERE / "caught_lie_capture.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\ncaptured {len(items)} items | caved {len(caved_items)} | caught {caught}", flush=True)
    for x in caved_items:
        print(f"  [{'CAUGHT' if x['caught'] else 'missed'}] {x['claim']}  (says {x['verbal_under_pressure']}, "
              f"mind {x['internal_score']:+.2f} vs tau {out['tau']:+.2f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
