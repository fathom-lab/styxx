"""Does the length confound vanish in activation space? (PREREG_grounded_vs_text_length)

Extracts last-token residual-stream activations from a local reader model processing the overconfidence
corpus, trains a linear probe (calibrated vs overconfident), and applies the SAME causal length control
(CEM) we used on the text instruments. Compares how much the TEXT probe vs the ACTIVATION probe loses to
length-matching. Frozen thresholds decide GROUNDING-WINS / HONEST-NULL / PARTIAL.

  python scripts/grounded_vs_text_length.py --extract --model {llama,qwen}
  python scripts/grounded_vs_text_length.py            # analyze all extracted
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from overconfidence_train_v0 import featurize
from length_control_causal import cv_oof, boot_ci
from suite_causal_length import cem_match
from scipy.stats import spearmanr

REPOS = {"llama": "meta-llama/Llama-3.2-3B-Instruct", "qwen": "Qwen/Qwen2.5-3B-Instruct"}
V0 = ROOT / "benchmarks" / "data" / "overconfidence" / "pairs_v0.jsonl"
ACTDIR = ROOT / "benchmarks" / "data" / "overconfidence"
ACT_CEM, GAP = 0.72, 0.05
def actpath(tag): return ACTDIR / f"_acts_{tag}.npz"
def load(p): return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def extract(tag):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = REPOS[tag]; rows = load(V0)
    print(f"[extract] {repo}: {len(rows)} rows")
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="cuda",
                                                 output_hidden_states=True).eval()
    acts, y, wc = [], [], []
    for i, r in enumerate(rows):
        text = r["question"] + "\n" + r["response"]
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            hs = model(**ids).hidden_states  # tuple [n_layers+1] of [1, seq, hidden]
        last = np.stack([h[0, -1, :].float().cpu().numpy() for h in hs])  # [n_layers+1, hidden]
        acts.append(last); y.append(int(r["label_overconfident"])); wc.append(len(r["response"].split()))
        if (i + 1) % 50 == 0: print(f"  [{i+1}/{len(rows)}]")
    np.savez_compressed(actpath(tag), acts=np.array(acts, np.float32), y=np.array(y), wc=np.array(wc))
    print(f"[extract] {tag}: saved {actpath(tag).name}  acts shape {np.array(acts).shape}")
    del model; torch.cuda.empty_cache()


def probe_layers(acts, y):
    """CV-AUC of a linear probe at each layer; return per-layer AUCs (full corpus)."""
    return [cv_oof(acts[:, L, :], y)[1] for L in range(acts.shape[1])]


def analyze_model(tag, rows):
    d = np.load(actpath(tag)); acts, y, wc = d["acts"], d["y"], d["wc"]
    # ---- text probe (the 9 overconfidence features) ----
    Xt, yt, names = featurize(rows)
    assert np.array_equal(yt, y), "row/label order mismatch"
    text_raw = cv_oof(Xt, yt)[1]
    idx = cem_match(wc.astype(float), y, binw=8)
    text_cem = cv_oof(Xt[idx], y[idx])[1]
    # ---- activation probe: pick layer by act_raw (BLIND to CEM), per frozen prereg ----
    per_layer = probe_layers(acts, y)
    sel = int(np.argmax(per_layer)); act_raw = per_layer[sel]
    oof_cem, act_cem = cv_oof(acts[idx][:, sel, :], y[idx])
    act_cem_ci = boot_ci(y[idx], oof_cem)
    # length leakage: does the matched-set ACT probe score still track word count?
    leak = float(spearmanr(oof_cem, wc[idx]).correlation)
    d_text, d_act = text_raw - text_cem, act_raw - act_cem
    print(f"\n===== {tag} (n={len(y)}, sel layer {sel}/{acts.shape[1]-1}) =====")
    print(f"  TEXT  probe: raw {text_raw:.3f} -> CEM {text_cem:.3f}   (Δ {d_text:+.3f})")
    print(f"  ACT   probe: raw {act_raw:.3f} -> CEM {act_cem:.3f} CI[{act_cem_ci[0]:.3f},{act_cem_ci[1]:.3f}]  (Δ {d_act:+.3f})")
    print(f"  act_cem - text_cem = {act_cem-text_cem:+.3f}   | length-leakage spearman(act_score,words)={leak:+.3f}")
    print(f"  layer sweep act_raw: {[round(a,2) for a in per_layer]}")
    win = (act_cem >= ACT_CEM) and ((act_cem - text_cem) >= GAP) and (d_act < d_text) and (abs(leak) < 0.35)
    nul = (act_cem < 0.65) or (abs(act_cem - text_cem) < GAP)
    verdict = "GROUNDING-WINS" if win else ("HONEST-NULL" if nul else "PARTIAL")
    if win and abs(leak) >= 0.35:
        verdict = "PARTIAL (act length-leakage too high)"
    print(f"  >>> {verdict}")
    return {"model": tag, "sel_layer": sel, "text_raw": text_raw, "text_cem": text_cem,
            "act_raw": act_raw, "act_cem": act_cem, "act_cem_ci": list(act_cem_ci),
            "delta_text": d_text, "delta_act": d_act, "act_minus_text_cem": act_cem - text_cem,
            "length_leak_spearman": leak, "verdict": verdict, "layer_sweep": per_layer}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true")
    ap.add_argument("--model", choices=list(REPOS), default="llama"); a = ap.parse_args()
    if a.extract:
        extract(a.model); return
    rows = load(V0)
    res = [analyze_model(t, rows) for t in REPOS if actpath(t).exists()]
    if res:
        verdicts = {r["verdict"].split(" ")[0] for r in res}
        print(f"\n###### CROSS-MODEL: {[(r['model'], r['verdict']) for r in res]}")
        print(f"###### ROBUST GROUNDING-WIN: {verdicts == {'GROUNDING-WINS'} and len(res) >= 2}")
        (ACTDIR / "_grounded_vs_text_result.json").write_text(json.dumps(res, indent=2))
        print("wrote benchmarks/data/overconfidence/_grounded_vs_text_result.json")


if __name__ == "__main__":
    main()
