# -*- coding: utf-8 -*-
"""
real_drift.py — REAL-degradation validation (closes the synthetic-corruption caveat). Instead of
hand-shuffling embeddings, we actually FINE-TUNE BERT and watch the monitor track the representation
damage, checkpoint by checkpoint. Two runs from the same init:
  DAMAGING : fine-tune on RANDOM labels (label-noise -> memorizes noise -> corrupts semantics)
  BENIGN   : fine-tune on REAL Binder semantic categories (meaningful supervision -> preserves meaning)
If the monitor's alignment FALLS for the damaging run but HOLDS for the benign one, it is catching
meaning DAMAGE, not merely "training happened." Reference: Binder 65-feature human space.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cn"))
from meaning_integrity import MeaningReference, alignment, dispersion, MeaningVitalSign

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOK = AutoTokenizer.from_pretrained("bert-base-uncased")


def embed(enc, words):
    """content-subword-token mean (excl CLS/SEP), eval mode — the monitor's view of the model's meaning."""
    enc.eval(); out = np.zeros((len(words), 768))
    with torch.no_grad():
        for i in range(0, len(words), 64):
            b = words[i:i + 64]
            e = TOK(b, return_tensors="pt", padding=True); am = e["attention_mask"]
            hs = enc(**{k: v.to(DEV) for k, v in e.items()}).last_hidden_state
            for j in range(len(b)):
                L = int(am[j].sum()); out[i + j] = hs[j, 1:L - 1].mean(0).cpu().numpy()
    return out


def train_run(name, words, labels, ref, vs, steps=400, ckpts=(0, 50, 100, 200, 400), lr=3e-5):
    torch.manual_seed(0)
    enc = AutoModel.from_pretrained("bert-base-uncased").to(DEV)
    head = nn.Linear(768, int(max(labels)) + 1).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr)
    Y = torch.tensor(labels, dtype=torch.long, device=DEV)
    ids = TOK(words, return_tensors="pt", padding=True)
    print(f"\n[{name}]  {'step':>5} {'align':>7} {'disp':>6} {'verdict':>9}")
    traj = []; step = 0; done = set()

    def ckpt(s):
        E = embed(enc, words); r = vs.check(E, words)
        traj.append({"step": s, "alignment": r["alignment"], "dispersion": r["dispersion_ratio"], "status": r["status"]})
        print(f"       {s:>5} {r['alignment']:>+7.3f} {r['dispersion_ratio']:>6.2f} {r['status']:>9}")
        done.add(s)

    ckpt(0)
    while step < max(ckpts):
        perm = torch.randperm(len(words))
        for i in range(0, len(words), 32):
            bi = perm[i:i + 32]
            enc.train(); opt.zero_grad()
            e = {k: v[bi].to(DEV) for k, v in ids.items()}
            logits = head(enc(**e).last_hidden_state[:, 0])
            F.cross_entropy(logits, Y[bi]).backward(); opt.step(); step += 1
            if step in ckpts and step not in done:
                ckpt(step)
        if step >= max(ckpts):
            break
    return traj


def main():
    df = pd.read_excel(os.path.join(HERE, "binder_data", "WordSet1_Ratings.xlsx"))
    words_all = [str(w).strip().lower() for w in df["Word"]]
    H_all = df.iloc[:, 5:70].apply(pd.to_numeric, errors="coerce").values
    cat_all = pd.factorize(df["Super Category"].astype(str))[0]
    keep = [i for i, w in enumerate(words_all) if w.isalpha() and not np.isnan(H_all[i]).any()]
    words = [words_all[i] for i in keep]
    human = H_all[keep].astype(float)
    real_labels = cat_all[keep]
    rng = np.random.default_rng(0)
    rand_labels = rng.integers(0, max(real_labels) + 1, len(words))
    ref = MeaningReference(human, words=words, name="binder65")
    print(f"{len(words)} words, {int(max(real_labels))+1} real categories")

    # calibrate the vital sign on the healthy pretrained model
    enc0 = AutoModel.from_pretrained("bert-base-uncased").to(DEV)
    vs = MeaningVitalSign(ref).calibrate(embed(enc0, words))
    del enc0
    print(f"healthy baseline: alignment {vs.base_align:.3f}")

    dmg = train_run("DAMAGING: random-label fine-tune", words, rand_labels, ref, vs)
    ben = train_run("BENIGN: real-category fine-tune", words, real_labels, ref, vs)

    print("\n" + "=" * 60)
    d0, dN = dmg[0]["alignment"], dmg[-1]["alignment"]
    b0, bN = ben[0]["alignment"], ben[-1]["alignment"]
    print(f"DAMAGING: alignment {d0:+.3f} -> {dN:+.3f} ({dN-d0:+.3f}), final status {dmg[-1]['status']}")
    print(f"BENIGN  : alignment {b0:+.3f} -> {bN:+.3f} ({bN-b0:+.3f}), final status {ben[-1]['status']}")
    win = (dN < d0 - 0.05) and (dmg[-1]["status"] != "HEALTHY") and (ben[-1]["status"] == "HEALTHY" or bN >= b0 - 0.03)
    print(f"-> monitor distinguishes HARMFUL from HELPFUL fine-tuning: {'PASS' if win else 'CHECK'}")
    import json
    json.dump({"damaging": dmg, "benign": ben, "pass": bool(win)},
              open(os.path.join(HERE, "real_drift_result.json"), "w"), indent=2)
    print("wrote real_drift_result.json")


if __name__ == "__main__":
    main()
