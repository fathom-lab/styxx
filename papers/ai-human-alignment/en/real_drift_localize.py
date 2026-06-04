# -*- coding: utf-8 -*-
"""
real_drift_localize.py — can the monitor LOCALIZE *real, targeted* damage? The strongest real-drift test.

Simulate targeted data-poisoning: fine-tune BERT where a chosen SUBSET of concepts get random (poisoned)
labels while the rest get their real category labels. The poisoned concepts' representations corrupt; the
clean ones are preserved/improved. Then ask the monitor's per-concept channel: does it pick out exactly the
poisoned concepts? With a CLEAN-fine-tune control (all real labels) proving the localization is the
poisoning, not chance. Reference: Binder 65-feature human space. Uses the shipped `styxx.meaning_integrity`.
"""
import os, sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
from styxx.meaning_integrity import MeaningReference, per_concept_alignment, meaning_alignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOK = AutoTokenizer.from_pretrained("bert-base-uncased")


def embed(enc, words):
    enc.eval(); out = np.zeros((len(words), 768))
    with torch.no_grad():
        for i in range(0, len(words), 64):
            b = words[i:i + 64]
            e = TOK(b, return_tensors="pt", padding=True); am = e["attention_mask"]
            hs = enc(**{k: v.to(DEV) for k, v in e.items()}).last_hidden_state
            for j in range(len(b)):
                L = int(am[j].sum()); out[i + j] = hs[j, 1:L - 1].mean(0).cpu().numpy()
    return out


def auc(scores, labels):
    o = np.argsort(scores); r = np.empty(len(scores)); r[o] = np.arange(len(scores))
    pos = labels == 1; n1 = pos.sum(); n0 = (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def train(words, labels, steps=350, lr=3e-5):
    torch.manual_seed(0)
    enc = AutoModel.from_pretrained("bert-base-uncased").to(DEV)
    head = nn.Linear(768, int(max(labels)) + 1).to(DEV)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr)
    Y = torch.tensor(labels, dtype=torch.long, device=DEV)
    ids = TOK(words, return_tensors="pt", padding=True)
    step = 0
    while step < steps:
        perm = torch.randperm(len(words))
        for i in range(0, len(words), 32):
            bi = perm[i:i + 32]
            enc.train(); opt.zero_grad()
            e = {k: v[bi].to(DEV) for k, v in ids.items()}
            F.cross_entropy(head(enc(**e).last_hidden_state[:, 0]), Y[bi]).backward()
            opt.step(); step += 1
            if step >= steps:
                break
    return embed(enc, words)


def main():
    df = pd.read_excel(os.path.join(HERE, "binder_data", "WordSet1_Ratings.xlsx"))
    words_all = [str(w).strip().lower() for w in df["Word"]]
    H_all = df.iloc[:, 5:70].apply(pd.to_numeric, errors="coerce").values
    cat_all = pd.factorize(df["Super Category"].astype(str))[0]
    keep = [i for i, w in enumerate(words_all) if w.isalpha() and not np.isnan(H_all[i]).any()]
    words = [words_all[i] for i in keep]
    human = H_all[keep].astype(float)
    real_labels = cat_all[keep]
    ref = MeaningReference(human, words=words, name="binder65")
    n = len(words); rng = np.random.default_rng(0)

    k = int(0.3 * n)
    poison_idx = rng.choice(n, k, replace=False)
    is_poison = np.zeros(n); is_poison[poison_idx] = 1
    poisoned_labels = real_labels.copy()
    poisoned_labels[poison_idx] = rng.integers(0, int(max(real_labels)) + 1, k)   # targeted poisoning
    print(f"{n} concepts, poisoning {k} ({100*k//n}%) with random labels; rest get real categories\n")

    for tag, labels in [("POISONED run (targeted)", poisoned_labels), ("CLEAN control (all real)", real_labels)]:
        emb = train(words, labels)
        pc = per_concept_alignment(emb, ref)
        a = auc(-pc, is_poison)                          # do the designated-poison concepts score low?
        pm, cm = pc[poison_idx].mean(), pc[is_poison == 0].mean()
        flagged = np.argsort(pc)[:k]; prec = is_poison[flagged].mean()
        print(f"{tag}: localization ROC-AUC {a:.3f}  precision@{k} {prec:.3f}  "
              f"(poison-mean {pm:+.3f} vs clean-mean {cm:+.3f})")

    print("\n  -> POISONED run AUC high + CLEAN control ~0.5 = the monitor localizes REAL targeted damage,")
    print("     and the designated set is NOT spuriously degraded without poisoning. Localization on real damage.")
    poisoned_words = [words[i] for i in poison_idx[:8]]
    print(f"  (example poisoned concepts: {poisoned_words})")


if __name__ == "__main__":
    main()
