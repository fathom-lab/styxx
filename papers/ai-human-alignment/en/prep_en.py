# -*- coding: utf-8 -*-
"""
prep_en.py — build an INDEPENDENT English test bed for the meaning-integrity monitor:
  reference = Lancaster Sensorimotor Norms (11 experiential dims, human-rated) — wholly independent of
              the Chinese 54-feature space.
  shallow   = GloVe-300 (co-occurrence).      deep = BERT-base (self-supervised, paradigm-matched to GloVe).
Saves en_data.npz (words, human[N,11], glove[N,300], bert[N,768]).
"""
import os, sys, csv
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))

COLS = ["Auditory.mean", "Gustatory.mean", "Haptic.mean", "Interoceptive.mean", "Olfactory.mean",
        "Visual.mean", "Foot_leg.mean", "Hand_arm.mean", "Head.mean", "Mouth.mean", "Torso.mean"]


def main():
    seen = {}
    with open(os.path.join(HERE, "lancaster.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = row["Word"].strip().lower()
            if w.isalpha() and 3 <= len(w) <= 15 and w not in seen:
                try:
                    seen[w] = [float(row[c]) for c in COLS]
                except (ValueError, KeyError):
                    continue
    print(f"Lancaster: {len(seen)} clean single words", flush=True)

    import gensim.downloader as gd
    print("loading GloVe-300 (downloads+caches once)...", flush=True)
    glove = gd.load("glove-wiki-gigaword-300")
    words = [w for w in seen if w in glove]
    rng = np.random.default_rng(0)
    if len(words) > 1500:
        words = sorted(rng.choice(words, 1500, replace=False))
    human = np.array([seen[w] for w in words], float)
    gv = np.array([glove[w] for w in words], float)
    print(f"{len(words)} words have GloVe; human {human.shape}, glove {gv.shape}", flush=True)

    print("embedding with BERT-base (mean-pooled last layer)...", flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    bert = []
    with torch.no_grad():
        for i in range(0, len(words), 64):
            batch = words[i:i + 64]
            enc = tok(batch, return_tensors="pt", padding=True).to(dev)
            out = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out * mask).sum(1) / mask.sum(1)
            bert.append(emb.cpu().numpy())
    bert = np.concatenate(bert)
    print(f"BERT {bert.shape} on {dev}", flush=True)

    np.savez(os.path.join(HERE, "en_data.npz"), words=np.array(words), human=human, glove=gv, bert=bert)
    print("wrote en_data.npz", flush=True)


if __name__ == "__main__":
    main()
