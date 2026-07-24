"""Cycle 67 -- build the SQuAD-v2 item pool and the retrieval haystack.

Everything here is DATA-DERIVED. Nothing is authored from an answer key:
  * questions and gold answers come from SQuAD v2 (Wikipedia + crowd annotators);
  * the plausible-wrong sibling Y is a REAL short answer span drawn from a DIFFERENT passage,
    chosen as the most embedding-similar candidate to the true answer (so it is type-plausible)
    -- a different passage, so the two candidates do not co-occur in the gold paragraph, which
    would make the retrieval channel abstain by construction;
  * the corpus is every unique SQuAD context (train + validation), a ~20k-paragraph haystack in
    which the gold passage must actually be FOUND, not looked up.

Writes squad_pool.json (items) and squad_corpus.json (passages). ASCII only, CPU/GPU embed.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_ITEMS = 200          # frozen pool size (runtime control)
SEED = 670000


def main():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    import numpy as np

    rng = np.random.default_rng(SEED)
    ds = load_dataset("rajpurkar/squad_v2")
    val, train = ds["validation"], ds["train"]

    # ---- haystack: every unique context in the dataset
    corpus = sorted({r["context"] for r in val} | {r["context"] for r in train})
    print(f"corpus passages: {len(corpus)}")

    # ---- candidate items: short answers, one per (context, answer) pair
    per_ctx = defaultdict(list)
    for r in val:
        a = r["answers"]["text"]
        if not a:
            continue
        t = a[0].strip()
        if 1 <= len(t.split()) <= 3 and re.search(r"\w", t):
            per_ctx[r["context"]].append((r["question"], t))

    cands = []
    for ctx, qs in per_ctx.items():
        seen = set()
        for q, t in qs:
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            cands.append({"q": q, "X": t, "ctx": ctx})
    print(f"candidate items: {len(cands)}")

    idx = rng.permutation(len(cands))[: N_ITEMS * 3]
    pick = [cands[i] for i in idx]

    # ---- distractor Y: most similar short answer from a DIFFERENT passage
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    answers = [c["X"] for c in pick]
    A = emb.encode(answers, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    S = A @ A.T
    items = []
    for i, c in enumerate(pick):
        order = np.argsort(-S[i])
        y = None
        for j in order:
            if j == i:
                continue
            if pick[j]["ctx"] == c["ctx"]:            # must come from a DIFFERENT passage
                continue
            if pick[j]["X"].lower() == c["X"].lower():
                continue
            if c["X"].lower() in pick[j]["X"].lower() or pick[j]["X"].lower() in c["X"].lower():
                continue                              # avoid substring overlap (breaks scoring)
            y = pick[j]["X"]
            break
        if y is None:
            continue
        items.append({"q": c["q"], "subject": c["q"], "X": c["X"], "Y": y,
                      "sim": float(S[i][order[0] if order[0] != i else order[1]])})
        if len(items) >= N_ITEMS:
            break

    print(f"final items: {len(items)}")
    for it in items[:5]:
        print(f"  Q={it['q'][:58]!r} X={it['X']!r} Y={it['Y']!r}")

    (HERE / "squad_pool.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    (HERE / "squad_corpus.json").write_text(json.dumps(corpus), encoding="utf-8")
    print(f"wrote squad_pool.json ({len(items)}) and squad_corpus.json ({len(corpus)})")


if __name__ == "__main__":
    sys.exit(main())
