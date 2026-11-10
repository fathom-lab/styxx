"""explore_pc1.py: EXPLORATORY. What is the one dominant direction in the stored arrays?

Lead-run, 2026-11-10. Stored arrays only (original 96 concepts, already seen), plus the cached tokenizers
of the five cached models, loaded offline. No model weights are loaded. Nothing here is a test: it is the
look that a sealed test on fresh words is then written from.

Hypothesis being looked at: the bare-word template "{w}" makes a ONE-token input for single-token words in
tokenizers that add no start token. Models are known to carry very large activations at the first position
(Sun et al. 2024, massive activations). If so, the dominant direction is "this word was alone at position 0".

Usage: py -3.12 -X utf8 -I -B explore_pc1.py <kit_dir> <out.json>
"""
import itertools
import json
import os
import sys

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

KIT, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, KIT)
import reproduce as kit  # word lists, templates, model table, distmat (the lab's method)

from transformers import AutoTokenizer  # noqa: E402

words, cats = kit.flatten(kit.ORIGINAL)
z = np.load(os.path.join(KIT, "contextual_reps.npz"))
iu = np.triu_indices(len(words), k=1)
out = {"set": "original", "n_words": len(words), "models": {}}


def auc(score, label):
    """probability that a random label-1 item scores above a random label-0 item (ties half)."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


seq_len = {}
for name, repo, fam, rev in kit.MODELS:
    key = "fixed__" + name
    x = z[key].astype(np.float64)
    xc = x - x.mean(0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    pc1 = u[:, 0] * s[0]
    share = float(s[0] ** 2 / (s ** 2).sum())
    norms = np.linalg.norm(x, axis=1)
    rec = {"pc1_share": round(share, 4), "row_norm_min": round(float(norms.min()), 2),
           "row_norm_median": round(float(np.median(norms)), 2), "row_norm_max": round(float(norms.max()), 2)}
    try:
        try:
            tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
            enc = lambda t: tok(t, return_tensors=None)["input_ids"]
            how = "transformers, offline"
        except Exception:
            # fall back to the raw tokenizer.json in the cache snapshot (post-processor adds any start token)
            from tokenizers import Tokenizer
            snap = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                                "models--" + repo.replace("/", "--"), "snapshots")
            cand = [os.path.join(snap, d, "tokenizer.json") for d in sorted(os.listdir(snap))]
            cand = [c for c in cand if os.path.exists(c)]
            raw = Tokenizer.from_file(cand[-1])
            enc = lambda t: raw.encode(t).ids
            how = "tokenizer.json from cache snapshot"
        bare = np.array([len(enc(w)) for w in words])
        with_a = np.array([len(enc("a " + w)) for w in words])
        alone = (bare == 1).astype(int)
        seq_len[name] = bare
        a = auc(pc1, alone)
        rec.update({
            "tokenizer": how,
            "bare_template_len_counts": {str(k): int((bare == k).sum()) for k in sorted(set(bare.tolist()))},
            "adds_start_token": bool(len(enc("a")) >= 2 and enc("a")[0] == enc("the")[0]),
            "n_words_alone_at_position_0": int(alone.sum()),
            "auc_pc1_vs_alone": None if a is None else round(max(a, 1 - a), 4),
            "corr_abs_pc1_vs_alone": None if alone.std() == 0 else round(abs(float(np.corrcoef(pc1, alone)[0, 1])), 4),
            "corr_abs_rownorm_vs_alone": None if alone.std() == 0 else round(abs(float(np.corrcoef(norms, alone)[0, 1])), 4),
            "words_not_alone": [w for w, f in zip(words, alone) if not f][:40],
        })
    except Exception as e:  # tokenizer not cached (gemma)
        rec["tokenizer"] = "not loaded: " + type(e).__name__
    out["models"][name] = rec

# what removal variants do to the 13 cross-family pairs (raw, lab method), all on the exploratory set
MODELS = [m[0] for m in kit.MODELS]


def up(x):
    return kit.distmat(x)[iu]


def remove_top(x, k=1):
    xc = x - x.mean(0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    s = s.copy()
    s[:k] = 0
    return (u * s) @ vt


def standardise(x):
    xc = x - x.mean(0)
    return xc / (xc.std(0) + 1e-9)


def xfam(reps):
    ups = {m: up(reps[m]) for m in MODELS}
    rows = {}
    for a, b in itertools.combinations(MODELS, 2):
        if kit.FAMILY[a] != kit.FAMILY[b]:
            rows[a + " x " + b] = float(np.corrcoef(ups[a], ups[b])[0, 1])
    return rows


def within_cat_null(reps, n=200, seed=0):
    rng = np.random.default_rng(seed)
    ups = {m: up(reps[m]) for m in MODELS}
    idx_by_cat = [np.where(cats == c)[0] for c in dict.fromkeys(cats.tolist())]
    means = []
    for _ in range(n):
        perm = np.arange(len(words))
        for idx in idx_by_cat:
            perm[idx] = rng.permutation(idx)
        vals = []
        for a, b in itertools.combinations(MODELS, 2):
            if kit.FAMILY[a] != kit.FAMILY[b]:
                d = kit.distmat(reps[b][perm])[iu]
                vals.append(np.corrcoef(ups[a], d)[0, 1])
        means.append(np.mean(vals))
    return float(np.mean(means)), float(np.std(means))


base = {m: z["fixed__" + m].astype(np.float64) for m in MODELS}
variants = {
    "as_published": base,
    "top1_pc_removed_all_models": {m: remove_top(base[m], 1) for m in MODELS},
    "standardised_per_dimension": {m: standardise(base[m]) for m in MODELS},
}
out["variants"] = {}
for vname, reps in variants.items():
    rows = xfam(reps)
    v = list(rows.values())
    nm, ns = within_cat_null(reps)
    out["variants"][vname] = {"xfam_mean": round(float(np.mean(v)), 4), "xfam_median": round(float(np.median(v)), 4),
                              "xfam_min": round(min(v), 4), "within_category_null_mean": round(nm, 4),
                              "within_category_null_sd": round(ns, 4),
                              "pair_Qwen1.5B_x_Llama1B": round(rows["Qwen2.5-1.5B x Llama-3.2-1B"], 4)}

json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
for name, rec in out["models"].items():
    print(name, {k: v for k, v in rec.items() if k != "words_not_alone"})
    if rec.get("words_not_alone"):
        print("   words not alone at position 0:", ", ".join(rec["words_not_alone"]))
for vname, v in out["variants"].items():
    print(vname, v)
