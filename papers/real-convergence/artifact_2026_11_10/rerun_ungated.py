"""rerun_ungated.py: check the lab's start-of-input finding yourself, with models that need no licence.

One file. No lab code is imported. It needs: python 3.10+, torch, transformers, numpy. A laptop CPU is enough for
the default model (about 1 GB download, a few minutes). Add --more for two larger ungated models (about 11 GB).

What it checks, on the 96 "fresh" words of the lab's sealed tests of 2026-11-10:

  1. Each word is put into 8 short templates. The first template is the bare word.
  2. In tokenizers that add no start token, a single-token bare word sits alone at position 0.
     The claim: its hidden state there is tens of times larger than in the other seven templates.
  3. So the mean over 8 templates is dominated by that one state: one direction holds nearly all the variance
     across words, and it separates "was alone at position 0" from the rest perfectly.
  4. Drop the bare template and that direction is gone.

It prints your numbers beside the lab's and says whether each sealed bar is met.

  python rerun_ungated.py            Qwen2.5-0.5B-Instruct only
  python rerun_ungated.py --more     also Qwen2.5-1.5B-Instruct and Phi-3.5-mini-instruct, and their agreement
  python rerun_ungated.py --local Qwen/Qwen2.5-0.5B-Instruct=<folder>     use a copy of a model you already hold
  python rerun_ungated.py --cpu      stay off a GPU (the lab's numbers are float32 on a processor)

Written after the lab's two sealed tests. It is not part of them: it repeats their bars on your machine.
"""
import itertools
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

FRESH = {
    "tool": ["hammer", "screwdriver", "wrench", "drill", "saw", "pliers", "chisel", "axe", "shovel", "rake", "ladder", "clamp"],
    "clothing": ["shirt", "pants", "jacket", "dress", "hat", "scarf", "gloves", "socks", "shoes", "coat", "sweater", "belt"],
    "sport": ["soccer", "tennis", "basketball", "baseball", "golf", "hockey", "boxing", "swimming", "cycling", "skiing", "surfing", "wrestling"],
    "kitchen": ["spoon", "fork", "knife", "plate", "bowl", "cup", "pot", "pan", "kettle", "blender", "oven", "fridge"],
    "building": ["house", "tower", "bridge", "castle", "church", "factory", "stadium", "museum", "hospital", "library", "school", "prison"],
    "plant": ["oak", "pine", "rose", "tulip", "fern", "cactus", "bamboo", "maple", "ivy", "moss", "daisy", "willow"],
    "drink": ["water", "coffee", "tea", "juice", "milk", "wine", "beer", "soda", "lemonade", "cocoa", "cider", "smoothie"],
    "emotion": ["joy", "anger", "fear", "sadness", "surprise", "disgust", "love", "hope", "pride", "shame", "envy", "guilt"],
}
TEMPLATES = ["{w}", "a {w}", "the {w}", "I saw a {w}", "there is a {w}", "this is a {w}", "they showed me the {w}", "look at that {w}"]
WORDS = [w for ws in FRESH.values() for w in ws]

# the lab's sealed results on these words (float32, CPU). (norm ratio, share with 8 templates, AUC, share with 7)
LAB = {
    "Qwen/Qwen2.5-0.5B-Instruct": (79.9, 0.9885, 1.0, 0.1038),
    "Qwen/Qwen2.5-1.5B-Instruct": (159.7, 0.9969, 1.0, 0.0894),
    "microsoft/Phi-3.5-mini-instruct": (29.1, 0.9554, 1.0, 0.0802),
}
# sealed bars: ratio at least 20 (10 for Phi), share8 at least 0.90 (0.85 for the second test), AUC 0.99, share7 at most 0.30
BARS = {"Qwen/Qwen2.5-0.5B-Instruct": 20.0, "Qwen/Qwen2.5-1.5B-Instruct": 20.0, "microsoft/Phi-3.5-mini-instruct": 10.0}
LAB_PAIR_7 = {("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"): 0.9234,
              ("Qwen/Qwen2.5-1.5B-Instruct", "microsoft/Phi-3.5-mini-instruct"): 0.8899}


def source(repo):
    """the hub name, or a local folder given as --local <hub name>=<folder> for a model you already hold"""
    for k, arg in enumerate(sys.argv):
        if arg == "--local" and k + 1 < len(sys.argv) and sys.argv[k + 1].startswith(repo + "="):
            return sys.argv[k + 1].split("=", 1)[1]
    return repo


def states_for(repo):
    dev = "cuda" if torch.cuda.is_available() and "--cpu" not in sys.argv else "cpu"
    tok = AutoTokenizer.from_pretrained(source(repo))
    mdl = AutoModel.from_pretrained(source(repo), torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    fl = max(1, int(0.66 * mdl.config.num_hidden_layers))          # the lab's fixed layer
    st = np.zeros((len(WORDS), len(TEMPLATES), mdl.config.hidden_size), np.float32)
    n_tok = np.zeros((len(WORDS), len(TEMPLATES)), int)
    with torch.no_grad():
        for i, w in enumerate(WORDS):
            for j, t in enumerate(TEMPLATES):
                ids = tok(t.format(w=w), return_tensors="pt").input_ids.to(dev)
                h = mdl(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[fl][0, -1]
                st[i, j] = h.float().cpu().numpy()
                n_tok[i, j] = ids.shape[1]
    del mdl
    return st.astype(np.float64), n_tok, dev


def top_share(x):
    xc = x - x.mean(0)
    u, s, _ = np.linalg.svd(xc, full_matrices=False)
    return float(s[0] ** 2 / (s ** 2).sum()), u[:, 0] * s[0]


def auc(score, label):
    pos, neg = score[label], score[~label]
    a = float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
    return max(a, 1 - a)


def upper(x):
    x = x - x.mean(0)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    d = np.sqrt(np.maximum(2.0 - 2.0 * (x @ x.T), 0.0))
    return d[np.triu_indices(len(x), k=1)]


def main():
    repos = list(LAB) if "--more" in sys.argv else ["Qwen/Qwen2.5-0.5B-Instruct"]
    reps, ok_all = {}, True
    for repo in repos:
        st, n_tok, dev = states_for(repo)
        alone = n_tok[:, 0] == 1
        norms = np.linalg.norm(st, axis=2)
        ratio = float(np.median((norms[:, 0] / np.median(norms[:, 1:], axis=1))[alone])) if alone.any() else float("nan")
        s8, p8 = top_share(st.mean(1))
        s7, _ = top_share(st[:, 1:].mean(1))
        a = auc(p8, alone) if alone.any() and (~alone).any() else float("nan")
        lab = LAB[repo]
        ok = ratio >= BARS[repo] and s8 >= 0.85 and a >= 0.99 and s7 <= 0.30
        ok_all = ok_all and ok
        print("\n%s  (%s, %d of %d words alone at position 0)" % (repo, dev, int(alone.sum()), len(WORDS)))
        print("  bare-template norm against the other seven:   yours %7.1fx   lab %7.1fx   bar: at least %gx" % (ratio, lab[0], BARS[repo]))
        print("  top-direction share, 8 templates:             yours %7.4f    lab %7.4f    bar: at least 0.85" % (s8, lab[1]))
        print("  that direction against 'alone', AUC:          yours %7.4f    lab %7.4f    bar: at least 0.99" % (a, lab[2]))
        print("  top-direction share, bare template dropped:   yours %7.4f    lab %7.4f    bar: at most 0.30" % (s7, lab[3]))
        print("  sealed bars met on your machine: %s" % ("YES" if ok else "NO"))
        reps[repo] = (st.mean(1), st[:, 1:].mean(1))
    for a, b in itertools.combinations(repos, 2):
        r8 = float(np.corrcoef(upper(reps[a][0]), upper(reps[b][0]))[0, 1])
        r7 = float(np.corrcoef(upper(reps[a][1]), upper(reps[b][1]))[0, 1])
        lab7 = LAB_PAIR_7.get((a, b)) or LAB_PAIR_7.get((b, a))
        print("\n%s  with  %s" % (a.split("/")[1], b.split("/")[1]))
        print("  agreement (RSA), 8 templates: %.4f   bare template dropped: %.4f%s" % (r8, r7, "" if lab7 is None else "   lab: %.4f" % lab7))
    print("\nRESULT: the lab's finding %s on this machine." % ("re-runs" if ok_all else "DOES NOT re-run"))
    print("Either way, please post this whole output where the lab asks for reruns. A failed rerun is worth more than a silent one.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
