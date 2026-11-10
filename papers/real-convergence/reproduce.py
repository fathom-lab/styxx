# -*- coding: utf-8 -*-
"""
reproduce.py -- re-derive the "real-convergence" cross-model RSA numbers.

Source: fathom-lab/styxx, papers/real-convergence/, commit a59db86e (2026-06-03).
The analysis code below is copied from run_real_convergence.py / _v2 / _v3_controls / _confirm
(distmat, cat_structure, partial_corr, the 50-permutation shuffled control) without change of
method. Only the model forward passes are separated out.

MODES
  py reproduce.py
      PATH A (no models, no network, numpy only). Loads the committed contextual_reps.npz
      (the v3 run's stored representations: ORIGINAL 96-concept set, fixed 0.66 layer and
      best-by-category layer, 6 models) and recomputes the v2/v3 numbers. Exit 0 iff every
      claimed number matches at the 3 decimals the lab published.

  py reproduce.py --extract OUTDIR [--set fresh|original|both] [--minilm]
      PATH B (downloads ~31 GB of models; needs torch + transformers, HF access to the gated
      Llama-3.2 and gemma-2 repos). Re-extracts representations and recomputes. --set fresh
      reproduces the PRE-REGISTERED out-of-sample CONFIRM (headline 0.258), which CANNOT be
      re-derived from committed files: the confirm run never saved its representations.

  py reproduce.py --tokens-from PATH/TO/Llama-3.2-1B-Instruct/tokenizer.json
      Recompute the frozen lexical token counts (lex_tokens.json) with the `tokenizers` package
      and compare. The lab used the Llama-3.2-1B-Instruct tokenizer (gated repo) for the
      token-count lexical control; the counts are frozen in lex_tokens.json so PATH A needs no
      gated download.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
NPZ_SHA256 = "88b73512671a592240a7da8db789455de5f34091bbe5e379550c1ca18d1ffc4d"

# (display name, HF repo, family, revision present in the lab's HF cache on 2026-11-10;
#  the 2026-06-03 runs did not record revisions, so these are NOT proven to be the ones used)
MODELS = [
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "qwen", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "qwen", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "llama", "9213176726f574b556790deb65791e0c5aa438b6"),
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", "llama", "0cb88a4f764b7a12671c53f0838cd831a0843b95"),
    ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "phi", "2fe192450127e6a83f7441aef6e3ca586c338b77"),
    ("gemma-2-2b", "google/gemma-2-2b-it", "gemma", None),  # no longer in the lab's cache
]
FAMILY = {m[0]: m[2] for m in MODELS}

ORIGINAL = {  # run_real_convergence.py CATEGORIES (v1/v2/v3)
    "animal": ["dog", "cat", "horse", "elephant", "lion", "tiger", "bear", "wolf", "rabbit", "mouse", "eagle", "shark"],
    "fruit": ["apple", "banana", "orange", "grape", "lemon", "peach", "cherry", "mango", "melon", "plum", "strawberry", "pear"],
    "vehicle": ["car", "truck", "bus", "train", "airplane", "bicycle", "boat", "ship", "motorcycle", "helicopter", "submarine", "scooter"],
    "profession": ["doctor", "teacher", "lawyer", "engineer", "nurse", "farmer", "pilot", "chef", "artist", "soldier", "scientist", "judge"],
    "body": ["hand", "foot", "head", "eye", "ear", "nose", "mouth", "arm", "leg", "heart", "brain", "finger"],
    "weather": ["rain", "snow", "wind", "storm", "sunshine", "cloud", "fog", "thunder", "lightning", "frost", "drizzle", "hail"],
    "furniture": ["chair", "table", "bed", "sofa", "desk", "shelf", "lamp", "mirror", "cabinet", "stool", "bench", "drawer"],
    "instrument": ["guitar", "piano", "violin", "drum", "flute", "trumpet", "harp", "cello", "clarinet", "saxophone", "banjo", "organ"],
}
FRESH = {  # run_real_convergence_confirm.py FRESH (pre-registered confirm)
    "tool": ["hammer", "screwdriver", "wrench", "drill", "saw", "pliers", "chisel", "axe", "shovel", "rake", "ladder", "clamp"],
    "clothing": ["shirt", "pants", "jacket", "dress", "hat", "scarf", "gloves", "socks", "shoes", "coat", "sweater", "belt"],
    "sport": ["soccer", "tennis", "basketball", "baseball", "golf", "hockey", "boxing", "swimming", "cycling", "skiing", "surfing", "wrestling"],
    "kitchen": ["spoon", "fork", "knife", "plate", "bowl", "cup", "pot", "pan", "kettle", "blender", "oven", "fridge"],
    "building": ["house", "tower", "bridge", "castle", "church", "factory", "stadium", "museum", "hospital", "library", "school", "prison"],
    "plant": ["oak", "pine", "rose", "tulip", "fern", "cactus", "bamboo", "maple", "ivy", "moss", "daisy", "willow"],
    "drink": ["water", "coffee", "tea", "juice", "milk", "wine", "beer", "soda", "lemonade", "cocoa", "cider", "smoothie"],
    "emotion": ["joy", "anger", "fear", "sadness", "surprise", "disgust", "love", "hope", "pride", "shame", "envy", "guilt"],
}
TEMPLATES = ["{w}", "a {w}", "the {w}", "I saw a {w}", "there is a {w}",
             "this is a {w}", "they showed me the {w}", "look at that {w}"]

# Claimed values, copied from the committed result JSONs (also checked against the copies in this kit).
CLAIMS_A = [
    # (label, source json, key path, set, layer, stat)
    ("v3 fixed-layer cross-family RSA raw", "real_convergence_v3_controls_result.json", ["fixed_layer", "summary", "xfam_raw"], "fixed", "xfam_raw"),
    ("v3 fixed-layer cross-family RSA partial-lex", "real_convergence_v3_controls_result.json", ["fixed_layer", "summary", "xfam_partial_lex"], "fixed", "xfam_partial_lex"),
    ("v3 fixed-layer all-pairs RSA raw", "real_convergence_v3_controls_result.json", ["fixed_layer", "summary", "mean_raw"], "fixed", "mean_raw"),
    ("v3 fixed-layer all-pairs RSA partial-lex", "real_convergence_v3_controls_result.json", ["fixed_layer", "summary", "mean_partial_lex"], "fixed", "mean_partial_lex"),
    ("v3 best-layer cross-family RSA raw", "real_convergence_v3_controls_result.json", ["best_layer", "summary", "xfam_raw"], "best", "xfam_raw"),
    ("v3 best-layer cross-family RSA partial-lex", "real_convergence_v3_controls_result.json", ["best_layer", "summary", "xfam_partial_lex"], "best", "xfam_partial_lex"),
    ("v2 fixed-layer shuffled control mean", "real_convergence_v2_result.json", ["fixed_layer_rsa", "ctrl_mean"], "fixed", "ctrl_mean"),
    ("v2 fixed-layer shuffled control sd", "real_convergence_v2_result.json", ["fixed_layer_rsa", "ctrl_sd"], "fixed", "ctrl_sd"),
    ("v2 best-layer shuffled control mean", "real_convergence_v2_result.json", ["best_layer_rsa", "ctrl_mean"], "best", "ctrl_mean"),
    ("v2 best-layer shuffled control sd", "real_convergence_v2_result.json", ["best_layer_rsa", "ctrl_sd"], "best", "ctrl_sd"),
    ("v2 best-layer same-family RSA", "real_convergence_v2_result.json", ["best_layer_rsa", "samefam"], "best", "samefam_raw"),
]
CLAIMS_CONFIRM = [
    ("CONFIRM cross-family RSA partial-lex (headline)", ["xfam_partial_lex"], "xfam_partial_lex"),
    ("CONFIRM cross-family RSA raw", ["xfam_raw"], "xfam_raw"),
    ("CONFIRM shuffled control mean", ["control_mean"], "ctrl_mean"),
    ("CONFIRM shuffled control sd", ["control_sd"], "ctrl_sd"),
]


# ---------------- analysis (copied from the lab's runners) ----------------
def distmat(R):
    R = R - R.mean(0)
    R = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
    G = R @ R.T
    return np.sqrt(np.maximum(2.0 - 2.0 * G, 0.0))


def cat_structure(D, cats):
    n = len(cats)
    same = cats[:, None] == cats[None, :]
    return float(D[~same].mean() / (D[same & ~np.eye(n, dtype=bool)].mean() + 1e-9))


def partial_corr(a, b, Z):
    X = np.column_stack([np.ones(len(a)), Z])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    return float(np.corrcoef(ra, rb)[0, 1])


def flatten(catdict):
    words, cats = [], []
    for c, ws in catdict.items():
        words += ws
        cats += [c] * len(ws)
    return words, np.array(cats)


def lex_design(words, toklen, iu):
    charlen = np.array([len(w) for w in words], dtype=float)
    toklen = np.asarray(toklen, dtype=float)
    zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9)
    zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
    return np.column_stack([np.abs(zc[:, None] - zc[None, :])[iu], np.abs(zt[:, None] - zt[None, :])[iu]])


def analyze(reps, words, cats, toklen, anchor_D=None):
    """reps: dict name->(N,d) in MODELS order. Returns the lab's summary stats, unrounded."""
    n = len(words)
    iu = np.triu_indices(n, 1)
    Z = lex_design(words, toklen, iu)
    models = list(reps)
    Ds = {m: distmat(reps[m]) for m in models}
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            rows.append({"a": a, "b": b, "xfam": FAMILY[a] != FAMILY[b],
                         "raw": float(np.corrcoef(Ds[a][iu], Ds[b][iu])[0, 1]),
                         "partial_lex": partial_corr(Ds[a][iu], Ds[b][iu], Z)})
    xf = [r for r in rows if r["xfam"]]
    sf = [r for r in rows if not r["xfam"]]
    rng = np.random.default_rng(0)  # control exactly as the lab ran it: models[0] vs permuted models[1]
    ctrl = [float(np.corrcoef(Ds[models[0]][iu], distmat(reps[models[1]][rng.permutation(n)])[iu])[0, 1]) for _ in range(50)]
    out = {
        "n_concepts": n, "models": models, "n_pairs": len(rows), "n_xfam_pairs": len(xf),
        "mean_raw": float(np.mean([r["raw"] for r in rows])),
        "mean_partial_lex": float(np.mean([r["partial_lex"] for r in rows])),
        "xfam_raw": float(np.mean([r["raw"] for r in xf])),
        "xfam_partial_lex": float(np.mean([r["partial_lex"] for r in xf])),
        "xfam_partial_lex_median": float(np.median([r["partial_lex"] for r in xf])),
        "xfam_partial_lex_min": float(np.min([r["partial_lex"] for r in xf])),
        "samefam_raw": float(np.mean([r["raw"] for r in sf])) if sf else None,
        "ctrl_mean": float(np.mean(ctrl)), "ctrl_sd": float(np.std(ctrl)),
        "cat_struct": {m: cat_structure(Ds[m], cats) for m in models},
        "pairs": sorted(rows, key=lambda r: -r["raw"]),
    }
    if anchor_D is not None:
        out["minilm_anchor_partial_lex"] = float(np.mean([partial_corr(Ds[m][iu], anchor_D[iu], Z) for m in models]))
    return out


def dig(d, path):
    for k in path:
        d = d[k]
    return d


def compare(label, claimed, got):
    ok = claimed is not None and got is not None and round(got, 3) == round(float(claimed), 3)
    print(f"  {'MATCH' if ok else 'DIFF '}  {label:52s} claimed {claimed!s:>8}  got {got:+.6f}")
    return {"what": label, "claimed": claimed, "got": round(got, 6), "match_3dp": ok}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- PATH A ----------------
def path_a(args):
    npz = Path(args.npz) if args.npz else KIT / "contextual_reps.npz"
    digest = sha256(npz)
    print(f"contextual_reps.npz sha256 {digest} ({'as committed' if digest == NPZ_SHA256 else 'DIFFERS from committed'})")
    toks = json.loads((KIT / "lex_tokens.json").read_text(encoding="utf-8"))
    words, cats = flatten(ORIGINAL)
    z = np.load(npz)
    res = {}
    for layer in ("fixed", "best"):
        reps = {m[0]: z[f"{layer}__{m[0]}"].astype(np.float64) for m in MODELS}
        res[layer] = analyze(reps, words, cats, toks["original"]["counts"])
    print("\nPATH A: original 96-concept set, stored v3 representations (6 models, 13 cross-family pairs)\n")
    report = []
    for label, src, path, layer, stat in CLAIMS_A:
        claimed = dig(json.loads((KIT / src).read_text(encoding="utf-8")), path)
        report.append(compare(label, claimed, res[layer][stat]))
    # per-pair check against v3 fixed / best pairs
    v3 = json.loads((KIT / "real_convergence_v3_controls_result.json").read_text(encoding="utf-8"))
    pair_ok = pair_n = 0
    for layer, key in (("fixed", "fixed_layer"), ("best", "best_layer")):
        got = {(r["a"], r["b"]): r for r in res[layer]["pairs"]}
        for c in v3[key]["pairs"]:
            g = got[(c["a"], c["b"])]
            pair_n += 2
            pair_ok += (round(g["raw"], 3) == c["raw"]) + (round(g["partial_lex"], 3) == c["partial_lex"])
    print(f"  {'MATCH' if pair_ok == pair_n else 'DIFF '}  {'v3 per-pair values (raw + partial, 2 layers)':52s} {pair_ok}/{pair_n} equal at 3 dp")
    report.append({"what": "v3 per-pair values", "claimed": f"{pair_n} values", "got": f"{pair_ok}/{pair_n} equal", "match_3dp": pair_ok == pair_n})
    f = res["fixed"]
    print(f"\n  not claimed in the committed JSONs, computed here: fixed-layer cross-family partial-lex median "
          f"{f['xfam_partial_lex_median']:+.3f}, min {f['xfam_partial_lex_min']:+.3f}")
    print("  cat_struct (across/within distance) fixed layer: " + ", ".join(f"{m} {v:.3f}" for m, v in f["cat_struct"].items()))
    print("\n  NOT re-derivable here: CONFIRM headline 0.258 / median 0.166 / control -0.002+-0.011 (fresh-set reps were")
    print("  never saved; use --extract OUTDIR --set fresh), and the MiniLM anchor values (need the MiniLM model; use --minilm).")
    n_ok = sum(r["match_3dp"] for r in report)
    print(f"\nRESULT: {n_ok}/{len(report)} checks match at 3 decimals")
    out = {"mode": "A", "npz_sha256": digest, "checks": report,
           "computed": {k: {s: v for s, v in r.items() if s != "pairs"} for k, r in res.items()}}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0 if n_ok == len(report) else 1


# ---------------- PATH B (not run by the lab's packager) ----------------
def path_b(args):
    import gc
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    outdir = Path(args.extract)
    outdir.mkdir(parents=True, exist_ok=True)
    sets = {"fresh": FRESH, "original": ORIGINAL}
    todo = ["fresh", "original"] if args.set == "both" else [args.set]
    toks = json.loads((KIT / "lex_tokens.json").read_text(encoding="utf-8"))

    @torch.no_grad()
    def concept_all_layers(mdl, tok, word):
        acc = None
        for t in TEMPLATES:
            ids = tok(t.format(w=word), return_tensors="pt").input_ids.to(mdl.device)
            out = mdl(input_ids=ids, output_hidden_states=True, use_cache=False)
            hs = torch.stack([h[0, -1] for h in out.hidden_states], 0).float().cpu().numpy()
            acc = hs if acc is None else acc + hs
        return acc / len(TEMPLATES)

    store = {s: {} for s in todo}
    for name, repo, fam, rev in MODELS:
        kw = {"revision": rev} if (rev and args.pin_revisions) else {}
        tok = AutoTokenizer.from_pretrained(repo, **kw)
        mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **kw).to(dev).eval()
        L = mdl.config.num_hidden_layers
        fl = max(1, int(0.66 * L))
        for s in todo:
            words, cats = flatten(sets[s])
            allrep = np.stack([concept_all_layers(mdl, tok, w) for w in words])
            cs = [cat_structure(distmat(allrep[:, l, :]), cats) for l in range(allrep.shape[1])]
            bl = int(np.argmax(cs))
            store[s][f"fixed__{name}"] = allrep[:, fl, :]
            store[s][f"best__{name}"] = allrep[:, bl, :]
            print(f"{name:14s} [{s}] fixed L{fl}/{L} best L{bl}", flush=True)
        del mdl, tok
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()
    anchor = {}
    if args.minilm:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
        for s in todo:
            anchor[s] = distmat(st.encode(flatten(sets[s])[0], normalize_embeddings=True))
    report = []
    for s in todo:
        np.savez(outdir / f"reps_{s}.npz", **store[s])
        words, cats = flatten(sets[s])
        reps = {m[0]: store[s][f"fixed__{m[0]}"].astype(np.float64) for m in MODELS}
        r = analyze(reps, words, cats, toks[s]["counts"], anchor.get(s))
        print(f"\n[{s}] fixed 0.66 layer: xfam raw {r['xfam_raw']:+.3f} partial-lex {r['xfam_partial_lex']:+.3f} "
              f"median {r['xfam_partial_lex_median']:+.3f} control {r['ctrl_mean']:+.3f}+-{r['ctrl_sd']:.3f}"
              + (f" MiniLM-anchor {r['minilm_anchor_partial_lex']:+.3f}" if "minilm_anchor_partial_lex" in r else ""))
        if s == "fresh":
            c = json.loads((KIT / "real_convergence_confirm_result.json").read_text(encoding="utf-8"))
            for label, path, stat in CLAIMS_CONFIRM:
                report.append(compare(label, dig(c, path), r[stat]))
            report.append(compare("CONFIRM cross-family partial-lex median (RESULT.md)", 0.166, r["xfam_partial_lex_median"]))
            if "minilm_anchor_partial_lex" in r:
                report.append(compare("CONFIRM MiniLM anchor partial-lex", c["minilm_anchor_partial_lex"], r["minilm_anchor_partial_lex"]))
            gate = r["xfam_partial_lex"] >= 0.25 and r["xfam_partial_lex"] >= 5 * max(abs(r["ctrl_mean"]), 0.02) and \
                ("minilm_anchor_partial_lex" not in r or r["minilm_anchor_partial_lex"] >= 0.25)
            print(f"[fresh] frozen gate (xfam partial >=0.25, >=5x control, anchor >=0.25): {'PASS' if gate else 'FAIL'}")
        else:
            v3 = json.loads((KIT / "real_convergence_v3_controls_result.json").read_text(encoding="utf-8"))
            report.append(compare("v3 fixed-layer cross-family partial-lex", v3["fixed_layer"]["summary"]["xfam_partial_lex"], r["xfam_partial_lex"]))
    Path(args.out).write_text(json.dumps({"mode": "B", "checks": report}, indent=2), encoding="utf-8")
    print(f"wrote {args.out}; representations in {outdir}")
    return 0


def tokens_from(path):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(path))
    frozen = json.loads((KIT / "lex_tokens.json").read_text(encoding="utf-8"))
    ok = True
    for s, d in (("original", ORIGINAL), ("fresh", FRESH)):
        words = flatten(d)[0]
        got = [len(tk.encode(" " + w, add_special_tokens=False).ids) for w in words]
        same = got == frozen[s]["counts"]
        ok &= same
        print(f"{s}: {sum(a == b for a, b in zip(got, frozen[s]['counts']))}/{len(words)} token counts equal the frozen file")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="PATH A input (default: contextual_reps.npz beside this script)")
    ap.add_argument("--out", default="reproduce_result.json", help="result file, written to the CURRENT directory")
    ap.add_argument("--extract", metavar="OUTDIR", help="PATH B: download models and re-extract")
    ap.add_argument("--set", choices=["fresh", "original", "both"], default="fresh")
    ap.add_argument("--minilm", action="store_true", help="PATH B: also compute the MiniLM anchor")
    ap.add_argument("--pin-revisions", action="store_true", help="PATH B: load the cache revisions listed in MODELS")
    ap.add_argument("--tokens-from", metavar="TOKENIZER_JSON")
    a = ap.parse_args()
    if a.tokens_from:
        return tokens_from(a.tokens_from)
    if a.extract:
        return path_b(a)
    return path_a(a)


if __name__ == "__main__":
    sys.exit(main())
