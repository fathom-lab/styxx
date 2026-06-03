# -*- coding: utf-8 -*-
"""
run_abstract_depth_simlex.py — does depth (deep LLM > shallow GloVe) matter MORE for ABSTRACT meaning?

The concrete-noun work found the deep LLM beats GloVe by +12.8% at human behavioral similarity. The
operator's deepest point: abstract meaning (justice, anger, argument) is where "universal form" lives,
and co-occurrence is a weaker proxy there. Falsifiable: the deep-model advantage over GloVe should be
LARGER for abstract word pairs than concrete ones.

Ground truth = SimLex-999 (human similarity ratings, with USF concreteness per word). For each pair,
predicted similarity from GloVe (cosine) and a deep-LLM consensus (final-layer cosine); Spearman vs the
human rating, split by concreteness. Gate: LLM-minus-GloVe Spearman is larger for abstract than concrete.
"""
from __future__ import annotations

import gc, io, json, ssl, urllib.request, zipfile
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
import sys
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import is_cached

LLMS = [("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"), ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct")]
TEMPLATES = ["{w}", "the {w}"]


def spearman(x, y):
    rx = np.argsort(np.argsort(np.asarray(x, float))); ry = np.argsort(np.argsort(np.asarray(y, float)))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    # SimLex-999
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    data = urllib.request.urlopen(urllib.request.Request("https://fh295.github.io/SimLex-999.zip", headers={"User-Agent": "Mozilla/5.0"}), context=ctx, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(data))
    txt = z.read("SimLex-999/SimLex-999.txt").decode("utf-8").splitlines()
    hdr = txt[0].split("\t")
    iw1, iw2, isl = hdr.index("word1"), hdr.index("word2"), hdr.index("SimLex999")
    ic1, ic2 = hdr.index("conc(w1)"), hdr.index("conc(w2)")
    pairs = []
    for line in txt[1:]:
        f = line.split("\t")
        pairs.append((f[iw1], f[iw2], float(f[isl]), (float(f[ic1]) + float(f[ic2])) / 2))
    words = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    print(f"SimLex: {len(pairs)} pairs, {len(words)} unique words", flush=True)

    import gensim.downloader as api
    glove = api.load("glove-wiki-gigaword-50")

    # deep-LLM consensus word reps (final layer, template-averaged)
    @torch.no_grad()
    def model_reps(repo):
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
        R = {}
        for w in words:
            acc = None
            for t in TEMPLATES:
                ids = tok(t.format(w=w), return_tensors="pt").input_ids.to(mdl.device)
                h = mdl(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-1][0, -1].float().cpu().numpy()
                acc = h if acc is None else acc + h
            R[w] = acc / len(TEMPLATES)
        del mdl, tok; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
        return R

    llm_reps = []
    for name, repo in LLMS:
        if is_cached(repo):
            llm_reps.append(model_reps(repo)); print(f"  {name} reps ok", flush=True)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    rows = []
    for w1, w2, human, conc in pairs:
        if w1 not in glove or w2 not in glove:
            continue
        g = cos(glove[w1], glove[w2])
        l = float(np.mean([cos(R[w1], R[w2]) for R in llm_reps]))
        rows.append({"human": human, "conc": conc, "glove": g, "llm": l})
    print(f"usable pairs (GloVe-covered): {len(rows)}", flush=True)

    concs = np.array([r["conc"] for r in rows])
    lo, hi = np.percentile(concs, 33), np.percentile(concs, 67)

    def block(sel, label):
        rs = [r for r in rows if sel(r["conc"])]
        h = [r["human"] for r in rs]
        sg = spearman([r["glove"] for r in rs], h); sl = spearman([r["llm"] for r in rs], h)
        return {"label": label, "n": len(rs), "glove_rho": round(sg, 3), "llm_rho": round(sl, 3), "llm_minus_glove": round(sl - sg, 3)}

    allb = block(lambda c: True, "ALL")
    absb = block(lambda c: c <= lo, f"ABSTRACT (conc<={lo:.1f})")
    conb = block(lambda c: c >= hi, f"CONCRETE (conc>={hi:.1f})")

    # bootstrap the abstract-minus-concrete depth advantage (is the ~2x real or noise?)
    abs_rows = [r for r in rows if r["conc"] <= lo]; con_rows = [r for r in rows if r["conc"] >= hi]
    def adv(rs):
        h = [r["human"] for r in rs]
        return spearman([r["llm"] for r in rs], h) - spearman([r["glove"] for r in rs], h)
    rng = np.random.default_rng(0); diffs = []
    for _ in range(2000):
        a = [abs_rows[i] for i in rng.integers(0, len(abs_rows), len(abs_rows))]
        c = [con_rows[i] for i in rng.integers(0, len(con_rows), len(con_rows))]
        diffs.append(adv(a) - adv(c))
    diffs = np.array(diffs)
    frac_pos = float((diffs > 0).mean()); ci = (round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3))
    print(f"\nbootstrap (abstract depth-adv minus concrete depth-adv): mean {diffs.mean():+.3f}, 95% CI {ci}, P(>0)={frac_pos:.3f}", flush=True)

    depth_more_abstract = (absb["llm_minus_glove"] > conb["llm_minus_glove"]) and (frac_pos >= 0.95)
    verdict = ((f"DEPTH MATTERS MORE FOR ABSTRACT MEANING: the deep-LLM advantage over shallow GloVe is "
                f"+{absb['llm_minus_glove']:.3f} for abstract pairs vs +{conb['llm_minus_glove']:.3f} for concrete "
                f"(human-similarity Spearman). Co-occurrence is a weaker proxy for abstract meaning; depth fills the gap.")
               if depth_more_abstract else
               (f"NOT confirmed: deep-LLM advantage is +{absb['llm_minus_glove']:.3f} abstract vs +{conb['llm_minus_glove']:.3f} "
                f"concrete -- depth does NOT help abstract more (or helps concrete more). Honest negative."))

    out = {"n_pairs": len(rows), "all": allb, "abstract": absb, "concrete": conb,
           "bootstrap_abstract_minus_concrete_advantage": {"mean": round(float(diffs.mean()), 3), "ci95": ci, "P_gt_0": round(frac_pos, 3)},
           "depth_helps_abstract_more": bool(depth_more_abstract), "verdict": verdict}
    (HERE / "abstract_depth_simlex_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n=== deep-LLM vs shallow GloVe at predicting HUMAN similarity (SimLex-999) ===")
    for b in (allb, absb, conb):
        print(f"  {b['label']:22s} n={b['n']:3d}  GloVe rho={b['glove_rho']:+.3f}  deep-LLM rho={b['llm_rho']:+.3f}  (LLM-GloVe {b['llm_minus_glove']:+.3f})")
    print(f"\n>>> {verdict}")
    print("wrote abstract_depth_simlex_result.json")


if __name__ == "__main__":
    main()
