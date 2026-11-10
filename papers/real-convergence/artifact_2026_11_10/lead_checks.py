"""lead_checks.py: the lead's own re-derivation of the skeptic's main claims about the geometry result.

Lead-run, 2026-11-10. Written without reading skeptic_checks.py or reproduce.py. Reads two committed files
(copied byte-identical into the kit folder): contextual_reps.npz and real_convergence_confirm_result.json.
Writes one json next to itself. numpy only.

Usage: py -3.12 -X utf8 -I -B lead_checks.py <kit_dir> <out.json>

Method, from REPLICATE.md prose: per model, centre the 96 x d matrix across concepts, unit-normalise rows,
distance = sqrt(2 - 2 cos); RSA = Pearson r of two models' upper triangles (4560 entries). Raw only:
the lexical partial needs token counts and is not re-derived here.
"""
import itertools
import json
import os
import statistics
import sys

import numpy as np

KIT, OUT = sys.argv[1], sys.argv[2]
FAMILY = {"Qwen2.5-1.5B": "qwen", "Qwen2.5-3B": "qwen", "Llama-3.2-1B": "llama", "Llama-3.2-3B": "llama",
          "Phi-3.5-mini": "phi", "gemma-2-2b": "gemma"}
MODELS = list(FAMILY)
out = {}

# ---- part 1: arithmetic on the committed confirm json (rounded pair values; not a re-derivation) ----
cj = json.load(open(os.path.join(KIT, "real_convergence_confirm_result.json"), encoding="utf-8"))
xf = [p for p in cj["pairs"] if p["xfam"]]
vals = [p["partial_lex"] for p in xf]
is_lg = lambda p: {FAMILY[p["a"]], FAMILY[p["b"]]} == {"llama", "gemma"}
rest = [p["partial_lex"] for p in xf if not is_lg(p)]
drop = {}
for m in MODELS:
    keep = [p["partial_lex"] for p in xf if m not in (p["a"], p["b"])]
    drop[m] = {"n_pairs": len(keep), "mean": round(sum(keep) / len(keep), 4)}
out["confirm_json"] = {
    "n_pairs": len(cj["pairs"]), "n_xfam": len(xf),
    "xfam_partial_mean": round(sum(vals) / len(vals), 4), "xfam_partial_median": statistics.median(vals),
    "xfam_at_or_above_0.25": sum(v >= 0.25 for v in vals),
    "n_llama_gemma_pairs": len(xf) - len(rest), "mean_without_llama_gemma": round(sum(rest) / len(rest), 4),
    "drop_one_model": drop, "drops_below_0.25": sum(d["mean"] < 0.25 for d in drop.values()),
    "stated_gate": cj.get("frozen_gate"), "stated_xfam_partial_lex": cj.get("xfam_partial_lex"),
}

# ---- part 2: the stored arrays (original 96 concepts, exploratory) ----
z = np.load(os.path.join(KIT, "contextual_reps.npz"))


def centred(x):
    x = x.astype(np.float64)
    return x - x.mean(axis=0, keepdims=True)


def dist_upper(xc):
    n = xc / np.linalg.norm(xc, axis=1, keepdims=True)
    cos = np.clip(n @ n.T, -1.0, 1.0)
    d = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cos))
    iu = np.triu_indices(d.shape[0], k=1)
    return d[iu]


def remove_top(xc, k):
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    s2 = s.copy()
    s2[:k] = 0.0
    return (u * s2) @ vt


def xfam_table(reps):
    ups = {m: dist_upper(reps[m]) for m in MODELS}
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        r = float(np.corrcoef(ups[a], ups[b])[0, 1])
        rows.append((a, b, FAMILY[a] != FAMILY[b], r))
    return rows


def summarise(rows):
    x = [r for a, b, f, r in rows if f]
    return {"n_xfam": len(x), "mean": round(float(np.mean(x)), 4), "median": round(float(np.median(x)), 4),
            "min": round(min(x), 4), "max": round(max(x), 4), "at_or_above_0.25": sum(v >= 0.25 for v in x)}


for layer in ("fixed", "best"):
    cen = {m: centred(z["%s__%s" % (layer, m)]) for m in MODELS}
    pc = {}
    for m in MODELS:
        s = np.linalg.svd(cen[m], compute_uv=False)
        var = s ** 2
        colvar = (cen[m] ** 2).sum(axis=0)
        pc[m] = {"pc1_share": round(float(var[0] / var.sum()), 4),
                 "top_single_dimension_share": round(float(colvar.max() / colvar.sum()), 4)}
    base = xfam_table(cen)
    no1 = xfam_table({m: remove_top(cen[m], 1) for m in MODELS})
    no3 = xfam_table({m: remove_top(cen[m], 3) for m in MODELS})
    out[layer] = {"pc1": pc, "raw_xfam": summarise(base), "raw_xfam_top1_removed": summarise(no1),
                  "raw_xfam_top3_removed": summarise(no3),
                  "pairs_raw": [{"a": a, "b": b, "xfam": f, "r": round(r, 4)} for a, b, f, r in base]}

json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
c = out["confirm_json"]
print("CONFIRM JSON (rounded pair values): xfam pairs %d, mean %.4f, median %.3f, at or above 0.25: %d of %d"
      % (c["n_xfam"], c["xfam_partial_mean"], c["xfam_partial_median"], c["xfam_at_or_above_0.25"], c["n_xfam"]))
print("  without the %d Llama-gemma pairs: mean %.4f" % (c["n_llama_gemma_pairs"], c["mean_without_llama_gemma"]))
print("  drop one model: " + ", ".join("%s %.3f" % (m, d["mean"]) for m, d in c["drop_one_model"].items()))
print("  drops below 0.25: %d of %d" % (c["drops_below_0.25"], len(MODELS)))
for layer in ("fixed", "best"):
    o = out[layer]
    print("STORED ARRAYS, %s layer, raw (lab method as written):" % layer)
    print("  pc1 share: " + ", ".join("%s %.1f%%" % (m, 100 * v["pc1_share"]) for m, v in o["pc1"].items()))
    print("  top single dimension share: " + ", ".join(
        "%s %.0f%%" % (m, 100 * v["top_single_dimension_share"]) for m, v in o["pc1"].items()))
    for k in ("raw_xfam", "raw_xfam_top1_removed", "raw_xfam_top3_removed"):
        print("  %-24s %s" % (k, o[k]))
print("wrote", OUT)
