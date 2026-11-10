"""mind_axis_check.py: does the shipped geometry axis (styxx.mind.geometry_citizenship) carry the start-token artifact?

Lead check, 2026-11-10. Not a sealed test: the arrays were already scored by the two sealed tests of today.
It runs the SHIPPED function, unchanged, on stored arrays. No model is loaded.

usage: python mind_axis_check.py <styxx working copy> <states dir (test 1)> <states2 dir (test 2)> <out.json>
"""
import hashlib
import json
import os
import sys
import tempfile

import numpy as np

repo, states1, states2, out_path = sys.argv[1:5]
sys.path.insert(0, repo)
from styxx import mind  # noqa: E402

assert os.path.abspath(mind.__file__).startswith(os.path.abspath(repo)), mind.__file__
JUNE = os.path.join(repo, "papers", "real-convergence", "contextual_reps.npz")
june = np.load(JUNE)


def top_share(x):
    xc = np.asarray(x, dtype=np.float64)
    xc = xc - xc.mean(0)
    s = np.linalg.svd(xc, compute_uv=False)
    return float(s[0] ** 2 / (s ** 2).sum())


def load(model):
    for d, pre in ((states2, "states2"), (states1, "states")):
        p = os.path.join(d, "%s__original__%s.npz" % (pre, model))
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            assert [str(w) for w in z["words"]] == list(mind.BATTERY), "battery order differs for " + model
            return z["states"].astype(np.float64), z["seq_len"], p
    return None, None, None


res = {"mind_file_sha256": hashlib.sha256(open(mind.__file__, "rb").read()).hexdigest(),
       "june_anchors_sha256": hashlib.sha256(open(JUNE, "rb").read()).hexdigest(),
       "templates": list(mind.TEMPLATES), "shares": [], "citizenship": []}

print("== share of the variance across the 96 battery concepts on one direction ==")
for name, _fam in mind.ANCHOR_MODELS:
    sh = top_share(june["fixed__" + name])
    res["shares"].append({"array": "June anchor fixed__" + name, "templates": 8, "share": sh})
    print("  June anchor %-14s (the shipped anchors)          %.4f" % (name, sh))

MODELS = [("Qwen2.5-0.5B", "qwen"), ("Qwen2.5-1.5B", "qwen"), ("Qwen2.5-3B", "qwen"),
          ("Llama-3.2-1B", "llama"), ("Llama-3.2-3B", "llama"), ("Phi-3.5-mini", "phi")]
rep8, rep7, alone = {}, {}, {}
for m, _fam in MODELS:
    st, seq, p = load(m)
    if st is None:
        print("  (no stored states for %s)" % m)
        continue
    rep8[m], rep7[m] = st.mean(1), st[:, 1:].mean(1)
    alone[m] = int((seq[:, 0] == 1).sum())
    s8, s7 = top_share(rep8[m]), top_share(rep7[m])
    res["shares"].append({"array": "re-extracted " + m, "templates": 8, "share": s8, "alone_at_position_0": alone[m]})
    res["shares"].append({"array": "re-extracted " + m, "templates": 7, "share": s7})
    print("  re-extracted %-14s 8 templates %.4f | bare template dropped %.4f | words alone at position 0: %d of 96" % (m, s8, s7, alone[m]))

# anchors with the bare template dropped, for the five anchor models that have stored states (gemma is missing)
tmp = tempfile.mkdtemp()
v2_path = os.path.join(tmp, "anchors_7.npz")
np.savez(v2_path, **{"fixed__" + m: rep7[m].astype(np.float32) for m, _f in MODELS if m != "Qwen2.5-0.5B" and m in rep7})

print()
print("== the shipped function, unchanged: citizenship = mean partial-lexical RSA over cross-family anchors ==")
for cand, fam in (("Qwen2.5-0.5B", "qwen"), ("Llama-3.2-1B", "llama")):
    if cand not in rep8:
        continue
    for label, reps, anchors in (
        ("as documented: 8 templates, shipped anchors", rep8[cand], JUNE),
        ("candidate with the bare template dropped, shipped anchors", rep7[cand], JUNE),
        ("bare template dropped in candidate AND anchors (five anchors)", rep7[cand], v2_path),
    ):
        out = mind.geometry_citizenship(reps, anchors, fam)
        row = {"candidate": cand, "family": fam, "setting": label, "citizenship": out["citizenship_xfam_partial_lex"],
               "per_anchor": {a["anchor"]: a["partial_lex_rsa"] for a in out["per_anchor"]}}
        res["citizenship"].append(row)
        print("  %-13s %-62s citizenship %s" % (cand, label, out["citizenship_xfam_partial_lex"]))
        print("      per anchor: " + "  ".join("%s %.3f" % (k, v) for k, v in row["per_anchor"].items()))

with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(res, f, indent=1)
    f.write("\n")
print()
print("wrote", os.path.basename(out_path))
