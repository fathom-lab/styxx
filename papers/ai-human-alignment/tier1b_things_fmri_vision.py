# -*- coding: utf-8 -*-
"""TIER 1b — the vision-confound GATE for the THINGS-fMRI IT result.
IT is high-level VISUAL cortex; subjects saw photographs. Does the text-only LLM predict IT geometry BEYOND
a vision model? Reuse the exact Mitchell control: a CLIP-image RDM (clip-ViT-B-32 over each concept's THINGS
image) partialled out of the LLM<->IT RSA. CLIP-image is vision-LANGUAGE (carries semantics) → conservative.
Honest, hard test: IT is mostly visual, so the LLM<->IT match may survive or may collapse. Reproduce:
tier1b_things_fmri_vision.py → tier1b_things_fmri_vision_result.json."""
import sys, json
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
TF = HERE / "things_fmri" / "betas_testset" / "betas_csv_testset"
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_brain_vision import get_clip_image_rdm          # reuse the committed CLIP-image pipeline
import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
DEV = "cuda" if torch.cuda.is_available() else "cpu"

concept_of = lambda s: s.rsplit("_", 1)[0].lower()
subs = ["sub-01", "sub-02", "sub-03"]
neural, csets = [], []
for s in subs:
    a = np.load(TF / f"{s}_TestResponsesIT.npy").astype(np.float64)
    df = pd.read_csv(TF / f"{s}_StimulusMetadataTestset.csv"); df["concept"] = df["stimulus"].map(concept_of)
    cs = sorted(df["concept"].unique())
    M = np.stack([a[df["concept"].values == c].mean(0) for c in cs]); M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    neural.append(1.0 - np.corrcoef(M)); csets.append(cs)
assert csets[0] == csets[1] == csets[2]
concepts = csets[0]; group = np.mean(neural, axis=0)

# CLIP-image RDM over the concepts that have a THINGS image
kept, vis_rdm = get_clip_image_rdm(concepts)
ix = [concepts.index(w) for w in kept]
print(f"vision: {len(kept)}/{len(concepts)} concepts have a CLIP-image vector; analysis set n={len(kept)}", flush=True)
group_k = group[np.ix_(ix, ix)]
N = len(kept); IU = np.triu_indices(N, 1)

# LLM RDM over kept
tok = AutoTokenizer.from_pretrained("gpt2-large")
mdl = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEV).eval()
llm = distmat(np.stack([concept_all_layers(mdl, tok, w)[-1] for w in kept])); del mdl; gc.collect()

rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
cl = np.array([len(w) for w in kept], float); tl = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in kept], float)
zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
def lexZ(ii, jj): return np.column_stack([np.abs(zc[ii] - zc[jj]), np.abs(zt[ii] - zt[jj])])
L = lexZ(*IU); vis_v = vis_rdm[IU]; brain_v = group_k[IU]; llm_v = llm[IU]

casc = {
    "raw":         float(partial_corr(llm_v, brain_v, np.zeros((len(IU[0]), 0)))),
    "lex":         float(partial_corr(llm_v, brain_v, L)),
    "lex+vision":  float(partial_corr(llm_v, brain_v, np.column_stack([L, vis_v]))),
}
refs = {
    "vision->IT | lex":              float(partial_corr(vis_v, brain_v, L)),
    "LLM->vision | lex (how visual)": float(partial_corr(llm_v, vis_v, L)),
}

def boot_lexvis(B=2000):
    rng = np.random.default_rng(0); v = []
    for _ in range(B):
        idx = rng.integers(0, N, N); ii, jj = np.triu_indices(N, 1)
        keep = idx[ii] != idx[jj]; a, b = idx[ii][keep], idx[jj][keep]
        Z = np.column_stack([lexZ(a, b), vis_rdm[a, b]])
        v.append(float(partial_corr(llm[a, b], group_k[a, b], Z)))
    return round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)

ci = boot_lexvis()
passed = casc["lex+vision"] > 0 and ci[0] > 0
out = {"n": N, "cascade": {k: round(v, 3) for k, v in casc.items()}, "lex+vision_CI": ci,
       "references": {k: round(v, 3) for k, v in refs.items()}, "survives_vision_control": bool(passed)}
(HERE / "tier1b_things_fmri_vision_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"\n=== LLM -> THINGS-fMRI IT, vision cascade (n={N}) ===")
for k, v in casc.items(): print(f"  partial(LLM, IT | {k:11s}) = {v:+.3f}")
print(f"  lex+vision 95% CI {ci}")
for k, v in refs.items(): print(f"  {k:32s} {v:+.3f}")
print(f"\n>>> {'SURVIVES' if passed else 'DOES NOT survive'} the vision control: "
      f"LLM->IT {casc['lex']:.3f} (|lex) -> {casc['lex+vision']:.3f} (|lex+VISION), CI {ci}. "
      f"{'Meaning beyond pixels.' if passed else 'The IT match is largely visual — honest bound; needs a non-visual ROI.'}")
print("wrote tier1b_things_fmri_vision_result.json")
