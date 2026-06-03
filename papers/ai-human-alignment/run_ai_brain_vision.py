# -*- coding: utf-8 -*-
"""
run_ai_brain_vision.py — the decisive control on the AI<->brain result: MEANING or pixels?

Mitchell stimuli were word + line-drawing, so the brain RDM carries VISUAL structure. This isolates
the semantic component: build a VISION RDM (CLIP-image features over the actual THINGS object images)
and a variance partition of the brain RDM across AI(semantic) + vision + behavioral(VICE) + lexical.

FROZEN QUESTION / BAR (stated before the partition is computed):
  Does the LLM predict the human brain BEYOND a vision model + word-form?
  PASS iff partial-corr(AI, brain | lexical + vision) >= 0.10 AND clearly > 0, AND the LLM's UNIQUE
  brain-variance beyond vision+lexical is > 0. Note CLIP-image is a vision-LANGUAGE model (carries
  semantics), so partialling it out is CONSERVATIVE -- survival is a strong meaning result.
"""
from __future__ import annotations

import gc, io, json, ssl, urllib.request
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
BRAIN = HERE / "brain"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_CTX = ssl.create_default_context(); _CTX.check_hostname = False; _CTX.verify_mode = ssl.CERT_NONE

import sys
sys.path.insert(0, str(HERE.parent / "real-convergence"))
from run_real_convergence import distmat, is_cached
from run_real_convergence_v2 import concept_all_layers
from run_real_convergence_v3_controls import partial_corr
from run_ai_human import COHORT, spearman


def r2(y, *cols):
    X = np.column_stack([np.ones_like(y)] + list(cols))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ beta
    return float(1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())))


def get_clip_image_rdm(nouns):
    """CLIP-image features over THINGS example images; cached. Returns (kept_nouns, RDM)."""
    cache = BRAIN / "clip_image_emb.npz"
    rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
    url_of = {r.split("\t")[0].strip().lower(): r.split("\t")[2].strip() for r in rows}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        embs = {str(k): z[k] for k in z.files}
    else:
        embs = {}
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    need = [w for w in nouns if w in url_of and w not in embs]
    if need:
        clip = SentenceTransformer("clip-ViT-B-32", device=DEV)
        for w in need:
            iid = url_of[w].rstrip("/").split("/")[-1]
            for ext in (".jpg", ".png", ".jpeg"):
                try:
                    data = urllib.request.urlopen(urllib.request.Request(f"https://i.imgur.com/{iid}{ext}",
                            headers={"User-Agent": "Mozilla/5.0"}), context=_CTX, timeout=25).read()
                    im = Image.open(io.BytesIO(data)).convert("RGB")
                    if im.size[0] > 50:
                        embs[w] = clip.encode([im], normalize_embeddings=True)[0]
                        break
                except Exception:
                    pass
        np.savez(cache, **embs)
        del clip; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
    kept = [w for w in nouns if w in embs]
    V = np.stack([embs[w] for w in kept])
    return kept, distmat(V)


def main():
    bz = np.load(HERE / "brain_rdm.npz", allow_pickle=True)
    nouns = [str(w) for w in bz["nouns"]]
    brain = bz["group"]
    ceil_lo = float(bz["ceiling"][0])

    # vision RDM (CLIP-image) -> the subset of nouns with an image defines the analysis set
    vnouns, vis_rdm_full = get_clip_image_rdm(nouns)
    print(f"vision: {len(vnouns)}/{len(nouns)} nouns have a CLIP-image vector", flush=True)

    # VICE behavioral over vnouns; restrict to nouns present in THINGS too
    rows = (DATA / "things_concepts.tsv").read_text(encoding="utf-8").splitlines()[1:]
    tindex = {r.split("\t")[0].strip().lower(): i for i, r in enumerate(rows)}
    vice = np.load(DATA / "final_embedding.npy")
    S = [w for w in vnouns if w in tindex]
    si_brain = [nouns.index(w) for w in S]
    si_vis = [vnouns.index(w) for w in S]
    N = len(S); IU = np.triu_indices(N, 1)
    print(f"analysis set: {N} nouns (vision inter VICE inter brain)", flush=True)

    rtok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    cl = np.array([len(w) for w in S], float); tl = np.array([len(rtok(" " + w, add_special_tokens=False).input_ids) for w in S], float)
    zc = (cl - cl.mean()) / (cl.std() + 1e-9); zt = (tl - tl.mean()) / (tl.std() + 1e-9)
    L = np.column_stack([np.abs(zc[:, None] - zc[None, :])[IU], np.abs(zt[:, None] - zt[None, :])[IU]])

    brain_v = brain[np.ix_(si_brain, si_brain)][IU]
    vis_v = vis_rdm_full[np.ix_(si_vis, si_vis)][IU]
    vice_v = distmat(vice[[tindex[w] for w in S]])[IU]

    # AI consensus RDM (mean of cohort final-layer RDMs) over S
    from sentence_transformers import SentenceTransformer
    mpnet = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=DEV)
    mpnet_v = distmat(mpnet.encode(S, normalize_embeddings=True))[IU]; del mpnet
    gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None

    rdms = []
    best_single = None; best_val = -9
    for name, repo, params, instruct in COHORT:
        if not is_cached(repo):
            continue
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, trust_remote_code=True).to(DEV).eval()
            rep = np.stack([concept_all_layers(mdl, tok, w)[-1] for w in S])
            rd = distmat(rep)[IU]
            rdms.append(rd)
            bv = partial_corr(rd, brain_v, L)
            if bv > best_val:
                best_val, best_single = bv, name
            del mdl, tok, rep; gc.collect(); torch.cuda.empty_cache() if DEV == "cuda" else None
            print(f"  {name:12s} ok", flush=True)
        except Exception as e:
            print(f"  {name:12s} FAIL {type(e).__name__}", flush=True)
    ai_v = np.mean(rdms, axis=0)  # consensus

    # --- partial-correlation cascade (AI consensus -> brain, adding controls) ---
    casc = {
        "raw": partial_corr(ai_v, brain_v, np.zeros((len(IU[0]), 0))),
        "lex": partial_corr(ai_v, brain_v, L),
        "lex+vision": partial_corr(ai_v, brain_v, np.column_stack([L, vis_v])),
        "lex+vision+behavioral": partial_corr(ai_v, brain_v, np.column_stack([L, vis_v, vice_v])),
    }
    ctrl_refs = {
        "vision->brain|lex": partial_corr(vis_v, brain_v, L),
        "behavioral->brain|lex": partial_corr(vice_v, brain_v, L),
        "AI->vision|lex (how visual is AI geom)": partial_corr(ai_v, vis_v, L),
        "AI->brain|lex (best single)": best_val,
    }

    # --- variance partition (R^2 of brain) ---
    R = {
        "lex": r2(brain_v, L),
        "vis+lex": r2(brain_v, vis_v, L),
        "ai+lex": r2(brain_v, ai_v, L),
        "ai+vis+lex": r2(brain_v, ai_v, vis_v, L),
        "vis+beh+lex": r2(brain_v, vis_v, vice_v, L),
        "ai+vis+beh+lex": r2(brain_v, ai_v, vis_v, vice_v, L),
        "ai+beh+lex": r2(brain_v, ai_v, vice_v, L),
    }
    unique = {
        "AI_beyond_vision+lex": R["ai+vis+lex"] - R["vis+lex"],
        "vision_beyond_AI+lex": R["ai+vis+lex"] - R["ai+lex"],
        "AI_beyond_vision+behavioral+lex": R["ai+vis+beh+lex"] - R["vis+beh+lex"],
        "behavioral_beyond_AI+vision+lex": R["ai+vis+beh+lex"] - R["ai+vis+lex"],
        "vision_beyond_AI+behavioral+lex": R["ai+vis+beh+lex"] - R["ai+beh+lex"],
    }

    passed = (casc["lex+vision"] >= 0.10) and (casc["lex+vision"] > 0) and (unique["AI_beyond_vision+lex"] > 0)
    reading = (f"{'MEANING, not pixels [PASS]' if passed else 'NOT cleanly meaning [FAIL]'}: AI->brain survives the "
               f"vision control -- partial corr {casc['raw']:.3f} (raw) -> {casc['lex']:.3f} (|lex) -> "
               f"{casc['lex+vision']:.3f} (|lex+VISION) -> {casc['lex+vision+behavioral']:.3f} (|lex+vision+behavioral). "
               f"The LLM uniquely explains {100*unique['AI_beyond_vision+lex']:.1f}% of brain variance BEYOND a "
               f"vision model + word-form (vision adds {100*unique['vision_beyond_AI+lex']:.1f}% beyond AI). "
               f"CLIP-image is vision-LANGUAGE (carries semantics) so this is conservative.")

    out = {"n_nouns": N, "noise_ceiling_lo": ceil_lo, "best_single_model": best_single,
           "ai_brain_partial_cascade": {k: round(v, 3) for k, v in casc.items()},
           "reference_partials": {k: round(v, 3) for k, v in ctrl_refs.items()},
           "brain_R2_by_predictor_set": {k: round(v, 3) for k, v in R.items()},
           "unique_contributions_to_brain_R2": {k: round(v, 4) for k, v in unique.items()},
           "passed_vision_control": bool(passed), "reading": reading}
    (HERE / "ai_brain_vision_result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== AI consensus -> HUMAN BRAIN, adding controls (n={N} nouns) ===")
    for k, v in casc.items():
        print(f"  partial(AI, brain | {k:24s}) = {v:+.3f}")
    print("\nreference partials (| lex):")
    for k, v in ctrl_refs.items():
        print(f"  {k:42s} {v:+.3f}")
    print("\nbrain variance R^2 explained:")
    for k, v in R.items():
        print(f"  {k:16s} {v:.3f}")
    print("unique contributions to brain R^2:")
    for k, v in unique.items():
        print(f"  {k:36s} {100*v:+.1f}%")
    print(f"\n>>> {reading}")
    print("wrote ai_brain_vision_result.json")


if __name__ == "__main__":
    main()
