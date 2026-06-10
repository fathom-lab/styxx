"""Anatomy of Convergence v0 — which parts of meaning are universal?

Frozen prereg: PREREG_anatomy_v0_2026_06_10.md (gates P1 W>=0.60 + perm p<0.05; pipeline validity
control vs the Atlas v0 receipt; VOIDs). Per-category cross-family partial-lexical RSA profiles for
the 4 out-of-anchor minds; Kendall's W over category rankings.

Usage:
    python papers/mind-instrument/run_anatomy_v0.py --smoke   # anchors only, *_SMOKE_INVALID
    python papers/mind-instrument/run_anatomy_v0.py           # scored run
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "papers" / "real-convergence"))

from styxx import mind  # noqa: E402

ANCHORS = REPO / "papers" / "real-convergence" / "contextual_reps.npz"
ATLAS = HERE / "atlas_v0_result.json"
LIVE_REPS = HERE / "atlas_live_reps.npz"
SEED = 0
N_PERM = 10_000

SUBJECTS = [
    ("gpt2", "gpt2", "gpt2"),
    ("gpt2-large", "gpt2-large", "gpt2"),
    ("pythia-410m", "EleutherAI/pythia-410m", "pythia"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "qwen"),
]
CATS = list(dict.fromkeys(mind.BATTERY_CATEGORY))
CAT_IDX = {c: [i for i, cc in enumerate(mind.BATTERY_CATEGORY) if cc == c] for c in CATS}


def live_reps(repo: str) -> np.ndarray:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from run_real_convergence_v2 import concept_all_layers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16,
                                               trust_remote_code=True).to(dev).eval()
    fl = max(1, int(0.66 * mdl.config.num_hidden_layers))
    out = np.stack([concept_all_layers(mdl, tok, w) for w in mind.BATTERY])[:, fl, :]
    del mdl, tok
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def cat_lexical_Z(idx: list[int]):
    """Frozen lexical design restricted to one category's pairs."""
    charlen = np.array([len(mind.BATTERY[i]) for i in idx], dtype=float)
    toklen = np.array([mind.BATTERY_REF_TOKLEN[i] for i in idx], dtype=float)
    zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9)
    zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
    iu = np.triu_indices(len(idx), 1)
    return np.column_stack([np.abs(zc[:, None] - zc[None, :])[iu],
                            np.abs(zt[:, None] - zt[None, :])[iu]]), iu


def anatomy_profile(reps: np.ndarray, family: str, anchors: dict) -> dict:
    """Per-category mean cross-family partial-lexical RSA vs the anchors."""
    out = {}
    for c, idx in CAT_IDX.items():
        Z, iu = cat_lexical_Z(idx)
        Dc = mind.distmat(reps[idx])[iu]
        vals = [mind.partial_corr(Dc, mind.distmat(a_reps[idx])[iu], Z)
                for a_name, (a_fam, a_reps) in anchors.items() if a_fam != family]
        out[c] = round(float(np.mean(vals)), 4)
    return out


def kendall_w(rank_rows: np.ndarray) -> float:
    """Kendall's W for m judges (rows) ranking n items (columns)."""
    m, n = rank_rows.shape
    R = rank_rows.sum(0)
    S = float(((R - R.mean()) ** 2).sum())
    return 12.0 * S / (m * m * (n ** 3 - n))


def ranks(profile: dict) -> np.ndarray:
    v = np.array([profile[c] for c in CATS])
    order = v.argsort()
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_SMOKE_INVALID" if args.smoke else ""
    rng = np.random.default_rng(SEED)

    z = np.load(ANCHORS)
    anchors = {name: (fam, z[f"fixed__{name}"].astype(float)) for name, fam in mind.ANCHOR_MODELS}

    receipt = {"experiment": "Anatomy of Convergence v0",
               "prereg": "papers/mind-instrument/PREREG_anatomy_v0_2026_06_10.md",
               "categories": CATS, "seed": SEED, "n_perm": N_PERM,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "anchors_sha256": hashlib.sha256(ANCHORS.read_bytes()).hexdigest()}

    if args.smoke:
        profs = {n: anatomy_profile(r, f, anchors) for n, (f, r) in list(anchors.items())[:3]}
        receipt.update({"smoke": True, "profiles": profs, "verdict": "SMOKE-OK"})
        (HERE / f"anatomy_v0_result{tag}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                           encoding="utf-8")
        print("SMOKE-OK"); return 0

    # subjects (live reps, persisted for provenance)
    subj_reps = {}
    for name, repo, fam in SUBJECTS:
        subj_reps[name] = (fam, live_reps(repo))
        print(f"[reps] {name}", flush=True)
    np.savez(LIVE_REPS, **{n: r for n, (f, r) in subj_reps.items()})

    # VALIDITY CONTROL: gpt2 full-battery citizenship must reproduce the Atlas receipt +-0.0005
    atlas = json.loads(ATLAS.read_text(encoding="utf-8"))
    pub = next(e["geometry"]["citizenship_xfam_partial_lex"]
               for e in atlas["axes_measured"]["atlas_geometry"] if e["subject"] == "gpt2")
    g = mind.geometry_citizenship(subj_reps["gpt2"][1], ANCHORS, "gpt2")
    ctrl_ok = abs(g["citizenship_xfam_partial_lex"] - pub) <= 0.0005
    receipt["validity_control"] = {"gpt2_recomputed": g["citizenship_xfam_partial_lex"],
                                   "gpt2_atlas": pub, "pass": ctrl_ok}
    if not ctrl_ok:
        receipt["verdict"] = "VOID-PIPELINE"
        (HERE / "anatomy_v0_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                     encoding="utf-8")
        print("VOID-PIPELINE"); return 2

    profiles = {n: anatomy_profile(r, f, anchors) for n, (f, r) in subj_reps.items()}
    anchor_profiles = {n: anatomy_profile(r, f, anchors) for n, (f, r) in anchors.items()}

    rank_rows = np.stack([ranks(profiles[n]) for n, _, _ in SUBJECTS])
    W = kendall_w(rank_rows)
    null = np.array([kendall_w(np.stack([rng.permutation(8) + 1.0 for _ in range(len(SUBJECTS))]))
                     for _ in range(N_PERM)])
    p = float((null >= W).mean())
    W_anchor = kendall_w(np.stack([ranks(anchor_profiles[n]) for n, _ in mind.ANCHOR_MODELS]))

    mean_prof = {c: round(float(np.mean([profiles[n][c] for n, _, _ in SUBJECTS])), 4) for c in CATS}
    receipt.update({
        "profiles_subjects": profiles, "profiles_anchors_reference": anchor_profiles,
        "mean_anatomy_independent": dict(sorted(mean_prof.items(), key=lambda kv: -kv[1])),
        "kendall_W_independent": round(W, 4), "perm_p": round(p, 5),
        "perm_null_p95": round(float(np.percentile(null, 95)), 4),
        "kendall_W_anchors_reference": round(W_anchor, 4),
        "gates": {"P1_W_bar": 0.60, "P1_pass_W": W >= 0.60, "P1_pass_perm": p < 0.05},
        "verdict": "SUPPORTED" if (W >= 0.60 and p < 0.05) else "CLOSED_NEGATIVE",
    })
    (HERE / "anatomy_v0_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("mean_anatomy_independent", "kendall_W_independent",
                                              "perm_p", "kendall_W_anchors_reference", "verdict")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
