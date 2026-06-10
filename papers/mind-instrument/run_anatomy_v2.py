"""Anatomy v2 — norm-equalized apparatus: does the anatomy survive its strongest critique?

PREREG_anatomy_v2_normeq_2026_06_10.md (frozen): per-template L2 normalization BEFORE averaging
(no template dominates by magnitude), all 10 minds recomputed live, anatomy re-tested at the v0
bars. G1 ANATOMY-ROBUST / APPARATUS-ARTIFACT; G2 v0<->v2 map correlation; G3 reliability recovery.

Usage:
    python papers/mind-instrument/run_anatomy_v2.py --smoke   # 2 minds
    python papers/mind-instrument/run_anatomy_v2.py
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

from styxx import mind  # noqa: E402

V0 = HERE / "anatomy_v0_result.json"
OUT_REPS = HERE / "normeq_reps.npz"
SEED = 0
N_PERM = 10_000
G3_BASELINE = 0.291   # v1's SB-corrected Qwen2.5-3B battery reliability under the old convention

ALL_MINDS = [
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", "qwen", "anchor"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct", "qwen", "anchor"),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct", "llama", "anchor"),
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct", "llama", "anchor"),
    ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct", "phi", "anchor"),
    ("gemma-2-2b", "google/gemma-2-2b-it", "gemma", "anchor"),
    ("gpt2", "gpt2", "gpt2", "subject"),
    ("gpt2-large", "gpt2-large", "gpt2", "subject"),
    ("pythia-410m", "EleutherAI/pythia-410m", "pythia", "subject"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", "qwen", "subject"),
]
CATS = list(dict.fromkeys(mind.BATTERY_CATEGORY))
CAT_IDX = {c: [i for i, cc in enumerate(mind.BATTERY_CATEGORY) if cc == c] for c in CATS}


def normeq_reps(repo: str, want_halves: bool = False):
    """Norm-equalized full reps (96 x d); optionally odd/even halves of the normalized states."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16,
                                               trust_remote_code=True).to(dev).eval()
    fl = max(1, int(0.66 * mdl.config.num_hidden_layers))

    @torch.no_grad()
    def states(word):
        out = []
        for t in mind.TEMPLATES:
            ids = tok(t.format(w=word), return_tensors="pt").input_ids.to(mdl.device)
            h = mdl(input_ids=ids, output_hidden_states=True, use_cache=False
                    ).hidden_states[fl][0, -1].float().cpu().numpy()
            out.append(h / (np.linalg.norm(h) + 1e-9))
        return np.stack(out)   # (8, d), each row unit-norm

    S = np.stack([states(w) for w in mind.BATTERY])   # (96, 8, d)
    full = S.mean(1)
    halves = (S[:, 0::2].mean(1), S[:, 1::2].mean(1)) if want_halves else None
    del mdl, tok
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return full, halves


def cat_Z(idx):
    charlen = np.array([len(mind.BATTERY[i]) for i in idx], dtype=float)
    toklen = np.array([mind.BATTERY_REF_TOKLEN[i] for i in idx], dtype=float)
    zc = (charlen - charlen.mean()) / (charlen.std() + 1e-9)
    zt = (toklen - toklen.mean()) / (toklen.std() + 1e-9)
    iu = np.triu_indices(len(idx), 1)
    return np.column_stack([np.abs(zc[:, None] - zc[None, :])[iu],
                            np.abs(zt[:, None] - zt[None, :])[iu]]), iu


def sub_rsa(R1, R2, idx) -> float:
    Z, iu = cat_Z(idx)
    return mind.partial_corr(mind.distmat(R1[idx])[iu], mind.distmat(R2[idx])[iu], Z)


def kendall_w(rank_rows: np.ndarray) -> float:
    m, n = rank_rows.shape
    R = rank_rows.sum(0)
    return 12.0 * float(((R - R.mean()) ** 2).sum()) / (m * m * (n ** 3 - n))


def rank_of(v: np.ndarray) -> np.ndarray:
    order = v.argsort()
    r = np.empty(len(v))
    r[order] = np.arange(1, len(v) + 1)
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rank_of(a), rank_of(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_SMOKE_INVALID" if args.smoke else ""
    rng = np.random.default_rng(SEED)
    minds = ALL_MINDS[:2] if args.smoke else ALL_MINDS

    reps, g3 = {}, None
    for name, repo, fam, role in minds:
        want = name == "Qwen2.5-3B"
        full, halves = normeq_reps(repo, want_halves=want)
        reps[name] = (fam, role, full)
        if want and halves is not None:
            Zb, iub = cat_Z(list(range(len(mind.BATTERY))))
            r = mind.partial_corr(mind.distmat(halves[0])[iub], mind.distmat(halves[1])[iub], Zb)
            g3 = round(2 * r / (1 + r), 4) if r > 0 else 0.0
        print(f"[reps] {name}", flush=True)

    receipt = {"experiment": "Anatomy v2 - norm-equalized apparatus",
               "prereg": "papers/mind-instrument/PREREG_anatomy_v2_normeq_2026_06_10.md",
               "seed": SEED, "n_perm": N_PERM,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    if args.smoke:
        receipt.update({"smoke": True, "verdict": "SMOKE-OK",
                        "n_minds": len(reps)})
        (HERE / f"anatomy_v2_result{tag}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                           encoding="utf-8")
        print("SMOKE-OK"); return 0

    np.savez(OUT_REPS, **{n: r for n, (f, ro, r) in reps.items()})
    anchors = [(n, f) for n, (f, ro, _) in reps.items() if ro == "anchor"]
    subjects = [(n, f) for n, (f, ro, _) in reps.items() if ro == "subject"]

    profiles = {}
    for sname, sfam in subjects:
        profiles[sname] = {}
        for c, idx in CAT_IDX.items():
            vals = [sub_rsa(reps[sname][2], reps[an][2], idx) for an, af in anchors if af != sfam]
            profiles[sname][c] = round(float(np.mean(vals)), 4)
        print(f"[prof] {sname:14s} " + " ".join(f"{c[:4]}={profiles[sname][c]:+.2f}" for c in CATS),
              flush=True)

    rank_rows = np.stack([rank_of(np.array([profiles[s][c] for c in CATS])) for s, _ in subjects])
    W = kendall_w(rank_rows)
    null = np.array([kendall_w(np.stack([rng.permutation(8) + 1.0 for _ in subjects]))
                     for _ in range(N_PERM)])
    p = float((null >= W).mean())

    v0 = json.loads(V0.read_text(encoding="utf-8"))["mean_anatomy_independent"]
    mean_v2 = {c: round(float(np.mean([profiles[s][c] for s, _ in subjects])), 4) for c in CATS}
    g2 = round(spearman(np.array([v0[c] for c in CATS]), np.array([mean_v2[c] for c in CATS])), 4)

    g1_pass = W >= 0.60 and p < 0.05
    g3_pass = g3 is not None and g3 > G3_BASELINE
    receipt.update({
        "mean_anatomy_normeq": dict(sorted(mean_v2.items(), key=lambda kv: -kv[1])),
        "profiles_subjects": profiles,
        "G1_kendall_W": round(W, 4), "G1_perm_p": round(p, 5),
        "G1_null_p95": round(float(np.percentile(null, 95)), 4), "G1_pass": g1_pass,
        "G2_spearman_v0_v2_map": g2,
        "G3_qwen3b_battery_reliability_normeq": g3, "G3_baseline_old_convention": G3_BASELINE,
        "G3_pass": g3_pass,
        "verdict": ("VOID-DIAGNOSIS" if not g3_pass else
                    ("ANATOMY-ROBUST" if g1_pass else "APPARATUS-ARTIFACT")),
    })
    (HERE / "anatomy_v2_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("mean_anatomy_normeq", "G1_kendall_W", "G1_perm_p",
                                              "G2_spearman_v0_v2_map",
                                              "G3_qwen3b_battery_reliability_normeq", "verdict")},
                     indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
