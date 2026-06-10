"""Anatomy v1 — idiosyncrasy or range-artifact? Disattenuation per the frozen prereg.

PREREG_anatomy_v1_disattenuation_2026_06_10.md (frozen): split-template reliabilities
(Spearman-Brown), measurability floor 0.20, disattenuated per-domain convergence over cross-family
pairs, branch verdict + P2 shared-anatomy re-test, controls C1/C2.

Usage:
    python papers/mind-instrument/run_anatomy_v1.py --smoke    # 2 minds, *_SMOKE_INVALID
    python papers/mind-instrument/run_anatomy_v1.py            # scored
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

ANCHORS_NPZ = REPO / "papers" / "real-convergence" / "contextual_reps.npz"
LIVE_NPZ = HERE / "atlas_live_reps.npz"
V0 = HERE / "anatomy_v0_result.json"
SEED = 0
N_PERM = 10_000
REL_FLOOR = 0.20

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
SUBJECTS = [m for m in ALL_MINDS if m[3] == "subject"]
CATS = list(dict.fromkeys(mind.BATTERY_CATEGORY))
CAT_IDX = {c: [i for i, cc in enumerate(mind.BATTERY_CATEGORY) if cc == c] for c in CATS}
HALF_A, HALF_B = mind.TEMPLATES[0::2], mind.TEMPLATES[1::2]   # odd/even interleave (amended pre-scoring)


def half_reps(repo: str) -> tuple[np.ndarray, np.ndarray]:
    """(96 x d) reps per template half at the frozen 0.66 layer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16,
                                               trust_remote_code=True).to(dev).eval()
    fl = max(1, int(0.66 * mdl.config.num_hidden_layers))

    @torch.no_grad()
    def rep(word, templates):
        acc = None
        for t in templates:
            ids = tok(t.format(w=word), return_tensors="pt").input_ids.to(mdl.device)
            h = mdl(input_ids=ids, output_hidden_states=True, use_cache=False
                    ).hidden_states[fl][0, -1].float().cpu().numpy()
            acc = h if acc is None else acc + h
        return acc / len(templates)

    A = np.stack([rep(w, HALF_A) for w in mind.BATTERY])
    B = np.stack([rep(w, HALF_B) for w in mind.BATTERY])
    del mdl, tok
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return A, B


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


def rank_of(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    r = np.empty(len(values))
    r[order] = np.arange(1, len(values) + 1)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_SMOKE_INVALID" if args.smoke else ""
    rng = np.random.default_rng(SEED)
    minds = ALL_MINDS[:2] if args.smoke else ALL_MINDS

    # full-template reps for raw RSA (stored receipts, as v0)
    zfull = np.load(ANCHORS_NPZ)
    zlive = np.load(LIVE_NPZ)
    full = {}
    for name, repo, fam, role in minds:
        full[name] = zfull[f"fixed__{name}"].astype(float) if role == "anchor" \
            else zlive[name].astype(float)

    # half reps -> reliabilities
    rel, battery_rel = {}, {}
    for name, repo, fam, role in minds:
        A, B = half_reps(repo)
        Zb, iub = (lambda z, i: (z, i))(*cat_Z(list(range(len(mind.BATTERY)))))
        r_full = mind.partial_corr(mind.distmat(A)[iub], mind.distmat(B)[iub], Zb)
        battery_rel[name] = round(2 * r_full / (1 + r_full), 4) if r_full > 0 else 0.0
        rel[name] = {}
        for c, idx in CAT_IDX.items():
            r = sub_rsa(A, B, idx)
            rel[name][c] = round(max(0.0, 2 * r / (1 + r)) if r > 0 else 0.0, 4)
        print(f"[rel] {name:14s} battery={battery_rel[name]:.3f} " +
              " ".join(f"{c[:4]}={rel[name][c]:.2f}" for c in CATS), flush=True)

    receipt = {"experiment": "Anatomy v1 — disattenuation (idiosyncrasy vs range-artifact)",
               "prereg": "papers/mind-instrument/PREREG_anatomy_v1_disattenuation_2026_06_10.md",
               "seed": SEED, "n_perm": N_PERM, "rel_floor": REL_FLOOR,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "battery_reliability": battery_rel, "domain_reliability": rel}

    if args.smoke:
        receipt.update({"smoke": True, "verdict": "SMOKE-OK"})
        (HERE / f"anatomy_v1_result{tag}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                           encoding="utf-8")
        print("SMOKE-OK"); return 0

    # C2: battery-level reliability > 0.5 for every mind
    dropped = [n for n, v in battery_rel.items() if v <= 0.5]
    receipt["C2_dropped"] = dropped
    live_subjects = [s for s in SUBJECTS if s[0] not in dropped]
    if len(live_subjects) < 3:
        receipt["verdict"] = "VOID-SUBSTRATE"
        (HERE / "anatomy_v1_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                     encoding="utf-8")
        print("VOID-SUBSTRATE"); return 2

    # C1: raw per-subject mean anatomy must reproduce v0 (+-0.005 per subject mean)
    v0 = json.loads(V0.read_text(encoding="utf-8"))
    anchors = [(n, f) for n, _, f, role in ALL_MINDS if role == "anchor" and n not in dropped]
    raw = {}
    for sname, _, sfam, _ in SUBJECTS:
        raw[sname] = {}
        for c, idx in CAT_IDX.items():
            vals = [sub_rsa(full[sname], full[an], idx) for an, af in anchors if af != sfam]
            raw[sname][c] = float(np.mean(vals))
    c1 = {}
    for sname, _, _, _ in SUBJECTS:
        mine = np.mean(list(raw[sname].values()))
        v0m = np.mean(list(v0["profiles_subjects"][sname].values()))
        c1[sname] = {"v1": round(float(mine), 4), "v0": round(float(v0m), 4),
                     "pass": abs(mine - v0m) <= 0.005}
    receipt["C1_raw_reproduction"] = c1
    if not all(v["pass"] for v in c1.values()):
        receipt["verdict"] = "VOID-PIPELINE"
        (HERE / "anatomy_v1_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                     encoding="utf-8")
        print("VOID-PIPELINE"); return 2

    # disattenuated profiles over measurable cross-family pairs
    dis, measurable = {}, {}
    for sname, _, sfam, _ in live_subjects:
        dis[sname] = {}
        for c, idx in CAT_IDX.items():
            pairs = []
            if rel[sname][c] >= REL_FLOOR:
                for an, af in anchors:
                    if af != sfam and rel[an][c] >= REL_FLOOR:
                        d = sub_rsa(full[sname], full[an], idx) / np.sqrt(rel[sname][c] * rel[an][c])
                        pairs.append(min(1.0, float(d)))
            dis[sname][c] = round(float(np.mean(pairs)), 4) if pairs else None
        print(f"[dis] {sname:14s} " + " ".join(
            f"{c[:4]}={dis[sname][c] if dis[sname][c] is not None else 'UNM'}" for c in CATS), flush=True)
    for c in CATS:
        measurable[c] = sum(1 for s, _, _, _ in live_subjects if dis[s][c] is not None)
    glob_meas = [c for c in CATS if measurable[c] >= 2]
    receipt["measurable_subject_counts"] = measurable
    receipt["globally_measurable_domains"] = glob_meas

    mean_dis = {c: round(float(np.mean([dis[s][c] for s, _, _, _ in live_subjects
                                        if dis[s][c] is not None])), 4)
                for c in glob_meas}
    receipt["mean_disattenuated_anatomy"] = dict(sorted(mean_dis.items(), key=lambda kv: -kv[1]))

    # branch verdict
    bottom2 = [c for c, _ in sorted(mean_dis.items(), key=lambda kv: kv[1])[:2]]
    pf_measurable = all(c in glob_meas for c in ("profession", "furniture"))
    if pf_measurable and set(bottom2) == {"profession", "furniture"}:
        branch = "IDIOSYNCRASY-CONFIRMED"
    else:
        branch = "RANGE-ARTIFACT"
    receipt["bottom2_disattenuated"] = bottom2

    # P2 on measurable domains
    rows = []
    for sname, _, _, _ in live_subjects:
        vals = np.array([dis[sname][c] if dis[sname][c] is not None else -9 for c in glob_meas])
        rows.append(rank_of(vals))
    W = kendall_w(np.stack(rows))
    n_dom = len(glob_meas)
    null = np.array([kendall_w(np.stack([rng.permutation(n_dom) + 1.0 for _ in rows]))
                     for _ in range(N_PERM)])
    p = float((null >= W).mean())
    receipt.update({"P2_kendall_W": round(W, 4), "P2_perm_p": round(p, 5),
                    "P2_null_p95": round(float(np.percentile(null, 95)), 4),
                    "P2_pass": W >= 0.60 and p < 0.05,
                    "verdict": branch + ("" if (W >= 0.60 and p < 0.05) else " + P2-REOPENED")})
    (HERE / "anatomy_v1_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                 encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("mean_disattenuated_anatomy", "bottom2_disattenuated",
                                              "P2_kendall_W", "P2_perm_p", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
