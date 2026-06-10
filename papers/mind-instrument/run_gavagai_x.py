"""GAVAGAI-X — does the interlingua cross the species barrier? (frozen prereg)

PREREG_gavagai_x_2026_06_10.md: label-free concept-identity recovery between causal LLMs
(normeq_reps.npz) and contrastively-trained sentence embedders (MiniLM, mpnet) — different
objective, architecture class, and output space. Gates X1/X2/X3; VOID self-pair.

Usage:
    python papers/mind-instrument/run_gavagai_x.py --smoke
    python papers/mind-instrument/run_gavagai_x.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from styxx import mind  # noqa: E402
from run_gavagai_v0 import translate, FAMILY  # noqa: E402

NORMEQ = HERE / "normeq_reps.npz"
SEED = 0
N_NULL = 100
N = 96
CAT = np.array(mind.BATTERY_CATEGORY)
EMBEDDERS = [("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2"),
             ("mpnet-base", "sentence-transformers/all-mpnet-base-v2")]
V0_XFAM_MEAN = 0.1661   # gavagai_v0_result.json, LLM<->LLM cross-family reference


def embedder_reps(repo: str) -> np.ndarray:
    """Battery reps mirrored to the frozen convention: per-template normalized, averaged."""
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(repo, device="cpu")
    acc = None
    for t in mind.TEMPLATES:
        E = st.encode([t.format(w=w) for w in mind.BATTERY], normalize_embeddings=True)
        acc = E if acc is None else acc + E
    return acc / len(mind.TEMPLATES)


def score_pair(RA: np.ndarray, RB: np.ndarray, rng: np.random.Generator):
    perm = rng.permutation(N)
    pi = translate(mind.distmat(RA), mind.distmat(RB[perm]))
    rec = perm[pi]
    return float((rec == np.arange(N)).mean()), float((CAT[rec] == CAT).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_SMOKE_INVALID" if args.smoke else ""
    rng = np.random.default_rng(SEED)

    zn = np.load(NORMEQ)
    llms = list(zn.keys())[:1] if args.smoke else list(zn.keys())
    embs = {n: embedder_reps(r) for n, r in (EMBEDDERS[:1] if args.smoke else EMBEDDERS)}
    print(f"[reps] {len(llms)} LLMs + {len(embs)} embedders", flush=True)

    # VOID: embedder self-pair
    e0 = list(embs)[0]
    a0, _ = score_pair(embs[e0], embs[e0], rng)
    receipt = {"experiment": "GAVAGAI-X - cross-species radical translation (LLM <-> embedder)",
               "prereg": "papers/mind-instrument/PREREG_gavagai_x_2026_06_10.md",
               "seed": SEED, "chance": round(1 / N, 4),
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "self_pair_control": a0}
    if a0 < 0.99:
        receipt["verdict"] = "VOID-PIPELINE"
        (HERE / f"gavagai_x_result{tag}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                          encoding="utf-8")
        print("VOID-PIPELINE", a0); return 2

    rows = []
    for ln in llms:
        for en, ER in embs.items():
            acc, cat = score_pair(zn[ln].astype(float), ER, rng)
            rows.append({"llm": ln, "embedder": en, "acc": round(acc, 4), "cat": round(cat, 4)})
            print(f"[x-species] {ln:13s}->{en:10s} acc={acc:.3f} cat={cat:.3f}", flush=True)
    e_pair = None
    if len(embs) == 2:
        ea, eb = list(embs)
        a, c = score_pair(embs[ea], embs[eb], rng)
        e_pair = {"pair": f"{ea}<->{eb}", "acc": round(a, 4), "cat": round(c, 4)}
        print(f"[within-other-species] {e_pair}", flush=True)

    if args.smoke:
        receipt.update({"smoke": True, "rows": rows, "verdict": "SMOKE-OK"})
        (HERE / f"gavagai_x_result{tag}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                          encoding="utf-8")
        print("SMOKE-OK"); return 0

    mean_acc = float(np.mean([r["acc"] for r in rows]))
    mean_cat = float(np.mean([r["cat"] for r in rows]))

    null = []
    for k in range(N_NULL):
        r = rows[k % len(rows)]
        RA = zn[r["llm"]].astype(float)
        RB = embs[r["embedder"]]
        fake = rng.permutation(N)
        perm = rng.permutation(N)
        pi = translate(mind.distmat(RA), mind.distmat(RB[perm]))
        null.append(float((perm[pi] == fake).mean()))
    null95 = float(np.percentile(null, 95))

    x1 = mean_acc >= 0.1042 and mean_acc > null95
    receipt.update({
        "rows": rows, "n_cross_species_pairs": len(rows),
        "mean_cross_species_accuracy": round(mean_acc, 4),
        "mean_cross_species_category": round(mean_cat, 4),
        "null_95th_percentile": round(null95, 4), "n_null": N_NULL,
        "X2_llm_llm_reference": V0_XFAM_MEAN,
        "X3_embedder_pair": e_pair,
        "X1_pass": x1,
        "verdict": "SPECIES-CROSSED" if x1 else "SPECIES-BOUND",
    })
    (HERE / "gavagai_x_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("mean_cross_species_accuracy",
                                              "mean_cross_species_category",
                                              "null_95th_percentile", "X3_embedder_pair",
                                              "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
