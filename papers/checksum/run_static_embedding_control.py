#!/usr/bin/env python3
"""run_static_embedding_control.py — the recipe the static-embedding control never had.

    python papers/checksum/run_static_embedding_control.py
    # writes papers/checksum/static_embedding_control_recipe_2026_09_13.json (never the committed
    # static_embedding_control.json, which stays as committed)

DUE_DILIGENCE_2026_09_13 §4 reports that a context-free baseline — the mean input-token embedding of
each of the 462 concept strings from SmolLM2-135M — correlates r 0.308 / 0.339 / 0.304 / 0.285 with
the four contextual banks' RDMs, against 0.87–0.96 between the banks themselves. The json holding
those numbers was committed with no script and no document naming it (red team, result-6). This is
the recipe, written after the fact, that tries to reproduce them from the tree: the concept list in
bank order comes from papers/disjoint-worlds/run_g0clear.py (CONCEPTS, 462 after dedup), the banks
are the committed _b31v2_pts*.npz, the RDM is built exactly as geometry_plates_demo.py builds it, and
the one thing the original left unstated — whether the concept string was embedded with or without a
leading space — is run both ways and reported both ways. Whichever variant matches the committed
numbers to six decimals is the reconstruction; if neither does, the json says so.
"""
import importlib.util
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(ROOT, "papers", "disjoint-worlds")
NAME = "HuggingFaceTB/SmolLM2-135M"
BANKS = {"llama_3b": "_b31v2_ptsA.npz", "llama_1b": "_b31v2_pts_llama_1b.npz",
         "gemma_2b": "_b31v2_pts_gemma_2b.npz", "qwen_1p5b": "_b31v2_pts_qwen_1p5b.npz"}
COMMITTED = os.path.join(HERE, "static_embedding_control.json")
OUT = os.path.join(HERE, "static_embedding_control_recipe_2026_09_13_v1.json")
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"   # the Hub snapshot the reconstruction used; pinned so a re-upload cannot move the table silently


def concepts() -> list:
    spec = importlib.util.spec_from_file_location("g0clear", os.path.join(DW, "run_g0clear.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return list(m.CONCEPTS)


def rdm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return 1.0 - X @ X.T


def r_upper(a: np.ndarray, b: np.ndarray) -> float:
    iu = np.triu_indices(a.shape[0], 1)
    return float(np.corrcoef(a[iu], b[iu])[0, 1])


def main():
    cs = concepts()
    banks = {k: rdm(np.load(os.path.join(DW, f))["pts"]) for k, f in BANKS.items()}
    for k, R in banks.items():
        assert R.shape[0] == len(cs), f"{k}: bank has {R.shape[0]} rows, concept list has {len(cs)}"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    # the table is NOT in the tree: it comes from the Hub at a pinned revision, and its bytes are hashed into the record
    tok = AutoTokenizer.from_pretrained(NAME, revision=REVISION)
    E = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, revision=REVISION).get_input_embeddings().weight.detach().numpy()
    import hashlib, transformers
    embedding_sha256 = hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest()
    concepts_sha256 = hashlib.sha256(json.dumps(cs, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    variants = {}
    for label, prefix in (("no_leading_space", ""), ("leading_space", " ")):
        rows = []
        for c in cs:
            ids = tok(prefix + c, add_special_tokens=False).input_ids
            rows.append(E[ids].mean(0))
        S = rdm(np.stack(rows))
        variants[label] = {k: r_upper(S, R) for k, R in banks.items()}
    random = {}
    for seed in (0, 343):
        Rr = rdm(np.random.default_rng(seed).normal(size=(len(cs), E.shape[1])))
        random[f"seed_{seed}"] = {k: r_upper(Rr, R) for k, R in banks.items()}
    committed = json.load(open(COMMITTED, encoding="utf-8")) if os.path.exists(COMMITTED) else None
    matches, deltas = [], {}
    if committed:
        for label, vals in variants.items():
            deltas[label] = {k: vals[k] - committed["static"][k] for k in BANKS}
            if all(abs(deltas[label][k]) < 5e-7 for k in BANKS):
                matches.append(label)
    out = {"schema": "styxx.checksum/static-embedding-control-recipe/v1", "model": NAME, "model_revision": REVISION,
           "embedding_sha256": embedding_sha256, "n_concepts": len(cs), "concepts_sha256": concepts_sha256,
           "embedding_dim": int(E.shape[1]),
           "concept_source": "papers/disjoint-worlds/run_g0clear.py CONCEPTS (465 entries, 462 after first-occurrence dedup)",
           "bank_writer": "papers/disjoint-worlds/run_b31v2.py: concepts = FULL_CONCEPTS (line 103); np.savez(pts=[pts[c] for c in concepts]) (lines 130/164)",
           "rdm": "centre columns, unit rows, 1 - X X^T (geometry_plates_demo.rdm)",
           "versions": {"transformers": transformers.__version__, "torch": torch.__version__, "numpy": np.__version__},
           "static_vs_bank_r": variants, "random_vs_bank_r": random,
           "committed_static_r": committed["static"] if committed else None,
           "deltas_vs_committed_static": deltas,
           "variants_matching_committed": matches,
           "committed_random_r": committed.get("random") if committed else None,
           "random_matching_committed": None,
           "random_note": "the committed random row has no recipe (its seed is unknown); the two seeds here are a fresh null of the same magnitude, not a reproduction"}
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    for label, vals in variants.items():
        print(label, {k: round(v, 4) for k, v in vals.items()})
    print("random", {s: {k: round(v, 4) for k, v in vals.items()} for s, vals in random.items()})
    print("matches committed:", matches, "deltas:", {l: {k: f"{v:.1e}" for k, v in d.items()} for l, d in deltas.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
