"""B34-v3 — label-free cross-family READ, per PREREG_b34v3_labelfree_read_2026_08_03.md.

Discover the correspondence with the committed linear machinery (no iteration — the v2 fix),
fit ONE MLP on the linear-discovered pseudo-pairs, read held-out concepts. FRESH split (seed
343) so the held-out concepts are disjoint in membership from v1/v2's glimpsed numbers. Runs
CPU-from-cache off the b31v2 extractions. `--smoke` = 40/10, INVALID-only.

Emits b34v3_result.json in the shape the frozen gates block scores against, then scores itself
mechanically through styxx.protocol.Experiment.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from run_g0clear import CONCEPTS as FULL_CONCEPTS      # noqa: E402
from styxx_transfer import TransferMap                 # noqa: E402
from run_b31v2 import fit_mlp                           # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 343
TARGETS = ["llama_1b", "gemma_2b", "qwen_1p5b"]


def load_pts(tag):
    z = np.load(HERE / f"_b31v2_pts_{tag}.npz", allow_pickle=True)
    return {c: z["pts"][i] for i, c in enumerate(FULL_CONCEPTS)}


def load_A():
    z = np.load(HERE / "_b31v2_ptsA.npz", allow_pickle=True)
    return {c: z["pts"][i] for i, c in enumerate(FULL_CONCEPTS)}


def read_top1(fin, ptsA, fin_ptsB, mapper):
    hits = sum(1 for i, c in enumerate(fin)
               if int(np.argmin(np.linalg.norm(fin_ptsB - mapper(ptsA[c]), axis=1))) == i)
    return hits / len(fin)


def assignment_from_map(XA, XB_shuf, mapper):
    MA = np.stack([mapper(x) for x in XA])
    D = np.linalg.norm(MA[:, None, :] - XB_shuf[None, :, :], axis=-1)
    _, col = linear_sum_assignment(D)
    return col


def main():
    s0 = json.loads((HERE / "g0clear_result_llama3b.json").read_text(encoding="utf-8"))
    kstar = s0["locked"]["k"]

    # FRESH split (seed 343) — disjoint membership from split_concepts(0)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(FULL_CONCEPTS))
    n_fin = 10 if SMOKE else 70
    n_tr = 40 if SMOKE else (len(FULL_CONCEPTS) - n_fin)
    fin_i = idx[:n_fin]
    tr_i = idx[n_fin:n_fin + n_tr]
    concepts = FULL_CONCEPTS
    tr = [concepts[i] for i in tr_i]
    fin = [concepts[i] for i in fin_i]

    ptsA = load_A()
    XA = np.array([ptsA[c] for c in tr])

    results = {"prereg": "PREREG_b34v3_labelfree_read_2026_08_03.md", "seed": SEED,
               "smoke": SMOKE, "n_tr": len(tr), "n_heldout": len(fin),
               "chance": round(1 / len(fin), 4), "targets": {}}
    shuffled_tops = []

    for tag in TARGETS:
        t0 = time.time()
        ptsB = load_pts(tag)
        XB = np.array([ptsB[c] for c in tr])
        fin_ptsB = np.array([ptsB[c] for c in fin])

        perm = rng.permutation(len(tr))
        XB_shuf = XB[perm]
        true_col = np.argsort(perm)

        # discover with the committed linear machinery (no iteration)
        tm = TransferMap.fit(XA, XB_shuf, k=kstar)
        disc_col = assignment_from_map(XA, XB_shuf, tm.transfer_point)
        seed_acc = float((disc_col == true_col).mean())

        # single MLP on the linear-discovered pseudo-pairs
        mlp_fn, _ = fit_mlp(XA, XB_shuf[disc_col], seed=SEED)
        read = read_top1(fin, ptsA, fin_ptsB, mlp_fn)

        # pairing-shuffled null: MLP on a random correspondence
        rand = rng.permutation(len(tr))
        null_fn, _ = fit_mlp(XA, XB_shuf[rand], seed=SEED)
        null_read = read_top1(fin, ptsA, fin_ptsB, null_fn)
        shuffled_tops.append(null_read)

        results["targets"][tag] = {
            "seed_acc": round(seed_acc, 4), "read_top1": round(read, 4),
            "shuffled_top1": round(null_read, 4),
            "x_chance_read": round(read * len(fin), 1)}
        print(f">> {tag}: seed_acc={seed_acc:.4f} read={read:.4f} "
              f"null={null_read:.4f} ({read*len(fin):.0f}x) [{time.time()-t0:.0f}s]", flush=True)

    results["max_shuffled_top1"] = round(max(shuffled_tops), 4)

    # score mechanically through the frozen gates block
    try:
        from styxx.protocol import Experiment
        exp = Experiment(HERE / "PREREG_b34v3_labelfree_read_2026_08_03.md")
        v = exp.score(results, smoke=SMOKE)
        results["verdict"] = v.verdict
        results["gates"] = v.gates
        results["prereg_commit"] = v.prereg_commit
        results["gates_sha256"] = v.gates_sha256
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}"
        results["score_error"] = str(e)

    out = HERE / f"b34v3_result{SUFFIX}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}  -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
