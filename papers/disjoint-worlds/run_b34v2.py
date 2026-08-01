"""b34-v2 driver: v1 verbatim with the initializer swapped to the committed linear pipeline.

Kept as a thin patch over run_b34.py so the diff IS the audit surface — the ONE change per
PREREG_b34v2_linear_seeded_2026_08_01.md: stage 1 = TransferMap.fit (GW warm start +
Sinkhorn-annealed Procrustes, the committed label-free linear machinery) and the initial
pseudo-pairing is the assignment its fitted map induces in full space. Everything else —
data, shuffle rail, MLP, iteration count, seeds, metric, gates — is run_b34.py unchanged.
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_disjoint_worlds as R
from run_g0clear import CONCEPTS as FULL_CONCEPTS, split_concepts
from run_b31v2 import fit_mlp
from styxx_transfer import TransferMap
from run_b34 import load_pts, read_top1, assignment_from_map, run_arm

SEED = 34; ITERS = 8
s0 = json.loads((HERE / 'g0clear_result_llama3b.json').read_text(encoding='utf-8'))
kstar = s0['locked']['k']

rng = np.random.default_rng(SEED)
tr_, sel_, fin = split_concepts(seed=0)
tr = list(tr_) + list(sel_)
ptsA = load_pts('_b31v2_ptsA.npz')
XA = np.array([ptsA[c] for c in tr])

results = {"prereg": "PREREG_b34v2_linear_seeded_2026_08_01.md", "seed": SEED,
           "iters": ITERS, "n_fit": len(tr), "n_heldout": len(fin),
           "chance": round(1/len(fin), 4), "targets": {}}

for tag in ["llama_1b", "gemma_2b", "qwen_1p5b"]:
    t0 = time.time()
    ptsB = load_pts(f'_b31v2_pts_{tag}.npz')
    XB = np.array([ptsB[c] for c in tr])
    fin_ptsB = np.array([ptsB[c] for c in fin])
    perm = rng.permutation(len(tr)); XB_shuf = XB[perm]
    true_col = np.argsort(perm)

    # THE ONE CHANGE: stage 1 = the committed linear label-free pipeline as initializer
    tm = TransferMap.fit(XA, XB_shuf, k=kstar)
    lin_col = assignment_from_map(XA, XB_shuf, tm.transfer_point)
    lin_seed_acc = float((lin_col == true_col).mean())
    lin_top1 = read_top1(fin, ptsA, fin_ptsB, tm.transfer_point)

    mlf, mlf_diag = run_arm(XA, XB_shuf, true_col, lin_col, ITERS, seed=SEED)
    mlf_top1 = read_top1(fin, ptsA, fin_ptsB, mlf)

    rand_col = rng.permutation(len(tr))
    n0, _ = run_arm(XA, XB_shuf, true_col, rand_col, 1, seed=SEED)
    n0_top1 = read_top1(fin, ptsA, fin_ptsB, n0)
    r0, r0_diag = run_arm(XA, XB_shuf, true_col, rand_col.copy(), ITERS, seed=SEED)
    r0_top1 = read_top1(fin, ptsA, fin_ptsB, r0)

    results["targets"][tag] = {
        "linear_seed_acc": round(lin_seed_acc, 4), "linear_top1": round(lin_top1, 4),
        "MLF_top1": round(mlf_top1, 4), "N0_top1": round(n0_top1, 4),
        "R0_top1": round(r0_top1, 4),
        "MLF_pseudo_pair_acc_by_iter": [round(x, 4) for x in mlf_diag],
        "x_chance_MLF": round(mlf_top1 * len(fin), 1)}
    print(f">> {tag}: lin_seed_acc={lin_seed_acc:.3f} lin_top1={lin_top1:.4f} "
          f"M-LF={mlf_top1:.4f} N0={n0_top1:.4f} R0={r0_top1:.4f} [{time.time()-t0:.0f}s]", flush=True)

t = results["targets"]; chance = 1/len(fin)
g0 = t["llama_1b"]["MLF_top1"] >= 0.29
g2 = all(v["N0_top1"] <= 2*chance for v in t.values())
g1 = t["gemma_2b"]["MLF_top1"] >= 0.143
results["gates"] = {"G0_machinery": g0, "G2_null": g2, "G1_bar": g1}
results["verdict"] = ("INVALID__pipeline_broken" if not g0 else
                      "INVALID__pipeline_artifact" if not g2 else
                      "TELEPATHY_BAR_CLEARED__pairing_discoverable" if g1 else
                      "PAIRING_NOT_DISCOVERABLE__at_this_class")
(HERE / 'b34v2_result.json').write_text(json.dumps(results, indent=2)+'\n', encoding='utf-8')
print(f"\nVERDICT: {results['verdict']}  -> b34v2_result.json", flush=True)
