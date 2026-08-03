"""B35-b — second source family, per PREREG_b35b_second_source_2026_08_03.md.

Qwen2.5-3B-Instruct as SOURCE: extract at the committed frac rule, then the b34-v3 pipeline
verbatim (seed 343) against qwen_1p5b (same-family G0), gemma_2b (G1), llama_1b (symmetry
probe). One model load, extraction banked; run DETACHED. `--smoke` = 15 concepts, INVALID-only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import AutoConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "introspection-gate"))
from introspection_gate import load_model       # noqa: E402
import run_thought_transfer as P                 # noqa: E402
from run_g0clear import CONCEPTS as FULL_C       # noqa: E402
from styxx_transfer import TransferMap            # noqa: E402
from run_b31v2 import fit_mlp                      # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 343
SRC = "Qwen/Qwen2.5-3B-Instruct"
FRAC = 11 / 28                                    # the committed frac rule (Llama-3B L11/28)
TARGETS = ["qwen_1p5b", "gemma_2b", "llama_1b"]


def load_tgt(tag):
    z = np.load(HERE / f"_b31v2_pts_{tag}.npz", allow_pickle=True)
    return {c: z["pts"][i] for i, c in enumerate(FULL_C)}


def main():
    concepts = FULL_C[:15] if SMOKE else FULL_C
    P.CONCEPTS = concepts
    kstar = json.loads((HERE / "g0clear_result_llama3b.json").read_text())["locked"]["k"]

    capath = HERE / f"_b35b_ptsA_qwen3b{SUFFIX}.npz"
    if capath.exists():
        z = np.load(capath, allow_pickle=True)
        ptsA = {c: z["pts"][i] for i, c in enumerate(concepts)}
        print("loaded banked Qwen-3B source extraction", flush=True)
    else:
        nl = AutoConfig.from_pretrained(SRC).num_hidden_layers
        LA = round(FRAC * nl)
        print(f"extracting {SRC} at layer {LA}/{nl}", flush=True)
        tok, m = load_model(SRC)
        with torch.no_grad():
            ptsA, _ = P.extract(m, tok, LA)
        del m
        torch.cuda.empty_cache()
        np.savez(capath, pts=np.array([ptsA[c] for c in concepts]))
        print("banked", flush=True)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(concepts))
    n_fin = 5 if SMOKE else 70
    fin = [concepts[i] for i in idx[:n_fin]]
    tr = [concepts[i] for i in idx[n_fin:]]
    XA = np.array([ptsA[c] for c in tr])

    results = {"prereg": "PREREG_b35b_second_source_2026_08_03.md", "seed": SEED,
               "smoke": SMOKE, "source": SRC, "n_heldout": len(fin),
               "chance": round(1 / len(fin), 4), "targets": {}}
    nulls = []
    for tag in TARGETS:
        t0 = time.time()
        ptsB = load_tgt(tag)
        XB = np.array([ptsB[c] for c in tr])
        fin_ptsB = np.array([ptsB[c] for c in fin])
        perm = rng.permutation(len(tr)); XBs = XB[perm]; true_col = np.argsort(perm)
        tm = TransferMap.fit(XA, XBs, k=kstar)
        MA = np.stack([tm.transfer_point(x) for x in XA])
        _, col = linear_sum_assignment(np.linalg.norm(MA[:, None, :] - XBs[None, :, :], axis=-1))
        seed_acc = float((col == true_col).mean())
        fn, _ = fit_mlp(XA, XBs[col], seed=SEED)

        def rd(mapper):
            return sum(1 for i, c in enumerate(fin)
                       if int(np.argmin(np.linalg.norm(fin_ptsB - mapper(ptsA[c]), axis=1))) == i) / len(fin)

        read = rd(fn)
        nf, _ = fit_mlp(XA, XBs[rng.permutation(len(tr))], seed=SEED)
        null = rd(nf)
        nulls.append(null)
        results["targets"][tag] = {"seed_acc": round(seed_acc, 4), "read": round(read, 4),
                                   "null": round(null, 4),
                                   "x_chance": round(read * len(fin), 1)}
        print(f">> {tag}: seed_acc={seed_acc:.4f} read={read:.4f} "
              f"({read*len(fin):.0f}x) null={null:.4f} [{time.time()-t0:.0f}s]", flush=True)

    results["max_null_top1"] = round(max(nulls), 4)
    try:
        from styxx.protocol import Experiment
        v = Experiment(HERE / "PREREG_b35b_second_source_2026_08_03.md").score(results, smoke=SMOKE)
        results["verdict"], results["gates"] = v.verdict, v.gates
        results["prereg_commit"] = v.prereg_commit
    except Exception as e:
        results["verdict"] = f"UNSCORED__{type(e).__name__}: {e}"
    (HERE / f"b35b_result{SUFFIX}.json").write_text(json.dumps(results, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
