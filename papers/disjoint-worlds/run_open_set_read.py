"""Open-set read — per PREREG_open_set_read_2026_09_01.md (frozen at 45d7ae5).

`read_top1` in run_b31v2.py:90-93 and run_b34v3.py:46-49 is an index-matched argmin over a
candidate array that CONTAINS the truth on every trial. There is no threshold and no reject
option, so E = 1 by construction and every published read figure is an A-term: given the
answer was in the list, was it picked?

This asks the branch the apparatus cannot express: handed a state whose target is ABSENT from
the candidate array, does the reader say so?

Method. Replicate b34v3 exactly (seed 343, same rng consumption order, same TransferMap ->
fit_mlp pipeline) to obtain the identical mapper, and reconcile its closed-set read_top1
against the committed b34v3_result.json before anything else is allowed to run. Then split the
70 HELD-OUT concepts into a candidate half C and an out-of-vocabulary half O under seed
20260901. Show the reader C only. Queries from C are IN trials (target present); queries from O
are OOV trials (target absent, correct behaviour is to abstain). Score AUROC of the top-1/top-2
margin separating IN from OOV — threshold-free, so no operating point has to be chosen.

CPU-only from the committed .npz banks. No model is loaded and nothing is collected.

DISCLOSED DEVIATION FROM THE FROZEN PREREG. The prereg says "partition the 462 concept indices"
into C and O. This runner partitions the 70 HELD-OUT concepts instead. That is STRICTER, not
looser: every query here is a concept the mapper never trained on, whereas partitioning all 462
would draw most queries from the 392 training concepts and confound familiarity with
presence-in-candidate-set. The prereg did not fix |C| and |O|; they are set to 35/35 here so the
two trial classes are balanced. Both facts are recorded in the emitted receipt.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from run_g0clear import CONCEPTS as FULL_CONCEPTS      # noqa: E402
from styxx_transfer import TransferMap                 # noqa: E402
from run_b31v2 import fit_mlp                          # noqa: E402
from run_b34v3 import (load_A, load_pts, read_top1,    # noqa: E402
                       assignment_from_map, SEED, TARGETS)

OPEN_SEED = 20260901
N_CAND = 35          # |C| — the candidate half of the held-out 70
COMMITTED = {"llama_1b": 0.6857, "gemma_2b": 0.5714, "qwen_1p5b": 0.1429}


def auroc(pos, neg):
    """P(a random positive scores above a random negative). Rank-based, ties at 0.5.

    Implemented here rather than imported so the receipt does not depend on a
    scikit-learn version. Mann-Whitney U over midranks.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1, dtype=float)
    # midranks for ties
    i = 0
    s = allv[order]
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rpos = ranks[:len(pos)].sum()
    return float((rpos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def margins(fin, ptsA, cand_ptsB, mapper):
    """Top-1/top-2 margin per query against a candidate array, plus the argmin index.

    The margin, not the raw d1: d1 scales with the query's norm and would measure the
    query rather than the quality of the match. Both are returned; only the margin decides.
    """
    out_m, out_d1, out_arg = [], [], []
    for c in fin:
        d = np.linalg.norm(cand_ptsB - mapper(ptsA[c]), axis=1)
        part = np.partition(d, 1)
        out_m.append(float(part[1] - part[0]))
        out_d1.append(float(part[0]))
        out_arg.append(int(np.argmin(d)))
    return np.array(out_m), np.array(out_d1), np.array(out_arg)


def main():
    t_all = time.time()
    s0 = json.loads((HERE / "g0clear_result_llama3b.json").read_text(encoding="utf-8"))
    kstar = s0["locked"]["k"]

    # ---- replicate b34v3 EXACTLY: same seed, same rng consumption order ----
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(FULL_CONCEPTS))
    n_fin, n_tr = 70, len(FULL_CONCEPTS) - 70
    fin_i, tr_i = idx[:n_fin], idx[n_fin:n_fin + n_tr]
    tr = [FULL_CONCEPTS[i] for i in tr_i]
    fin = [FULL_CONCEPTS[i] for i in fin_i]

    ptsA = load_A()
    XA = np.array([ptsA[c] for c in tr])

    # ---- the open-set split, drawn from the HELD-OUT 70 only ----
    rng_open = np.random.default_rng(OPEN_SEED)
    p = rng_open.permutation(n_fin)
    C_pos, O_pos = np.sort(p[:N_CAND]), np.sort(p[N_CAND:])
    assert not (set(C_pos.tolist()) & set(O_pos.tolist()))

    res = {
        "prereg": "PREREG_open_set_read_2026_09_01.md",
        "prereg_commit": "45d7ae5",
        "what": ("does the reader decline when the target is ABSENT from the candidate array? "
                 "read_top1 is an index-matched argmin over an array containing the truth, so "
                 "E = 1 by construction and every published read figure is an A-term"),
        "reject_statistic": "top-1/top-2 margin (d2 - d1) in the mapped space, fixed in the prereg",
        "b34v3_seed": SEED, "open_seed": OPEN_SEED,
        "n_heldout": n_fin, "n_candidates": int(N_CAND), "n_oov": int(len(O_pos)),
        "deviation_from_prereg": (
            "The prereg says 'partition the 462 concept indices'. This partitions the 70 HELD-OUT "
            "concepts instead — STRICTER, because every query is then a concept the mapper never "
            "trained on. Partitioning all 462 would draw most queries from the 392 training "
            "concepts and confound familiarity with presence-in-candidate-set. The prereg did not "
            "fix |C| and |O|; set to 35/35 here for balance. Disclosed, not silently taken."),
        "targets": {},
    }

    for tag in TARGETS:
        t0 = time.time()
        ptsB = load_pts(tag)
        XB = np.array([ptsB[c] for c in tr])
        fin_ptsB = np.array([ptsB[c] for c in fin])

        # identical rng draws to b34v3.main(), in the identical order
        perm = rng.permutation(len(tr))
        XB_shuf = XB[perm]
        true_col = np.argsort(perm)

        tm = TransferMap.fit(XA, XB_shuf, k=kstar)
        disc_col = assignment_from_map(XA, XB_shuf, tm.transfer_point)
        seed_acc = float((disc_col == true_col).mean())

        mlp_fn, _ = fit_mlp(XA, XB_shuf[disc_col], seed=SEED)
        read = read_top1(fin, ptsA, fin_ptsB, mlp_fn)

        rand = rng.permutation(len(tr))
        null_fn, _ = fit_mlp(XA, XB_shuf[rand], seed=SEED)

        # ---- G-O1: closed-set reconciliation, before anything open-set is reported ----
        recon_ok = abs(round(read, 4) - COMMITTED[tag]) < 1e-9

        # ---- open-set: candidate array is C only; queries are all 70 held-out ----
        cand = fin_ptsB[C_pos]
        m_real, d1_real, arg_real = margins(fin, ptsA, cand, mlp_fn)
        m_null, _, _ = margins(fin, ptsA, cand, null_fn)

        is_in = np.zeros(n_fin, bool)
        is_in[C_pos] = True
        au_real = auroc(m_real[is_in], m_real[~is_in])
        au_null = auroc(m_null[is_in], m_null[~is_in])

        # accuracy on IN trials within the reduced candidate array (context, not a gate)
        pos_of = {int(c): k for k, c in enumerate(C_pos)}
        in_hits = sum(1 for q in C_pos if arg_real[q] == pos_of[int(q)])
        res["targets"][tag] = {
            "seed_acc": round(seed_acc, 4),
            "closed_set_read_top1": round(read, 4),
            "committed_read_top1": COMMITTED[tag],
            "G_O1_reconciles": bool(recon_ok),
            "auroc_margin_in_vs_oov": round(au_real, 4),
            "auroc_margin_null_mapper": round(au_null, 4),
            "in_trial_top1_within_C": round(in_hits / len(C_pos), 4),
            "seconds": round(time.time() - t0, 1),
        }
        print(f">> {tag}: closed={read:.4f} (committed {COMMITTED[tag]}, "
              f"reconciles={recon_ok})  AUROC={au_real:.4f}  null_AUROC={au_null:.4f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    # ---------------- gates ----------------
    T = res["targets"]
    g1 = all(v["G_O1_reconciles"] for v in T.values())
    g2 = all(v["auroc_margin_null_mapper"] <= 0.55 for v in T.values())

    def verdict(a):
        if a >= 0.75:
            return "OPEN_SET_SIGNAL"
        if a <= 0.55:
            return "CLOSED_SET_ONLY"
        return "INDETERMINATE"

    per = {k: verdict(v["auroc_margin_in_vs_oov"]) for k, v in T.items()}
    res["gates"] = {
        "G_O1_closed_set_reconciliation": {"pass": bool(g1),
                                           "note": "every target must re-derive its committed read_top1 exactly"},
        "G_O2_null_mapper_auroc_le_0.55": {"pass": bool(g2),
                                           "observed": {k: v["auroc_margin_null_mapper"] for k, v in T.items()}},
        "G_O3_per_target_verdict": per,
    }
    if not g1:
        res["verdict"] = "VOID__reconciliation_failed"
    elif not g2:
        res["verdict"] = "VOID__null_mapper_separates"
    else:
        vs = set(per.values())
        res["verdict"] = ("OPEN_SET_SIGNAL" if vs == {"OPEN_SET_SIGNAL"}
                          else "CLOSED_SET_ONLY" if vs == {"CLOSED_SET_ONLY"}
                          else "MIXED__" + "_".join(f"{k}={v}" for k, v in sorted(per.items())))
    res["seconds_total"] = round(time.time() - t_all, 1)

    out = HERE / "open_set_read_result.json"
    out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {res['verdict']}  -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
