"""B31-v2 — heavy-machinery content transport, per PREREG_b31v2_content_transport_2026_08_01.md.

Is the cross-family content-reading cliff a map-capacity limit or bedrock? The rung-2
apparatus VERBATIM (P.extract, the N=462 battery, split_concepts(seed=0), the held-out top-1
read metric), plus two new cells per target:

  M0  linear TransferMap (the rung-2 class)             — replication control
  M1  two-layer MLP adapter fit on PAIRED train anchors — the capacity bet
  N1  pairing-shuffled M1 (same arch/training, shuffled correspondence) — specificity null

Targets: Llama-3.2-1B (G0 machinery), gemma-2-2b-it (G1 decisive cell), Qwen2.5-1.5B (context).
Frozen gates in the prereg; the closed-negative branch is pre-committed. Checkpointed per
model (`_b31v2_pts_*.npz`); `--smoke` runs 5 concepts and writes `_smoke`-suffixed files only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "introspection-gate"))

from introspection_gate import load_model                     # noqa: E402
from styxx_transfer import TransferMap                        # noqa: E402
import run_thought_transfer as P                              # noqa: E402
from run_g0clear import CONCEPTS as FULL_CONCEPTS, split_concepts  # noqa: E402

SMOKE = "--smoke" in sys.argv
SUFFIX = "_smoke" if SMOKE else ""
SEED = 31
SRC = "meta-llama/Llama-3.2-3B-Instruct"
CACHE_A = HERE / "_rung2_ptsA_vecsA.npz"      # reuse the committed rung-2 extraction of A
TARGETS = [
    ("llama_1b", "meta-llama/Llama-3.2-1B-Instruct"),   # G0 same-family machinery control
    ("gemma_2b", "google/gemma-2-2b-it"),               # G1 decisive cell (RSA 0.955, M0 = chance)
    ("qwen_1p5b", "Qwen/Qwen2.5-1.5B-Instruct"),        # context (the 4x cell)
]


# ---------------------------------------------------------------- M1: the MLP adapter

class MLPMap(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        h = 2 * d_in
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, h), torch.nn.GELU(), torch.nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(XA, XB, seed=SEED, epochs=1500, lr=1e-3, wd=1e-2):
    """Standardize by TRAIN stats, full-batch Adam, deterministic. Returns a numpy fn."""
    torch.manual_seed(seed)
    muA, sdA = XA.mean(0), XA.std(0) + 1e-6
    muB, sdB = XB.mean(0), XB.std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.tensor((XA - muA) / sdA, dtype=torch.float32, device=dev)
    b = torch.tensor((XB - muB) / sdB, dtype=torch.float32, device=dev)
    net = MLPMap(XA.shape[1], XB.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(a), b)
        loss.backward()
        opt.step()
    net.eval()

    def apply(xA):
        with torch.no_grad():
            z = torch.tensor((xA - muA) / sdA, dtype=torch.float32, device=dev)
            return net(z).cpu().numpy() * sdB + muB
    return apply, float(loss.item())


def read_top1(fin, ptsA, fin_ptsB, mapper):
    hits = sum(1 for i, c in enumerate(fin)
               if int(np.argmin(np.linalg.norm(fin_ptsB - mapper(ptsA[c]), axis=1))) == i)
    return hits / len(fin)


# ---------------------------------------------------------------- run

def main():
    if SMOKE:
        concepts = FULL_CONCEPTS[:5]
        tr, fin = concepts[:3], concepts[3:]
    else:
        concepts = FULL_CONCEPTS
        tr, sel, fin = split_concepts(seed=0)   # 323 / 69 / 70 — the committed split
        tr = list(tr) + list(sel)               # M1 may use train+sel; held-out NEVER seen
    P.CONCEPTS = concepts

    s0 = json.loads((HERE / "g0clear_result_llama3b.json").read_text(encoding="utf-8"))
    Lstar, kstar = s0["locked"]["layer"], s0["locked"]["k"]
    frac = Lstar / AutoConfig.from_pretrained(SRC).num_hidden_layers
    print(f"[b31v2{SUFFIX}] layer_A={Lstar} (frac {frac:.3f}) k={kstar} | "
          f"{len(concepts)} concepts, {len(tr)} train, {len(fin)} held-out", flush=True)

    # ---- A (source) ----
    if not SMOKE and CACHE_A.exists():
        z = np.load(CACHE_A, allow_pickle=True)
        ptsA = {c: z["pts"][i] for i, c in enumerate(FULL_CONCEPTS)}
        print("loaded committed rung-2 A cache", flush=True)
    else:
        capath = HERE / f"_b31v2_ptsA{SUFFIX}.npz"
        if capath.exists():
            z = np.load(capath, allow_pickle=True)
            ptsA = {c: z["pts"][i] for i, c in enumerate(concepts)}
        else:
            tokA, mA = load_model(SRC)
            with torch.no_grad():
                ptsA, _ = P.extract(mA, tokA, Lstar)
            del mA
            torch.cuda.empty_cache()
            np.savez(capath, pts=np.array([ptsA[c] for c in concepts]))
        print("extracted A", flush=True)
    RA = np.array([ptsA[c] for c in concepts])
    idx_tr = [concepts.index(c) for c in tr]

    rng = np.random.default_rng(SEED)
    results = {"prereg": "PREREG_b31v2_content_transport_2026_08_01.md",
               "seed": SEED, "smoke": SMOKE, "n_heldout": len(fin),
               "chance": round(1 / len(fin), 4), "targets": {}}

    for tag, hf in TARGETS:
        cpath = HERE / f"_b31v2_pts_{tag}{SUFFIX}.npz"
        t0 = time.time()
        nlB = AutoConfig.from_pretrained(hf).num_hidden_layers
        LB = round(frac * nlB)
        if cpath.exists():
            z = np.load(cpath, allow_pickle=True)
            ptsB = {c: z["pts"][i] for i, c in enumerate(concepts)}
            print(f">> {tag}: loaded cached extraction", flush=True)
        else:
            tokB, mB = load_model(hf)
            with torch.no_grad():
                ptsB, _ = P.extract(mB, tokB, LB)
            del mB
            torch.cuda.empty_cache()
            np.savez(cpath, pts=np.array([ptsB[c] for c in concepts]))
            print(f">> {tag}: extracted in {time.time()-t0:.0f}s", flush=True)
        RB = np.array([ptsB[c] for c in concepts])
        fin_ptsB = np.array([ptsB[c] for c in fin])

        # M0 — linear (rung-2 class), replication control
        tm = TransferMap.fit(RA[idx_tr], RB[idx_tr], k=kstar)
        m0 = read_top1(fin, ptsA, fin_ptsB, tm.transfer_point)

        # M1 — MLP on paired anchors
        m1_fn, m1_loss = fit_mlp(RA[idx_tr], RB[idx_tr])
        m1 = read_top1(fin, ptsA, fin_ptsB, m1_fn)

        # N1 — pairing-shuffled null (same arch/training, shuffled correspondence)
        perm = rng.permutation(len(idx_tr))
        n1_fn, _ = fit_mlp(RA[idx_tr], RB[np.array(idx_tr)[perm]])
        n1 = read_top1(fin, ptsA, fin_ptsB, n1_fn)

        results["targets"][tag] = {
            "hf": hf, "layer_B": LB,
            "M0_linear_top1": round(m0, 4), "M1_mlp_top1": round(m1, 4),
            "N1_shuffled_top1": round(n1, 4), "m1_train_loss": round(m1_loss, 5),
            "x_chance_M1": round(m1 * len(fin), 1),
        }
        print(f">> {tag}: M0={m0:.4f} M1={m1:.4f} N1={n1:.4f} "
              f"(chance {1/len(fin):.4f}) [{time.time()-t0:.0f}s]", flush=True)

    # ---- gates (frozen; smoke is INVALID-only) ----
    if SMOKE:
        results["verdict"] = "INVALID__smoke_plumbing_only"
    else:
        t = results["targets"]
        chance = 1 / len(fin)
        g0 = t["llama_1b"]["M1_mlp_top1"] >= 0.53
        g2 = all(v["N1_shuffled_top1"] <= 2 * chance for v in t.values())
        g1 = (t["gemma_2b"]["M1_mlp_top1"] >= 0.143
              and t["gemma_2b"]["M1_mlp_top1"] >= 5 * max(t["gemma_2b"]["M0_linear_top1"], chance))
        results["gates"] = {"G0_machinery": g0, "G2_specificity_null": g2, "G1_door": g1}
        if not g0:
            results["verdict"] = "INVALID__map_class_broken"
        elif not g2:
            results["verdict"] = "INVALID__architecture_artifact"
        elif g1:
            results["verdict"] = "DOOR_OPENS__content_capacity_limited"
        else:
            results["verdict"] = "DOOR_CLOSES__cliff_not_capacity_limited_at_this_class"

    out = HERE / f"b31v2_result{SUFFIX}.json"
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nVERDICT: {results['verdict']}  -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
