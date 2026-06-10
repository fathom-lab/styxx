"""TELEPATHY v0 — structure-only transmission of novel concepts between minds (frozen prereg).

PREREG_telepathy_v0_2026_06_10.md: unsupervised channel bootstrap (GAVAGAI translator), relational
message (96 distances to the sender's battery), receiver decodes against its own profiles.
Gates T1/T2/T3; VOID self-pair. GPU pass persists telepathy_reps.npz (battery + probes, 10 minds).

Usage:
    python papers/mind-instrument/run_telepathy_v0.py --smoke
    python papers/mind-instrument/run_telepathy_v0.py
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from styxx import mind  # noqa: E402
from run_gavagai_v0 import translate, FAMILY  # noqa: E402  (frozen channel bootstrap)

OUT_REPS = HERE / "telepathy_reps.npz"
SEED = 0
N_NULL = 100
NB = 96

# PROBE SET — the fresh confirm battery, frozen (disjoint from the anchor battery; incl. emotion).
_PROBES = {
    "tool": ["hammer", "screwdriver", "wrench", "drill", "saw", "pliers", "chisel", "axe", "shovel", "rake", "ladder", "clamp"],
    "clothing": ["shirt", "pants", "jacket", "dress", "hat", "scarf", "gloves", "socks", "shoes", "coat", "sweater", "belt"],
    "sport": ["soccer", "tennis", "basketball", "baseball", "golf", "hockey", "boxing", "swimming", "cycling", "skiing", "surfing", "wrestling"],
    "kitchen": ["spoon", "fork", "knife", "plate", "bowl", "cup", "pot", "pan", "kettle", "blender", "oven", "fridge"],
    "building": ["house", "tower", "bridge", "castle", "church", "factory", "stadium", "museum", "hospital", "library", "school", "prison"],
    "plant": ["oak", "pine", "rose", "tulip", "fern", "cactus", "bamboo", "maple", "ivy", "moss", "daisy", "willow"],
    "drink": ["water", "coffee", "tea", "juice", "milk", "wine", "beer", "soda", "lemonade", "cocoa", "cider", "smoothie"],
    "emotion": ["joy", "anger", "fear", "sadness", "surprise", "disgust", "love", "hope", "pride", "shame", "envy", "guilt"],
}
PROBES = [w for ws in _PROBES.values() for w in ws]
PROBE_CAT = [c for c, ws in _PROBES.items() for _ in ws]
NP_ = len(PROBES)

MINDS = [
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct"),
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct"),
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Phi-3.5-mini", "microsoft/Phi-3.5-mini-instruct"),
    ("gemma-2-2b", "google/gemma-2-2b-it"),
    ("gpt2", "gpt2"),
    ("gpt2-large", "gpt2-large"),
    ("pythia-410m", "EleutherAI/pythia-410m"),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"),
]
WORDS = mind.BATTERY + PROBES   # 192 per mind


def normeq_reps(repo: str) -> np.ndarray:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16,
                                               trust_remote_code=True).to(dev).eval()
    fl = max(1, int(0.66 * mdl.config.num_hidden_layers))

    @torch.no_grad()
    def rep(word):
        acc = None
        for t in mind.TEMPLATES:
            ids = tok(t.format(w=word), return_tensors="pt").input_ids.to(mdl.device)
            h = mdl(input_ids=ids, output_hidden_states=True, use_cache=False
                    ).hidden_states[fl][0, -1].float().cpu().numpy()
            h = h / (np.linalg.norm(h) + 1e-9)
            acc = h if acc is None else acc + h
        return acc / len(mind.TEMPLATES)

    R = np.stack([rep(w) for w in WORDS])
    del mdl, tok
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return R


def profiles(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(battery RDM for channel bootstrap, probe->battery distance profiles).
    Frame: center ALL rows on the battery mean; row-normalize; Euclidean distances."""
    bat, prob = R[:NB], R[NB:]
    mu = bat.mean(0)
    def prep(X):
        X = X - mu
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    b, p = prep(bat), prep(prob)
    bat_rdm = np.sqrt(np.maximum(0.0, ((b[:, None, :] - b[None, :, :]) ** 2).sum(-1)))
    msg = np.sqrt(np.maximum(0.0, ((p[:, None, :] - b[None, :, :]) ** 2).sum(-1)))   # (96probes, 96battery)
    return bat_rdm, msg


def decode_acc(msgA: np.ndarray, profB: np.ndarray, pi: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Sender A transmits msgA rows; receiver B correlates through alignment pi (A-i -> B-j)."""
    M = msgA[:, :]                      # (probes, 96 in A's battery order)
    P = profB[:, pi]                    # B's candidate profiles re-indexed into A's order... careful:
    # pi maps A-index i -> B-index pi[i]; message slot i corresponds to B's battery concept pi[i].
    # So compare msg[i] against B-profile column pi[i]: align B's columns BY pi.
    A = M - M.mean(1, keepdims=True)
    B = P - P.mean(1, keepdims=True)
    num = A @ B.T
    den = np.sqrt((A * A).sum(1))[:, None] * np.sqrt((B * B).sum(1))[None, :] + 1e-12
    S = num / den                       # (probes_sent, candidates)
    pred = S.argmax(1)
    truth = np.arange(NP_)
    top1 = float((pred == truth).mean())
    order = np.argsort(-S, axis=1)
    top5 = float(np.mean([(truth[i] in order[i, :5]) for i in range(NP_)]))
    return top1, top5, pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_SMOKE_INVALID" if args.smoke else ""
    rng = np.random.default_rng(SEED)
    minds = MINDS[:2] if args.smoke else MINDS

    reps = {}
    for name, repo in minds:
        reps[name] = normeq_reps(repo)
        print(f"[reps] {name} ({len(WORDS)} words)", flush=True)
    if not args.smoke:
        np.savez(OUT_REPS, **reps)

    pre = {n: profiles(R) for n, R in reps.items()}

    # VOID-PIPELINE: A->A with identity alignment
    n0 = minds[0][0]
    t1, _, _ = decode_acc(pre[n0][1], pre[n0][1], np.arange(NB))
    receipt = {"experiment": "TELEPATHY v0 - structure-only novel-concept transmission",
               "prereg": "papers/mind-instrument/PREREG_telepathy_v0_2026_06_10.md",
               "seed": SEED, "chance": round(1 / NP_, 4), "n_probes": NP_,
               "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "self_pair_identity_decode": round(t1, 4)}
    if t1 < 0.99:
        receipt["verdict"] = "VOID-PIPELINE"
        (HERE / f"telepathy_v0_result{tag}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                             encoding="utf-8")
        print("VOID-PIPELINE", t1); return 2

    rows = []
    for a, _ in minds:
        for b, _ in minds:
            if a == b:
                continue
            bat_a, msg_a = pre[a]
            bat_b, prof_b = pre[b]
            perm = rng.permutation(NB)                      # hide B's labels from the channel
            pi_scr = translate(bat_a, bat_b[perm][:, perm])  # unsupervised, on scrambled B
            pi = perm[pi_scr]                                # back to true B indices
            top1, top5, _ = decode_acc(msg_a, prof_b, pi)
            o1, o5, _ = decode_acc(msg_a, prof_b, np.arange(NB))   # oracle alignment
            rows.append({"a": a, "b": b, "xfam": FAMILY[a] != FAMILY[b],
                         "top1": round(top1, 4), "top5": round(top5, 4),
                         "oracle_top1": round(o1, 4), "oracle_top5": round(o5, 4)})
            print(f"[{'x' if FAMILY[a]!=FAMILY[b] else '='}] {a:13s}->{b:13s} "
                  f"top1={top1:.3f} top5={top5:.3f} (oracle {o1:.3f})", flush=True)

    if args.smoke:
        receipt.update({"smoke": True, "rows": rows, "verdict": "SMOKE-OK"})
        (HERE / f"telepathy_v0_result{tag}.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                             encoding="utf-8")
        print("SMOKE-OK"); return 0

    xf = [r for r in rows if r["xfam"]]
    mean1 = float(np.mean([r["top1"] for r in xf]))
    mean5 = float(np.mean([r["top5"] for r in xf]))
    oracle1 = float(np.mean([r["oracle_top1"] for r in xf]))

    # broken-channel null: random derangement alignments
    null = []
    xs = xf[: min(10, len(xf))]
    for k in range(N_NULL):
        r = xs[k % len(xs)]
        d = rng.permutation(NB)
        while np.any(d == np.arange(NB)):
            d = rng.permutation(NB)
        n1, _, _ = decode_acc(pre[r["a"]][1], pre[r["b"]][1], d)
        null.append(n1)
    null95 = float(np.percentile(null, 95))

    # T3: per-category decode over cross-family pairs (recompute predictions, identity-truth)
    cat_hits = {c: [] for c in _PROBES}
    for r in xf:
        bat_a, msg_a = pre[r["a"]]
        prof_b = pre[r["b"]][1]
        perm = rng.permutation(NB)
        pi = perm[translate(bat_a, pre[r["b"]][0][perm][:, perm])]
        _, _, pred = decode_acc(msg_a, prof_b, pi)
        for i, c in enumerate(PROBE_CAT):
            cat_hits[c].append(float(pred[i] == i))
    t3 = {c: round(float(np.mean(v)), 4) for c, v in cat_hits.items()}

    t1_pass = mean1 >= 0.1042 and mean1 > null95
    receipt.update({
        "rows": rows,
        "mean_xfam_top1": round(mean1, 4), "mean_xfam_top5": round(mean5, 4),
        "T2_oracle_mean_xfam_top1": round(oracle1, 4),
        "null_95th_percentile": round(null95, 4), "n_null": N_NULL,
        "T3_per_category_top1_xfam": dict(sorted(t3.items(), key=lambda kv: -kv[1])),
        "T1_pass": t1_pass,
        "verdict": "TRANSMISSION-DEMONSTRATED" if t1_pass else "CHANNEL-INSUFFICIENT",
    })
    (HERE / "telepathy_v0_result.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("mean_xfam_top1", "mean_xfam_top5",
                                              "T2_oracle_mean_xfam_top1", "null_95th_percentile",
                                              "T3_per_category_top1_xfam", "verdict")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
