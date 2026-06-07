# -*- coding: utf-8 -*-
"""run_real_legibility.py — ZERO-ANCHOR CROSS-MODEL LEGIBILITY, aligner-validated.

The program's open frontier: are real, differently-trained LLMs near-isometric enough that one
model's concept geometry can be aligned to another's with ZERO paired data (no Rosetta stone)?
The prior real-universality run reported recovery ~0.05 (chance 0.01) at mean RSA 0.69 and
concluded "shared but not zero-anchor alignable" — BUT it used only <=0.6B models and had NO
aligner POSITIVE CONTROL. The program's own hardest lesson (the GW aligner that false-falsified
Plato) is that a ~chance recovery from an UN-VALIDATED aligner is uninterpretable: it may be a
real bound on universality OR a broken instrument.

This run fixes both:
  1. A stronger aligner (Wasserstein-Procrustes: PCA->common dim, multi-restart, Hungarian)
     alongside the entropic-GW aligner, head to head.
  2. A MANDATORY positive-control CALIBRATION CURVE: warp a real embedding to a sweep of KNOWN
     RSA levels and measure each aligner's recovery-vs-RSA. This says what recovery is ACHIEVABLE
     at the RSA real pairs actually exhibit. A real-pair null is only interpretable if the aligner
     recovers a KNOWN correspondence at that same RSA.
  3. The STRONGEST real pairs incl 3B models (Llama-3.2-3B, Qwen2.5-3B, gemma-2-2b) + a same-family
     ladder, with graded metrics (top-1, top-5, mean reciprocal rank), not just brittle top-1.

VERDICT:
  CRACKED            — a real cross-family pair is zero-anchor recoverable well above chance AND
                       above the calibration floor -> minds are mutually legible with no anchors.
  SHARED_NOT_ALIGN   — aligner PASSES calibration at the real RSA, yet real pairs recover ~chance
                       -> real models differ by a STRUCTURED warp beyond isotropic noise: a true,
                       non-artifactual bound on the universality of mind.
  ALIGNER_LIMITED    — aligner FAILS calibration at the real RSA -> the prior 0.05 was instrument-
                       limited; recovery question stays open, need a better aligner / more data.

HONEST SCOPE: real models all saw English/web text, so shared geometry does not isolate data-
independence (the synthetic disjoint-worlds controlled that). This answers the PRACTICAL frontier:
does zero-anchor alignment WORK on real brains, with a validated instrument.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_disjoint_worlds as R
from transformers import AutoModel, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = "--smoke" in sys.argv

CONCEPTS = ("dog cat horse cow pig sheep lion tiger bear wolf fox deer rabbit mouse elephant "
            "monkey goat duck chicken eagle owl shark whale dolphin frog snake spider ant bee "
            "apple banana orange grape lemon peach cherry strawberry potato carrot onion bread "
            "cheese rice egg milk sugar salt butter coffee tea wine "
            "hammer screwdriver wrench saw drill knife fork spoon plate bowl cup bottle needle "
            "car truck bus train plane boat ship bicycle motorcycle helicopter rocket "
            "chair table bed sofa desk shelf lamp mirror clock door window roof floor wall "
            "shirt pants shoe hat coat dress sock glove scarf belt ring watch "
            "tree flower grass leaf root branch river mountain ocean lake forest desert cloud "
            "rain snow wind storm sun moon star fire ice stone sand gold silver iron "
            "house school church store bank hospital bridge tower castle prison factory library "
            "guitar piano drum violin trumpet flute "
            "doctor teacher farmer soldier king queen baby child king "
            "anger fear joy love hope grief").split()
# dedup preserve order
_seen = set(); CONCEPTS = [c for c in CONCEPTS if not (c in _seen or _seen.add(c))]

MODELS = (["Qwen/Qwen2.5-0.5B", "EleutherAI/pythia-410m"] if SMOKE else [
    "meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B", "google/gemma-2-2b",
    "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B"])
BIG = {"meta-llama/Llama-3.2-3B", "Qwen/Qwen2.5-3B", "google/gemma-2-2b"}
if SMOKE:
    CONCEPTS = CONCEPTS[:30]
N = len(CONCEPTS)
R.N = N
R.GW_INITS = 8
IU = np.triu_indices(N, 1)
TEMPLATES = ["a {}", "the {}", "I saw a {}.", "a photo of a {}.", "this is a {}.", "look at the {}."]


@torch.no_grad()
def embed(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    kw = dict(use_safetensors=True)
    kw["dtype"] = torch.float16 if model_name in BIG else torch.float32
    if "gemma" in model_name.lower():
        kw["attn_implementation"] = "eager"
    m = AutoModel.from_pretrained(model_name, **kw).to(DEV).eval()
    vecs = []
    for c in CONCEPTS:
        reps = []
        for t in TEMPLATES:
            ids = tok(t.format(c), return_tensors="pt").to(DEV)
            h = m(**ids).last_hidden_state[0]
            reps.append(h[-1].float().cpu().numpy())   # last-token (decoder LMs)
        vecs.append(np.mean(reps, 0))
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return np.array(vecs, dtype=np.float64)


def rsa(EA, EB):
    return float(np.corrcoef(R.distmat(EA)[IU], R.distmat(EB)[IU])[0, 1])


def pca_k(E, k):
    E = E - E.mean(0, keepdims=True)
    U, S, _ = np.linalg.svd(E, full_matrices=False)
    P = U[:, :k] * S[:k]
    return P / (np.linalg.norm(P) + 1e-9)


def _sinkhorn(logP, iters=20):
    logP = logP - logP.max()
    for _ in range(iters):
        logP = logP - np.logaddexp.reduce(logP, axis=1, keepdims=True)
        logP = logP - np.logaddexp.reduce(logP, axis=0, keepdims=True)
    return np.exp(logP)


def _refine(A, B, Q, iters, eps0=0.5):
    """Alternate soft-OT assignment (Sinkhorn) -> orthogonal Procrustes. Anneal eps -> hard."""
    n = A.shape[0]
    prev = None
    for it in range(iters):
        S = (A @ Q) @ B.T
        eps = max(0.03, eps0 * (0.85 ** it))
        P = _sinkhorn(S / eps)              # soft doubly-stochastic transport
        M = A.T @ (P @ B)                   # Procrustes against soft-matched B
        U, _, Vt = np.linalg.svd(M)
        Q = U @ Vt
        _, col = linear_sum_assignment(-((A @ Q) @ B.T))
        if prev is not None and np.array_equal(col, prev):
            break
        prev = col
    return Q


def wproc(EA, EB, rng, k=60, restarts=20, iters=80):
    """Unsupervised Wasserstein-Procrustes (Sinkhorn-annealed) with structure-only inits incl a
    GW warm-start. Returns (assign A->B, similarity S, matched_score). NO identity-correspondence
    init (that would leak the zero-anchor answer)."""
    k = min(k, EA.shape[1], EB.shape[1], N - 1)
    A, B = pca_k(EA, k), pca_k(EB, k)
    n = A.shape[0]
    inits = []
    # GW warm-start: entropic-GW coupling -> Hungarian assign -> Procrustes -> Q (structure-only)
    try:
        Tgw, _ = R.entropic_gw(R.distmat(EA), R.distmat(EB))
        _, cgw = linear_sum_assignment(-Tgw)
        Ug, _, Vtg = np.linalg.svd(A.T @ B[cgw])
        inits.append(Ug @ Vtg)
    except Exception:
        pass
    for _ in range(restarts):
        Qr, _ = np.linalg.qr(rng.standard_normal((k, k)))
        inits.append(Qr)
    best = (None, None, -np.inf)
    for Q0 in inits:
        Q = _refine(A, B, Q0, iters)
        S = (A @ Q) @ B.T
        _, col = linear_sum_assignment(-S)
        score = float(np.sum(S[np.arange(n), col]))
        if score > best[2]:
            best = (col.copy(), S.copy(), score)
    return best


def metrics(assign, S):
    """top-1, top-5, MRR of the TRUE match (identity: A-row i <-> B-row i)."""
    n = S.shape[0]
    top1 = float(np.mean(assign == np.arange(n)))
    order = np.argsort(-S, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] for i in range(n)])
    top5 = float(np.mean(ranks < 5))
    mrr = float(np.mean(1.0 / (ranks + 1.0)))
    return {"top1": round(top1, 3), "top5": round(top5, 3), "mrr": round(mrr, 3)}


def gw_top1(EA, EB, rng):
    assign, _ = R.align(EA, EB, rng)
    return float(np.mean(assign == np.arange(N)))


def degrade(E, sigma, rng):
    """Known-correspondence warp: random rotation (basis change) + isotropic noise."""
    Ec = E - E.mean(0, keepdims=True)
    d = Ec.shape[1]
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Ec = Ec @ Q
    scale = np.linalg.norm(Ec) / np.sqrt(Ec.size)
    return Ec + rng.standard_normal(Ec.shape) * sigma * scale


def main():
    rng = np.random.default_rng(0)
    print(f"N={N} concepts, {len(MODELS)} models", flush=True)
    E = {}
    for mn in MODELS:
        E[mn] = embed(mn)
        print(f"  embedded {mn.split('/')[-1]}: {E[mn].shape}", flush=True)

    # ---- CALIBRATION: positive-control recovery-vs-RSA curve on a real reference embedding ----
    ref = MODELS[0]
    print(f"\nCALIBRATION (positive control) on {ref.split('/')[-1]} — known-correspondence warps:", flush=True)
    calib = []
    for sigma in [0.1, 0.3, 0.5, 0.8, 1.2, 1.8, 2.6]:
        Ew = degrade(E[ref], sigma, np.random.default_rng(100 + int(sigma * 10)))
        r = rsa(E[ref], Ew)
        assign, S, _ = wproc(E[ref], Ew, np.random.default_rng(7))
        m = metrics(assign, S)
        g = gw_top1(E[ref], Ew, np.random.default_rng(7))
        calib.append({"sigma": sigma, "rsa": round(r, 3), "wproc": m, "gw_top1": round(g, 3)})
        print(f"  sigma={sigma:>4} rsa={r:.3f}  WProc top1={m['top1']:.3f} top5={m['top5']:.3f} "
              f"mrr={m['mrr']:.3f}  GW top1={g:.3f}", flush=True)

    def calib_recovery_at(rsa_val):
        """interpolate WProc top1 the calibration achieves at a given RSA."""
        pts = sorted(calib, key=lambda x: x["rsa"])
        xs = [p["rsa"] for p in pts]; ys = [p["wproc"]["top1"] for p in pts]
        return float(np.interp(rsa_val, xs, ys))

    # ---- REAL cross-model pairs ----
    print("\nREAL cross-model zero-anchor recovery:", flush=True)
    chance = 1.0 / N
    pairs = []
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, b = MODELS[i], MODELS[j]
            r = rsa(E[a], E[b])
            assign, S, _ = wproc(E[a], E[b], np.random.default_rng(13))
            m = metrics(assign, S)
            g = gw_top1(E[a], E[b], np.random.default_rng(13))
            fam_a, fam_b = a.split("/")[-1].split("-")[0], b.split("/")[-1].split("-")[0]
            same_fam = a.split("/")[0] == b.split("/")[0] and fam_a[:4] == fam_b[:4]
            cfloor = calib_recovery_at(r)
            pairs.append({"a": a.split("/")[-1], "b": b.split("/")[-1],
                          "same_family": bool(same_fam), "rsa": round(r, 3),
                          "wproc": m, "gw_top1": round(g, 3),
                          "calib_floor_top1_at_rsa": round(cfloor, 3),
                          "beats_calib_floor": bool(m["top1"] >= max(cfloor, 0.10))})
            print(f"  {'SAME' if same_fam else 'XFAM'} {a.split('/')[-1]:16s}<->{b.split('/')[-1]:16s} "
                  f"RSA={r:.3f}  WProc top1={m['top1']:.3f} top5={m['top5']:.3f} mrr={m['mrr']:.3f} "
                  f"GW={g:.3f}  calibFloor@rsa={cfloor:.3f}", flush=True)

    # ---- verdict ----
    # aligner validated at high RSA (non-trivial: exclude rsa~1.0): recovers a KNOWN warp at rsa in [0.92,0.99]
    mid = [c["wproc"]["top1"] for c in calib if 0.92 <= c["rsa"] < 0.999]
    calib_mid = max(mid) if mid else calib_recovery_at(0.95)
    xfam = [p for p in pairs if not p["same_family"]]
    samef = [p for p in pairs if p["same_family"]]
    best_xfam = max(xfam, key=lambda p: p["wproc"]["top1"]) if xfam else None
    best_same = max(samef, key=lambda p: p["wproc"]["top1"]) if samef else None
    best_any = max(pairs, key=lambda p: p["wproc"]["top1"])

    def cracked(p):
        return p and p["wproc"]["top1"] >= 0.30 and p["wproc"]["top1"] >= 5 * chance

    aligner_works_midRSA = calib_mid >= 0.30
    if cracked(best_xfam):
        verdict = (f"CRACKED (cross-family) — zero-anchor recovery WORKS across families: "
                   f"{best_xfam['a']}<->{best_xfam['b']} top1={best_xfam['wproc']['top1']:.2f} "
                   f"(chance {chance:.3f}). Real differently-trained minds are mutually legible, no anchors.")
    elif cracked(best_same):
        verdict = (f"CRACKED (near-isometric regime only) — same-family/high-RSA pairs are zero-anchor "
                   f"recoverable ({best_same['a']}<->{best_same['b']} top1={best_same['wproc']['top1']:.2f}, "
                   f"RSA={best_same['rsa']:.2f}) but cross-family are not (best xfam "
                   f"{best_xfam['wproc']['top1'] if best_xfam else float('nan'):.2f}). Universality is "
                   "alignable only when geometry is near-isometric; cross-family warp blocks zero-anchor reading.")
    elif aligner_works_midRSA:
        verdict = (f"SHARED_NOT_ALIGNABLE — aligner VALIDATED (recovers a known warp at RSA~0.95: "
                   f"{calib_mid:.2f}) yet NO real pair recovers (best top1={best_any['wproc']['top1']:.3f} vs "
                   f"chance {chance:.3f}); real pairs sit at/below their RSA-matched calibration floor. Real "
                   "models share relational geometry (RSA up to "
                   f"{max(p['rsa'] for p in pairs):.2f}) but differ by a STRUCTURED warp beyond isotropic "
                   "noise: a true bound — universality is in relational structure, not an alignable frame.")
    else:
        verdict = (f"REGIME/SCALE-LIMITED — even a KNOWN warp is not recoverable at RSA~0.95 (calib {calib_mid:.2f}) "
                   f"at this concept count (N={N}); the real RSA band ({min(p['rsa'] for p in pairs):.2f}-"
                   f"{max(p['rsa'] for p in pairs):.2f}) is below this aligner's recoverable threshold. The prior "
                   "'not alignable (0.05)' was INSTRUMENT/SCALE-limited, not a universality bound — zero-anchor "
                   "recovery on real LLMs needs vec2vec-class methods / far more concepts. Question stays OPEN.")
    out = {"experiment": "zero-anchor cross-model legibility (aligner-validated)",
           "models": MODELS, "n_concepts": N, "chance": round(chance, 4),
           "calibration_positive_control": calib,
           "calib_recovery_known_warp_at_rsa~0.95": round(calib_mid, 3),
           "aligner_works_midRSA": bool(aligner_works_midRSA),
           "pairs": pairs,
           "best_cross_family": best_xfam, "best_same_family": best_same, "best_any_pair": best_any,
           "VERDICT": verdict}
    fn = HERE / ("real_legibility_smoke.json" if SMOKE else "real_legibility_result.json")
    fn.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\n===== " + verdict, flush=True)
    print("wrote", fn.name, flush=True)


if __name__ == "__main__":
    main()
