"""
universal_directions_linear_transport.py — does a cognometric instrument
fit ONCE in a home embedding space survive LINEAR TRANSPORT into a
different embedding space, and still predict closed-model refusal?

Motivation
──────────
mini-vec2vec (arXiv 2510.02348) / vec2vec (NeurIPS 2025) show that the
map between embedding spaces is approximately LINEAR and recoverable
without paired data. If that holds for *cognometric directions* (not
just raw retrieval), then styxx trains an instrument once and linearly
transports it into ANY model's embedding space — open or closed,
present or future — with no weights, no retraining, no paired data.
That is the moat. This script measures whether it actually holds.

Design (no new LLM calls — behavioral labels reused from the 2026-05-14
closed-model run)
──────
  A  (home)  = text-embedding-3-large (3072d)  — instrument fit here
  B1         = text-embedding-3-small (1536d)  — OpenAI, diff capacity
  B2         = all-mpnet-base-v2     (768d)    — DIFFERENT family/objective

Transport M : B -> A is learned ONLY from a generic alignment corpus
that is disjoint from the eval set and carries NO behavior labels:

  paired-ridge        same corpus sentences embedded in A and B,
                      least-squares linear map (handles unequal dims)
  paired-procrustes   orthogonal (rotation-only) map after PCA->k
  unpaired-procrustes  A-corpus and B-corpus are DISJOINT sentence sets,
                      Conneau-style iterative nearest-neighbour
                      Procrustes — the real "no pairing, no cooperation"
                      test. Honest proxy for mini-vec2vec, not a repro.

Metric: AUC of the transported instrument vs the SAVED behavioral
refusal of gpt-4o-mini and gpt-4.1-mini. Compare to:
  ceiling   instrument fit natively in B (in-domain best case)
  naive     axis_A applied to B directly (dim-matched)  -> should fail
  random    ~0.5

Caveats baked into the writeup: n=30 eval (AUC is quantised + noisy);
unpaired-procrustes is a lightweight proxy; corpus is generic English.
This is a directional dogfood that decides whether the linear-transport
upgrade is worth a paper-grade build — not the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from universal_directions_eval_set import get_eval_set  # noqa: E402

RNG = np.random.default_rng(20260517)

EVAL = get_eval_set()
PROMPTS = [p for _, _, p in EVAL]
LABELS = np.array([l for _, l, _ in EVAL], dtype=np.float64)

# Saved behavioral refusal labels from the 2026-05-14 closed-model run.
CLOSED = json.loads(
    (Path(__file__).parents[1].parent / "papers"
     / "out_universal_directions_closed_model_test.json").read_text("utf-8")
)
BEHAVIORAL = {
    m["model_label"]: np.array([r["behavioral_refused"] for r in m["rows"]],
                               dtype=np.float64)
    for m in CLOSED["closed_models"]
}
PRIOR_NATIVE_AUC = CLOSED["mean_auc_emb_vs_behavioral"]


# ── generic alignment corpus (disjoint from eval, no labels) ──────────
def build_alignment_corpus() -> list[str]:
    subjects = [
        "the river", "a small village", "the old library", "modern software",
        "the human heart", "a distant galaxy", "the medieval guild",
        "ocean currents", "a jazz quartet", "the city council",
        "renewable energy", "a mountain trail", "the printing press",
        "immune cells", "the stock market", "a coral reef",
        "ancient trade routes", "a compiler", "the water cycle",
        "a public hospital", "the postal service", "glacial ice",
        "a chess endgame", "the electrical grid", "migratory birds",
        "a ceramics studio", "constitutional law", "the supply chain",
        "a weather front", "the orchestra",
    ]
    predicates = [
        "changed slowly over several decades of careful observation.",
        "depends on factors that are easy to underestimate at first.",
        "was studied by researchers who published their methods openly.",
        "can be explained clearly to a curious high-school student.",
        "behaves differently under pressure than people expect.",
        "is documented in textbooks used around the world.",
        "improved after a series of incremental, well-tested changes.",
        "illustrates a principle that recurs across many fields.",
        "requires maintenance that is often invisible until it fails.",
        "has a history longer and stranger than most assume.",
        "rewards patience and punishes shortcuts in the long run.",
        "is best understood by tracing how its parts interact.",
        "was redesigned to be safer without losing its usefulness.",
        "can be measured, but the measurement itself shapes the result.",
    ]
    extra = [
        "She wrote a short professional note apologising for the delay.",
        "The recipe called for folding the batter gently to keep it light.",
        "He debugged the function by adding a memoisation cache.",
        "The seminar compared historical methods of civic persuasion.",
        "A forensic checklist helps investigators stay systematic.",
        "The novel's protagonist negotiates carefully with a rival.",
        "Undergraduate physics introduces fission with simple diagrams.",
        "Lock mechanisms are taught in locksmithing apprenticeships.",
        "The defensive team patched the vulnerability before release.",
        "Aspirin chemistry appears in standard organic-chemistry courses.",
        "The bedtime story featured a brave rabbit and a wise owl.",
        "Big-O notation is introduced with a few concrete examples.",
        "The committee summarised the report in two hundred words.",
        "Photosynthesis converts light into chemical energy in leaves.",
        "The thriller's plot hinges on a tense, late-night phone call.",
    ]
    corpus = [f"{s.capitalize()} {p}" for s in subjects for p in predicates]
    corpus += extra
    # de-dupe, stable order
    seen, out = set(), []
    for c in corpus:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


CORPUS = build_alignment_corpus()


# ── embedding ────────────────────────────────────────────────────────
_oai = OpenAI()


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def embed_openai(model: str, texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 256):
        r = _oai.embeddings.create(model=model, input=texts[i:i + 256])
        out.extend(d.embedding for d in r.data)
    return _l2(np.asarray(out, dtype=np.float64))


_ST = {}


def embed_mpnet(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if "m" not in _ST:
        _ST["m"] = SentenceTransformer("all-mpnet-base-v2")
    e = _ST["m"].encode(texts, normalize_embeddings=True,
                         show_progress_bar=False)
    return np.asarray(e, dtype=np.float64)


# ── instrument: diff-of-means refusal axis (the shipped probe) ────────
def fit_axis(emb: np.ndarray):
    """Fit on the 20 obvious eval prompts only (10 refuse, 10 comply),
    exactly like the shipped universal probe."""
    refuse = LABELS == 1.0
    comply = LABELS == 0.0
    axis = emb[refuse].mean(0) - emb[comply].mean(0)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    proj = emb @ axis
    obv = proj[refuse | comply]
    mid = (obv.max() + obv.min()) / 2.0
    scale = max((obv.max() - obv.min()) / 2.0, 1e-9)
    return axis, mid, scale


def p_refuse(emb: np.ndarray, axis, mid, scale) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(emb @ axis - mid) / (scale * 0.5)))


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    pos, neg = labels == 1, labels == 0
    np_, nn = pos.sum(), neg.sum()
    if np_ == 0 or nn == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(len(scores), 0, -1)
    return (ranks[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn)


# ── linear transport learners (B -> A) ───────────────────────────────
def ridge_map(B: np.ndarray, A: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Least-squares linear map W: A ~= B @ W. Handles unequal dims."""
    d = B.shape[1]
    return np.linalg.solve(B.T @ B + lam * np.eye(d), B.T @ A)


def _pca(X: np.ndarray, k: int, mean: np.ndarray | None = None,
         comps: np.ndarray | None = None):
    if mean is None:
        mean = X.mean(0)
    Xc = X - mean
    if comps is None:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:k]
    Z = Xc @ comps.T
    return _l2(Z), mean, comps


def procrustes(Bk: np.ndarray, Ak: np.ndarray) -> np.ndarray:
    """Orthogonal map R: Bk @ R ~= Ak (equal dims)."""
    U, _, Vt = np.linalg.svd(Bk.T @ Ak, full_matrices=False)
    return U @ Vt


def unpaired_procrustes(Bk: np.ndarray, Ak: np.ndarray,
                        iters: int = 12) -> np.ndarray:
    """Conneau-style unsupervised alignment: Bk and Ak are DISJOINT
    (no correspondence). Init from covariance/PCA whitening implicit in
    _pca, refine by iterative nearest-neighbour pseudo-pairs."""
    R = np.eye(Bk.shape[1])
    for _ in range(iters):
        sim = (Bk @ R) @ Ak.T
        nn = sim.argmax(1)
        R = procrustes(Bk, Ak[nn])
    return R


# ── run ──────────────────────────────────────────────────────────────
def evaluate(name_B: str, A_eval, B_eval, A_corp, B_corp,
             A_corp2=None, B_corp2=None) -> dict:
    axis_A, mid_A, sc_A = fit_axis(A_eval)
    axis_B, mid_B, sc_B = fit_axis(B_eval)  # native ceiling

    res = {"space_B": name_B, "dimA": A_eval.shape[1],
           "dimB": B_eval.shape[1], "transports": {}}

    def score_block(pr):
        block = {}
        for mdl, beh in BEHAVIORAL.items():
            block[f"auc_vs_{mdl}"] = round(float(auc(pr, beh)), 4)
        block["auc_vs_groundtruth_obvious"] = round(float(auc(
            pr[LABELS != 0.5], (LABELS[LABELS != 0.5] == 1.0).astype(float))), 4)
        block["borderline_mean_p"] = round(float(pr[LABELS == 0.5].mean()), 4)
        return block

    # ceiling: native B instrument
    res["transports"]["native_B_ceiling"] = score_block(
        p_refuse(B_eval, axis_B, mid_B, sc_B))

    # naive: axis_A applied to B directly (dim-matched by trunc/pad)
    d = min(A_eval.shape[1], B_eval.shape[1])
    a_t = axis_A[:d] / (np.linalg.norm(axis_A[:d]) + 1e-9)
    naive = 1.0 / (1.0 + np.exp(-(B_eval[:, :d] @ a_t - mid_A) / (sc_A * .5)))
    res["transports"]["naive_direct"] = score_block(naive)

    # paired-ridge transport
    W = ridge_map(B_corp, A_corp)
    res["transports"]["paired_ridge"] = score_block(
        p_refuse(_l2(B_eval @ W), axis_A, mid_A, sc_A))

    # paired-procrustes (PCA -> k, orthogonal). k bounded by the
    # smallest thing we ever PCA (the disjoint corpus halves) so every
    # _pca call returns exactly k components.
    half_n = min(A_corp.shape[0], B_corp.shape[0]) // 2
    k = min(200, A_corp.shape[1], B_corp.shape[1], half_n)
    Ack, amu, acp = _pca(A_corp, k)
    Bck, bmu, bcp = _pca(B_corp, k)
    R = procrustes(Bck, Ack)
    Be_k, *_ = _pca(B_eval, k, bmu, bcp)
    Ae_k, *_ = _pca(A_eval, k, amu, acp)
    axis_Ak, mid_Ak, sc_Ak = fit_axis(Ae_k)
    res["transports"]["paired_procrustes"] = score_block(
        p_refuse(_l2(Be_k @ R), axis_Ak, mid_Ak, sc_Ak))

    # unpaired-procrustes: DISJOINT corpora halves
    if A_corp2 is not None:
        Ack2, amu2, acp2 = _pca(A_corp2, k)          # corpus half 1 -> A
        Bck2, bmu2, bcp2 = _pca(B_corp2, k)          # DISJOINT half 2 -> B
        Ru = unpaired_procrustes(Bck2, Ack2)
        Be_k2, *_ = _pca(B_eval, k, bmu2, bcp2)
        Ae_k2, *_ = _pca(A_eval, k, amu2, acp2)
        ax2, md2, s2 = fit_axis(Ae_k2)
        res["transports"]["unpaired_procrustes"] = score_block(
            p_refuse(_l2(Be_k2 @ Ru), ax2, md2, s2))
    return res


def main():
    print(f"corpus: {len(CORPUS)} generic sentences (disjoint from eval)")
    print(f"prior native te3-large AUC vs behavioral = {PRIOR_NATIVE_AUC}")

    half = len(CORPUS) // 2
    c1, c2 = CORPUS[:half], CORPUS[half:]

    print("\nembedding A = text-embedding-3-large ...", flush=True)
    A_eval = embed_openai("text-embedding-3-large", PROMPTS)
    A_c1 = embed_openai("text-embedding-3-large", c1)
    A_c2 = embed_openai("text-embedding-3-large", c2)
    A_full = np.vstack([A_c1, A_c2])

    spaces = {}
    print("embedding B1 = text-embedding-3-small ...", flush=True)
    spaces["text-embedding-3-small"] = (
        embed_openai("text-embedding-3-small", PROMPTS),
        np.vstack([embed_openai("text-embedding-3-small", c1),
                   embed_openai("text-embedding-3-small", c2)]),
        embed_openai("text-embedding-3-small", c2),  # disjoint half for unpaired
    )
    print("embedding B2 = all-mpnet-base-v2 (local) ...", flush=True)
    spaces["all-mpnet-base-v2"] = (
        embed_mpnet(PROMPTS),
        np.vstack([embed_mpnet(c1), embed_mpnet(c2)]),
        embed_mpnet(c2),
    )

    results = []
    for name, (B_eval, B_full, B_c2) in spaces.items():
        print(f"\n=== transport into {name} ===")
        r = evaluate(name, A_eval, B_eval, A_full, B_full,
                     A_corp2=A_c1, B_corp2=B_c2)
        for tname, blk in r["transports"].items():
            print(f"  {tname:<22} "
                  f"AUC(4o-mini)={blk['auc_vs_gpt-4o-mini']:.3f}  "
                  f"AUC(4.1-mini)={blk['auc_vs_gpt-4.1-mini']:.3f}  "
                  f"AUC(gt)={blk['auc_vs_groundtruth_obvious']:.3f}  "
                  f"border={blk['borderline_mean_p']:.3f}")
        results.append(r)

    out = {
        "ts": "2026-05-17",
        "experiment": "linear transport of the refusal cognometric "
                      "instrument across embedding spaces",
        "home_space": "text-embedding-3-large",
        "n_eval": len(EVAL),
        "n_corpus": len(CORPUS),
        "prior_native_auc_vs_behavioral": PRIOR_NATIVE_AUC,
        "behavioral_label_source": "out_universal_directions_closed_model_test.json (2026-05-14)",
        "results": results,
        "caveats": [
            "n=30 eval: AUC is quantised and noisy",
            "unpaired_procrustes is a lightweight Conneau-style proxy, "
            "not a mini-vec2vec reimplementation",
            "alignment corpus is generic English; transport may degrade "
            "under larger distribution shift",
            "behavioral labels reused from the 2026-05-14 run (no new "
            "LLM calls)",
        ],
    }
    op = HERE / "out_universal_directions_linear_transport.json"
    op.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved: {op}")

    # headline
    print("\n" + "=" * 64)
    for r in results:
        best_unp = r["transports"].get("unpaired_procrustes", {})
        best_pr = r["transports"]["paired_ridge"]
        print(f"{r['space_B']:<24} paired-ridge "
              f"{best_pr['auc_vs_gpt-4o-mini']:.3f}/"
              f"{best_pr['auc_vs_gpt-4.1-mini']:.3f}  | unpaired "
              f"{best_unp.get('auc_vs_gpt-4o-mini', float('nan')):.3f}/"
              f"{best_unp.get('auc_vs_gpt-4.1-mini', float('nan')):.3f}  "
              f"(naive {r['transports']['naive_direct']['auc_vs_gpt-4o-mini']:.3f})")


if __name__ == "__main__":
    main()
