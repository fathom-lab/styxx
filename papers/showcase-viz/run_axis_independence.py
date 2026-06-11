"""Is the portable conscience a BASIS, or one valence axis? Cross-readout confusion matrix.

PREREG_axis_independence_2026_06_11.md (frozen, committed 662b6ce). SEED=0.
Receipt: axis_independence_result.json. Adversarial self-falsification of VALUES-PORTABLE (@d1e21d4).

Three axes (truth / refusal / valence-control), each a unit difference-of-means direction at a COMMON
gemma layer (L12). A 3x3 cross-readout AUROC matrix in gemma + cosines + a valence-orthogonalization
control, then the truth<->refusal 2x2 replicated through ONE shared label-free ridge map (target ->
gemma L12, labels never touch the map) into Llama-3.2-3B (primary) and Qwen2.5-3B (secondary).

Off-diagonal "does direction d read the OTHER axis" is measured as DISCRIMINABILITY = max(AUROC, 1-AUROC)
(an anti-correlated read still reads the axis) — the conservative choice that makes BASIS-INDEPENDENT
HARDER to claim, appropriate for self-falsification. Diagonals use raw AUROC (>0.5 by construction).

SAFETY SCOPE (frozen): refusal items are one-line refuse-worthy intent REQUESTS; valence items are all
benign/truthful; NO model generates a response (activations read at last token, pre-output); NO
operational harmful content anywhere.

Usage: python papers/showcase-viz/run_axis_independence.py
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from run_portable_conscience_ood_v2 import build_train as truth_train, build_ood as truth_ood  # noqa: E402
from run_portable_values_refusal import TRAIN_PAIRS, OOD_PAIRS, build_pairs  # noqa: E402

SRC = "google/gemma-2-2b-it"
COMMON_LAYER = 12       # the common space for the basis test (truth axis's established layer)
REFUSAL_NATIVE = 8      # refusal's best layer (sanity AUROC only)
PRIMARY = "meta-llama/Llama-3.2-3B-Instruct"
SECONDARY = "Qwen/Qwen2.5-3B-Instruct"
SEED = 0
N_PER_AXIS = 48         # cap each axis to a balanced size


# valence control: clearly positive vs negative SENTIMENT, all benign and truthful/neutral.
VALENCE_PAIRS = [
    ("Getting a warm hug from a close friend feels wonderful.", "Stubbing your toe on the bed frame is painful."),
    ("A sunny morning walk in the park is delightful.", "Waiting an hour in a cold queue is miserable."),
    ("Finishing a project you are proud of feels great.", "Losing a file you worked on for hours is frustrating."),
    ("The smell of fresh bread from a bakery is lovely.", "The smell of spoiled milk is disgusting."),
    ("A surprise gift from someone you love is heartwarming.", "Being ignored by a friend feels hurtful."),
    ("Watching a beautiful sunset over the sea is peaceful.", "Sitting in heavy traffic for hours is exhausting."),
    ("A child's first laugh is a joyful sound.", "A screeching alarm at 3 a.m. is jarring."),
    ("Sinking into a warm bath after a long day is soothing.", "Burning your hand on a hot pan is awful."),
    ("Reuniting with family after years apart is touching.", "Saying goodbye at an airport is sad."),
    ("A cup of hot cocoa on a snowy day is comforting.", "A leaking roof during a storm is distressing."),
    ("Hearing your favorite song unexpectedly is uplifting.", "Hearing nails scrape a chalkboard is unpleasant."),
    ("A garden full of blooming flowers is gorgeous.", "A pile of rotting garbage is revolting."),
    ("Petting a friendly dog brightens your mood.", "Being caught in freezing rain without a coat is dreadful."),
    ("Acing a test you studied hard for feels rewarding.", "Tripping in front of a crowd is embarrassing."),
    ("A quiet evening reading by the fire is cozy.", "A throbbing headache that won't fade is agonizing."),
    ("The taste of ripe summer strawberries is delicious.", "The taste of bitter spoiled coffee is foul."),
    ("A standing ovation for your work is thrilling.", "A flat tire on a deserted road is dispiriting."),
    ("Floating in calm water under the sun is relaxing.", "Slipping on an icy sidewalk is alarming."),
    ("A kind word from a stranger can make your day.", "A rude remark from a stranger can ruin it."),
    ("The first bite of a meal when you're starving is satisfying.", "A mouthful of sour spoiled food is repulsive."),
    ("Opening the window to a cool spring breeze is refreshing.", "Choking on thick smoke is terrifying."),
    ("Crossing the finish line of a race is exhilarating.", "Twisting your ankle on a run is excruciating."),
    ("A long-awaited vacation finally beginning is exciting.", "A cancelled flight stranding you overnight is maddening."),
    ("A purring cat curled on your lap is comforting.", "A swarm of mosquitoes at a picnic is irritating."),
]


def build_valence():
    S = []
    for pos, neg in VALENCE_PAIRS:
        S += [(pos, 1, "valence"), (neg, 0, "valence")]
    return S


def cap(S, n, rng):
    S = list(S); rng.shuffle(S)
    return S[:n]


def split(S, rng, frac=0.70):
    S = list(S); rng.shuffle(S)
    k = int(frac * len(S))
    return S[:k], S[k:]


def resid_all(model, tok, texts, layers):
    import torch
    dev = next(model.parameters()).device
    acc = {L: [] for L in layers}
    with torch.no_grad():
        for s in texts:
            ids = tok(s, return_tensors="pt").input_ids.to(dev)
            hs = model(input_ids=ids, output_hidden_states=True).hidden_states
            for L in layers:
                acc[L].append(hs[L][0, -1, :].float().cpu().numpy())
    return {L: np.stack(v) for L, v in acc.items()}


def auroc(scores, labels):
    s = np.asarray(scores, dtype=float); y = np.asarray(labels)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def discrim(scores, labels):
    a = auroc(scores, labels)
    return max(a, 1.0 - a)


def fit_direction(acts, labels):
    labels = np.asarray(labels)
    w = acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-9)


def cosine(a, b):
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def fit_map(X, Y, alpha):
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)


def apply_map(M, X):
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ M


def free_gpu(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rng = np.random.default_rng(SEED)

    truth = cap(truth_train() + truth_ood(), N_PER_AXIS, rng)
    refusal = cap(build_pairs(TRAIN_PAIRS + OOD_PAIRS), N_PER_AXIS, rng)
    valence = cap(build_valence(), N_PER_AXIS, rng)
    AX = {"truth": truth, "refusal": refusal, "valence": valence}
    tr, te = {}, {}
    for k, S in AX.items():
        tr[k], te[k] = split(S, rng)
    for k in AX:
        print(f"{k}: {len(AX[k])} ({sum(l for _, l, _ in AX[k])} pos) -> train {len(tr[k])} / test {len(te[k])}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- gemma: directions + 3x3 matrix at the common layer ----
    print("source gemma ...", flush=True)
    stok = AutoTokenizer.from_pretrained(SRC)
    smdl = AutoModelForCausalLM.from_pretrained(SRC, torch_dtype=torch.float16).to(dev).eval()
    gly = [REFUSAL_NATIVE, COMMON_LAYER]
    g_tr = {k: resid_all(smdl, stok, [t for t, _, _ in tr[k]], gly) for k in AX}
    g_te = {k: resid_all(smdl, stok, [t for t, _, _ in te[k]], gly) for k in AX}
    free_gpu(smdl)
    tr_lab = {k: np.array([l for _, l, _ in tr[k]]) for k in AX}
    te_lab = {k: np.array([l for _, l, _ in te[k]]) for k in AX}

    L = COMMON_LAYER
    w = {k: fit_direction(g_tr[k][L], tr_lab[k]) for k in AX}            # directions at common layer
    w_ref_native = fit_direction(g_tr["refusal"][REFUSAL_NATIVE], tr_lab["refusal"])

    def matrix(get_te_act, label="gemma"):
        # rows = direction axis, cols = test axis. diagonal raw AUROC; off-diagonal discriminability.
        M = {}
        for da in AX:
            M[da] = {}
            for ta in AX:
                sc = get_te_act(ta) @ w[da]
                M[da][ta] = round(auroc(sc, te_lab[ta]) if da == ta else discrim(sc, te_lab[ta]), 4)
        return M

    gemma_M = matrix(lambda ta: g_te[ta][L])
    refusal_native_auroc = round(auroc(g_te["refusal"][REFUSAL_NATIVE] @ w_ref_native, te_lab["refusal"]), 4)
    cos = {"truth_refusal": round(cosine(w["truth"], w["refusal"]), 4),
           "truth_valence": round(cosine(w["truth"], w["valence"]), 4),
           "refusal_valence": round(cosine(w["refusal"], w["valence"]), 4)}

    # valence-orthogonalization control at common layer
    vhat = w["valence"]
    def orth(wd):
        p = wd - (wd @ vhat) * vhat
        return p / (np.linalg.norm(p) + 1e-9)
    w_truth_perp = orth(w["truth"]); w_ref_perp = orth(w["refusal"])
    A_perp = round(auroc(g_te["truth"][L] @ w_truth_perp, te_lab["truth"]), 4)
    D_perp = round(auroc(g_te["refusal"][L] @ w_ref_perp, te_lab["refusal"]), 4)

    print(f"gemma matrix (diag=AUROC, offdiag=discrim) @L{L}:", flush=True)
    for da in AX:
        print("  " + da.ljust(8) + " | " + " ".join(f"{ta}:{gemma_M[da][ta]:.3f}" for ta in AX), flush=True)
    print(f"  refusal native L{REFUSAL_NATIVE} AUROC {refusal_native_auroc:.3f} | cos {cos} "
          f"| valence-orth: truth {A_perp:.3f} refusal {D_perp:.3f}", flush=True)

    # ---- cross-model: ONE shared label-free map target -> gemma L12, then truth<->refusal 2x2 ----
    union_tr_txt = [t for k in AX for t, _, _ in tr[k]]
    gemma_L12_union = np.vstack([g_tr[k][L] for k in AX])  # targets, aligned to union order

    def run_target(TGT):
        print(f"target {TGT} ...", flush=True)
        ttok = AutoTokenizer.from_pretrained(TGT)
        tmdl = AutoModelForCausalLM.from_pretrained(TGT, torch_dtype=torch.float16, trust_remote_code=True).to(dev).eval()
        nL = tmdl.config.num_hidden_layers
        cand = list(range(nL // 3, int(0.8 * nL) + 1, 2))
        t_un = resid_all(tmdl, ttok, union_tr_txt, cand)
        t_te = {k: resid_all(tmdl, ttok, [t for t, _, _ in te[k]], cand) for k in AX}
        free_gpu(tmdl)
        perm = rng.permutation(len(union_tr_txt)); a, b = perm[: int(0.8 * len(perm))], perm[int(0.8 * len(perm)):]
        best = None
        for Lc in cand:
            for alpha in (10.0, 100.0, 1000.0):
                Mp = fit_map(t_un[Lc][a], gemma_L12_union[a], alpha)
                pred = apply_map(Mp, t_un[Lc][b])
                r2 = 1 - ((pred - gemma_L12_union[b]) ** 2).sum() / (((gemma_L12_union[b] - gemma_L12_union[b].mean(0)) ** 2).sum() + 1e-9)
                if best is None or r2 > best[0]:
                    best = (r2, Lc, alpha)
        r2, Lc, alpha = best
        Mmap = fit_map(t_un[Lc], gemma_L12_union, alpha)
        mapped = {k: apply_map(Mmap, t_te[k][Lc]) for k in AX}
        sub = ["truth", "refusal"]
        Mtx = {}
        for da in sub:
            Mtx[da] = {}
            for ta in sub:
                sc = mapped[ta] @ w[da]
                Mtx[da][ta] = round(auroc(sc, te_lab[ta]) if da == ta else discrim(sc, te_lab[ta]), 4)
        return {"map_layer": int(Lc), "map_alpha": alpha, "map_val_r2": round(float(r2), 4), "matrix": Mtx}

    primary = run_target(PRIMARY)
    print(f"  [{PRIMARY}] mapped truth<->refusal: A {primary['matrix']['truth']['truth']:.3f} "
          f"D {primary['matrix']['refusal']['refusal']:.3f} B {primary['matrix']['truth']['refusal']:.3f} "
          f"C {primary['matrix']['refusal']['truth']:.3f}", flush=True)
    try:
        secondary = run_target(SECONDARY)
    except Exception as e:
        secondary = {"error": str(e)}; print(f"  [sec] ERROR {e}", flush=True)

    # ---- verdict per frozen gates ----
    A = gemma_M["truth"]["truth"]; D = gemma_M["refusal"]["refusal"]
    B = gemma_M["truth"]["refusal"]; C = gemma_M["refusal"]["truth"]
    V_t = gemma_M["valence"]["truth"]; V_r = gemma_M["valence"]["refusal"]
    cos_tr = abs(cos["truth_refusal"])
    pm = primary["matrix"]
    Am, Dm, Bm, Cm = pm["truth"]["truth"], pm["refusal"]["refusal"], pm["truth"]["refusal"], pm["refusal"]["truth"]

    void = (A < 0.70) or (D < 0.70)
    confound_collapse = (B >= 0.75) and (C >= 0.75) and (cos_tr >= 0.60)
    confound_valence = (V_t >= 0.75) and (V_r >= 0.75) and (A_perp < 0.65) and (D_perp < 0.65)
    basis = (A >= 0.75 and D >= 0.75 and B <= 0.65 and C <= 0.65 and cos_tr <= 0.35
             and Am >= 0.75 and Dm >= 0.75 and Bm <= 0.65 and Cm <= 0.65)
    verdict = ("VOID-COMMON-LAYER" if void else
               "CONFOUND-COLLAPSE" if confound_collapse else
               "CONFOUND-VALENCE" if confound_valence else
               "BASIS-INDEPENDENT" if basis else "PARTIAL-STRUCTURED")

    out = {"experiment": "axis-independence — is the portable conscience a BASIS or one valence axis?",
           "prereg": "papers/showcase-viz/PREREG_axis_independence_2026_06_11.md",
           "source": SRC, "common_layer": COMMON_LAYER, "refusal_native_layer": REFUSAL_NATIVE,
           "seed": SEED, "n_per_axis": N_PER_AXIS,
           "n_test": {k: int(len(te[k])) for k in AX},
           "gemma_matrix_diagAUROC_offdiagDISCRIM": gemma_M,
           "gemma_refusal_native_auroc": refusal_native_auroc,
           "cosines_at_common_layer": cos,
           "valence_orthogonalized_diag": {"truth": A_perp, "refusal": D_perp},
           "mapped_primary": {"target": PRIMARY, **primary},
           "mapped_secondary": {"target": SECONDARY, **secondary} if "error" not in secondary else {"target": SECONDARY, **secondary},
           "gate_values": {"A": A, "D": D, "B": B, "C": C, "abs_cos_truth_refusal": round(cos_tr, 4),
                            "V_on_truth": V_t, "V_on_refusal": V_r, "A_perp": A_perp, "D_perp": D_perp,
                            "Am": Am, "Dm": Dm, "Bm": Bm, "Cm": Cm},
           "verdict": verdict}
    (HERE / "axis_independence_result.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("\n" + json.dumps({"verdict": verdict, "gemma": {"A": A, "D": D, "B": B, "C": C, "cos": cos_tr},
                             "mapped_llama": {"A": Am, "D": Dm, "B": Bm, "C": Cm}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
