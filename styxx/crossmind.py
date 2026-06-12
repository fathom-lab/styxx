"""styxx.crossmind — portable value-axis readout across model INTERNALS (read-only).

Read one model's value-state (truth, harm-avoidance, ...) using a value axis FIT ON A
DIFFERENT *reference* model, by transporting the reference's difference-of-means direction
through a label-free map of last-token hidden states and reading it in a WHITENED
(Mahalanobis) frame. No labels are needed on the target model.

Sibling of `styxx.transport`: that module moves a cognometric instrument across EMBEDDING
spaces (encoders, cosine/procrustes, paired corpus). `crossmind` moves a VALUE axis across
the residual streams of GENERATIVE models and adds the whitened-basis readout. Different
substrate (hidden states, not embeddings); different readout (whitened, not cosine).

Productizes the portable-conscience arc (`papers/showcase-viz/`, 2026-06-11):
  - VALUES-PORTABLE     a difference-of-means value direction transfers across minds (truth
                        AND harm-avoidance) through a label-free ridge map of hidden states.
  - WHITENING-RESOLVES  under a ZCA-whitened readout the value axes form a CLEAN ORTHONORMAL
                        basis; raw dot-product cross-talk is a covariance artifact.
  - conscience-coords   the truth coordinate generalizes to new content and cross-model.

READ-ONLY by construction: it returns a coordinate, never an edit. Using these directions to
STEER or intervene on a model is a separate, closed line (read != write) and is REFUSED.

Quick start (states are last-token hidden activations at one layer; bring your own extractor):
  axis  = crossmind.fit_axis(reference_states, labels, name="truth")   # on a reference model
  smap  = crossmind.fit_state_map(target_anchor_states, reference_anchor_states)  # paired, label-free
  coords = crossmind.read(axis, target_states, state_map=smap)         # NO target labels

REFUSED:
  steering / intervention   -> read != write; writing/steering is out of scope (raises).
  content_danger (borrowed) -> a refusal-fit axis does not read statement-level danger
                               (HARM-AXIS-NULL, cycle 4); fit a danger axis directly instead.

CLI:
  python -m styxx.crossmind selftest        # deterministic end-to-end equivalence self-test

Scope: linear difference-of-means directions in a whitened space, label-free ridge transport,
last-token pre-output regime, register-bounded. Local open models validated (gemma-2-2b source;
Llama-3.2 + Qwen2.5 targets). Existence-and-significance, not a deployed-accuracy guarantee.
No claim about consciousness, welfare, or general capability.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = [
    # primitives (validated, lifted verbatim from papers/showcase-viz)
    "fit_direction", "fit_map", "apply_map", "zca_whiten", "zca_shrink", "auroc", "discrim",
    "permutation_null",
    # instrument
    "PortableAxis", "StateMap", "fit_axis", "fit_state_map", "identity_map",
    "read", "read_cross_model", "evaluate",
    # demarcation + certificate
    "REFUSALS", "refused", "crossmind_certificate",
    # self-test / equivalence gate
    "selftest",
]

_EPS = 1e-9
_WHITEN_EPS = 1e-3


# --------------------------------------------------------------------------------------------
# primitives — EXACT math from the validated runners (papers/showcase-viz/run_entanglement_
# resolution.py, run_portable_conscience_ood_v2.py). Do not "improve" these: the equivalence
# gate (T1) pins them to the published findings.
# --------------------------------------------------------------------------------------------

def fit_direction(acts: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Unit difference-of-means direction: mean(label==1) - mean(label==0), L2-normalized."""
    acts = np.asarray(acts, dtype=float)
    labels = np.asarray(labels)
    w = acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
    return w / (np.linalg.norm(w) + _EPS)


def fit_map(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Label-free affine ridge map X -> Y. Returns M (shape (d_x+1, d_y)) with a bias row."""
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return np.linalg.solve(Xb.T @ Xb + alpha * np.eye(Xb.shape[1]), Xb.T @ Y)


def apply_map(M: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ M


def zca_whiten(train: np.ndarray, eps: float = _WHITEN_EPS):
    """ZCA whitening from pooled covariance. Returns (mu, W) so x_white = (x - mu) @ W."""
    train = np.asarray(train, dtype=float)
    mu = train.mean(0)
    Xc = train - mu
    Sigma = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    s, V = np.linalg.eigh(Sigma)
    s = np.clip(s, 0, None)
    W = V @ np.diag(1.0 / np.sqrt(s + eps)) @ V.T
    return mu, W


def zca_shrink(train: np.ndarray, lam: float = 0.5, eps: float = 1e-8):
    """ZCA whitening with the covariance shrunk toward its scaled identity (Ledoit-Wolf-style).

    Σ_λ = (1-λ)·Σ̂ + λ·(tr Σ̂ / d)·I, for λ in [0, 1]. Use this for the MAPPED-target metric on
    cross-model reads, where the anchor count is far below the hidden dimension so the raw covariance is
    rank-deficient and must be shrunk (B29; FINDING_mapped_whitening_2026_06_12.md). λ=0.5 is the
    pre-registered default. Returns (mu, W) so x_white = (x - mu) @ W.
    """
    train = np.asarray(train, dtype=float)
    mu = train.mean(0)
    Xc = train - mu
    S = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    d = S.shape[0]
    tgt = np.trace(S) / d
    S = (1.0 - lam) * S + lam * tgt * np.eye(d)
    s, V = np.linalg.eigh(S)
    s = np.clip(s, eps, None)
    return mu, V @ np.diag(1.0 / np.sqrt(s)) @ V.T


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Tie-aware AUROC. Returns nan if either class is empty."""
    s = np.asarray(scores, dtype=float); y = np.asarray(labels)
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def discrim(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Direction-agnostic discriminability: max(AUROC, 1 - AUROC)."""
    a = auroc(scores, labels)
    return max(a, 1.0 - a)


def permutation_null(scores, labels, *, seed: int = 0, k_perm: int = 1000) -> dict:
    """Label-permutation null for a score vector. Returns p95 + one-tailed p_value of AUROC."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    obs = auroc(scores, labels)
    null = np.array([auroc(scores, rng.permutation(labels)) for _ in range(k_perm)])
    p95 = float(np.percentile(null, 95))
    p_value = float((1 + int((null >= obs).sum())) / (1 + k_perm))
    return {"auroc": round(obs, 6), "perm_p95": round(p95, 6), "p_value": round(p_value, 6),
            "k_perm": int(k_perm)}


# --------------------------------------------------------------------------------------------
# instrument
# --------------------------------------------------------------------------------------------

@dataclass
class PortableAxis:
    """A fitted, transportable value direction in a (whitened) reference-model space."""
    name: str
    w: np.ndarray
    mu: np.ndarray
    W: np.ndarray
    dim: int
    whitened: bool
    n_pos: int
    n_neg: int

    def score(self, reference_states: np.ndarray) -> np.ndarray:
        """Coordinate of in-REFERENCE-space states along this axis (whiten then project)."""
        x = np.asarray(reference_states, dtype=float)
        return ((x - self.mu) @ self.W) @ self.w


@dataclass
class StateMap:
    """A label-free affine map from a target model's hidden space into the reference's space."""
    M: np.ndarray
    alpha: float
    val_r2: float
    src_dim: int
    dst_dim: int

    def apply(self, target_states: np.ndarray) -> np.ndarray:
        return apply_map(self.M, target_states)


def fit_axis(states: np.ndarray, labels: Sequence[int], *, name: str,
             background: Optional[np.ndarray] = None, whiten: bool = True,
             eps: float = _WHITEN_EPS) -> PortableAxis:
    """Fit a value axis from labeled reference-model activations.

    states: (n, d) reference-model last-token hidden states at one layer.
    labels: length-n {0,1}; the direction points from 0 -> 1.
    background: optional (m, d) states to estimate the whitening covariance on (defaults to
        `states`); pass a broad background set to make the whitened readout axis-fair.
    whiten: if False, mu=0 and W=I (raw dot-product readout; not recommended — WHITENING-RESOLVES).
    """
    states = np.asarray(states, dtype=float)
    labels = np.asarray(labels)
    if states.ndim != 2:
        raise ValueError(f"states must be 2-D (n, d); got shape {states.shape}")
    if len(labels) != states.shape[0]:
        raise ValueError(f"labels length {len(labels)} != n_states {states.shape[0]}")
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    if pos == 0 or neg == 0:
        raise ValueError("need both classes present in labels to fit a direction")
    d = states.shape[1]
    if whiten:
        bg = states if background is None else np.asarray(background, dtype=float)
        mu, W = zca_whiten(bg, eps)
        w = fit_direction((states - mu) @ W, labels)
    else:
        mu, W = np.zeros(d), np.eye(d)
        w = fit_direction(states, labels)
    return PortableAxis(name=name, w=w, mu=mu, W=W, dim=d, whitened=bool(whiten),
                        n_pos=pos, n_neg=neg)


def fit_state_map(target_states: np.ndarray, reference_states: np.ndarray, *,
                  alphas: Sequence[float] = (10.0, 100.0, 1000.0),
                  val_frac: float = 0.2, seed: int = 0) -> StateMap:
    """Fit a label-free ridge map target -> reference on PAIRED anchor activations.

    target_states / reference_states: (n, d_t) and (n, d_r) hidden states on the SAME n anchor
    texts (paired row-for-row). Labels are NOT used. alpha is selected by held-out R^2.
    """
    X = np.asarray(target_states, dtype=float)
    Y = np.asarray(reference_states, dtype=float)
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"paired anchors must align: {X.shape[0]} target vs {Y.shape[0]} reference rows")
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = max(1, int((1.0 - val_frac) * n))
    tr, va = perm[:k], perm[k:]
    if len(va) == 0:  # tiny n: validate on the train split rather than crash
        va = tr
    best = None
    for alpha in alphas:
        M = fit_map(X[tr], Y[tr], alpha)
        pred = apply_map(M, X[va])
        denom = ((Y[va] - Y[va].mean(0)) ** 2).sum() + _EPS
        r2 = 1.0 - ((pred - Y[va]) ** 2).sum() / denom
        if best is None or r2 > best[0]:
            best = (r2, alpha)
    r2, alpha = best
    M = fit_map(X, Y, alpha)  # refit on all anchors at the chosen alpha
    return StateMap(M=M, alpha=float(alpha), val_r2=round(float(r2), 6),
                    src_dim=int(X.shape[1]), dst_dim=int(Y.shape[1]))


def identity_map(dim: int) -> StateMap:
    """A no-op map (for reading a model with its OWN axis, no cross-model transport)."""
    M = np.vstack([np.eye(dim), np.zeros((1, dim))])
    return StateMap(M=M, alpha=0.0, val_r2=1.0, src_dim=int(dim), dst_dim=int(dim))


def read(axis: PortableAxis, target_states: np.ndarray,
         state_map: Optional[StateMap] = None) -> np.ndarray:
    """Read target-model activations along a borrowed axis. No target labels needed.

    state_map=None reads in the axis's own (reference) space. Otherwise target states are mapped
    into the reference space first, then whitened and projected onto the axis.
    """
    x = np.asarray(target_states, dtype=float)
    mapped = x if state_map is None else state_map.apply(x)
    return axis.score(mapped)


def read_cross_model(reference_states: np.ndarray, labels: Sequence[int], state_map: StateMap,
                     target_states: np.ndarray, *, mapped_anchors: np.ndarray,
                     shrink_lambda: float = 0.5, eps: float = 1e-8) -> np.ndarray:
    """Cross-model read with MAPPED-space whitening — the correct metric for transport (B29).

    `read` whitens in the REFERENCE metric (good for in-model reads). For a CROSS-model read the
    label-free ridge map distorts covariance, so reading mapped points in the reference metric leaves a
    residual anisotropy that bleeds neighbouring axes in (FINDING_mapped_whitening_2026_06_12.md,
    BASIS-CLEARED). Here the whitening metric is re-estimated on the MAPPED target distribution (shrunk
    toward scaled identity, since anchors ≪ hidden dim), the difference-of-means direction is fit in that
    metric, and the target is read in it. Prefer this over `read(..., state_map=...)` for cross-model use.

    reference_states / labels: the reference-model labeled states the axis is defined by (n, d_ref).
    state_map: a StateMap target→reference (from fit_state_map).
    target_states: target-model states to read (m, d_tgt).
    mapped_anchors: target-model anchor states (unlabeled, k, d_tgt) defining the mapped distribution.
    shrink_lambda: covariance shrinkage toward scaled identity (0.5 pre-registered).
    """
    ref = np.asarray(reference_states, dtype=float)
    labels = np.asarray(labels)
    mu_m, W_m = zca_shrink(state_map.apply(mapped_anchors), shrink_lambda, eps)
    w = fit_direction((ref - mu_m) @ W_m, labels)
    return ((state_map.apply(target_states) - mu_m) @ W_m) @ w


def evaluate(coords: Sequence[float], labels: Sequence[int]) -> dict:
    """Assess a coordinate vector against ground-truth labels (when you happen to have them)."""
    return {"auroc": round(auroc(coords, labels), 6),
            "discrim": round(discrim(coords, labels), 6),
            "n": int(len(coords))}


# --------------------------------------------------------------------------------------------
# demarcation — what this instrument refuses to do, with receipts
# --------------------------------------------------------------------------------------------

REFUSALS = {
    "steering": {
        "status": "REFUSED",
        "reason": "read != write. styxx.crossmind READS a value coordinate; it does not steer, "
                  "edit, or intervene on a model. Using these directions to change behavior is a "
                  "separate, closed line and is out of scope for this instrument.",
        "receipt": "papers/showcase-viz/ (portable-conscience arc is read-only by construction)",
    },
    "intervention": {  # alias of steering
        "status": "REFUSED",
        "reason": "alias of 'steering' — this is a read-only instrument; intervention is out of scope.",
        "receipt": "papers/showcase-viz/",
    },
    "content_danger": {
        "status": "REFUSED",
        "reason": "a refusal-fit axis (harmful-vs-benign REQUESTS) does NOT read the danger of a "
                  "STATEMENT; the harm coordinate was at chance for statement danger-topic "
                  "(HARM-AXIS-NULL). To read content danger, fit a danger axis directly on "
                  "danger-vs-safe statements — do not borrow the refusal axis for it.",
        "receipt": "papers/showcase-viz/FINDING_conscience_coordinates_2026_06_11.md",
    },
}


def refused(name: str):
    """Raise for an axis/operation this instrument deliberately does not provide."""
    if name in REFUSALS:
        info = REFUSALS[name]
        raise PermissionError(
            f"styxx.crossmind refuses '{name}': {info['reason']} (receipt: {info['receipt']})")
    raise KeyError(f"unknown axis/operation '{name}' (no refusal record; not a provided capability)")


# --------------------------------------------------------------------------------------------
# certificate
# --------------------------------------------------------------------------------------------

def _instrument_sha256() -> str:
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def crossmind_certificate(axis: PortableAxis, *, reference_id: str, target_id: str,
                          evaluation: Optional[dict] = None,
                          state_map: Optional[StateMap] = None) -> dict:
    """Assemble a re-runnable certificate for a transported read. Always carries the refusals."""
    return {
        "instrument": "styxx.crossmind v0",
        "prereg": "papers/crossmind-instrument/PREREG_crossmind_v0_2026_06_12.md",
        "instrument_sha256": _instrument_sha256(),
        "axis": axis.name,
        "reference_model": reference_id,
        "target_model": target_id,
        "dim": axis.dim,
        "whitened": axis.whitened,
        "fit_n_pos": axis.n_pos,
        "fit_n_neg": axis.n_neg,
        "state_map": (None if state_map is None else
                      {"alpha": state_map.alpha, "val_r2": state_map.val_r2,
                       "src_dim": state_map.src_dim, "dst_dim": state_map.dst_dim}),
        "evaluation": evaluation,
        "axes_refused": REFUSALS,
        "scope": ("Linear difference-of-means direction in a ZCA-whitened space, label-free ridge "
                  "transport of last-token hidden states, pre-output regime, register-bounded. "
                  "READ-ONLY (no steering). Existence-and-significance, not a deployed-accuracy "
                  "guarantee. No claim about consciousness, welfare, or general capability."),
    }


# --------------------------------------------------------------------------------------------
# self-test / equivalence gate (T1 math port, T4 determinism) — deterministic, no models
# --------------------------------------------------------------------------------------------

def _synthetic(seed: int = 0):
    """Two synthetic 'models' related by a known linear map + noise, sharing a value axis.

    Fully deterministic given the seed. Returns paired anchors (target, reference), labeled value
    states (target, reference, labels), and a held-out target set with labels.
    """
    rng = np.random.default_rng(seed)
    d_ref, d_tgt, dim_latent = 24, 28, 8
    A_ref = rng.standard_normal((dim_latent, d_ref))
    A_tgt = rng.standard_normal((dim_latent, d_tgt))
    axis_latent = rng.standard_normal(dim_latent)
    axis_latent /= np.linalg.norm(axis_latent)

    def emit(n, signed):
        z = rng.standard_normal((n, dim_latent))
        if signed is not None:  # push +/- along the value axis for labeled sets
            z = z + signed[:, None] * 2.0 * axis_latent[None, :]
        ref = z @ A_ref + 0.10 * rng.standard_normal((n, d_ref))
        tgt = z @ A_tgt + 0.10 * rng.standard_normal((n, d_tgt))
        return tgt, ref

    anchor_tgt, anchor_ref = emit(120, None)
    lab = np.array([1, 0] * 40)
    val_tgt, val_ref = emit(80, np.where(lab == 1, 1.0, -1.0))
    hlab = np.array([1, 0] * 30)
    hold_tgt, _ = emit(60, np.where(hlab == 1, 1.0, -1.0))
    return {"anchor_tgt": anchor_tgt, "anchor_ref": anchor_ref,
            "val_tgt": val_tgt, "val_ref": val_ref, "lab": lab,
            "hold_tgt": hold_tgt, "hlab": hlab}


def selftest(seed: int = 0) -> dict:
    """Deterministic end-to-end self-test: fit an axis on the 'reference', transport from the
    'target', read held-out target states with NO target labels, score against held-out truth.
    Used as the equivalence/determinism gate. Returns a small metrics dict.
    """
    d = _synthetic(seed)
    axis = fit_axis(d["val_ref"], d["lab"], name="selftest", whiten=True)
    smap = fit_state_map(d["val_tgt"], d["val_ref"], seed=seed)
    coords = read(axis, d["hold_tgt"], state_map=smap)
    ev = evaluate(coords, d["hlab"])
    ref_ev = evaluate(axis.score(d["val_ref"]), d["lab"])
    return {"seed": int(seed), "map_alpha": smap.alpha, "map_val_r2": smap.val_r2,
            "reference_self_auroc": ref_ev["auroc"], "transported_auroc": ev["auroc"],
            "transported_discrim": ev["discrim"], "n_held": ev["n"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json
    p = argparse.ArgumentParser(prog="styxx.crossmind",
                                description="portable value-axis readout across model internals (read-only)")
    sub = p.add_subparsers(dest="cmd")
    st = sub.add_parser("selftest", help="run the deterministic end-to-end equivalence self-test")
    st.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.cmd == "selftest":
        print(json.dumps(selftest(args.seed), indent=2))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
