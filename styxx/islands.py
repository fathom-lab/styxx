"""styxx.islands — is a cohort of minds mutually legible, or does it contain islands?

The disjoint-worlds arc measured, under frozen gates, that independently trained minds share a
concept-frame geometry — and that a mind can sit *mostly inside* that shared geometry and still
be unreadable, because legibility is **switch-like** in the frame coordinate rather than graded
(``papers/disjoint-worlds/``, nine sealed acts). This module is that measurement, generalized and
handed to anyone: give it a cohort of representations over a **shared item set** and it reports
whether the cohort is one legible clique or contains islands, how steep the transition is, and
whether a low-rank correction rescues the outliers.

Nothing here is specific to language models. The inputs are ``(n_items, dim)`` arrays over the
same items in the same order — model activations, fMRI betas over a shared stimulus set, MEG
epochs, embeddings. Dimensions may differ across members; item count and order may not.

    from styxx.islands import survey, cliff, rescue

    s = survey({"m1": X1, "m2": X2, "m3": X3, "m4": X4})   # X* : (n_items, dim_*)
    s.verdict            # ISLANDS_PRESENT / UNIMODAL_COHORT / UNDERPOWERED__n_below_8
    s.islands            # candidate island members, by the stated rule
    s.pairwise           # pairwise frame affinity, measured against a random-frame null

``cliff`` and ``rescue`` take a caller-supplied ``legibility_fn`` — in a real application the
legibility measure already exists and belongs to the user (a decoder's accuracy, a discovery
routine). An internal one shipped in the first draft, failed its own exam against a known
answer, and was removed rather than defaulted.

Open by design (see ``docs/governance/OPEN_CORE.md``): a measurement primitive, never gated.

**Honest ceilings, stated in the API rather than the fine print.**

* ``survey`` names *candidate* islands below 8 members and refuses an inferential verdict —
  bimodality cannot be tested on a handful of points, and pretending otherwise is how
  instruments lie. The verdict string says so.
* Every statistic here is symmetric. "A is an island" means the cohort cannot read A and A
  cannot read the cohort; it carries no direction and no cause.
* Affinity is measured between concept-Gram eigenframes. Two members with identical frames can
  still differ in what they load onto those frames; ``rescue`` is what distinguishes a rotation
  from missing information.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from itertools import combinations

import numpy as np

__all__ = ["frame", "affinity", "survey", "cliff", "rescue", "hartigan_dip_p",
           "normalize_items",
           "CohortSurvey", "MIN_COHORT"]

MIN_COHORT = 8          # below this, bimodality has no power and the verdict refuses
_DEF_K = 20


def normalize_items(X: np.ndarray) -> np.ndarray:
    """Center each item's response vector and scale it to unit norm.

    **The amplitude control, and it is not optional in practice.** Frame affinity can be driven
    entirely by a per-item *magnitude* profile that every member shares, with independent
    geometry: in fMRI, "some stimuli simply drive the region harder in everybody" is a certainty,
    not a hypothesis. Red team 2026-08-06: independent members sharing only a lognormal item
    amplitude profile (spread 0.5) reached 22.4x the random-frame null — more than twice the
    9.0x this lab published as an "emphatic" shared frame. Normalising items removes that channel
    and leaves genuine shared geometry intact (verified: a real shared latent survives at
    0.4667 vs 0.4141 unnormalised).
    """
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(1, keepdims=True)
    return Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)


def frame(X: np.ndarray, k: int = _DEF_K) -> np.ndarray:
    """Top-k eigenvectors of the double-centered item Gram — the member's *frame*.

    Item-space, not feature-space: this is what makes members of different dimensionality
    directly comparable, and it is the construction the disjoint-worlds arc measured.
    """
    X = np.asarray(X, dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("frame(): input contains NaN or inf. numpy.linalg.eigh propagates "
                         "these silently, and a NaN affinity vector made _gap_p return a "
                         "constant 1/(n_perm+1) — i.e. ISLANDS_PRESENT, deterministically, "
                         "with an empty islands list and no caveat (red team 2026-08-06).")
    Xc = X - X.mean(0)
    G = Xc @ Xc.T
    n = G.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    G = J @ G @ J
    w, V = np.linalg.eigh(G)
    return V[:, np.argsort(w)[::-1][:min(k, n - 1)]]


def affinity(Ua: np.ndarray, Ub: np.ndarray) -> float:
    """Mean squared cosine of principal angles between two frames — 0 (orthogonal) to 1 (same)."""
    k = min(Ua.shape[1], Ub.shape[1])
    return float(np.linalg.norm(Ua[:, :k].T @ Ub[:, :k], "fro") ** 2 / k)


def _haar_null(n: int, k: int, n_draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    def f():
        Q, _ = np.linalg.qr(rng.standard_normal((n, k)))
        return Q[:, :k]
    vals = [affinity(f(), f()) for _ in range(n_draws)]
    return {"expectation": round(k / n, 4), "p95": round(float(np.percentile(vals, 95)), 4),
            "max": round(float(np.max(vals)), 4), "n_draws": n_draws}


def hartigan_dip_p(values) -> float:
    """Hartigan's dip test p-value — the reference unimodality statistic.

    ``_gap_p`` below is this module's own screen: exactly reproducible in twenty lines with an
    explicit null, and always available. Hartigan's dip is what the methods literature (and this
    lab's frozen H1 prediction gate) actually names, so it is offered whenever the optional
    ``diptest`` package is installed, and ``nan`` otherwise. Run both when it matters: on the
    2026-08-06 human cohorts the two agreed on the conclusion while the gap screen sat closer to
    the bar (alignment 0.0779 vs dip 0.6036; readability 0.6523 vs dip 0.7553), i.e. the screen
    is the more liberal of the two, so a non-flag from it is the stronger statement.
    """
    try:
        import diptest
    except ImportError:
        return float("nan")
    v = np.asarray(values, dtype=float)
    if v.size < 4:
        return float("nan")
    return round(float(diptest.diptest(v)[1]), 4)


def _gap_p(values: np.ndarray, n_perm: int, seed: int) -> float:
    """Dip-like unimodality check: is the largest gap in the sorted values bigger than the
    largest gap of a unimodal (normal) sample with the same mean, spread and count?

    Reported as an add-one-smoothed permutation p. This is a *screen*, not Hartigan's dip; it is
    used because it is exactly reproducible in twenty lines and its null is explicit. If you are
    scoring the H1 human-islands prediction (``papers/disjoint-worlds/PREDICTION_h1_*``), that
    gate's reference statistic is Hartigan's dip — report whichever you use under its own name.
    """
    v = np.asarray(values, dtype=float)
    if not np.isfinite(v).all():
        return float("nan")          # never emit a verdict from a non-finite statistic
    v = np.sort(v)
    if len(v) < 3 or v[-1] - v[0] <= 0:
        return 1.0
    obs = float(np.max(np.diff(v)) / (v[-1] - v[0]))
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        s = np.sort(rng.normal(v.mean(), v.std() + 1e-12, len(v)))
        if float(np.max(np.diff(s)) / (s[-1] - s[0] + 1e-12)) >= obs:
            ge += 1
    return round((ge + 1) / (n_perm + 1), 4)


@dataclass
class CohortSurvey:
    """What a cohort of minds looks like from the outside."""
    verdict: str
    members: list
    k: int
    n_items: int
    pairwise: dict                       # "a__b" -> affinity
    mean_affinity: dict                  # member -> mean affinity to the rest
    islands: list                        # candidate islands under the stated rule
    null: dict                           # the random-frame null this is measured against
    bimodality_p: float                  # gap screen, NOT Hartigan's dip — see _gap_p
    cohort_median: float                 # median of per-member MEAN affinities (not pairwise)
    median_pairwise_affinity: float = 0.0
    island_rule: str = ""
    caveats: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        head = (f"cohort of {len(self.members)} · k={self.k} · {self.n_items} shared items\n"
                f"  {self.verdict}\n"
                f"  median pairwise affinity {self.cohort_median} "
                f"(random-frame null p95 {self.null['p95']})\n")
        rows = "".join(f"    {m:<24} {self.mean_affinity[m]:.4f}"
                       f"{'   ← island' if m in self.islands else ''}\n"
                       for m in self.members)
        return head + rows + "".join(f"  ! {c}\n" for c in self.caveats)


def survey(reps: dict, k: int = _DEF_K, n_null: int = 1000, n_perm: int = 100_000,
           seed: int = 343, island_z: float = 1.0,
           normalize_amplitude: bool = True) -> CohortSurvey:
    """Measure a cohort of minds over a shared item set.

    ``reps`` maps member name -> ``(n_items, dim)`` array. Item order must be identical across
    members; dimensionality need not be. An island is a member whose mean affinity to the rest
    sits at least ``island_z`` robust deviations below the cohort's median mean-affinity — a
    *stated rule*, not a fitted threshold, so a reader can recompute it.
    """
    names = list(reps)
    if len(names) < 2:
        raise ValueError("a cohort needs at least two members")
    ns = {np.asarray(v).shape[0] for v in reps.values()}
    if len(ns) != 1:
        raise ValueError(f"members must share the item set; got item counts {sorted(ns)}")
    n_items = ns.pop()

    if normalize_amplitude:
        reps = {m: normalize_items(v) for m, v in reps.items()}
    U = {m: frame(reps[m], k) for m in names}
    kk = min(u.shape[1] for u in U.values())
    pairwise = {f"{a}__{b}": round(affinity(U[a][:, :kk], U[b][:, :kk]), 4)
                for a, b in combinations(names, 2)}
    mean_aff = {m: round(float(np.mean([pairwise.get(f"{m}__{o}", pairwise.get(f"{o}__{m}"))
                                        for o in names if o != m])), 4) for m in names}

    vals = np.array([mean_aff[m] for m in names])
    med = float(np.median(vals))
    med_pairwise = float(np.median(list(pairwise.values())))
    mad = float(np.median(np.abs(vals - med))) or float(vals.std()) or 1e-9
    cut = med - island_z * 1.4826 * mad
    islands = [m for m in names if mean_aff[m] < cut]
    null = _haar_null(n_items, kk, n_null, seed)
    bp = _gap_p(vals, n_perm, seed)

    caveats = []
    if n_items < 3 * kk:
        caveats.append(
            f"{n_items} items against k={kk}: the random-frame null is drawn in R^n while frames "
            f"live in the centered (n-1) subspace, so the null is systematically too easy at this "
            f"ratio. At n_items <= ~1.5k, pure noise passes the shared-frame gate. Use "
            f"n_items >= 3k.")
    if len(names) < MIN_COHORT:
        verdict = "UNDERPOWERED__n_below_8"
        caveats.append(f"{len(names)} members: bimodality is not testable and no inferential "
                       f"verdict is offered. Candidate islands are reported as leads only.")
    elif bp <= 0.05:
        verdict = "ISLANDS_PRESENT"
    else:
        verdict = "UNIMODAL_COHORT"
    if med_pairwise <= null["p95"] < med:
        caveats.append(
            f"the MEAN-of-means ({round(med, 4)}) clears the null but the true median PAIRWISE "
            f"affinity ({round(med_pairwise, 4)}) does not. That pattern is the signature of a "
            f"COHORT SPLIT: two internally-legible groups that cannot read each other leave the "
            f"per-member means flat. This screen cannot see a balanced split — inspect `pairwise` "
            f"directly.")
    if med <= null["p95"]:
        caveats.append("cohort median affinity does not clear the random-frame null — these "
                       "members do not share a frame at all, so 'island' is not the right "
                       "description of any of them.")
    return CohortSurvey(verdict=verdict, members=names, k=kk, n_items=n_items,
                        pairwise=pairwise, mean_affinity=mean_aff, islands=islands, null=null,
                        bimodality_p=bp, cohort_median=round(med, 4),
                        median_pairwise_affinity=round(med_pairwise, 4),
                        island_rule=f"mean affinity < median - {island_z}*1.4826*MAD = "
                                    f"{round(cut, 4)}", caveats=caveats)


def _swap(X_T, U_T, U_S):
    """Express the target's own frame loadings through a donor frame (rank-k concept surgery)."""
    L = U_T.T @ X_T
    return X_T - U_T @ L + U_S @ L


def cliff(reader, island, legibility_fn, k: int = _DEF_K,
          doses=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), chance: float = None) -> dict:
    """How does legibility rise as the island's frame is rotated toward the reader's?

    ``legibility_fn(reader, island_corrected) -> float`` is **supplied by the caller**, and that
    is deliberate: in any real application the legibility measure already exists and belongs to
    the user — a decoder's accuracy on held-out items, a cross-subject decoding score, a
    label-free discovery routine. This module does not substitute one.

    *Why it is not optional:* the first version of this function shipped an internal measure
    (orthogonal Procrustes + Hungarian). It failed its own exam — on the model pair where this
    lab measured 0.9745, it returned chance — because Procrustes requires a known correspondence
    while recovering the correspondence is the entire task. It was removed rather than shipped.
    See ``papers/disjoint-worlds/`` for the alignment machinery that does pass.

    Returns the dose curve, the knee, the transition width, and a shape reading — or a REFUSED
    shape when the full-correction endpoint does not clear ``chance``, because a knee computed
    from noise is worse than no knee.
    """
    UT, US = frame(island, k), frame(reader, k)
    curve = {}
    for t in doses:
        if t == 0.0:
            Xd = np.asarray(island, dtype=float)
        else:
            Q, _ = np.linalg.qr((1 - t) * UT + t * US)
            Xd = _swap(np.asarray(island, dtype=float), UT, Q[:, :UT.shape[1]])
        curve[str(t)] = round(float(legibility_fn(reader, Xd)), 4)
    top = curve[str(doses[-1])]
    floor = chance if chance is not None else 1.0 / len(np.asarray(reader))
    if top <= max(floor * 2, floor + 1e-9):
        return {"legibility_by_dose": curve, "chance": round(floor, 4), "knee_t_half": None,
                "transition_width": None,
                "shape": "REFUSED__endpoint_at_chance_no_curve_to_read",
                "why": "full frame correction did not raise legibility above chance; either the "
                       "legibility measure cannot see this pair or there is nothing to see. A "
                       "knee read off this curve would be noise."}
    at = lambda frac: next((t for t in doses if curve[str(t)] >= frac * top), None)  # noqa: E731
    q1, q3 = at(0.25), at(0.75)
    return {"legibility_by_dose": curve, "chance": round(floor, 4), "knee_t_half": at(0.5),
            "transition_width": (round(q3 - q1, 4) if q1 is not None and q3 is not None else None),
            "shape": ("switch-like" if q1 is not None and q3 is not None and q3 - q1 <= 0.25
                      else "graded")}


def rescue(reader, island, legibility_fn, ranks=(1, 2, 5, 10, 20, 40), seed: int = 343) -> dict:
    """Does a LOW-RANK frame correction recover legibility — and at what rank?

    ``legibility_fn`` is caller-supplied for the reason given in :func:`cliff`. Each rank is
    scored against a matched random-orthonormal frame of the same rank, so a recovery cannot be
    credited to "any k directions would do". A small ``min_sufficient_rank`` means the deficit is
    a *rotation*: the information was present and misaddressed, not absent.
    """
    island = np.asarray(island, dtype=float)
    out, rng = {}, np.random.default_rng(seed + 7000)
    base = float(legibility_fn(reader, island))
    for k in ranks:
        UT, US = frame(island, k), frame(reader, k)
        kk = min(UT.shape[1], US.shape[1])
        corrected = float(legibility_fn(reader, _swap(island, UT[:, :kk], US[:, :kk])))
        Q, _ = np.linalg.qr(rng.standard_normal((len(island), kk)))
        null = float(legibility_fn(reader, _swap(island, UT[:, :kk], Q[:, :kk])))
        out[str(k)] = {"corrected": round(corrected, 4), "random_frame_null": round(null, 4)}
    full = out[str(ranks[-1])]["corrected"]
    floor = 1.0 / len(np.asarray(reader))
    if full <= max(floor * 2, floor + 1e-9):
        return {"baseline_legibility": round(base, 4), "by_rank": out,
                "full_rank_legibility": round(full, 4), "min_sufficient_rank": None,
                "reading": "REFUSED__full_rank_correction_at_chance",
                "why": "no rank recovered legibility above chance, so no sufficient rank exists "
                       "to report. This is a statement about the pair or the measure, not "
                       "evidence that the deficit is high-rank."}
    suff = next((k for k in ranks if out[str(k)]["corrected"] >= 0.5 * full
                 and out[str(k)]["random_frame_null"] <= 0.15), None)
    return {"baseline_legibility": round(base, 4), "by_rank": out,
            "full_rank_legibility": round(full, 4), "min_sufficient_rank": suff,
            "reading": ("low-rank rotation" if suff is not None and suff <= max(ranks) // 4
                        else "not low-rank at these ranks")}


def _demo() -> int:
    """Ten seconds, no data, no GPU: plant a cohort with one island and watch it get found."""
    rng = np.random.default_rng(0)
    n = 120
    shared = rng.standard_normal((n, 8))                     # the geometry six minds share
    reps = {f"mind_{i}": shared @ rng.standard_normal((8, 24 + i))
                          + 0.05 * rng.standard_normal((n, 24 + i)) for i in range(6)}
    reps["mind_6"] = shared @ rng.standard_normal((8, 24)) + 0.05 * rng.standard_normal((n, 24))
    reps["ISLAND"] = (rng.standard_normal((n, 8)) @ rng.standard_normal((8, 24)))  # its own geometry

    print("styxx.islands — a cohort of 8 minds over 120 shared items.")
    print("Seven were built from one shared geometry. One was not. Nothing is labelled.\n")
    s = survey(reps, n_null=400, n_perm=400)
    print(s)
    print("The instrument was given no labels, no pairing, and no hint which member is which.")
    print("Frame affinity alone separates the planted island from the clique.\n")

    print("And when there is nothing to see, it says so rather than guessing:")
    out = cliff(reps["mind_0"], reps["ISLAND"], legibility_fn=lambda r, i: 1.0 / len(r))
    print(f"  cliff(...) with a legibility measure that cannot see -> {out['shape']}")
    print(f"  {out['why']}\n")
    print("Your turn:  survey({name: array_of_shape_n_items_by_dim, ...})")
    print("  fMRI betas over shared stimuli · MEG epochs · activations · embeddings —")
    print("  anything where several minds saw the same items in the same order.")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="styxx.islands",
        description="Survey a cohort of minds for islands: shared frame, cliff, low-rank rescue.")
    ap.add_argument("npz", nargs="?",
                    help="an .npz whose arrays are (n_items, dim) over the SAME items")
    ap.add_argument("--demo", action="store_true",
                    help="run a self-contained demonstration on a planted cohort (no data needed)")
    ap.add_argument("--k", type=int, default=_DEF_K)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.demo or not a.npz:
        if not a.npz and not a.demo:
            print("no .npz given — running --demo. Pass a file to survey your own cohort.\n")
        return _demo()
    z = np.load(a.npz, allow_pickle=True)
    reps = {n: np.asarray(z[n]) for n in z.files}
    s = survey(reps, k=a.k)
    print(s)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(s.to_dict(), f, indent=2)
        print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
