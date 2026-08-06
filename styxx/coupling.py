"""styxx.coupling — is this agent coupled to that stream, beyond the obvious confound?

One instrument for a question this program kept meeting in different costumes:

* **mind ↔ mind** — do two models share recoverable structure? (``papers/disjoint-worlds/``)
* **mind ↔ world** — is a physical room's state coupled to an agent's internal state?
  (``papers/first-afference/``)
* **mind ↔ brain** — does a decoder's feature stream track a subject's neural stream?
  **UNTESTED: no neural data has been through this module.** The defaults here
  (``bin_seconds=60``, ``min_bins=200``) are wrong by two to five orders of magnitude for fMRI
  (TR 1–2 s) or MEG/EEG (milliseconds), and the field standard for this question is a
  cross-validated voxelwise encoding model with held-out prediction *r*, which also answers the
  directional question a symmetric coefficient cannot. Treat this bullet as an intended
  application, not a demonstrated one, until it has been run on real recordings.

They are the same measurement: two timestamped streams, resampled to a common grid, scored for
dependence against a null that **preserves the confound you are worried about**. The confound is
the whole game. A room and an agent both get busier during the day; a brain and a stimulus
stream both follow the experiment's clock; two models both track token frequency. Beat a naive
shuffle and you have proven a clock exists. Beat a confound-preserving shuffle and you have
something.

    from styxx.coupling import couple

    r = couple(agent_rows, agent_ts, room_rows, room_ts,
               confound=lambda ts: [t.hour for t in ts])     # preserve time-of-day
    r.verdict          # COUPLED_BEYOND_CONFOUND__attribution_pending | CONFOUND_ONLY |
                       # NO_DETECTABLE_COUPLING__above_measured_floor | INVALID__*
    r.power_floor      # what this run could have detected, quoted with every null

**What is and is not new here.** The confound-preserving null is **not our invention and not
novel** — it is a restricted/stratified permutation test, standard in neuroimaging since Nichols
& Holmes (*HBM* 2002) and Anderson & ter Braak (*JSCS* 2003), generalized as the conditional
permutation test by Berrett, Wang, Barber & Samworth (*JRSS-B* 2020), and shipped in FSL PALM as
exchangeability blocks (Winkler et al., *NeuroImage* 2015). Confound control in decoding
specifically has its own literature (Snoek et al., *NeuroImage* 2019; Görgen et al., *NeuroImage*
2018). Autocorrelation-preserving surrogates are equally standard (Theiler et al. 1992; Schreiber
& Schmitz 1996), and circular shift is the default null in intersubject-correlation work (Simony
et al., *Nat. Commun.* 2016). If you are doing neuroimaging, you already have all of this.

What this module contributes is **the composition and the defaults**: an instrument that refuses
to license a positive unless a confound-preserving null, an autocorrelation-preserving null, a
leverage check and a sampling-density check all pass — and that names which one stopped it. The
sampling-density verdict in particular we could not find in the literature.

**Every refusal was earned by a specific published failure.** They are scars, not
speculation:

* ``INVALID__insufficient_overlap`` — an under-observed apparatus licenses nothing
  (``papers/disjoint-worlds/`` b35-b).
* The confound-preserving null is the *licensing* null, and the free shuffle is reported only as
  a contrast — because in validation the pure-clock world was maximally significant against the
  free null (p 0.002) and entirely absorbed by the matched one (``FINDING_r0v2_*``).
* ``attribution_pending`` is in the verdict **string**, because every statistic here is
  symmetric and cannot tell "the agent tracks the room" from "the room registers the agent" —
  and an agent's own hardware is usually *in* the room it is measuring
  (``ROADMAP_r_line_*`` invariant 6).
* A null result quotes ``power_floor`` and never says "no coupling" — because an instrument that
  cannot see was shipped here once and caught by its own exam (``FINDING_r0_instrument_blind_*``).
* ``COUPLED__sampling_density_confound_unbounded`` — found by pairing real agent telemetry
  against its own **time-reversed copy**, where no bin-level coupling can exist, and getting
  RV 0.3704 at p 0.0033. Irregular sampling makes both streams' bin averages shrink together;
  the alignment is real, the cause is the clock on the recorder, and no permutation null absorbs
  it. This verdict fires *instead of* a positive when the channel is open.

Open by design (``docs/governance/OPEN_CORE.md``): a measurement primitive, never gated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

__all__ = ["couple", "Coupling", "rv_coefficient", "debiased_cka", "resample_pair"]


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    """Matrix correlation between two multivariate streams. Symmetric, dimension-robust, no fit."""
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(0)
    Yc = np.asarray(Y, float) - np.asarray(Y, float).mean(0)
    Sxy = Xc.T @ Yc
    num = np.trace(Sxy @ Sxy.T)
    den = np.sqrt(np.trace((Xc.T @ Xc) @ (Xc.T @ Xc)) * np.trace((Yc.T @ Yc) @ (Yc.T @ Yc)))
    return float(num / (den + 1e-12))


def debiased_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Unbiased-HSIC estimator of the same quantity ``rv_coefficient`` computes.

    ``rv_coefficient`` IS linear CKA, and linear CKA is upward-biased when feature count is large
    relative to sample count: on INDEPENDENT random streams at this module's own minimum of 200
    bins it reads 0.058 at 12 features, 0.333 at 100, 0.714 at 500 and 0.909 at 2000. The
    permutation p-value is unaffected (the null is drawn at the same n and p, so the bias
    cancels), but the raw coefficient is a dimensionality readout, not an effect size, and is not
    comparable across runs. Reported alongside it, per Song et al. (*JMLR* 2012) and Murphy,
    Zylberberg & Fyshe (*ICLR Re-Align* 2024), who name fMRI and MEG as exactly this regime.
    """
    X = np.asarray(X, float) - np.asarray(X, float).mean(0)
    Y = np.asarray(Y, float) - np.asarray(Y, float).mean(0)
    n = X.shape[0]
    if n < 4:
        return float("nan")
    K, L = X @ X.T, Y @ Y.T
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    ones = np.ones(n)

    def hsic(K, L):
        t1 = float(np.sum(K * L))
        t2 = float(ones @ K @ ones) * float(ones @ L @ ones) / ((n - 1) * (n - 2))
        t3 = 2.0 * float(ones @ K @ L @ ones) / (n - 2)
        return (t1 + t2 - t3) / (n * (n - 3))
    hkl, hkk, hll = hsic(K, L), hsic(K, K), hsic(L, L)
    den = np.sqrt(max(hkk, 0.0) * max(hll, 0.0))
    return float(hkl / den) if den > 0 else 0.0


def _zscore(X):
    X = np.asarray(X, float)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def resample_pair(A, ts_a, B, ts_b, bin_seconds: float = 60.0):
    """Mean-pool both streams onto a shared time grid; drop bins missing on either side.

    Returns (A_binned, B_binned, bin_index, bin_counts). Timestamps are epoch seconds.
    """
    def to_bins(X, ts):
        out = {}
        for t, x in zip(np.asarray(ts, float), np.asarray(X, float)):
            out.setdefault(int(t // bin_seconds), []).append(x)
        return {b: np.mean(np.asarray(v), 0) for b, v in out.items()}
    ba, bb = to_bins(A, ts_a), to_bins(B, ts_b)
    ca = {}
    for t_, _ in zip(np.asarray(ts_a, float), np.asarray(A, float)):
        ca[int(t_ // bin_seconds)] = ca.get(int(t_ // bin_seconds), 0) + 1
    common = sorted(set(ba) & set(bb))
    if not common:
        return np.empty((0, 0)), np.empty((0, 0)), [], np.empty(0)
    return (np.asarray([ba[b] for b in common]), np.asarray([bb[b] for b in common]),
            common, np.asarray([ca.get(b, 0) for b in common]))


def _confound_matched_perm(n, groups, rng):
    """Permute only WITHIN confound groups, so the confound survives into the null."""
    idx = np.arange(n)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        idx[m] = m[rng.permutation(len(m))]
    return idx


def _density_confound(Az, Bz, counts) -> dict:
    """How much of each binned stream's magnitude is explained by how many records fell in the bin?

    Irregular sampling makes a bin holding many records average toward the mean and a sparse bin
    stay extreme — in BOTH streams identically, because they share the grid. That is aligned
    structure with no shared cause, and a permutation null cannot absorb it (shuffling rows is
    exactly what destroys the alignment the artifact lives in). Discovered on real agent telemetry
    2026-08-06: a stream paired against its own TIME-REVERSED copy — no bin-level coupling
    possible — read RV 0.3704 at p 0.0033 through this channel alone.
    """
    if counts.size == 0 or counts.std() == 0:
        return {"applicable": False, "note": "uniform sampling: bins hold equal counts"}
    na, nb = np.linalg.norm(Az, axis=1), np.linalg.norm(Bz, axis=1)
    ra = float(np.corrcoef(counts, na)[0, 1]) if na.std() > 0 else 0.0
    rb = float(np.corrcoef(counts, nb)[0, 1]) if nb.std() > 0 else 0.0
    return {"applicable": True, "count_min": int(counts.min()), "count_max": int(counts.max()),
            "corr_count_vs_magnitude_a": round(ra, 4), "corr_count_vs_magnitude_b": round(rb, 4),
            "shared": bool(abs(ra) >= 0.3 and abs(rb) >= 0.3)}


def _lag1(X):
    """Lag-1 autocorrelation of the binned stream, averaged over columns."""
    if len(X) < 3:
        return 0.0
    r = []
    for j in range(X.shape[1]):
        a, b = X[:-1, j], X[1:, j]
        if a.std() > 1e-12 and b.std() > 1e-12:
            r.append(abs(float(np.corrcoef(a, b)[0, 1])))
    return round(float(np.mean(r)) if r else 0.0, 4)


def _trend_r2(X) -> float:
    """Mean fraction of each column's variance explained by a straight line in bin order.

    A shared drift defeats BOTH nulls: within-group permutation destroys it (so the null is too
    narrow) and a circular shift preserves enough of it to stay high (so that null fails too).
    Red-team 2026-08-06: independent linear drifts produced 21/21 false positives.
    """
    n = len(X)
    if n < 5:
        return 0.0
    t = np.arange(n, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    r2 = []
    for j in range(X.shape[1]):
        y = X[:, j]
        if y.std() <= 1e-12:
            continue
        r2.append(float(np.corrcoef(t, y)[0, 1] ** 2))
    return round(float(np.mean(r2)) if r2 else 0.0, 4)


def _leverage(Az, Bz) -> float:
    """Fraction of the RV numerator carried by the single most influential bin.

    RV is a second-moment statistic: a couple of shared glitch bins (a power blip written to both
    logs, a clock-sync marker) can carry an entire verdict. Red-team 2026-08-06: two sentinel bins
    out of 336 produced RV 0.973 at p 0.002 on otherwise independent streams.
    """
    n = len(Az)
    full = rv_coefficient(Az, Bz)
    if full <= 0 or n < 20:
        return 0.0
    # target the bins that can actually carry the statistic: highest joint magnitude.
    joint = np.linalg.norm(Az, axis=1) * np.linalg.norm(Bz, axis=1)
    k = max(2, n // 100)
    top = np.argsort(joint)[-k:]
    without = rv_coefficient(np.delete(Az, top, 0), np.delete(Bz, top, 0))
    return round(float(max(0.0, (full - without) / (full + 1e-12))), 4)


@dataclass
class Coupling:
    verdict: str
    n_paired_bins: int
    rv: float
    matched_p: float
    free_p: float
    power_floor: dict
    confound_used: bool
    rv_debiased: float = float("nan")
    sampling_density: dict = field(default_factory=dict)
    dependence: dict = field(default_factory=dict)
    caveats: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        s = (f"{self.verdict}\n"
             f"  {self.n_paired_bins} paired bins · RV {self.rv}\n"
             f"  confound-matched p {self.matched_p}   (licensing null)\n"
             f"  free-shuffle    p {self.free_p}   (contrast only)\n"
             f"  power floor: {self.power_floor['detectable_rv_at_p01']} RV at p<=0.01, "
             f"estimated from {self.power_floor['n_calibration_draws']} draws\n")
        return s + "".join(f"  ! {c}\n" for c in self.caveats)


def _power_floor(A, B, n_draws, rng, perm_fn):
    """Smallest RV this run could have called significant — quoted with every null result.

    Calibrated against the SAME null that licenses a positive (the confound-matched permutation
    when one is supplied), not the free shuffle — red-team 2026-08-06 caught the docstring and
    the code disagreeing, which meant the one number quoted with every null result was
    calibrated against the wrong distribution. Reported, never gated.
    """
    n = len(A)
    nulls = sorted(rv_coefficient(A, B[perm_fn(n, rng)]) for _ in range(n_draws))
    p99 = float(nulls[min(int(0.99 * len(nulls)), len(nulls) - 1)])
    return {"detectable_rv_at_p01": round(p99, 4), "n_calibration_draws": int(n_draws),
            "note": "an observed RV at or below this could not have been called significant; "
                    "a null result means no coupling ABOVE this floor, never 'no coupling'"}


def couple(A, ts_a, B, ts_b, confound=None, bin_seconds: float = 60.0,
           n_perm: int = 500, min_bins: int = 200, seed: int = 343,
           alpha: float = 0.01) -> Coupling:
    """Measure coupling between two timestamped multivariate streams.

    ``confound`` maps the bin index array to a group label per bin (e.g. hour-of-day). The
    licensing null permutes **within** those groups so the confound cannot masquerade as
    coupling. Passing ``None`` runs free-shuffle only and the result says so loudly — that
    configuration answers a weaker question than most callers think.
    """
    Ab, Bb, bins, counts = resample_pair(A, ts_a, B, ts_b, bin_seconds)
    n = len(Ab)
    caveats = []
    if n < min_bins:
        return Coupling(verdict="INVALID__insufficient_overlap", n_paired_bins=n, rv=0.0,
                        matched_p=1.0, free_p=1.0,
                        power_floor={"detectable_rv_at_p01": None, "n_calibration_draws": 0,
                                     "note": "not computed: coverage gate failed"},
                        confound_used=confound is not None, sampling_density={},
                        caveats=[f"{n} paired bins < the {min_bins} required before anything is "
                                 f"read. An under-observed apparatus licenses nothing."])
    if not (np.isfinite(Ab).all() and np.isfinite(Bb).all()):
        return Coupling(verdict="INVALID__nonfinite_input", n_paired_bins=n, rv=float("nan"),
                        matched_p=1.0, free_p=1.0,
                        power_floor={"detectable_rv_at_p01": None, "n_calibration_draws": 0,
                                     "note": "not computed: non-finite input"},
                        confound_used=confound is not None, sampling_density={}, dependence={},
                        caveats=["NaN or inf in the binned streams. Left unchecked this produced "
                                 "a maximal-confidence COUPLED verdict, because every permuted RV "
                                 "is also NaN and `nan >= nan` is False, so zero permutations "
                                 "exceed the observation. Clean or drop the affected bins."])
    Az, Bz = _zscore(Ab), _zscore(Bb)
    density = _density_confound(Az, Bz, counts)
    ac_a, ac_b = _lag1(Az), _lag1(Bz)
    obs = rv_coefficient(Az, Bz)
    rng_f = np.random.default_rng(seed + 62000)
    ge_f = sum(rv_coefficient(Az, Bz[rng_f.permutation(n)]) >= obs for _ in range(n_perm))
    free_p = (ge_f + 1) / (n_perm + 1)

    if confound is None:
        caveats.append("no confound supplied: only the FREE shuffle was run, which any shared "
                       "clock or shared trend will beat. This does not license a coupling claim.")
        matched_p = 1.0
    else:
        groups = np.asarray(confound(np.asarray(bins)))
        if len(groups) != n:
            raise ValueError(f"confound returned {len(groups)} labels for {n} bins")
        rng_h = np.random.default_rng(seed + 31000)
        ge_h = sum(rv_coefficient(Az, Bz[_confound_matched_perm(n, groups, rng_h)]) >= obs
                   for _ in range(n_perm))
        matched_p = (ge_h + 1) / (n_perm + 1)
        _, gcounts = np.unique(groups, return_counts=True)
        frozen_note = round(int(gcounts[gcounts == 1].sum()) / n, 4)
        frozen_frac = frozen_note
        if len(np.unique(groups)) >= n:
            caveats.append("every bin is its own confound group: the matched null is degenerate "
                           "and cannot absorb anything. Use a coarser confound.")
        elif frozen_frac >= 0.2:
            caveats.append(
                f"{frozen_frac:.0%} of bins sit in singleton confound strata and are therefore "
                f"FROZEN at their true pairing in every null draw — that fraction of the data is "
                f"never actually tested against the confound. This costs power silently; the "
                f"power floor below is the honest consequence. Use a coarser confound.")

    # every valid shift, capped: the null must be able to RESOLVE p <= alpha, i.e. it needs at
    # least 1/alpha draws. Under-drawing it silently converts real coupling into an INVALID.
    max_shift_draws = max(n_perm, int(np.ceil(2.0 / max(alpha, 1e-6))))
    step = max(1, (n - 10) // max_shift_draws)
    shift_nulls = [rv_coefficient(Az, np.roll(Bz, s, axis=0)) for s in range(5, n - 5, step)]
    if len(shift_nulls) < np.ceil(1.0 / max(alpha, 1e-6)):
        caveats.append(
            f"circular-shift null has only {len(shift_nulls)} draws (n={n} bins), so its "
            f"smallest attainable p is {1/(len(shift_nulls)+1):.4f} — it cannot resolve "
            f"alpha={alpha}. Autocorrelated data at this length cannot license a positive here.")
    shift_p = round((sum(v >= obs for v in shift_nulls) + 1) / (len(shift_nulls) + 1), 4)
    autocorrelated = max(ac_a, ac_b) >= 0.2
    tr_a, tr_b = _trend_r2(Az), _trend_r2(Bz)
    shared_trend = tr_a >= 0.2 and tr_b >= 0.2
    dependence = {"frozen_bin_fraction": (frozen_note if confound is not None else 0.0),
                  "lag1_autocorr_a": ac_a, "lag1_autocorr_b": ac_b,
                  "trend_r2_a": tr_a, "trend_r2_b": tr_b, "shared_trend": bool(shared_trend),
                  "autocorrelated": bool(autocorrelated), "circular_shift_p": shift_p,
                  "n_shift_draws": len(shift_nulls),
                  "leverage_top_bin_share": _leverage(Az, Bz)}
    licensing_p = max(matched_p, shift_p) if autocorrelated else matched_p
    if confound is not None:
        _g = np.asarray(confound(np.asarray(bins)))
        _pf = lambda m, r: _confound_matched_perm(m, _g, r)      # noqa: E731
    else:
        _pf = lambda m, r: r.permutation(m)                      # noqa: E731
    floor = _power_floor(Az, Bz, max(n_perm // 2, 50), np.random.default_rng(seed + 99), _pf)

    if confound is not None and licensing_p <= alpha and shared_trend:
        verdict = "INVALID__shared_temporal_trend"
        caveats.append(
            f"Both streams drift monotonically over the window (linear-in-time R^2 {tr_a} and "
            f"{tr_b}). A shared trend defeats BOTH nulls: within-group permutation destroys it, "
            f"so that null is too narrow, and a circular shift preserves it, so that null stays "
            f"high. Two streams that merely warmed up during the observation window reach the "
            f"permutation floor. Detrend or difference both streams and re-run — and note that "
            f"doing so changes the question to 'coupled beyond a linear trend'.")
    elif confound is not None and licensing_p <= alpha and dependence["leverage_top_bin_share"] >= 0.5:
        verdict = "COUPLED__driven_by_a_single_bin"
        caveats.append(
            f"REFUSING the coupled verdict: one bin carries "
            f"{dependence['leverage_top_bin_share']:.0%} of the RV. A shared glitch written to "
            f"both logs (power blip, clock-sync marker, log rotation) reproduces this exactly. "
            f"Winsorize or rank-transform, or exclude the bin and re-run.")
    elif confound is not None and licensing_p <= alpha and density.get("shared"):
        verdict = "COUPLED__sampling_density_confound_unbounded"
        caveats.append(
            f"REFUSING the coupled verdict: bin record-count explains the magnitude of BOTH "
            f"streams (r {density['corr_count_vs_magnitude_a']} and "
            f"{density['corr_count_vs_magnitude_b']}; bins hold {density['count_min']}-"
            f"{density['count_max']} records). Two streams sharing only their sampling times "
            f"acquire aligned magnitude structure that no permutation null can absorb, because "
            f"shuffling rows is exactly what destroys the alignment the artifact lives in. "
            f"Bound it before claiming coupling: bin uniformly, subsample to equal counts, or "
            f"pass a confound that strata on bin count.")
    elif confound is not None and licensing_p <= alpha:
        verdict = "COUPLED_BEYOND_CONFOUND__attribution_pending"
        caveats.append("'beyond confound' means beyond the ONE confound you supplied; other "
                       "shared drivers on other timescales are not excluded. A day-of-week "
                       "driver survives an hour-of-day null intact.")
        if autocorrelated:
            caveats.append(f"streams are autocorrelated (lag-1 {ac_a} / {ac_b}); the licensing "
                           f"p is the CONSERVATIVE max of the confound-matched null and an "
                           f"autocorrelation-preserving circular-shift null (shift p {shift_p}).")
        caveats.append("attribution is NOT established: this statistic is symmetric and cannot "
                       "distinguish A tracking B from B registering A. If the two streams share "
                       "any physical channel (an agent's own hardware sitting in the room it "
                       "measures), that channel must be bounded before any directional claim.")
    elif autocorrelated and matched_p <= alpha:
        verdict = "INVALID__autocorrelation_defeats_the_permutation_null"
        caveats.append(
            f"The confound-matched null assumes bins are exchangeable within confound groups. "
            f"These are not: lag-1 autocorrelation {ac_a} / {ac_b}. Within-group shuffling "
            f"destroys each stream's own temporal structure, so the null describes white noise "
            f"the data is not, and two INDEPENDENT drifting streams reach the permutation floor. "
            f"The circular-shift null, which preserves that structure, gives p {shift_p}. "
            f"Prewhiten, difference, or use a block/stationary-bootstrap null before claiming "
            f"coupling.")
    elif free_p <= alpha:
        verdict = ("CONFOUND_ONLY__explained_by_the_supplied_confound" if confound is not None
                   else "FREE_SHUFFLE_ONLY__no_confound_supplied_claim_not_licensed")
    else:
        verdict = "NO_DETECTABLE_COUPLING__above_measured_floor"
        caveats.append(f"reads as: no coupling above RV {floor['detectable_rv_at_p01']} at this "
                       f"n, NOT as 'no coupling'.")
    return Coupling(verdict=verdict, n_paired_bins=n, rv=round(obs, 4),
                    rv_debiased=round(debiased_cka(Az, Bz), 4),
                    matched_p=round(matched_p, 4), free_p=round(free_p, 4),
                    power_floor=floor, confound_used=confound is not None,
                    sampling_density=density, dependence=dependence, caveats=caveats)


def _demo() -> int:
    """Three worlds with known truth — the exam this module's design had to pass."""
    rng = np.random.default_rng(11)
    n, day = 240, 86400
    ts = np.sort(rng.choice(14 * 24, size=n, replace=False)).astype(float) * 3600.0
    hours = ((ts // 3600) % 24).astype(int)
    ang = 2 * np.pi * hours / 24
    clock = np.stack([np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)], 1)

    def world(coupled, clocked):
        z = rng.standard_normal((n, 4))
        zb = z if coupled else rng.standard_normal((n, 4))
        c = clock if clocked else np.zeros_like(clock)
        A = np.tanh(z @ rng.standard_normal((4, 12)) + c @ rng.standard_normal((4, 12)))
        B = np.tanh(zb @ rng.standard_normal((4, 24)) + c @ rng.standard_normal((4, 24)))
        return A + 0.15 * rng.standard_normal((n, 12)), B + 0.15 * rng.standard_normal((n, 24))

    print("styxx.coupling — the same instrument for mind↔mind, mind↔world, mind↔brain.\n")
    for name, (coupled, clocked) in {
            "COUPLED (shared latent + clock)": (True, True),
            "CLOCK ONLY (independent latents, same clock)": (False, True),
            "NOTHING (independent, no clock)": (False, False)}.items():
        A, B = world(coupled, clocked)
        r = couple(A, ts, B, ts, confound=lambda b: np.asarray(b) % 24,
                   bin_seconds=3600, n_perm=200, min_bins=200)
        print(f"{name}\n  {r}")
    print("The clock-only world is the point: a naive shuffle calls it significant, the")
    print("confound-matched null absorbs it. That gap is where false discoveries live.")
    return 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="styxx.coupling",
                                 description="Is this agent coupled to that stream?")
    ap.add_argument("--demo", action="store_true")
    ap.parse_args(argv)
    return _demo()


if __name__ == "__main__":
    raise SystemExit(main())
