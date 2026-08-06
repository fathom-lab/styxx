"""styxx.coupling — is this agent coupled to that stream, beyond the obvious confound?

One instrument for a question this program kept meeting in different costumes:

* **mind ↔ mind** — do two models share recoverable structure? (``papers/disjoint-worlds/``)
* **mind ↔ world** — is a physical room's state coupled to an agent's internal state?
  (``papers/first-afference/``)
* **mind ↔ brain** — does a decoder's feature stream track a subject's neural stream?

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

**Every refusal in this module was earned by a specific published failure.** They are scars, not
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

Open by design (``docs/governance/OPEN_CORE.md``): a measurement primitive, never gated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

__all__ = ["couple", "Coupling", "rv_coefficient", "resample_pair"]


def rv_coefficient(X: np.ndarray, Y: np.ndarray) -> float:
    """Matrix correlation between two multivariate streams. Symmetric, dimension-robust, no fit."""
    Xc = np.asarray(X, float) - np.asarray(X, float).mean(0)
    Yc = np.asarray(Y, float) - np.asarray(Y, float).mean(0)
    Sxy = Xc.T @ Yc
    num = np.trace(Sxy @ Sxy.T)
    den = np.sqrt(np.trace((Xc.T @ Xc) @ (Xc.T @ Xc)) * np.trace((Yc.T @ Yc) @ (Yc.T @ Yc)))
    return float(num / (den + 1e-12))


def _zscore(X):
    X = np.asarray(X, float)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def resample_pair(A, ts_a, B, ts_b, bin_seconds: float = 60.0):
    """Mean-pool both streams onto a shared time grid; drop bins missing on either side.

    Returns (A_binned, B_binned, bin_index). Timestamps are epoch seconds.
    """
    def to_bins(X, ts):
        out = {}
        for t, x in zip(np.asarray(ts, float), np.asarray(X, float)):
            out.setdefault(int(t // bin_seconds), []).append(x)
        return {b: np.mean(np.asarray(v), 0) for b, v in out.items()}
    ba, bb = to_bins(A, ts_a), to_bins(B, ts_b)
    common = sorted(set(ba) & set(bb))
    if not common:
        return np.empty((0, 0)), np.empty((0, 0)), []
    return (np.asarray([ba[b] for b in common]), np.asarray([bb[b] for b in common]), common)


def _confound_matched_perm(n, groups, rng):
    """Permute only WITHIN confound groups, so the confound survives into the null."""
    idx = np.arange(n)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        idx[m] = m[rng.permutation(len(m))]
    return idx


@dataclass
class Coupling:
    verdict: str
    n_paired_bins: int
    rv: float
    matched_p: float
    free_p: float
    power_floor: dict
    confound_used: bool
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


def _power_floor(A, B, n_draws, rng):
    """Smallest RV this run could have called significant — quoted with every null result.

    Mixes B toward A's own row order in increasing amounts and finds where the matched null
    stops absorbing it. Reported, never gated.
    """
    n = len(A)
    nulls = sorted(rv_coefficient(A, B[rng.permutation(n)]) for _ in range(n_draws))
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
    Ab, Bb, bins = resample_pair(A, ts_a, B, ts_b, bin_seconds)
    n = len(Ab)
    caveats = []
    if n < min_bins:
        return Coupling(verdict="INVALID__insufficient_overlap", n_paired_bins=n, rv=0.0,
                        matched_p=1.0, free_p=1.0,
                        power_floor={"detectable_rv_at_p01": None, "n_calibration_draws": 0,
                                     "note": "not computed: coverage gate failed"},
                        confound_used=confound is not None,
                        caveats=[f"{n} paired bins < the {min_bins} required before anything is "
                                 f"read. An under-observed apparatus licenses nothing."])
    Az, Bz = _zscore(Ab), _zscore(Bb)
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
        if len(np.unique(groups)) >= n:
            caveats.append("every bin is its own confound group: the matched null is degenerate "
                           "and cannot absorb anything. Use a coarser confound.")

    floor = _power_floor(Az, Bz, max(n_perm // 2, 50), np.random.default_rng(seed + 99))

    if confound is not None and matched_p <= alpha:
        verdict = "COUPLED_BEYOND_CONFOUND__attribution_pending"
        caveats.append("attribution is NOT established: this statistic is symmetric and cannot "
                       "distinguish A tracking B from B registering A. If the two streams share "
                       "any physical channel (an agent's own hardware sitting in the room it "
                       "measures), that channel must be bounded before any directional claim.")
    elif free_p <= alpha:
        verdict = ("CONFOUND_ONLY__explained_by_the_supplied_confound" if confound is not None
                   else "FREE_SHUFFLE_ONLY__no_confound_supplied_claim_not_licensed")
    else:
        verdict = "NO_DETECTABLE_COUPLING__above_measured_floor"
        caveats.append(f"reads as: no coupling above RV {floor['detectable_rv_at_p01']} at this "
                       f"n, NOT as 'no coupling'.")
    return Coupling(verdict=verdict, n_paired_bins=n, rv=round(obs, 4),
                    matched_p=round(matched_p, 4), free_p=round(free_p, 4),
                    power_floor=floor, confound_used=confound is not None, caveats=caveats)


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
