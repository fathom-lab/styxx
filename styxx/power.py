"""styxx.power — decide whether a preregistered bar is reachable, BEFORE the compute is spent.

**Status: under examination. Not exported from the package. See
``papers/first-afference/PREREG_p1_power_refusal_2026_08_08.md``.**

Three of this lab's five preregistration defects in the week of 2026-08-02 were the same defect:
a bar fixed before anyone computed whether an instrument could clear it. ``styxx.protocol`` v2
added ``power_basis``, which asks an author to *declare* that a bar is reachable. This module
asks whether the declaration can be *checked*.

The question it answers is narrow and worth stating precisely. Given

* a **null** — draws of the metric when the effect is absent, and
* a **bar** with a direction,

it reports the false-positive rate the bar concedes. Given additionally

* an **alternative** — draws of the metric when the effect is present at the size you care about,

it reports power, and calls the bar REACHABLE or UNREACHABLE. It cannot tell you whether your
alternative is the right one to care about; nothing can. It can tell you that the bar you wrote
is beyond the reach of the apparatus you have, which is the mistake we actually keep making.

Three constructions have their own entry points because all three burned us:

``effective_n``      Bartlett-corrected sample size for an autocorrelated series. The C-series
                     died here: nominal 300 timepoints, effective 6.9 to 45.8.
``order_stat_bar``   The bar a MAXIMUM of n draws must clear. b48 died here: a bar correct for
                     one draw, applied to the largest of 45.
``bounded_ceiling``  The largest value a bounded discrete statistic can attain, and the spacing
                     between attainable values. A bar between two attainable values is
                     unreachable no matter how strong the effect.

Everything refuses rather than guesses. A verdict with no basis is the failure mode this whole
program exists to prevent, so a null too small to estimate a tail from, a null with zero
variance, or any non-finite input raises ``PowerError``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = ["PowerError", "Reach", "effective_n", "order_stat_bar", "bounded_ceiling",
           "false_positive_rate", "reachable", "min_detectable_bar"]

MIN_DRAWS = 20          # below this a tail estimate is noise; refuse rather than report it
_OPS = {">=": lambda x, b: x >= b, ">": lambda x, b: x > b,
        "<=": lambda x, b: x <= b, "<": lambda x, b: x < b}


class PowerError(ValueError):
    """The reachability of this bar cannot be computed from what was provided."""


@dataclass
class Reach:
    """A reachability verdict. ``reachable`` is the single source of truth."""
    reachable: bool
    verdict: str
    alpha: float                      # false-positive rate the bar concedes under the null
    power: float | None               # detection rate under the alternative, if one was given
    bar: float
    op: str
    reasons: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def __bool__(self) -> bool:       # ``if reachable(...)`` must mean what it says
        return self.reachable


def _clean(x: Sequence[float], what: str) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise PowerError(f"{what} is empty")
    if not np.all(np.isfinite(a)):
        raise PowerError(f"{what} contains non-finite values; refusing to estimate a tail from it")
    return a


def _check_bar(bar, op):
    if isinstance(bar, bool) or not isinstance(bar, (int, float)) or not math.isfinite(float(bar)):
        raise PowerError(f"bar must be a finite real number, got {bar!r}")
    if op not in _OPS:
        raise PowerError(f"op must be one of {sorted(_OPS)}, got {op!r}")


def effective_n(series: Sequence[float], *, nominal: int | None = None) -> dict:
    """Bartlett-corrected effective sample size for an autocorrelated series.

    A series of 300 autocorrelated timepoints does not carry 300 independent observations. The
    correction divides the nominal count by the integrated autocorrelation, ``1 + 2*sum(rho_k)``,
    truncated at the first non-positive lag (Bartlett's rule).

    Returns ``nominal``, ``rho1``, the integrated autocorrelation and ``effective``. Refuses on a
    constant series, where autocorrelation is undefined rather than zero.
    """
    a = _clean(series, "series")
    n = int(nominal if nominal is not None else a.size)
    if a.size < MIN_DRAWS:
        raise PowerError(f"series has {a.size} points; need at least {MIN_DRAWS}")
    a = a - a.mean()
    denom = float(a @ a)
    if denom <= 0.0:
        raise PowerError("series is constant; its autocorrelation is undefined, not zero")

    total, lags = 1.0, 0
    for k in range(1, a.size // 2):
        rho = float(a[k:] @ a[:-k]) / denom
        if rho <= 0.0:
            break
        total += 2.0 * rho
        lags = k
    rho1 = float(a[1:] @ a[:-1]) / denom
    eff = n / total if total > 0 else float(n)
    return {"nominal": n, "rho1": round(rho1, 4), "integrated_autocorrelation": round(total, 4),
            "lags_summed": lags, "effective": round(float(eff), 4)}


def order_stat_bar(null: Sequence[float], n_draws: int, alpha: float = 0.05, *,
                   op: str = ">=") -> dict:
    """The bar that the MAXIMUM of ``n_draws`` must clear to concede only ``alpha``.

    b48's defect, mechanised. A bar chosen so a *single* draw exceeds it with probability alpha is
    exceeded by the largest of n draws with probability ``1 - (1-alpha)**n`` — at n=45 and
    alpha=0.05 that is 0.90, so the gate fails almost every time under a true null.

    Returns the correct single-draw quantile (``per_draw_alpha``), the bar it implies, and the
    ``naive_bar`` an author would have written, so the two can be compared.
    """
    a = _clean(null, "null")
    if a.size < MIN_DRAWS:
        raise PowerError(f"null has {a.size} draws; need at least {MIN_DRAWS} to estimate a tail")
    if not isinstance(n_draws, int) or isinstance(n_draws, bool) or n_draws < 1:
        raise PowerError(f"n_draws must be a positive int, got {n_draws!r}")
    if not 0.0 < alpha < 1.0:
        raise PowerError(f"alpha must lie strictly in (0,1), got {alpha!r}")
    if op not in (">=", ">", "<=", "<"):
        raise PowerError(f"op must be a comparison, got {op!r}")

    per_draw = 1.0 - (1.0 - alpha) ** (1.0 / n_draws)
    upper = op in (">=", ">")
    q_naive = (1.0 - alpha) if upper else alpha
    q_corr = (1.0 - per_draw) if upper else per_draw
    naive, corrected = float(np.quantile(a, q_naive)), float(np.quantile(a, q_corr))
    inflated = 1.0 - (1.0 - alpha) ** n_draws
    return {"n_draws": n_draws, "alpha": alpha, "per_draw_alpha": round(per_draw, 6),
            "naive_bar": round(naive, 6), "corrected_bar": round(corrected, 6),
            "naive_bar_true_alpha": round(inflated, 6),
            "bars_differ": bool(abs(corrected - naive) > 1e-12),
            "note": (f"a single-draw bar at alpha={alpha} concedes {inflated:.4f} when applied to "
                     f"the max of {n_draws} draws")}


def bounded_ceiling(n_items: int, *, kind: str = "accuracy") -> dict:
    """Attainable values of a bounded discrete statistic, and the spacing between them.

    A proportion over ``n_items`` trials can only take the values ``k/n_items``. A bar that falls
    strictly between two attainable values is met by the same outcomes as the next attainable one
    up, so it is not the bar the author thinks they wrote; a bar above the ceiling is unreachable
    at any effect size.
    """
    if not isinstance(n_items, int) or isinstance(n_items, bool) or n_items < 1:
        raise PowerError(f"n_items must be a positive int, got {n_items!r}")
    if kind != "accuracy":
        raise PowerError(f"unknown kind {kind!r}; only 'accuracy' is implemented")
    return {"n_items": n_items, "ceiling": 1.0, "floor": 0.0, "chance": round(1.0 / n_items, 6),
            "spacing": round(1.0 / n_items, 6), "n_attainable_values": n_items + 1}


def false_positive_rate(null: Sequence[float], bar: float, op: str = ">=") -> float:
    """Fraction of null draws that clear the bar — the alpha the bar actually concedes."""
    a = _clean(null, "null")
    _check_bar(bar, op)
    if a.size < MIN_DRAWS:
        raise PowerError(f"null has {a.size} draws; need at least {MIN_DRAWS}")
    return round(float(_OPS[op](a, float(bar)).mean()), 6)


def min_detectable_bar(null: Sequence[float], alpha: float = 0.05, op: str = ">=",
                       *, n_draws: int = 1) -> float:
    """The most permissive bar that still concedes at most ``alpha`` under the null.

    Write this down *before* choosing a bar. If the bar you wanted is more permissive than this
    one, it does not control alpha; if the effect you can produce is smaller than this one, no
    bar controls both alpha and power and the experiment is not worth running as designed.
    """
    a = _clean(null, "null")
    if a.size < MIN_DRAWS:
        raise PowerError(f"null has {a.size} draws; need at least {MIN_DRAWS}")
    if not 0.0 < alpha < 1.0:
        raise PowerError(f"alpha must lie strictly in (0,1), got {alpha!r}")
    if n_draws > 1:
        return float(order_stat_bar(a, n_draws, alpha, op=op)["corrected_bar"])
    upper = op in (">=", ">")
    return round(float(np.quantile(a, (1.0 - alpha) if upper else alpha)), 6)


def reachable(null: Sequence[float], bar: float, op: str = ">=", *,
              alt: Sequence[float] | None = None, alpha: float = 0.05,
              target_power: float = 0.80, n_draws: int = 1,
              n_items: int | None = None) -> Reach:
    """Is this bar reachable by this apparatus? REACHABLE only if alpha AND power are both met.

    ``null`` is the metric under no effect. ``alt``, if given, is the metric under the effect you
    care about; without it only alpha is checked and the verdict cannot be REACHABLE, because a
    bar nothing can clear controls alpha perfectly.

    ``n_draws`` > 1 declares that the gate reads an order statistic over that many draws — the
    b48 construction. ``n_items`` declares a bounded proportion over that many trials, which adds
    a ceiling and a spacing check.
    """
    a = _clean(null, "null")
    _check_bar(bar, op)
    bar = float(bar)
    if a.size < MIN_DRAWS:
        raise PowerError(f"null has {a.size} draws; need at least {MIN_DRAWS} to judge a bar")
    if float(np.ptp(a)) == 0.0:
        raise PowerError("null has zero variance; no bar can be judged against a point mass")
    if not 0.0 < alpha < 1.0 or not 0.0 < target_power < 1.0:
        raise PowerError("alpha and target_power must lie strictly in (0,1)")

    reasons: list[str] = []
    detail: dict = {}
    upper = op in (">=", ">")

    # -- ceiling and spacing, for a bounded discrete statistic -------------------------------
    if n_items is not None:
        b = bounded_ceiling(n_items)
        detail["bounded"] = b
        if upper and bar > b["ceiling"]:
            reasons.append(f"bar {bar} exceeds the attainable ceiling {b['ceiling']} for a "
                           f"proportion over {n_items} trials; no effect size can reach it")
        if not upper and bar < b["floor"]:
            reasons.append(f"bar {bar} is below the attainable floor {b['floor']}")
        scaled = bar * n_items
        if abs(scaled - round(scaled)) > 1e-9 and abs(bar) <= 1.0:
            reasons.append(f"bar {bar} falls between attainable values spaced {b['spacing']} "
                           f"apart; the effective bar is not the one written")

    # -- alpha, corrected for the order statistic the gate actually reads --------------------
    fpr_single = float(_OPS[op](a, bar).mean())
    fpr = 1.0 - (1.0 - fpr_single) ** n_draws if n_draws > 1 else fpr_single
    detail["false_positive_rate"] = round(fpr, 6)
    detail["false_positive_rate_single_draw"] = round(fpr_single, 6)
    detail["n_draws"] = n_draws
    if fpr > alpha:
        if n_draws > 1:
            corr = order_stat_bar(a, n_draws, alpha, op=op)
            detail["order_stat"] = corr
            reasons.append(
                f"bar concedes {fpr:.4f} across the max of {n_draws} draws against alpha={alpha}; "
                f"a single-draw bar was written where an order statistic is read "
                f"(corrected bar {corr['corrected_bar']})")
        else:
            reasons.append(f"bar concedes a false-positive rate of {fpr:.4f} against alpha={alpha}")

    # -- power, only computable with an alternative ------------------------------------------
    power = None
    if alt is not None:
        alt_a = _clean(alt, "alt")
        if alt_a.size < MIN_DRAWS:
            raise PowerError(f"alt has {alt_a.size} draws; need at least {MIN_DRAWS}")
        power = round(float(_OPS[op](alt_a, bar).mean()), 6)
        detail["power"] = power
        if power < target_power:
            mdb = min_detectable_bar(a, alpha, op, n_draws=n_draws)
            detail["min_detectable_bar"] = mdb
            reasons.append(
                f"power is {power:.4f} against a target of {target_power}; the apparatus clears "
                f"this bar too rarely for the gate to be informative "
                f"(most permissive alpha-controlling bar is {mdb})")
    else:
        reasons.append("no alternative distribution supplied, so power is unknown; a bar that "
                       "nothing can clear controls alpha perfectly and is still useless")

    ok = not reasons
    return Reach(reachable=ok,
                 verdict="REACHABLE" if ok else "UNREACHABLE__" + "; ".join(reasons),
                 alpha=round(fpr, 6), power=power, bar=bar, op=op,
                 reasons=reasons, detail=detail)
