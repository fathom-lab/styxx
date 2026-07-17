"""styxx.calibration -- transfer-safe threshold calibration for detectors and batteries.

A deployment threshold is a promise: "fire on at most `target_fpr` of the target-absent
population." An instrument keeps that promise only if the threshold was calibrated on scores that
are EXCHANGEABLE with the ones it will meet in deployment. The retro-cert arc under
``papers/read-neq-write/`` proved -- on a real honesty probe -- that the exchangeability premise is
the whole game, and that it silently fails in one specific, common way:

  * ``retro_certify_private13.py``            -- a mount-style point-estimate threshold at
    target FPR 0.20, calibrated on CALIB negatives, fired on 0.548 of held-out EVAL negatives
    -> VOID_INSTRUMENT__nonspecific (174% past its operating point).
  * ``retro_certify_private13_conformal.py``  -- split-conformal and beta-tolerance corrections on
    the SAME in-sample scores STILL fired 0.48-0.52. The gap is NOT a quantile-derivation artifact.
  * ``retro_certify_private13_threeway.py``   -- fitting the probe on HALF of CALIB and
    conformal-calibrating the threshold on the OTHER half (fit-disjoint) transferred correctly:
    EVAL FPR 0.161, ADMISSIBLE.

ROOT CAUSE, stated once: a threshold calibrated on scores the probe was FIT on is not exchangeable
with held-out scores -- the fit split's decision values are in-sample and optimistic. Conformal /
tolerance guarantees are distribution-free, but they are NOT fit-optimism-free: they assume the
calibration draw and a future negative are exchangeable, which in-sample fit scores are not. The
repair is a THREE-WAY protocol -- fit / threshold-calibrate / deploy on three DISJOINT splits.

This module bakes that lesson into a reusable primitive. ``conformal_threshold`` gives the
split-conformal order-statistic threshold (Vovk quantile with the finite-sample ``(n+1)``
correction) or the stronger beta-tolerance variant; both state their guarantee CONDITIONALLY on
exchangeability. ``calibrate_transfer_safe`` is the guarded door: it will not hand out a guarantee
it cannot keep -- given evidence that the calibration negatives came from the probe's own fit split
(an explicit split id, or an overlap check that finds calib scores literally inside the fit set), it
REFUSES to certify and returns a loud ``transfer_valid=False`` pointing at the three-way protocol.

The conformal math is ported verbatim from the three retro-cert scripts (the exact order-statistic
rule and scipy-free incomplete beta), generalized to both fire directions. CPU-only: numpy + stdlib.

    from styxx.calibration import conformal_threshold, calibrate_transfer_safe

    # a detector (fires HIGH). calib_neg = target-absent scores from a fit-DISJOINT split.
    ct = conformal_threshold(calib_neg, target_fpr=0.05, direction="higher_fires")
    fired = ct.fires(deployment_scores)          # tau with a marginal FPR guarantee

    # the guarded door: refuses if the calib scores are the probe's own fit values
    ct = calibrate_transfer_safe(calib_neg, target_fpr=0.05, fit_scores=probe_fit_scores)
    if not ct.transfer_valid:
        print(ct.transfer_note)                  # -> use the three-way protocol
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "ConformalThreshold",
    "conformal_threshold",
    "calibrate_transfer_safe",
]

# The three-way repair, named once and reused in every refusal message.
_THREEWAY = ("fit / threshold-calibrate / deploy on three DISJOINT splits (the three-way protocol; "
             "papers/read-neq-write/retro_certify_private13_threeway.py)")


# --------------------------------------------------------------------------------------------
# direction: mirror styxx.admissibility's expect= convention
# --------------------------------------------------------------------------------------------

def _higher_fires(direction: str) -> bool:
    """Normalize the fire direction to a bool. Accepts this module's ``higher_fires``/``lower_fires``
    and, as aliases, styxx.admissibility's ``higher_on_positive``/``lower_on_positive`` so a detector
    calibrated here reads the same way it certifies there.

    higher-fires -- a DETECTOR: it fires when the score exceeds tau (few false alarms at HIGH tau).
    lower-fires  -- a capability BATTERY: it fires when the score falls below tau (few at LOW tau).
    """
    d = str(direction)
    if d in ("higher_fires", "higher_on_positive"):
        return True
    if d in ("lower_fires", "lower_on_positive"):
        return False
    raise ValueError("direction must be 'higher_fires'/'lower_fires' (or the admissibility aliases "
                     f"'higher_on_positive'/'lower_on_positive'); got {direction!r}")


# --------------------------------------------------------------------------------------------
# the conformal math -- ported verbatim from papers/read-neq-write/retro_certify_private13*.py,
# generalized to both directions by reflection (a lower-fires battery is a higher-fires detector on
# negated scores; the "never fire" sentinel reflects +inf -> -inf).
# --------------------------------------------------------------------------------------------

def _split_conformal_higher(neg: np.ndarray, alpha: float):
    """Split-conformal (Vovk) quantile for a HIGHER-fires detector at target FPR alpha.

    tau = ceil((n+1)*(1-alpha))-th SMALLEST calibration-negative score. Under exchangeability of the
    n calibration negatives with a future negative, P(score_new > tau) <= alpha exactly (marginal
    over the calibration draw). The (n+1) is the finite-sample correction the naive empirical
    quantile lacks. If ceil((n+1)(1-alpha)) > n the guarantee needs tau above every calibration
    point -> +inf (never fire). Returns (tau, rank, n)."""
    s = np.sort(np.asarray(neg, dtype=float))   # ascending
    n = len(s)
    if n == 0:
        return float("nan"), 0, 0
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf"), k, n
    return float(s[k - 1]), k, n


def _incomplete_beta_int(a_shape: int, b_shape: int, alpha: float) -> float:
    """Regularized incomplete beta I_alpha(a, b) for POSITIVE INTEGER a,b via the exact binomial
    identity I_x(a,b) = sum_{j=a}^{a+b-1} C(a+b-1, j) x^j (1-x)^(a+b-1-j), n = a+b-1. Scipy-free;
    cross-checked against scipy.stats.beta.cdf and a numpy binomial sum in the tests."""
    n = a_shape + b_shape - 1
    x = float(alpha)
    return float(sum(math.comb(n, j) * (x ** j) * ((1.0 - x) ** (n - j))
                     for j in range(a_shape, n + 1)))


def _beta_tolerance_higher(neg: np.ndarray, alpha: float, delta: float):
    """(alpha, delta) one-sided upper tolerance bound via order statistics, HIGHER-fires.

    tau = X_(r), the r-th smallest calib-negative, with r the SMALLEST rank whose order-statistic
    coverage clears confidence 1-delta: the population fraction above X_(r) is Beta(n-r+1, r), so
    confidence(true tail <= alpha) = I_alpha(n-r+1, r). Strictly more conservative than split
    (r >= the split rank). If no r<=n reaches 1-delta the tolerance is unattainable at this n -> +inf.
    Returns (tau, rank_or_None, n, confidence)."""
    s = np.sort(np.asarray(neg, dtype=float))
    n = len(s)
    if n == 0:
        return float("nan"), None, 0, float("nan")
    for r in range(1, n + 1):
        conf = _incomplete_beta_int(n - r + 1, r, alpha)
        if conf >= 1.0 - delta:
            return float(s[r - 1]), r, n, conf
    return float("inf"), None, n, float(_incomplete_beta_int(1, n, alpha))


def _reflect_tau(tau: float) -> float:
    """Map a higher-fires tau computed on negated scores back to the original axis. A finite tau
    negates; the higher-fires never-fire sentinel +inf reflects to the lower-fires sentinel -inf."""
    if math.isinf(tau):
        return float("-inf")
    return -float(tau)


def _fires(scores, tau: float, higher_fires: bool) -> np.ndarray:
    """Boolean fire mask at tau for the given direction. inf/-inf sentinels never fire."""
    s = np.asarray(scores, dtype=float).ravel()
    if s.size == 0:
        return np.zeros(0, dtype=bool)
    return (s > tau) if higher_fires else (s < tau)


def _realized_fpr(neg, tau: float, higher_fires: bool) -> float:
    s = np.asarray(neg, dtype=float).ravel()
    if s.size == 0:
        return float("nan")
    return float(np.mean(_fires(s, tau, higher_fires)))


# --------------------------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------------------------

@dataclass
class ConformalThreshold:
    """A calibrated firing threshold with an explicit, conditional transfer guarantee.

    Fields:
      tau                 -- the firing threshold. A detector fires when score > tau; a battery when
                             score < tau. The never-fire sentinel is +inf (detector) / -inf (battery).
      target_fpr          -- the false-alarm rate the threshold targets.
      n_calib             -- number of calibration negatives it was derived from.
      correction          -- 'split' (Vovk marginal quantile) or 'beta' (finite-sample tolerance).
      direction           -- 'higher_fires' (detector) or 'lower_fires' (battery).
      realized_calib_fpr  -- in-sample fire-rate at tau (a lower bound on deployment FPR; NOT the
                             guarantee -- see `guarantee`).
      rank                -- the order-statistic rank used (None when tau is the never-fire sentinel).
      delta, confidence   -- for correction='beta': the tolerance miss-rate and achieved confidence.
      guarantee           -- human-readable statement of what tau guarantees, and under what premise.
      transfer_valid      -- None  = exchangeability premise UNASSERTED (default; guarantee is stated
                                     conditionally),
                             True  = calibration negatives asserted / verified fit-disjoint,
                             False = REFUSED -- fit-contaminated; the guarantee cannot be kept.
      transfer_note       -- explanation of the transfer_valid state (the refusal message points at
                             the three-way protocol).
    """
    tau: float
    target_fpr: float
    n_calib: int
    correction: str
    direction: str
    realized_calib_fpr: float
    rank: Optional[int] = None
    delta: Optional[float] = None
    confidence: Optional[float] = None
    guarantee: str = ""
    transfer_valid: Optional[bool] = None
    transfer_note: str = ""
    notes: list = field(default_factory=list)

    @property
    def never_fires(self) -> bool:
        return math.isinf(self.tau)

    @property
    def higher_fires(self) -> bool:
        return _higher_fires(self.direction)

    def fires(self, scores) -> np.ndarray:
        """Boolean fire mask for a score vector at this threshold and direction."""
        return _fires(scores, self.tau, self.higher_fires)

    def realized_fpr(self, negatives) -> float:
        """Realized false-alarm rate on a held-out negative population (the honest transfer check)."""
        return _realized_fpr(negatives, self.tau, self.higher_fires)

    def as_dict(self) -> dict:
        tau = "inf" if self.tau == float("inf") else ("-inf" if self.tau == float("-inf")
                                                      else round(self.tau, 6))
        return {
            "tau": tau,
            "target_fpr": self.target_fpr,
            "n_calib": self.n_calib,
            "correction": self.correction,
            "direction": self.direction,
            "realized_calib_fpr": (round(self.realized_calib_fpr, 6)
                                   if math.isfinite(self.realized_calib_fpr) else self.realized_calib_fpr),
            "rank": self.rank,
            "delta": self.delta,
            "confidence": (round(self.confidence, 6)
                           if self.confidence is not None and math.isfinite(self.confidence)
                           else self.confidence),
            "guarantee": self.guarantee,
            "transfer_valid": self.transfer_valid,
            "transfer_note": self.transfer_note,
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        tau_s = ("+inf (never fires)" if self.tau == float("inf")
                 else "-inf (never fires)" if self.tau == float("-inf")
                 else f"{self.tau:.6f}")
        fire_word = "score > tau" if self.higher_fires else "score < tau"
        if self.transfer_valid is True:
            xfer = "transfer: fit-disjoint (guarantee holds)"
        elif self.transfer_valid is False:
            xfer = "transfer: REFUSED (fit-contaminated -- guarantee VOID)"
        else:
            xfer = "transfer: premise UNASSERTED (guarantee conditional on exchangeability)"
        L = [
            f"CONFORMAL THRESHOLD ({self.correction}, {self.direction}): tau = {tau_s}  [fires when {fire_word}]",
            f"  target FPR {self.target_fpr:.3f}   realized calib FPR {self.realized_calib_fpr:.3f}   "
            f"n_calib {self.n_calib}   rank {self.rank}",
        ]
        if self.correction == "beta" and self.confidence is not None:
            L.append(f"  tolerance: confidence {self.confidence:.4f} >= 1-delta ({1.0 - (self.delta or 0):.4f})")
        L.append(f"  guarantee: {self.guarantee}")
        L.append(f"  {xfer}")
        if self.transfer_note:
            L.append(f"  note: {self.transfer_note}")
        for n in self.notes:
            L.append(f"  note: {n}")
        return "\n".join(L)


# --------------------------------------------------------------------------------------------
# the primitive
# --------------------------------------------------------------------------------------------

def conformal_threshold(calib_neg_scores: Sequence[float], *, target_fpr: float = 0.05,
                        direction: str = "higher_fires", correction: str = "split",
                        delta: float = 0.1,
                        fit_disjoint: Optional[bool] = None) -> ConformalThreshold:
    """Distribution-free, finite-sample firing threshold from a calibration NEGATIVE population.

    calib_neg_scores -- scores of KNOWN target-ABSENT units (the population that must stay quiet).
    target_fpr       -- the false-alarm rate to hold (alpha).
    direction        -- 'higher_fires' (a detector; fires when score > tau) or 'lower_fires' (a
                        capability battery; fires when score < tau). Admissibility's
                        'higher_on_positive'/'lower_on_positive' are accepted as aliases.
    correction       -- 'split': the Vovk split-conformal order-statistic quantile with the marginal
                        guarantee P(FPR <= target_fpr) under exchangeability.
                        'beta': the stronger (alpha, delta) one-sided distribution-free tolerance
                        bound -- "with confidence >= 1-delta the TRUE exceedance probability <=
                        target_fpr" -- strictly more conservative (lower realized FPR).
    delta            -- tolerance miss-rate for correction='beta' (ignored for 'split').
    fit_disjoint     -- the exchangeability premise, stated honestly:
                          None  (default) -- UNASSERTED. tau is computed and the guarantee is stated
                                  CONDITIONALLY ("holds iff the calibration negatives are exchangeable
                                  with deployment negatives -- i.e. fit-disjoint"). Zero-surprise
                                  default: it computes a threshold, it does not adjudicate provenance.
                          True  -- the caller asserts the calibration negatives are disjoint from the
                                  probe's fit split; the guarantee is certified (transfer_valid=True).
                          False -- the caller declares the calibration negatives ARE the probe's own
                                  fit values; the primitive REFUSES to certify (transfer_valid=False)
                                  and points at the three-way protocol. tau is still returned for
                                  description, but flagged VOID.

    Returns a ConformalThreshold. Use ``calibrate_transfer_safe`` when you want the primitive to
    DETECT contamination (overlap check / split-id check) rather than trust a caller flag.
    """
    if correction not in ("split", "beta"):
        raise ValueError(f"correction must be 'split' or 'beta'; got {correction!r}")
    if not (0.0 < float(target_fpr) < 1.0):
        raise ValueError(f"target_fpr must be in (0, 1); got {target_fpr}")
    higher = _higher_fires(direction)
    dir_norm = "higher_fires" if higher else "lower_fires"

    neg = np.asarray(calib_neg_scores, dtype=float).ravel()
    # reflect a lower-fires battery into a higher-fires detector on negated scores
    work = neg if higher else -neg

    if correction == "split":
        tau_w, rank, n = _split_conformal_higher(work, float(target_fpr))
        conf = None
        delta_out = None
        guarantee = (f"split-conformal (Vovk): P(deployment FPR <= {target_fpr:g}) marginal over the "
                     f"calibration draw, via the ceil((n+1)(1-alpha))={rank}-th order statistic of "
                     f"n={n} calibration negatives -- distribution-free, finite-sample")
    else:
        tau_w, rank, n, conf = _beta_tolerance_higher(work, float(target_fpr), float(delta))
        delta_out = float(delta)
        if rank is not None:
            guarantee = (f"beta-tolerance: with confidence >= {1.0 - float(delta):g} the TRUE "
                         f"false-alarm probability <= {target_fpr:g}, via the {rank}-th order "
                         f"statistic of n={n} calibration negatives -- one-sided distribution-free "
                         f"tolerance interval")
        else:
            guarantee = (f"beta-tolerance UNATTAINABLE at n={n} for (alpha={target_fpr:g}, "
                         f"delta={float(delta):g}) -> tau=+inf (never fire); need more calibration "
                         f"negatives for this (alpha, delta)")

    tau = float(tau_w) if higher else _reflect_tau(tau_w)
    realized = _realized_fpr(neg, tau, higher)

    ct = ConformalThreshold(
        tau=tau, target_fpr=float(target_fpr), n_calib=int(n), correction=correction,
        direction=dir_norm, realized_calib_fpr=realized, rank=(None if rank in (None, 0) else int(rank)),
        delta=delta_out, confidence=conf, guarantee=guarantee,
    )

    # ---- the transfer premise ----
    _apply_transfer_state(ct, fit_disjoint, reason=None)
    return ct


def _apply_transfer_state(ct: ConformalThreshold, fit_disjoint: Optional[bool], reason: Optional[str]):
    """Stamp the transfer_valid tri-state + note onto a ConformalThreshold, and VOID the guarantee
    string when contamination makes it unkeepable. Centralized so conformal_threshold and
    calibrate_transfer_safe speak with one voice."""
    if fit_disjoint is True:
        ct.transfer_valid = True
        ct.transfer_note = (reason or "calibration negatives asserted fit-disjoint; the finite-sample "
                            "guarantee is certified.")
    elif fit_disjoint is False:
        ct.transfer_valid = False
        why = reason or "calibration negatives came from the probe's own fit split"
        ct.transfer_note = (
            f"REFUSED -- no transfer guarantee: {why}. A threshold calibrated on scores the probe was "
            f"FIT on is NOT exchangeable with held-out scores (in-sample fit optimism, not a quantile "
            f"artifact -- papers/read-neq-write retro-cert arc). Repair: {_THREEWAY}.")
        ct.guarantee = "VOID (fit-contaminated): " + ct.guarantee
    else:
        ct.transfer_valid = None
        ct.transfer_note = (
            "exchangeability premise UNASSERTED: the guarantee holds ONLY IF these calibration "
            "negatives are exchangeable with deployment negatives -- i.e. drawn from a split the probe "
            f"was NOT fit on. If they are the probe's own fit values the guarantee is void; use "
            f"calibrate_transfer_safe to have that checked, or the three-way protocol to guarantee it: {_THREEWAY}.")
    return ct


# --------------------------------------------------------------------------------------------
# the guarded door: won't hand out a guarantee it can't keep
# --------------------------------------------------------------------------------------------

def calibrate_transfer_safe(calib_neg_scores: Sequence[float], *, target_fpr: float = 0.05,
                            direction: str = "higher_fires", correction: str = "split",
                            delta: float = 0.1,
                            fit_scores: Optional[Sequence[float]] = None,
                            fit_split_id: Optional[str] = None,
                            calib_split_id: Optional[str] = None,
                            overlap_atol: float = 1e-9) -> ConformalThreshold:
    """Conformal threshold that DETECTS fit-contamination and refuses to certify what it can't keep.

    This is the transfer lesson of the retro-cert arc made a first-class guard. It computes the same
    threshold as ``conformal_threshold`` but decides ``fit_disjoint`` from EVIDENCE rather than a
    caller flag, and REFUSES (transfer_valid=False) unless disjointness is positively established:

      * ``fit_scores``   -- the probe's own fit-split decision values. An OVERLAP CHECK looks for any
                            calibration score that appears inside the fit set (within ``overlap_atol``).
                            Any overlap == in-sample calibration == contamination -> REFUSE. This is the
                            literal failure the receipts showed: the CALIB negatives WERE the probe's
                            fit-split decision-function values.
      * ``fit_split_id`` + ``calib_split_id`` -- provenance tags. Equal ids == same split -> REFUSE.
                            Distinct ids == positive evidence of disjointness -> certify.

    If NO evidence is supplied, the guard cannot verify disjointness, so it does NOT certify a
    guarantee it can't keep: transfer_valid=False with a note asking for fit_scores or split ids (and
    pointing at the three-way protocol). Contrast ``conformal_threshold(fit_disjoint=None)``, which
    computes the threshold with the premise left honestly UNASSERTED.
    """
    neg = np.asarray(calib_neg_scores, dtype=float).ravel()

    contaminated = False
    disjoint_evidence = False
    reasons: list = []

    # (1) overlap check -- are the calib scores literally inside the fit set?
    overlap_n = 0
    if fit_scores is not None:
        fit = np.asarray(fit_scores, dtype=float).ravel()
        if fit.size and neg.size:
            # a calib score is "in the fit set" if some fit score is within atol of it
            diffs = np.abs(neg[:, None] - fit[None, :])
            overlap_n = int(np.any(diffs <= float(overlap_atol), axis=1).sum())
        if overlap_n > 0:
            contaminated = True
            reasons.append(f"{overlap_n}/{neg.size} calibration scores are inside the probe's fit set "
                           f"(in-sample decision values, atol={overlap_atol:g})")
        else:
            disjoint_evidence = True
            reasons.append(f"overlap check clean: 0/{neg.size} calibration scores found in the "
                           f"fit set (atol={overlap_atol:g})")

    # (2) split-id check -- same provenance tag == same split
    if fit_split_id is not None and calib_split_id is not None:
        if str(fit_split_id) == str(calib_split_id):
            contaminated = True
            reasons.append(f"calibration split id == fit split id ({calib_split_id!r})")
        else:
            disjoint_evidence = True
            reasons.append(f"split ids differ (fit={fit_split_id!r}, calib={calib_split_id!r})")

    if contaminated:
        fit_disjoint: Optional[bool] = False
        reason = "; ".join(reasons)
    elif disjoint_evidence:
        fit_disjoint = True
        reason = "; ".join(reasons)
    else:
        # no evidence either way: the transfer-safe door will NOT certify a guarantee it can't keep
        fit_disjoint = False
        reason = ("disjointness could not be verified -- pass fit_scores (for an overlap check) or "
                  "both fit_split_id and calib_split_id (for a provenance check)")

    ct = conformal_threshold(neg, target_fpr=target_fpr, direction=direction,
                             correction=correction, delta=delta, fit_disjoint=None)
    _apply_transfer_state(ct, fit_disjoint, reason=reason)
    return ct
