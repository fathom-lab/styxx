"""styxx.admissibility -- two-sided instrument admissibility.

The one-sided sibling of this is `styxx.probe_validity.validate_probe` (is a probe tracking the
CONCEPT or a surface artifact?). This module answers the OTHER question every interpretability
instrument owes its user: is it valid in BOTH directions on its OWN score?

  SPECIFIC   -- it stays QUIET on a null / target-absent population (low false-alarm rate).
  SENSITIVE  -- it FIRES on a positive / target-present-or-destroyed population (real separation,
                in the DIRECTION a working instrument should show).

A capability battery that scores a knowledge-erased model LOWER is only admissible if it ALSO
stays quiet on intact controls; a deception detector that fires on harmful statements is only
admissible if it ALSO stays quiet on benign ones. An instrument that is sensitive but not specific
cries wolf; one that is specific but not sensitive is asleep; a sign-FLIPPED one has high
discriminability but reads the world backwards. This primitive certifies against all three, on the
instrument's own score, with a permutation null and a self-verifying certificate.

General by construction -- bring ANY instrument, ANY population:

    from styxx.admissibility import instrument_admissibility
    rep = instrument_admissibility(
        score=my_instrument,            # callable(list[input]) -> np.ndarray of scores
        positive=erased_prompts,        # target-present-or-DESTROYED units (label 1)
        null=intact_prompts,            # target-ABSENT / control units (label 0)
        expect="lower_on_positive",     # a good capability battery scores the destroyed class LOWER
    )
    print(rep.summary())
    cert = rep.certificate()            # groundable, re-verifiable from its own points

Verdict logic (precedence: unmeasurable -> insensitive -> nonspecific -> ADMISSIBLE):
  1. MEASURABLE  -- n_positive >= 2, n_null >= 2, scores non-degenerate; else VOID (unmeasurable).
  2. SENSITIVE   -- discriminability (max(AUROC,1-AUROC)) >= auroc_floor AND a direction-agnostic
                    permutation p < alpha AND the positive class actually falls on the `expect`
                    side (the direction check that catches a sign-flipped instrument).
  3. SPECIFIC    -- the false-alarm rate on the NULL population, at the firing threshold, <= max_fire.

Reuses the frozen crossmind primitives verbatim (`discrim`, `permutation_null`) -- do not re-derive
the math. CPU-only: numpy + crossmind + stdlib. No sklearn, no torch.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .crossmind import auroc, discrim, permutation_null  # tie-aware AUROC, max(AUROC,1-AUROC), label-perm null

__all__ = [
    "instrument_admissibility", "AdmissibilityReport",
    "certificate", "verify_admissibility_certificate", "slope_permutation_null",
]


# --------------------------------------------------------------------------------------------
# small pure helpers (CPU, no scipy)
# --------------------------------------------------------------------------------------------

def _sha256(p: Path) -> str:
    # copied verbatim from styxx.ladder._sha256 -- same receipt-hashing contract
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _hash_receipts(receipts, root: Path | str = ".") -> dict:
    """Normalize the `receipts` argument into a {relative_path: sha256} map.

    Accepts: None -> {}; a dict {path: sha} used verbatim (already the map ladder-style); or an
    iterable of paths, each hashed relative to `root`. Absolute paths hash as given."""
    if receipts is None:
        return {}
    if isinstance(receipts, dict):
        return dict(receipts)
    out: dict = {}
    for rel in receipts:
        out[str(rel)] = _sha256(Path(root) / rel)
    return out


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation; |abs err| < 1.2e-9).

    Local so the module stays scipy-free. Only used for the (approximate) minimum-detectable-effect."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _mde_auroc(n_pos: int, n_null: int, alpha: float) -> float:
    """APPROXIMATE minimum detectable effect: the smallest AUROC separation ABOVE 0.5 this n has
    ~80% power to flag as significant at `alpha` (one-tailed, since the test orients to the data).

    Normal approximation using the Mann-Whitney null SE of AUROC,
        SE0 = sqrt((n_pos + n_null + 1) / (12 * n_pos * n_null)),
    and MDE = (z_{1-alpha} + z_{0.80}) * SE0, clamped to [0, 0.5]. Honest and conservative (the true
    SE under H1 is smaller), labelled approximate; a power analysis, not a guarantee."""
    if n_pos < 1 or n_null < 1:
        return float("nan")
    se0 = math.sqrt((n_pos + n_null + 1) / (12.0 * n_pos * n_null))
    z_alpha = _norm_ppf(1.0 - alpha)
    z_power = _norm_ppf(0.80)
    return float(min(0.5, (z_alpha + z_power) * se0))


def _sensitivity_p(scores: np.ndarray, labels: np.ndarray, seed: int, k_perm: int) -> float:
    """Direction-AGNOSTIC permutation p-value for the separation.

    `crossmind.permutation_null` is ONE-TAILED UPPER on the raw AUROC (p = (1+#{null>=obs})/(1+k)),
    so it only tests "positive scores HIGHER". A working instrument may point either way, so we run
    the same one-tailed test on the scores AND on their negation, and take the SMALLER p -- i.e. the
    one-tailed test in whichever direction the data actually points. (Equivalent to orienting the
    scores to the observed effect before the upper-tailed test.)"""
    s = np.asarray(scores, dtype=float)
    p_hi = permutation_null(s, labels, seed=seed, k_perm=k_perm)["p_value"]
    p_lo = permutation_null(-s, labels, seed=seed, k_perm=k_perm)["p_value"]
    return float(min(p_hi, p_lo))


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------

@dataclass
class AdmissibilityReport:
    """Two-sided admissibility verdict for one instrument on one (positive, null) population pair.

    Mirrors styxx.probe_validity.ProbeValidityReport ergonomics: a flat dataclass with `.as_dict()`
    and a human-readable `.summary()`, plus `.certificate()` for a groundable, re-verifiable record."""
    discrim: float
    sensitivity_p: float
    direction_ok: bool
    fire_rate: float
    fire_threshold: float
    min_detectable_effect: float
    n_positive: int
    n_null: int
    sensitive: bool
    specific: bool
    measurable: bool
    admissible: bool
    admissibility_verdict: str
    notes: list = field(default_factory=list)
    # carried for the certificate / recompute-verify (NOT part of the headline as_dict)
    scores: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    expect: str = "lower_on_positive"
    thresholds: dict = field(default_factory=dict)
    instrument: Optional[str] = None
    receipts: Any = None

    _HEADLINE = ("discrim", "sensitivity_p", "direction_ok", "fire_rate", "fire_threshold",
                 "min_detectable_effect", "n_positive", "n_null", "sensitive", "specific",
                 "measurable", "admissible", "admissibility_verdict", "notes")

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._HEADLINE}

    def summary(self) -> str:
        if self.expect == "lower_on_positive":
            dir_word = "positive lower" if self.direction_ok else "positive HIGHER -- INVERTED"
        else:
            dir_word = "positive higher" if self.direction_ok else "positive LOWER -- INVERTED"
        floor = self.thresholds.get("auroc_floor", 0.70)
        alpha = self.thresholds.get("alpha", 0.05)
        max_fire = self.thresholds.get("max_fire", 0.15)
        L = [
            f"INSTRUMENT ADMISSIBILITY: {self.admissibility_verdict}",
            f"  sensitivity (discrim, direction-agnostic): {self.discrim:.3f}   "
            f"[floor {floor:.2f} -> {'>= floor' if self.discrim >= floor else 'BELOW floor'}]",
            f"    permutation p (oriented, min of +/-):    {self.sensitivity_p:.3f}   "
            f"[{'significant' if self.sensitivity_p < alpha else 'NOT above a random split'}]",
            f"    direction on positive class:             [{dir_word}]",
            f"  specificity (fires on null):               fire_rate {self.fire_rate:.3f} "
            f"@ threshold {self.fire_threshold:.3f}   "
            f"[max_fire {max_fire:.2f} -> {'specific' if self.specific else 'NONSPECIFIC'}]",
            f"  measurable: {'yes' if self.measurable else 'NO'}  "
            f"(n_positive={self.n_positive}, n_null={self.n_null})",
            f"  min detectable effect (AUROC sep, ~80% power @ alpha, approx): "
            f"{self.min_detectable_effect:.3f}",
        ]
        L += [f"  note: {n}" for n in self.notes]
        return "\n".join(L)

    def certificate(self, receipts=None, out_path=None) -> dict:
        return certificate(self, receipts=receipts, out_path=out_path)


# --------------------------------------------------------------------------------------------
# the primitive
# --------------------------------------------------------------------------------------------

def instrument_admissibility(*, scores: Optional[Sequence[float]] = None,
                             labels: Optional[Sequence[int]] = None,
                             score: Optional[Callable] = None,
                             positive: Optional[Sequence] = None,
                             null: Optional[Sequence] = None,
                             expect: str = "lower_on_positive",
                             fire_threshold: Optional[float] = None,
                             auroc_floor: float = 0.70, alpha: float = 0.05, max_fire: float = 0.15,
                             k_perm: int = 1000, seed: int = 0,
                             receipts: Any = None,
                             out_path: Optional[str] = None) -> AdmissibilityReport:
    """Certify that an interpretability instrument is admissible in BOTH directions on its own score.

    Two ways to supply data:
      * precomputed  -- `scores` (1-D) + `labels` (1 = positive/target-present-or-destroyed,
                        0 = null/target-absent); OR
      * on the fly   -- `score` callable mapping a list of inputs -> np.ndarray, evaluated on
                        `positive` rows (label 1) and `null` rows (label 0).

    `expect` -- the direction a WORKING instrument shows on the positive class:
      "lower_on_positive"  (default) a good capability battery scores the destroyed class LOWER;
      "higher_on_positive"          a good detector scores the target-present class HIGHER.

    Returns an AdmissibilityReport. If `out_path` is given, also writes the certificate there."""
    if expect not in ("lower_on_positive", "higher_on_positive"):
        raise ValueError(f"expect must be 'lower_on_positive' or 'higher_on_positive'; got {expect!r}")

    # ---- assemble scores/labels from whichever input mode was used ----
    if scores is None:
        if score is None or positive is None or null is None:
            raise ValueError("supply either (scores, labels) or (score, positive, null)")
        pos_s = np.asarray(score(list(positive)), dtype=float).ravel()
        null_s = np.asarray(score(list(null)), dtype=float).ravel()
        scores_a = np.concatenate([pos_s, null_s])
        labels_a = np.concatenate([np.ones(len(pos_s), dtype=int), np.zeros(len(null_s), dtype=int)])
    else:
        if labels is None:
            raise ValueError("`labels` is required when `scores` is given")
        scores_a = np.asarray(scores, dtype=float).ravel()
        labels_a = np.asarray(labels, dtype=int).ravel()
        if scores_a.shape[0] != labels_a.shape[0]:
            raise ValueError(f"scores ({scores_a.shape[0]}) and labels ({labels_a.shape[0]}) length mismatch")
    bad = set(np.unique(labels_a).tolist()) - {0, 1}
    if bad:
        raise ValueError(f"labels must be 0/1 (1=positive, 0=null); saw {sorted(bad)}")

    n_pos = int((labels_a == 1).sum())
    n_null = int((labels_a == 0).sum())
    non_degenerate = int(np.unique(scores_a).size) >= 2
    measurable = n_pos >= 2 and n_null >= 2 and non_degenerate
    both = n_pos >= 1 and n_null >= 1

    notes: list = []

    if both:
        # ---- SENSITIVITY: magnitude (direction-agnostic) + significance + the direction check ----
        d = float(discrim(scores_a, labels_a))                       # max(AUROC, 1-AUROC)
        sens_p = _sensitivity_p(scores_a, labels_a, seed, k_perm)    # direction-agnostic perm p
        pos_mean = float(scores_a[labels_a == 1].mean())
        null_mean = float(scores_a[labels_a == 0].mean())
        # DIRECTION CHECK -- confirm the positive class falls on the `expect` side. A sign-flipped
        # instrument has high `discrim` but reads the world backwards; it must NOT pass sensitivity.
        direction_ok = (pos_mean < null_mean) if expect == "lower_on_positive" else (pos_mean > null_mean)

        # ---- SPECIFICITY: false-alarm rate on the NULL population at the firing threshold ----
        null_scores = scores_a[labels_a == 0]
        fire_high = (expect == "higher_on_positive")   # a detector fires HIGH; a battery fires LOW
        if fire_threshold is None:
            # auto-derive a self-referential 5%-false-positive floor from the null itself: the
            # percentile that leaves 5% of the null on the FIRING side. This makes the auto-threshold
            # specific by construction; pass a real deployment `fire_threshold` for a load-bearing test.
            ft = float(np.percentile(null_scores, 95.0 if fire_high else 5.0))
            derived = True
        else:
            ft = float(fire_threshold)
            derived = False
        fired = (null_scores > ft) if fire_high else (null_scores < ft)
        fire_rate = float(np.mean(fired))
    else:
        d = float("nan")
        sens_p = float("nan")
        direction_ok = False
        ft = float(fire_threshold) if fire_threshold is not None else float("nan")
        derived = fire_threshold is None
        fire_rate = float("nan")

    # nan comparisons are False, so a class-empty / degenerate instrument fails both flags safely
    sensitive = bool(d >= auroc_floor and sens_p < alpha and direction_ok)
    specific = bool(fire_rate <= max_fire)
    admissible = bool(measurable and sensitive and specific)

    if not measurable:
        verdict = "VOID_INSTRUMENT__unmeasurable"
    elif not sensitive:
        verdict = "VOID_INSTRUMENT__insensitive"
    elif not specific:
        verdict = "VOID_INSTRUMENT__nonspecific"
    else:
        verdict = "ADMISSIBLE"

    mde = _mde_auroc(n_pos, n_null, alpha)

    # ---- notes (honest, actionable) ----
    if not measurable:
        if n_pos < 2 or n_null < 2:
            notes.append(f"too few units per class (n_positive={n_pos}, n_null={n_null}); need >= 2 each.")
        if not non_degenerate:
            notes.append("scores are degenerate (a single value) -- no separation is measurable.")
    if both and (d >= auroc_floor) and not direction_ok:
        notes.append("high discriminability but the positive class is on the WRONG side of `expect` "
                     "-- a sign-flipped / inverted instrument; NOT sensitive.")
    if verdict == "VOID_INSTRUMENT__insensitive" and direction_ok:
        notes.append("separation is at/near chance or not significant against a permuted null.")
    if not specific and both:
        notes.append(f"fires on {fire_rate:.1%} of the NULL population (> max_fire {max_fire:.1%}) "
                     f"at threshold {ft:.4f} -- cries wolf.")
    if derived and both:
        side = "95th" if (expect == "higher_on_positive") else "5th"
        notes.append(f"fire_threshold auto-derived as the {side} percentile of the null (self-referential "
                     "5% false-positive floor); pass a deployment threshold for a load-bearing test.")
    if not admissible:
        notes.append(f"min detectable effect ~{mde:.3f} AUROC sep at this n (approx, ~80% power @ alpha={alpha}).")

    rep = AdmissibilityReport(
        discrim=(round(d, 6) if math.isfinite(d) else d),
        sensitivity_p=(round(sens_p, 6) if math.isfinite(sens_p) else sens_p),
        direction_ok=bool(direction_ok),
        fire_rate=(round(fire_rate, 6) if math.isfinite(fire_rate) else fire_rate),
        fire_threshold=(round(ft, 6) if math.isfinite(ft) else ft),
        min_detectable_effect=(round(mde, 6) if math.isfinite(mde) else mde),
        n_positive=n_pos, n_null=n_null,
        sensitive=sensitive, specific=specific, measurable=measurable,
        admissible=admissible, admissibility_verdict=verdict, notes=notes,
        scores=[float(x) for x in scores_a.tolist()],
        labels=[int(x) for x in labels_a.tolist()],
        expect=expect,
        thresholds={"auroc_floor": float(auroc_floor), "alpha": float(alpha),
                    "max_fire": float(max_fire), "k_perm": int(k_perm), "seed": int(seed)},
        instrument=None,
        receipts=receipts,
    )
    if out_path is not None:
        rep.certificate(receipts=receipts, out_path=out_path)
    return rep


# --------------------------------------------------------------------------------------------
# certificate + verify (stronger than ladder: RECOMPUTES the verdict from the stored points)
# --------------------------------------------------------------------------------------------

def certificate(report: AdmissibilityReport, receipts: Any = None, out_path=None) -> dict:
    """Emit a groundable admissibility certificate. Every headline number is recomputable from the
    `points` list (score+label per unit). Writes JSON to `out_path` when given."""
    receipts = receipts if receipts is not None else report.receipts
    th = report.thresholds
    cert: dict = {
        "what": "styxx two-sided instrument-admissibility certificate",
    }
    if report.instrument:
        cert["instrument"] = report.instrument
    cert.update({
        "sensitivity": {
            "discrim": report.discrim,
            "p_value": report.sensitivity_p,
            "direction_ok": report.direction_ok,
            "expect": report.expect,
            "auroc_floor": th.get("auroc_floor"),
            "alpha": th.get("alpha"),
            "sensitive": report.sensitive,
        },
        "specificity": {
            "fire_rate": report.fire_rate,
            "fire_threshold": report.fire_threshold,
            "max_fire": th.get("max_fire"),
            "specific": report.specific,
        },
        "points": [{"unit_index": i, "score": float(s), "label": int(l)}
                   for i, (s, l) in enumerate(zip(report.scores, report.labels))],
        "admissible": report.admissible,
        "admissibility_verdict": report.admissibility_verdict,
        "thresholds": dict(th),
        "reuses": {"discrim": "styxx.crossmind.discrim",
                   "permutation_null": "styxx.crossmind.permutation_null"},
        "min_detectable_effect": report.min_detectable_effect,
        "receipts_sha256": _hash_receipts(receipts),
        "issued_by": "styxx.admissibility.instrument_admissibility",
    })
    if out_path is not None:
        Path(out_path).write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    return cert


def verify_admissibility_certificate(cert: dict | str | Path, root: Path | str = ".") -> dict:
    """Verify an admissibility certificate two ways:

      1. RECEIPT INTEGRITY (ladder-style) -- re-hash every path in `receipts_sha256` against the live
         tree and confirm it still matches (`ok`, `checked`, `mismatches`, `missing`).
      2. FAITHFULNESS (stronger than ladder) -- RERUN the two tests on the certificate's own stored
         `points` at the recorded thresholds and confirm the recomputed `admissible` equals the stored
         flag (`recomputed_admissible`, `faithful`). This recompute is only possible because the tests
         are pure/CPU; it turns the certificate from an index into a self-checking artifact."""
    if isinstance(cert, (str, Path)):
        cert = json.loads(Path(cert).read_text(encoding="utf-8"))
    root = Path(root)

    # 1. receipt hashes
    recorded = cert.get("receipts_sha256", {})
    mismatches, missing, checked = [], [], 0
    for rel, sha in recorded.items():
        p = root / rel
        if not p.exists():
            missing.append(rel)
            continue
        checked += 1
        live = _sha256(p)
        if live != sha:
            mismatches.append({"receipt": rel, "recorded": sha, "live": live})

    # 2. recompute the verdict from the stored points at the recorded thresholds
    pts = cert.get("points", [])
    scores = [p["score"] for p in pts]
    labels = [p["label"] for p in pts]
    sens = cert.get("sensitivity", {})
    spec = cert.get("specificity", {})
    th = cert.get("thresholds", {})
    if scores:
        rep = instrument_admissibility(
            scores=scores, labels=labels,
            expect=sens.get("expect", "lower_on_positive"),
            fire_threshold=spec.get("fire_threshold"),   # the RESOLVED threshold -> deterministic match
            auroc_floor=sens.get("auroc_floor", 0.70),
            alpha=sens.get("alpha", 0.05),
            max_fire=spec.get("max_fire", 0.15),
            k_perm=th.get("k_perm", 1000), seed=th.get("seed", 0),
        )
        recomputed_admissible = bool(rep.admissible)
    else:
        recomputed_admissible = False
    faithful = bool(recomputed_admissible == bool(cert.get("admissible")))

    return {"ok": not mismatches and not missing, "checked": checked,
            "n_recorded": len(recorded), "mismatches": mismatches, "missing": missing,
            "recomputed_admissible": recomputed_admissible, "faithful": faithful}


# --------------------------------------------------------------------------------------------
# general stats primitive -- dose-response slope with a within-unit permutation null
# (exported for reuse; deliberately NOT called inside instrument_admissibility)
# --------------------------------------------------------------------------------------------

def slope_permutation_null(stat: Sequence[float], dose: Sequence[float], *,
                           unit: Optional[Sequence] = None, seed: int = 0,
                           k_perm: int = 1000) -> dict:
    """Permutation null for a dose-response SLOPE, regressed on the ACTUAL dose values.

    slope = np.polyfit(dose, stat, 1)[0] -- the OLS coefficient of `stat` on the real `dose` levels
    (NOT the index; do not use trajectory.slope when dose != arange). Significance is a permutation
    null on |slope| (direction-agnostic, one-tailed on the magnitude): the dose labels are shuffled
    and the slope re-fit `k_perm` times; p = (1 + #{|null| >= |obs|}) / (1 + k_perm).

    `unit` -- when given, doses are permuted WITHIN each unit group (a repeated-measures /
    within-subject null), so a purely BETWEEN-unit dose confound is correctly NOT called significant.
    When None, doses are permuted globally."""
    stat = np.asarray(stat, dtype=float).ravel()
    dose = np.asarray(dose, dtype=float).ravel()
    if stat.shape[0] != dose.shape[0]:
        raise ValueError(f"stat ({stat.shape[0]}) and dose ({dose.shape[0]}) length mismatch")

    def _slope(d: np.ndarray) -> float:
        return float(np.polyfit(d, stat, 1)[0])

    obs = _slope(dose)
    obs_abs = abs(obs)
    rng = np.random.default_rng(seed)

    groups = None
    if unit is not None:
        unit_a = np.asarray(unit)
        if unit_a.shape[0] != dose.shape[0]:
            raise ValueError("unit length must match stat/dose")
        groups = [np.where(unit_a == u)[0] for u in np.unique(unit_a)]

    null_abs = np.empty(int(k_perm), dtype=float)
    for i in range(int(k_perm)):
        if groups is None:
            d = rng.permutation(dose)
        else:
            d = dose.copy()
            for idx in groups:
                d[idx] = rng.permutation(dose[idx])
        null_abs[i] = abs(_slope(d))

    p95 = float(np.percentile(null_abs, 95.0))
    p_value = float((1 + int((null_abs >= obs_abs).sum())) / (1 + int(k_perm)))
    return {"slope": round(obs, 6), "perm_p95": round(p95, 6),
            "p_value": round(p_value, 6), "k_perm": int(k_perm)}


# --------------------------------------------------------------------------------------------
# CLI (mirrors styxx.ladder._main)
# --------------------------------------------------------------------------------------------

def _main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m styxx.admissibility",
        description="two-sided instrument admissibility -- sensitive AND specific, on the instrument's own score")
    ap.add_argument("--root", default=".", help="root for receipt re-hashing (default: cwd)")
    ap.add_argument("--certificate", nargs=2, metavar=("RESULT", "OUT"),
                    help="load RESULT.json's points, re-derive an admissibility certificate to OUT")
    ap.add_argument("--verify", metavar="CERT",
                    help="verify an issued certificate: re-hash its receipts AND recompute admissibility from its points")
    a = ap.parse_args()

    if a.verify:
        v = verify_admissibility_certificate(a.verify, a.root)
        print(f"receipts: checked {v['checked']}/{v['n_recorded']} -> {'OK (un-tampered)' if v['ok'] else 'DRIFT DETECTED'}")
        print(f"recompute: admissible={v['recomputed_admissible']} -> "
              f"{'FAITHFUL (matches stored flag)' if v['faithful'] else 'UNFAITHFUL (stored flag != recompute)'}")
        for m in v["mismatches"]:
            print(f"  MISMATCH {m['receipt']}: recorded {m['recorded'][:12]} != live {m['live'][:12]}")
        for miss in v["missing"]:
            print(f"  MISSING  {miss}")
        return 0 if (v["ok"] and v["faithful"]) else 1

    if a.certificate:
        result_path, out_path = a.certificate
        src = json.loads(Path(result_path).read_text(encoding="utf-8"))
        pts = src.get("points", [])
        scores = [p["score"] for p in pts]
        labels = [p["label"] for p in pts]
        sens = src.get("sensitivity", {})
        spec = src.get("specificity", {})
        th = src.get("thresholds", {})
        rep = instrument_admissibility(
            scores=scores, labels=labels,
            expect=sens.get("expect", "lower_on_positive"),
            fire_threshold=spec.get("fire_threshold"),
            auroc_floor=sens.get("auroc_floor", 0.70), alpha=sens.get("alpha", 0.05),
            max_fire=spec.get("max_fire", 0.15), k_perm=th.get("k_perm", 1000), seed=th.get("seed", 0),
        )
        rep.certificate(receipts=src.get("receipts_sha256"), out_path=out_path)
        print(f"verdict: {rep.admissibility_verdict} (admissible={rep.admissible}) -> {out_path}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
