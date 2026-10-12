# -*- coding: utf-8 -*-
"""styxx.horizon — the horizon datasheet: where, if anywhere inside the tested range, a causal LM stops
judging a distant claim while it still recalls the fact the claim is about.

Built from `next/horizon-zoo/DESIGN.md` (2026-10-11) with every fatal and major fix of its red team
(`ATTACK.md`) applied. It is the instrument only: a pure function from per-item bits to a datasheet.
No torch, no GPU, no model load, no network. The GPU runner that produces the bits is not here; the
zoo run that would fill nine rows is not here; nothing in this module changes what any committed
RESULT says.

WHAT IT TAKES. Per-item bits for one model, in cells keyed by *family* (the probe's word for the two
operations, as `run_realmodel_bridge_v3.py` uses it: ``compare`` = judge a claim about a planted code,
``recall`` = produce the planted code) and *distance* (nominal NeoX tokens of filler between the facts
and the question, grid {16, 64, 256, 1024}). A compare item carries ``label`` (the claim is true),
``prediction`` (the model said yes) and ``logodds`` (log p(yes) - log p(no) at the answer position); a
recall item carries ``label`` (the queried slot, 0..2) and ``prediction`` (the argmax slot). A cell may
instead be a plain list of correctness bits; then the guard, the AUROC and the threshold-free curve are
"not recorded" and the sheet says so. The *architecture* family (DECAY / ATTN) is a different word: it is
asserted from a config and, for a state-space model, from its A values — never from a name
(:func:`assert_family`).

WHAT IT COMPUTES, per model.
  * chance-normalised accuracy per cell, ``a' = (a - chance) / (1 - chance)`` (compare chance 1/2,
    recall chance 1/3); a constant responder under exact balance maps to exactly 0 in both families;
  * a 95% paired item-bootstrap interval per cell (one index resample per replicate, applied to every
    cell of every model, both families, all distances — the pairing the red team's M9 requires);
  * the degeneracy guard per cell (compare: yes-rate within [0.20, 0.80]; recall: largest slot share
    <= 0.80) and the rank AUROC of the compare log-odds against the label (0.5 for any constant);
  * the capability screen at D0: both normalised accuracies >= the bar and both guards passed; the
    screen reads one half of the D0 items and the horizon's denominator the other half (M4), so that
    selecting on c(D0) does not bias the ratios of a model near the bar;
  * the comparison horizon: with ``tau = frac * c(D0)`` (frac 0.5), the smallest D >= D0 at which the
    piecewise-linear interpolant of c in log2 D reaches tau. The bracketing grid pair is the
    measurement; the interpolated point is a convention and is printed second. No crossing inside the
    grid is CENSORED ("> 1024 (none inside the tested range)"); c(D0) <= 0 is NOT_ESTIMABLE. The
    interval is the 2.5th / 97.5th percentile of the finite replicates with the censored fraction
    ``f_cens`` beside it: above 0.025 the upper end prints "> 1024", above 0.5 the point prints
    censored. The recall horizon by the same rule; the threshold-free compare curve
    ``c_auc = 2 * AUROC - 1`` by the same rule (F1c) — a horizon in c that is absent from c_auc is
    flagged CALIBRATION_ARTEFACT_SUSPECTED, because a drifting yes-prior moves a thresholded accuracy
    and leaves a rank statistic alone;
  * ``ratio_min`` = min over D > D0 of c(D) / c(D0) per family, and the dissociation metric
    ``ratio_min_compare - ratio_min_recall`` with its paired interval (F2: the sentence worth gating is
    whether a model loses comparison faster than recall, not whether attention has a horizon);
  * flags: COMPARE_ONLY_HORIZON, RECALL_ONLY_HORIZON, STORAGE_HORIZON, NO_HORIZON_INSIDE_RANGE,
    DEGENERATE cells (printed with both c and c_auc), NONMONOTONE, and the control flags below;
  * controls, when the receipt carries them: ``ablated`` (codes removed; must stay at chance — above
    0.16 normalised in any cell reads PROBE_READS_PRIORS), ``near`` (a code planted 16 tokens before the
    question at the far distance; its ratio to c(D0) below 0.5 reads FORMAT_DECAY, below 0.8
    FORMAT_DRIFT), ``overwrite`` (the far filler re-plants one name; compare on the original code must
    fall to chance or the model is not reading the far binding: POSITIVE_CONTROL_SILENT). A receipt
    without ``near`` reads FORMAT_DISTANCE_NOT_EXCLUDED beside every compare horizon; one without
    ``overwrite`` reads FAR_BINDING_CONTROL_ABSENT beside every censored one.

WHAT IT DOES NOT SAY. A horizon is a description of one curve under one frozen probe, one filler
corpus and one few-shot format. It licenses no sentence about rotation, honesty, causation or ranking;
every model differs from every other in more than its recurrence. SCREEN_FAIL means the model did not
perform this format at D0, not that it cannot compare. The distance axis is exact only for a NeoX
tokenizer; for others the receipt records the native token count per cell and the sheet prints the
horizon both ways. The tested range ends at the last grid point, which is under 1% of context for some
models: the word is "inside the tested range", never "inside context". The self-test
(:func:`selftest`) is a software test on synthetic bits; it shows the fitter recovers a planted
crossing, not that any model has one.

CLI::

    python -m styxx.horizon selftest
    python -m styxx.horizon fit   --receipt horizon_<model>.json            # one model, no GPU
    python -m styxx.horizon fit   --receipt realmodel_bridge_v3_result.json --arm M1
    python -m styxx.horizon sheet --receipts a.json b.json ... [--prereg PREREG.md]
    python -m styxx.horizon family --config <snapshot>/config.json [--safetensors <snapshot>/*.safetensors]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import struct
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CHANCE", "GRID", "D0", "SCREEN_BAR", "GUARD_YES_RANGE", "GUARD_SLOT_MAX", "ABLATED_BAR",
    "HorizonRefusal", "normalize", "guard", "auroc", "cell_summary", "crossing", "fit_horizon",
    "fit_model", "fit_zoo", "screen", "assert_family", "mamba_A_from_safetensors", "selftest",
    "datasheet", "synthetic_cell", "load_receipt", "main",
]

RECEIPT_SCHEMA = "styxx.horizon/receipt/0.1"
FAMILIES = ("compare", "recall")
CHANCE = {"compare": 0.5, "recall": 1.0 / 3.0}
GRID = (16, 64, 256, 1024)
D0 = 16
SCREEN_BAR = 0.40
GUARD_YES_RANGE = (0.20, 0.80)
GUARD_SLOT_MAX = 0.80
ABLATED_BAR = 0.16
FRAC = 0.5
N_BOOT = 2000
SEED_BOOT = 20261011
LOGODDS_TIE = 1e-3
NEAR_FORMAT_DECAY = 0.5       # c_near(D1) / c(D0) below this: the format itself decays with distance
NEAR_FORMAT_DRIFT = 0.8       # below this: drift reported, horizon read with the caveat printed
OVERWRITE_FIRED_MAX = 0.16    # overwrite control: compare on the overwritten code must sit at chance
NONMONOTONE_RISE = 0.05       # an adjacent rise above this normalised amount flags NONMONOTONE
ATTN_TYPES = ("gpt_neox", "llama", "qwen2", "gemma2", "phi3", "gpt2", "mistral", "opt", "gptj")
DECAY_TYPES = ("mamba",)
REQUIRED_REACH = 1250         # every attention layer must reach the longest prompt of the grid

# The words the lab never prints, assembled from fragments so that this file does not print them
# either; the datasheet, this file, its README and its tests are scanned against the lab's own list.
_HOUSE_WORDS = tuple("".join(p) for p in (("fir", "st"), ("nov", "el"), ("revolut", "ionary"),
                                           ("ground", "breaking"), ("break", "through"), ("tamper", "-proof"),
                                           ("self", "-verifying"), ("hallucination", " detector")))


class HorizonRefusal(RuntimeError):
    """The instrument refuses: a malformed receipt, a failed self-test, a name offered as a fact."""


# ---------------------------------------------------------------------------------------------------
# scoring rules (the bridge-v3 runner's, restated; no bar is new here)
# ---------------------------------------------------------------------------------------------------

def normalize(acc: float, chance: float) -> float:
    """``(acc - chance) / (1 - chance)``: 0 at chance, 1 at ceiling, negative below chance."""
    if not (0.0 <= chance < 1.0):
        raise HorizonRefusal(f"chance must lie in [0, 1), got {chance!r}")
    return (float(acc) - chance) / (1.0 - chance)


def guard(yes_rate: Optional[float] = None, max_slot_frac: Optional[float] = None) -> Optional[bool]:
    """The degeneracy guard of one cell; None when the cell recorded neither statistic."""
    if yes_rate is not None:
        return bool(GUARD_YES_RANGE[0] <= float(yes_rate) <= GUARD_YES_RANGE[1])
    if max_slot_frac is not None:
        return bool(float(max_slot_frac) <= GUARD_SLOT_MAX)
    return None


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank AUROC with average ranks on ties; 0.5 for any constant score; None if a class is empty."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    return float(_auroc_from_ranks(_rank_avg(s), y, n_pos, n_neg))


def _rank_avg(s: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties sharing their mean rank; vectorised, no Python loop over items."""
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    _, inv, counts = np.unique(ss, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(float)
    starts = ends - counts + 1.0
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = ((starts + ends) / 2.0)[inv]
    return ranks


def _auroc_from_ranks(ranks: np.ndarray, y: np.ndarray, n_pos: int, n_neg: int) -> float:
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _as_bool_array(values, name: str) -> np.ndarray:
    arr = np.asarray(list(values))
    if arr.dtype == object or arr.ndim != 1:
        raise HorizonRefusal(f"{name}: expected a flat list of booleans or 0/1 integers")
    if arr.dtype != bool:
        if not np.all(np.isin(arr, (0, 1))):
            raise HorizonRefusal(f"{name}: values must be booleans or 0/1 integers")
        arr = arr.astype(bool)
    return arr


def cell_summary(family: str, labels=None, predictions=None, logodds=None, correct=None) -> dict:
    """One cell's statistics from its per-item bits.

    Either ``labels`` + ``predictions`` (+ ``logodds`` for compare) or a plain ``correct`` vector. A
    plain vector yields accuracy only; the guard and the AUROC are then None and the sheet prints
    "not recorded" for them rather than a number nobody measured.
    """
    if family not in FAMILIES:
        raise HorizonRefusal(f"family must be one of {FAMILIES}, got {family!r}")
    chance = CHANCE[family]
    out: Dict[str, Any] = {"family": family, "yes_rate": None, "max_slot_fraction": None,
                           "slot_fractions": None, "auroc": None, "c_auc": None, "near_ties": None,
                           "guard_ok": None, "exact_balance": None}
    if correct is not None:
        if labels is not None or predictions is not None:
            raise HorizonRefusal("give either correct bits or labels + predictions, not both")
        cor = _as_bool_array(correct, f"{family} correct")
        out["recorded"] = "correctness bits only"
    else:
        if labels is None or predictions is None:
            raise HorizonRefusal(f"{family} cell: labels and predictions are both required")
        if family == "compare":
            y = _as_bool_array(labels, "compare labels")
            p = _as_bool_array(predictions, "compare predictions")
            if len(y) != len(p):
                raise HorizonRefusal("compare cell: labels and predictions differ in length")
            cor = p == y
            out["yes_rate"] = float(p.mean()) if len(p) else None
            out["exact_balance"] = bool(len(y) and 2 * int(y.sum()) == len(y))
            if logodds is not None:
                lo = np.asarray(list(logodds), dtype=float)
                if len(lo) != len(y):
                    raise HorizonRefusal("compare cell: logodds and labels differ in length")
                if not np.all(np.isfinite(lo)):
                    raise HorizonRefusal("compare cell: a log-odds value is not finite")
                # the prediction IS the sign of the log-odds (ties at exactly 0 break toward No)
                if not np.array_equal(lo > 0.0, p):
                    raise HorizonRefusal("compare cell: predictions disagree with the sign of the "
                                         "log-odds; a receipt that records both must agree")
                a = auroc(lo, y)
                out["auroc"] = a
                out["c_auc"] = None if a is None else 2.0 * a - 1.0
                out["near_ties"] = int((np.abs(lo) < LOGODDS_TIE).sum())
            out["guard_ok"] = guard(yes_rate=out["yes_rate"]) if len(p) else None
            out["recorded"] = "labels, predictions" + (", logodds" if logodds is not None else "")
        else:
            q = np.asarray(list(labels))
            s = np.asarray(list(predictions))
            if q.ndim != 1 or s.ndim != 1 or len(q) != len(s):
                raise HorizonRefusal("recall cell: labels and predictions must be flat and equal")
            if not (np.all(np.isin(q, (0, 1, 2))) and np.all(np.isin(s, (0, 1, 2)))):
                raise HorizonRefusal("recall cell: slots must be 0, 1 or 2")
            q = q.astype(int)
            s = s.astype(int)
            cor = s == q
            fr = [float((s == k).mean()) if len(s) else 0.0 for k in range(3)]
            out["slot_fractions"] = fr
            out["max_slot_fraction"] = max(fr) if len(s) else None
            counts = [int((q == k).sum()) for k in range(3)]
            out["exact_balance"] = bool(len(q) and max(counts) - min(counts) <= 1)
            out["guard_ok"] = guard(max_slot_frac=out["max_slot_fraction"]) if len(s) else None
            out["recorded"] = "labels, predictions"
    n = int(len(cor))
    raw = float(cor.mean()) if n else float("nan")
    out.update({"n": n, "acc_raw": raw, "acc_norm": normalize(raw, chance) if n else float("nan"),
                "_correct": cor})
    return out


# ---------------------------------------------------------------------------------------------------
# the horizon estimator
# ---------------------------------------------------------------------------------------------------

def _log2_grid(grid: Sequence[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=float)
    if g.ndim != 1 or len(g) < 2 or np.any(g <= 0) or np.any(np.diff(g) <= 0):
        raise HorizonRefusal(f"the distance grid must be increasing positive integers, got {list(grid)}")
    return np.log2(g)


def crossing(curve: Sequence[float], grid: Sequence[int] = GRID, frac: float = FRAC) -> dict:
    """Where the piecewise-linear interpolant of ``curve`` in log2 D reaches ``frac * curve[0]``.

    Returns ``{"status": "FINITE"|"CENSORED"|"NOT_ESTIMABLE", "point": D or None,
    "bracket": (D_lo, D_hi) or None, "tau": float or None}``. A crossing exists iff some grid value is
    strictly below tau (the interpolant is linear between grid points, so it cannot dip below tau
    between two points that are both at or above it). The earliest crossing is used; a later rise is
    the NONMONOTONE flag's business, not this function's.
    """
    c = np.asarray(curve, dtype=float)
    lg = _log2_grid(grid)
    if len(c) != len(lg):
        raise HorizonRefusal(f"curve has {len(c)} points, grid has {len(lg)}")
    if not np.all(np.isfinite(c)):
        return {"status": "NOT_ESTIMABLE", "point": None, "bracket": None, "tau": None,
                "reason": "a cell on the curve is not finite"}
    if c[0] <= 0.0:
        return {"status": "NOT_ESTIMABLE", "point": None, "bracket": None, "tau": None,
                "reason": "c(D0) <= 0: no positive D0 value to halve"}
    tau = frac * c[0]
    below = np.flatnonzero(c < tau)
    if len(below) == 0:
        return {"status": "CENSORED", "point": None, "bracket": None, "tau": float(tau)}
    k = int(below[0])
    t = (c[k - 1] - tau) / (c[k - 1] - c[k])
    x = lg[k - 1] + t * (lg[k] - lg[k - 1])
    return {"status": "FINITE", "point": float(2.0 ** x), "bracket": (int(grid[k - 1]), int(grid[k])),
            "tau": float(tau)}


def _crossings_matrix(cmat: np.ndarray, grid: Sequence[int], frac: float):
    """Vectorised :func:`crossing` over replicates (rows of ``cmat``). Returns (points, status codes):
    status 0 = FINITE, 1 = CENSORED, 2 = NOT_ESTIMABLE; points are NaN unless FINITE."""
    lg = _log2_grid(grid)
    B, nD = cmat.shape
    points = np.full(B, np.nan)
    status = np.full(B, 1, dtype=int)
    finite_rows = np.all(np.isfinite(cmat), axis=1)
    c0 = cmat[:, 0]
    nonest = (~finite_rows) | (c0 <= 0.0)
    status[nonest] = 2
    tau = frac * c0
    below = cmat < tau[:, None]
    below[nonest] = False
    any_below = below.any(axis=1)
    rows = np.flatnonzero(any_below)
    if len(rows):
        k = np.argmax(below[rows], axis=1)
        c_prev = cmat[rows, k - 1]
        c_k = cmat[rows, k]
        t = (c_prev - tau[rows]) / (c_prev - c_k)
        x = lg[k - 1] + t * (lg[k] - lg[k - 1])
        points[rows] = 2.0 ** x
        status[rows] = 0
    return points, status


def _ratio_min(curve: np.ndarray) -> float:
    if not np.all(np.isfinite(curve)) or curve[0] <= 0.0:
        return float("nan")
    return float(np.min(curve[1:] / curve[0]))


def _ratio_min_matrix(cmat: np.ndarray) -> np.ndarray:
    c0 = cmat[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.min(cmat[:, 1:] / c0[:, None], axis=1)
    r[(c0 <= 0.0) | ~np.all(np.isfinite(cmat), axis=1)] = np.nan
    return r


def _percentiles(values: np.ndarray) -> Tuple[Optional[float], Optional[float], float]:
    """2.5th / 97.5th percentiles of the finite entries and the fraction that was not finite."""
    v = np.asarray(values, dtype=float)
    fin = v[np.isfinite(v)]
    f_nonfinite = 1.0 - len(fin) / len(v) if len(v) else 1.0
    if len(fin) == 0:
        return None, None, f_nonfinite
    lo, hi = np.percentile(fin, [2.5, 97.5])
    return float(lo), float(hi), float(f_nonfinite)


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    xa, ya = np.asarray(x, float), np.asarray(y, float)
    if len(xa) < 3 or not np.all(np.isfinite(ya)):
        return None
    rx, ry = _rank_avg(xa), _rank_avg(ya)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def fit_horizon(curve: Sequence[float], replicate_curves: Optional[np.ndarray] = None,
                grid: Sequence[int] = GRID, frac: float = FRAC, native_mean: Optional[Sequence[float]] = None) -> dict:
    """The horizon of one curve with, when replicate curves are given, its censored interval.

    ``replicate_curves`` is a (B, len(grid)) matrix of bootstrap curves computed on the SAME index
    resamples as every other curve of the run (see :func:`fit_model`). ``native_mean`` is the mean
    native token count per grid point; the horizon is then also printed in the model's own tokens.
    """
    point = crossing(curve, grid, frac)
    out: Dict[str, Any] = {"status": point["status"], "point": point["point"], "bracket": point["bracket"],
                           "tau": point["tau"], "grid_max": int(grid[-1]),
                           "ci_lo": None, "ci_hi": None, "f_cens": None, "f_nonest": None,
                           "ci_hi_censored": None, "print_censored": point["status"] != "FINITE",
                           "native_point": None, "n_resamples": 0}
    if point.get("reason"):
        out["reason"] = point["reason"]
    if replicate_curves is not None:
        pts, st = _crossings_matrix(np.asarray(replicate_curves, float), grid, frac)
        B = len(st)
        out["n_resamples"] = int(B)
        out["f_cens"] = float(np.mean(st == 1))
        out["f_nonest"] = float(np.mean(st == 2))
        lo, hi, _ = _percentiles(pts)
        out["ci_lo"], out["ci_hi"] = lo, hi
        out["ci_hi_censored"] = bool(out["f_cens"] > 0.025)
        if out["status"] == "FINITE" and out["f_cens"] > 0.5:
            out["print_censored"] = True
    if native_mean is not None and out["point"] is not None:
        nm = np.asarray(native_mean, float)
        if len(nm) == len(grid) and np.all(np.isfinite(nm)):
            out["native_point"] = float(np.interp(math.log2(out["point"]), _log2_grid(grid), nm))
    return out


# ---------------------------------------------------------------------------------------------------
# receipts: the shape the fitter reads, and the two shapes it accepts
# ---------------------------------------------------------------------------------------------------

_CELL_KEYS = {"label": ("label", "labels", "q_slot", "q_slots"),
              "prediction": ("prediction", "predictions", "pred", "pred_slot", "pred_slots", "pred_yes"),
              "logodds": ("logodds", "log_odds", "logit_diff"),
              "correct": ("correct", "bits", "correctness")}


def _pick(d: dict, kind: str):
    for k in _CELL_KEYS[kind]:
        if k in d:
            return d[k]
    return None


def _cell_from_spec(family: str, spec, where: str) -> dict:
    """A cell spec is a dict of per-item vectors or a plain list of correctness bits."""
    if isinstance(spec, dict):
        cor = _pick(spec, "correct")
        if cor is not None and _pick(spec, "label") is None:
            return cell_summary(family, correct=cor)
        labels, preds, lo = _pick(spec, "label"), _pick(spec, "prediction"), _pick(spec, "logodds")
        if preds is None and lo is not None and family == "compare":
            preds = [float(v) > 0.0 for v in lo]
        try:
            cell = cell_summary(family, labels=labels, predictions=preds, logodds=lo)
        except HorizonRefusal as e:
            raise HorizonRefusal(f"{where}: {e}") from e
        # keep the vectors the cell was built from: the D0 split re-reads them, the AUROC replicates need them
        if family == "compare":
            cell["_labels"] = _as_bool_array(labels, "compare labels")
            cell["_preds"] = _as_bool_array(preds, "compare predictions")
        else:
            cell["_labels"] = np.asarray(list(labels)).astype(int)
            cell["_preds"] = np.asarray(list(preds)).astype(int)
        if lo is not None:
            cell["_logodds"] = np.asarray(list(lo), dtype=float)
        return cell
    if isinstance(spec, (list, tuple)):
        try:
            return cell_summary(family, correct=spec)
        except HorizonRefusal as e:
            raise HorizonRefusal(f"{where}: {e}") from e
    raise HorizonRefusal(f"{where}: a cell must be a dict of per-item vectors or a list of bits, "
                         f"got {type(spec).__name__}")


def _cells_from_block(block, where: str) -> Dict[str, Dict[int, dict]]:
    """``{family: {str(D): cell_spec}}`` -> ``{family: {int(D): cell_summary}}``."""
    if not isinstance(block, dict):
        raise HorizonRefusal(f"{where}: expected a dict keyed by family, got {type(block).__name__}")
    out: Dict[str, Dict[int, dict]] = {}
    for fam, by_d in block.items():
        if fam not in FAMILIES:
            raise HorizonRefusal(f"{where}: unknown family {fam!r}; the probe has {FAMILIES}")
        if not isinstance(by_d, dict):
            raise HorizonRefusal(f"{where}.{fam}: expected a dict keyed by distance")
        out[fam] = {}
        for d_key, spec in by_d.items():
            try:
                d = int(d_key)
            except (TypeError, ValueError):
                raise HorizonRefusal(f"{where}.{fam}: distance key {d_key!r} is not an integer") from None
            out[fam][d] = _cell_from_spec(fam, spec, f"{where}.{fam}[{d}]")
    return out


def _cells_from_items(items: Sequence[dict], where: str):
    """A flat list of per-item records -> (main cells, control cells). Records are grouped by
    (control, family, distance) and ordered by their ``item`` index when they carry one — the
    index is the pairing key across distances and controls."""
    groups: Dict[Tuple[Optional[str], str, int], List[dict]] = {}
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise HorizonRefusal(f"{where}[{i}]: an item record must be a dict")
        fam = it.get("family")
        if fam not in FAMILIES:
            raise HorizonRefusal(f"{where}[{i}]: family must be one of {FAMILIES}, got {fam!r}")
        try:
            d = int(it["distance"])
        except (KeyError, TypeError, ValueError):
            raise HorizonRefusal(f"{where}[{i}]: an integer 'distance' is required") from None
        groups.setdefault((it.get("control"), fam, d), []).append(it)
    main: Dict[str, Dict[int, dict]] = {}
    controls: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for (ctrl, fam, d), recs in groups.items():
        if all("item" in r for r in recs):
            recs = sorted(recs, key=lambda r: int(r["item"]))
        spec = {"label": [r.get("label") for r in recs], "prediction": [r.get("prediction") for r in recs]}
        if fam == "compare" and all(r.get("logodds") is not None for r in recs):
            spec["logodds"] = [r["logodds"] for r in recs]
        if any(r.get("label") is None or r.get("prediction") is None for r in recs):
            if all(r.get("correct") is not None for r in recs):
                spec = {"correct": [r["correct"] for r in recs]}
            else:
                raise HorizonRefusal(f"{where}: ({ctrl}, {fam}, {d}) has items without label/prediction "
                                     f"or correct")
        cell = _cell_from_spec(fam, spec, f"{where}:({ctrl},{fam},{d})")
        target = main if ctrl is None else controls.setdefault(str(ctrl), {})
        target.setdefault(fam, {})[d] = cell
    return main, controls


def _config_from(raw: dict, where: str) -> dict:
    cfg = raw.get("config") or {}
    grid = cfg.get("distances") or cfg.get("grid") or list(GRID)
    try:
        grid = [int(x) for x in grid]
    except (TypeError, ValueError):
        raise HorizonRefusal(f"{where}: config.distances must be integers") from None
    _log2_grid(grid)
    d0 = int(cfg.get("D0", grid[0]))
    if d0 != grid[0]:
        raise HorizonRefusal(f"{where}: D0 {d0} is not the smallest grid distance {grid[0]}")
    boot = cfg.get("bootstrap") or {}
    seeds = cfg.get("seeds") or {}
    native = cfg.get("native_D")
    if native is not None:
        if not isinstance(native, dict):
            raise HorizonRefusal(f"{where}: config.native_D must map distance -> {{min, mean, max}}")
        native = {int(k): v for k, v in native.items()}
    pairing = cfg.get("pairing", "items" if cfg.get("items_prefix_nested", True) else "independent")
    if pairing not in ("items", "independent"):
        raise HorizonRefusal(f"{where}: config.pairing must be 'items' or 'independent'")
    return {"grid": grid, "D0": d0, "screen_bar": float(cfg.get("screen_bar", SCREEN_BAR)),
            "frac": float(cfg.get("horizon_frac", FRAC)), "pairing": pairing,
            "d0_split": bool(cfg.get("d0_split", True)),
            "n_resamples": int(boot.get("n_resamples", cfg.get("n_resamples", N_BOOT))),
            "seed": int(boot.get("seed", seeds.get("bootstrap", SEED_BOOT))),
            "native_D": native, "dtype": cfg.get("dtype"), "params": cfg.get("params"),
            "ctx": cfg.get("ctx", cfg.get("max_position_embeddings")),
            "ctx_is_training_length": bool(cfg.get("ctx_is_training_length", False)),
            "copied_from": cfg.get("copied_from"), "items_sha256": cfg.get("items_sha256"),
            "answer_format": cfg.get("answer_format"), "snapshot": cfg.get("snapshot"),
            "notes": cfg.get("notes")}


def load_receipt(path_or_dict, arm: Optional[str] = None) -> dict:
    """Read a receipt into the fitter's shape.

    Two shapes are accepted. The module's own (``schema`` = ``styxx.horizon/receipt/0.1``): ``model``,
    optional ``arch_family``, ``config``, and either ``per_item`` (``{family: {D: cell}}``) or ``items``
    (a flat list), with ``controls`` (``{ablated|near|overwrite: {family: {D: cell}}}``). The bridge-v3
    receipt, with ``arm`` given: ``per_item[arm]`` and ``controls_per_item[arm]`` (the fields decision Z4
    asks v3 to add); a v3 receipt without them is refused with that sentence, not read as zeros.
    """
    if isinstance(path_or_dict, (str, os.PathLike)):
        p = str(path_or_dict)
        try:
            with open(p, "rb") as fh:
                data = fh.read()
        except OSError as e:
            raise HorizonRefusal(f"cannot read receipt {p}: {e}") from e
        try:
            raw = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise HorizonRefusal(f"receipt {p} is not UTF-8 JSON: {e}") from e
        source = {"path": p, "sha256": hashlib.sha256(data).hexdigest()}
    elif isinstance(path_or_dict, dict):
        raw = path_or_dict
        source = {"path": None, "sha256": None}
    else:
        raise HorizonRefusal("load_receipt takes a path or a dict")
    if not isinstance(raw, dict):
        raise HorizonRefusal("a receipt is a JSON object")
    where = source["path"] or "<dict>"

    if arm is not None:
        per_item = raw.get("per_item")
        if not isinstance(per_item, dict) or arm not in per_item:
            raise HorizonRefusal(
                f"{where}: the receipt carries no per_item bits for arm {arm!r}. The horizon of a v3 arm "
                f"needs per-item correctness bits and compare log-odds in the receipt (zoo DESIGN section "
                f"8 item 2, decision Z4); without them the arm's row prints from acc_norm without an "
                f"interval, which this fitter does not do silently.")
        cells = _cells_from_block(per_item[arm], f"{where}.per_item[{arm}]")
        ctrl_raw = (raw.get("controls_per_item") or {}).get(arm) or {}
        controls = {k: _cells_from_block(v, f"{where}.controls_per_item[{arm}].{k}") for k, v in ctrl_raw.items()}
        checkpoints = (raw.get("config") or {}).get("checkpoints") or {}
        model = checkpoints.get(arm, arm)
        cfg = _config_from(raw, where)
        cfg["copied_from"] = cfg.get("copied_from") or f"v3 receipt arm {arm}"
        arch = raw.get("arch_family", {}).get(arm) if isinstance(raw.get("arch_family"), dict) else None
    else:
        schema = raw.get("schema")
        if schema is not None and schema != RECEIPT_SCHEMA:
            raise HorizonRefusal(f"{where}: schema {schema!r} is not {RECEIPT_SCHEMA!r}")
        model = raw.get("model")
        if not isinstance(model, str) or not model:
            raise HorizonRefusal(f"{where}: 'model' (a non-empty string) is required")
        if "per_item" in raw:
            cells = _cells_from_block(raw["per_item"], f"{where}.per_item")
            controls = {k: _cells_from_block(v, f"{where}.controls.{k}")
                        for k, v in (raw.get("controls") or {}).items()}
        elif "items" in raw:
            if not isinstance(raw["items"], list):
                raise HorizonRefusal(f"{where}: 'items' must be a list of per-item records")
            cells, controls = _cells_from_items(raw["items"], f"{where}.items")
        else:
            raise HorizonRefusal(f"{where}: a receipt needs 'per_item' cells or an 'items' list")
        cfg = _config_from(raw, where)
        arch = raw.get("arch_family")
    for fam in FAMILIES:
        if fam not in cells:
            raise HorizonRefusal(f"{where}: no {fam} cells; both families are needed for a horizon row")
        missing = [d for d in cfg["grid"] if d not in cells[fam]]
        if missing:
            raise HorizonRefusal(f"{where}: {fam} has no cell at distance(s) {missing} of the grid {cfg['grid']}")
    for k in controls:
        if k not in ("ablated", "near", "overwrite"):
            raise HorizonRefusal(f"{where}: unknown control {k!r}; the instrument knows ablated, near, overwrite")
    if arch is not None and not (isinstance(arch, dict) and arch.get("family") in ("DECAY", "ATTN")):
        raise HorizonRefusal(f"{where}: arch_family must be {{'family': 'DECAY'|'ATTN', 'fact': ...}} as "
                             f"assert_family returns it, or absent")
    return {"model": model, "arch_family": arch, "config": cfg, "cells": cells, "controls": controls,
            "source": source, "status": raw.get("status")}


# ---------------------------------------------------------------------------------------------------
# the per-model fit
# ---------------------------------------------------------------------------------------------------

def screen(compare_d0: dict, recall_d0: dict, bar: float = SCREEN_BAR) -> dict:
    """The capability screen at D0: both normalised accuracies >= bar and both guards passed.

    A guard that was not recorded cannot pass: the screen then fails with the reason, because a row
    whose constant-responder check nobody ran must not enter a family set.
    """
    reasons = []
    c, r = compare_d0["acc_norm"], recall_d0["acc_norm"]
    if not (c >= bar):
        reasons.append(f"compare normalised {c:+.3f} < bar {bar:.2f}")
    if not (r >= bar):
        reasons.append(f"recall normalised {r:+.3f} < bar {bar:.2f}")
    for fam, cell in (("compare", compare_d0), ("recall", recall_d0)):
        g = cell.get("guard_ok")
        if g is None:
            reasons.append(f"{fam} guard not recorded (correctness bits only)")
        elif not g:
            stat = (f"yes-rate {cell['yes_rate']:.2f}" if fam == "compare"
                    else f"largest slot share {cell['max_slot_fraction']:.2f}")
            reasons.append(f"{fam} guard failed ({stat})")
    return {"pass": not reasons, "reasons": reasons, "bar": bar, "compare_norm_D0": c, "recall_norm_D0": r,
            "guard_compare": compare_d0.get("guard_ok"), "guard_recall": recall_d0.get("guard_ok"),
            "n_compare": compare_d0["n"], "n_recall": recall_d0["n"]}


def _split_masks(cell: dict) -> Tuple[np.ndarray, np.ndarray, str]:
    """Two halves of a D0 cell: the screen's and the denominator's. Stratified by label when the cell
    records labels, so that a cell in exact balance yields two halves in exact balance and a constant
    responder maps to 0 on both; positional halves otherwise, and the note says so."""
    n = cell["n"]
    y = cell.get("_labels")
    screen_mask = np.zeros(n, dtype=bool)
    if y is None:
        screen_mask[: n // 2] = True
        return screen_mask, ~screen_mask, (f"positional halves (no labels recorded): screen [0, {n // 2}), "
                                            f"denominator [{n // 2}, {n})")
    for v in np.unique(y):
        pos = np.flatnonzero(y == v)
        screen_mask[pos[: len(pos) // 2]] = True
    return screen_mask, ~screen_mask, (f"stratified halves by label: screen on {int(screen_mask.sum())} items, "
                                        f"denominator on {int((~screen_mask).sum())}")


def _subcell(cell: dict, mask: np.ndarray, label: str) -> dict:
    """The cell restricted to the items ``mask`` selects: accuracy, guard and AUROC recomputed."""
    cor = cell["_correct"][mask]
    out = dict(cell)
    out["_correct"] = cor
    out["n"] = int(len(cor))
    raw = float(cor.mean()) if len(cor) else float("nan")
    out["acc_raw"] = raw
    out["acc_norm"] = normalize(raw, CHANCE[cell["family"]]) if len(cor) else float("nan")
    y, p, lo_ = cell.get("_labels"), cell.get("_preds"), cell.get("_logodds")
    if y is not None and p is not None:
        y, p = y[mask], p[mask]
        if cell["family"] == "compare":
            out["yes_rate"] = float(p.mean()) if len(p) else None
            out["guard_ok"] = guard(yes_rate=out["yes_rate"]) if len(p) else None
            if lo_ is not None:
                a = auroc(lo_[mask], y)
                out["auroc"], out["c_auc"] = a, (None if a is None else 2.0 * a - 1.0)
        else:
            fr = [float((p == k).mean()) if len(p) else 0.0 for k in range(3)]
            out["slot_fractions"], out["max_slot_fraction"] = fr, max(fr)
            out["guard_ok"] = guard(max_slot_frac=max(fr)) if len(p) else None
    out["items"] = label
    return out


def _rep_means(correct: np.ndarray, idx: np.ndarray, keep_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Replicate means of a bool vector under the index resample; with ``keep_mask``, only resampled
    indices the mask selects count (the D0 denominator half)."""
    vals = correct[idx].astype(float)
    if keep_mask is None:
        return vals.mean(axis=1)
    mask = keep_mask[idx]
    cnt = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(mask, vals, 0.0).sum(axis=1) / cnt
    m[cnt == 0] = np.nan
    return m


def _rep_c_auc(logodds: np.ndarray, labels: np.ndarray, idx: np.ndarray, keep_mask: Optional[np.ndarray] = None) -> np.ndarray:
    out = np.full(idx.shape[0], np.nan)
    for b in range(idx.shape[0]):
        sel = idx[b] if keep_mask is None else idx[b][keep_mask[idx[b]]]
        if len(sel) == 0:
            continue
        a = auroc(logodds[sel], labels[sel])
        if a is not None:
            out[b] = 2.0 * a - 1.0
    return out


def _resample_matrix(n: int, n_resamples: int, seed: int) -> np.ndarray:
    if n < 2:
        raise HorizonRefusal(f"a cell of {n} item(s) cannot be resampled")
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, n, size=(int(n_resamples), n))


def _fmt_flag_cells(flags: List[str], fam: str, d: int, cell: dict) -> None:
    c = cell["acc_norm"]
    ca = cell.get("c_auc")
    extra = f" c={c:+.2f}" + (f" c_auc={ca:+.2f}" if ca is not None else "")
    flags.append(f"DEGENERATE@{fam}:{d}{extra}")


def fit_model(rec: dict, idx: Optional[np.ndarray] = None, n_resamples: Optional[int] = None,
              seed: Optional[int] = None) -> dict:
    """The row of one model: cells with intervals, screen, horizons, ratios, dissociation, flags.

    ``idx`` is a shared (B, n) index matrix when several models are fitted together (one resample
    stream across models, M9); when None the model draws its own from ``seed``. With
    ``config.pairing == "independent"`` every cell draws its own indices (the receipt declared that
    its items are not shared across distances) and the sheet says so.
    """
    cfg = rec["config"]
    grid = cfg["grid"]
    d0 = cfg["D0"]
    frac = cfg["frac"]
    B = int(n_resamples if n_resamples is not None else cfg["n_resamples"])
    seed = int(seed if seed is not None else cfg["seed"])
    cells = rec["cells"]
    ns = {cells[f][d]["n"] for f in FAMILIES for d in grid}
    paired = cfg["pairing"] == "items"
    if paired and len(ns) != 1:
        raise HorizonRefusal(f"{rec['model']}: cells differ in item count {sorted(ns)} while the receipt "
                             f"declares items paired across distances; declare config.pairing = "
                             f"'independent' or fix the receipt")
    n = min(ns)
    if idx is not None and (idx.ndim != 2 or (paired and idx.shape[1] != n)):
        raise HorizonRefusal(f"{rec['model']}: the shared index matrix has shape {idx.shape}, the cells have n={n}")
    if idx is None:
        idx = _resample_matrix(n, B, seed)
    B = int(idx.shape[0])
    split = bool(cfg["d0_split"])
    if split and n < 4:
        raise HorizonRefusal(f"{rec['model']}: {n} items are too few for the D0 split")
    masks = {f: _split_masks(cells[f][d0]) for f in FAMILIES} if split else None

    rng_cells = np.random.default_rng(seed + 1)   # only used under pairing == "independent"

    def cell_idx(cell: dict) -> np.ndarray:
        if paired:
            return idx
        return rng_cells.integers(0, cell["n"], size=(B, cell["n"]))

    # -- the screen on one half of D0, the denominator on the other ---------------------------------
    if masks is not None:
        screen_cells = {f: _subcell(cells[f][d0], masks[f][0], "screen half") for f in FAMILIES}
        denom_cells = {f: _subcell(cells[f][d0], masks[f][1], "denominator half") for f in FAMILIES}
        split_note = ("D0 split (" + "; ".join(f"{f}: {masks[f][2]}" for f in FAMILIES)
                      + f"); other distances on all {n}")
    else:
        screen_cells = {f: cells[f][d0] for f in FAMILIES}
        denom_cells = {f: cells[f][d0] for f in FAMILIES}
        split_note = ("D0 split: none (screen and denominator share the D0 items; a model near the bar "
                      "has its ratios biased low by selection)")
    scr = screen(screen_cells["compare"], screen_cells["recall"], cfg["screen_bar"])
    scr["items"] = "the screen half of D0 (split_note)" if masks is not None else f"all {n}"

    # -- curves, replicate curves, intervals ------------------------------------------------------
    curves: Dict[str, np.ndarray] = {}
    rep_curves: Dict[str, np.ndarray] = {}
    out_cells: Dict[str, Dict[int, dict]] = {fam: {} for fam in FAMILIES}
    flags: List[str] = []
    auc_curve = np.full(len(grid), np.nan)
    auc_rep = np.full((B, len(grid)), np.nan)
    has_logodds = all(cells["compare"][d].get("_logodds") is not None for d in grid)
    for fam in FAMILIES:
        pts, reps = [], []
        for j, d in enumerate(grid):
            cell = cells[fam][d]
            use = denom_cells[fam] if d == d0 else cell
            keep = masks[fam][1] if (d == d0 and masks is not None) else None
            ci = cell_idx(cell)
            rm = _rep_means(cell["_correct"], ci, keep)
            rnorm = (rm - CHANCE[fam]) / (1.0 - CHANCE[fam])
            lo, hi, _ = _percentiles(rnorm)
            pts.append(use["acc_norm"])
            reps.append(rnorm)
            oc = {k: v for k, v in use.items() if not k.startswith("_")}
            oc.update({"ci_lo": lo, "ci_hi": hi, "degenerate": (use.get("guard_ok") is False)})
            if fam == "compare" and has_logodds:
                auc_curve[j] = use["c_auc"] if use.get("c_auc") is not None else np.nan
                auc_rep[:, j] = _rep_c_auc(cell["_logodds"].astype(float), cell["_labels"].astype(bool), ci, keep)
            if d != d0 and use.get("guard_ok") is False:
                _fmt_flag_cells(flags, fam, d, use)
            out_cells[fam][d] = oc
        curves[fam] = np.asarray(pts, float)
        rep_curves[fam] = np.stack(reps, axis=1)

    native_mean = None
    if cfg.get("native_D"):
        try:
            native_mean = [float(cfg["native_D"][d]["mean"]) for d in grid]
        except (KeyError, TypeError, ValueError):
            native_mean = None
            flags.append("NATIVE_D_INCOMPLETE")

    horizons = {fam: fit_horizon(curves[fam], rep_curves[fam], grid, frac, native_mean) for fam in FAMILIES}
    if has_logodds:
        horizons["compare_auc"] = fit_horizon(auc_curve, auc_rep, grid, frac, native_mean)
    else:
        horizons["compare_auc"] = {"status": "NOT_RECORDED", "point": None, "bracket": None,
                                   "reason": "no per-item log-odds in the receipt"}
        flags.append("LOGODDS_ABSENT")

    ratios = {}
    rep_ratio = {}
    for fam in FAMILIES:
        r = _ratio_min(curves[fam])
        rr = _ratio_min_matrix(rep_curves[fam])
        lo, hi, fnon = _percentiles(rr)
        ratios[fam] = {"point": None if not np.isfinite(r) else float(r), "ci_lo": lo, "ci_hi": hi,
                       "f_nonfinite": fnon}
        rep_ratio[fam] = rr
    diss_rep = rep_ratio["compare"] - rep_ratio["recall"]
    dp = (ratios["compare"]["point"] - ratios["recall"]["point"]
          if ratios["compare"]["point"] is not None and ratios["recall"]["point"] is not None else None)
    lo, hi, fnon = _percentiles(diss_rep)
    dissoc = {"metric": "ratio_min_compare - ratio_min_recall", "point": dp, "ci_lo": lo, "ci_hi": hi,
              "f_nonfinite": fnon}

    # -- dissociation and shape flags -------------------------------------------------------------
    hc, hr = horizons["compare"]["status"], horizons["recall"]["status"]
    if hc == "FINITE" and hr == "CENSORED":
        flags.append("COMPARE_ONLY_HORIZON")
    elif hc == "CENSORED" and hr == "FINITE":
        flags.append("RECALL_ONLY_HORIZON")
    elif hc == "FINITE" and hr == "FINITE":
        flags.append("STORAGE_HORIZON" if horizons["recall"]["point"] <= horizons["compare"]["point"]
                     else "BOTH_HORIZONS_RECALL_LATER")
    elif hc == "CENSORED" and hr == "CENSORED":
        flags.append("NO_HORIZON_INSIDE_RANGE")
    else:
        flags.append("HORIZON_NOT_ESTIMABLE")
    if hc == "FINITE" and horizons["compare_auc"].get("status") == "CENSORED":
        flags.append("CALIBRATION_ARTEFACT_SUSPECTED")
    spearman = {fam: _spearman(np.log2(grid), curves[fam]) for fam in FAMILIES}
    for fam in FAMILIES:
        c = curves[fam]
        if np.any(np.diff(c) > NONMONOTONE_RISE):
            flags.append(f"NONMONOTONE@{fam}")

    # -- controls ---------------------------------------------------------------------------------
    ctrl_out: Dict[str, Any] = {}
    abl = rec["controls"].get("ablated")
    if abl:
        vals = {f"{fam}:{d}": abl[fam][d]["acc_norm"] for fam in abl for d in abl[fam]}
        mx = max(vals.values())
        ctrl_out["ablated"] = {"max_norm_acc": float(mx), "by_cell": vals, "bar": ABLATED_BAR,
                               "reads_priors": bool(mx > ABLATED_BAR)}
        if mx > ABLATED_BAR:
            flags.append("PROBE_READS_PRIORS")
    else:
        ctrl_out["ablated"] = None
        flags.append("ABLATED_CONTROL_ABSENT")
    near = (rec["controls"].get("near") or {}).get("compare")
    c0 = curves["compare"][0]
    if near:
        d_far = max(near)
        cn = near[d_far]["acc_norm"]
        ratio = (cn / c0) if c0 > 0 else None
        ctrl_out["near"] = {"distance": d_far, "norm_acc": float(cn), "ratio_to_D0": ratio,
                            "format_decay_below": NEAR_FORMAT_DECAY, "format_drift_below": NEAR_FORMAT_DRIFT}
        if ratio is None:
            flags.append("NEAR_CONTROL_UNREADABLE")
        elif ratio < NEAR_FORMAT_DECAY:
            flags.append(f"FORMAT_DECAY@{d_far}")
        elif ratio < NEAR_FORMAT_DRIFT:
            flags.append(f"FORMAT_DRIFT@{d_far}")
        else:
            flags.append("FORMAT_DISTANCE_EXCLUDED")
    else:
        ctrl_out["near"] = None
        flags.append("FORMAT_DISTANCE_NOT_EXCLUDED")
    ow = (rec["controls"].get("overwrite") or {}).get("compare")
    if ow:
        d_far = max(ow)
        co = ow[d_far]["acc_norm"]
        fired = bool(co <= OVERWRITE_FIRED_MAX)
        ctrl_out["overwrite"] = {"distance": d_far, "norm_acc": float(co), "fired": fired,
                                 "fired_max": OVERWRITE_FIRED_MAX}
        flags.append("POSITIVE_CONTROL_FIRED" if fired else f"POSITIVE_CONTROL_SILENT@{d_far}")
    else:
        ctrl_out["overwrite"] = None
        flags.append("FAR_BINDING_CONTROL_ABSENT")
    if not scr["pass"]:
        flags.insert(0, "SCREEN_FAIL")

    return {"model": rec["model"], "arch_family": rec["arch_family"], "source": rec["source"],
            "config": {k: v for k, v in cfg.items() if k != "native_D"},
            "native_D": cfg.get("native_D"), "n_items": n, "pairing": cfg["pairing"],
            "n_resamples": B, "seed": seed, "split_note": split_note, "screen": scr,
            "cells": out_cells, "curves": {fam: [float(x) for x in curves[fam]] for fam in FAMILIES},
            "curve_auc": [None if not np.isfinite(x) else float(x) for x in auc_curve],
            "horizon": horizons, "ratio_min": ratios, "dissociation": dissoc, "spearman_log2D": spearman,
            "controls": ctrl_out, "flags": flags,
            "_rep": {"dissoc": diss_rep, "ratio": rep_ratio, "curves": rep_curves}}


# ---------------------------------------------------------------------------------------------------
# the zoo sheet: rows plus the family aggregates a gates block can read with agg / over / excluding
# ---------------------------------------------------------------------------------------------------

def _round6(x):
    return None if x is None or not np.isfinite(x) else float(round(float(x), 6))


def fit_zoo(recs: Sequence[dict], exclude: Sequence[str] = (), n_resamples: Optional[int] = None,
            seed: int = SEED_BOOT, family: str = "ATTN") -> dict:
    """Fit every receipt on ONE index resample stream and aggregate the named architecture family.

    ``exclude`` names models whose rows are printed but never counted (v3's own arms, copied rows).
    The aggregates follow the red team's fixes: the ablated maximum is taken over the screened set
    BEFORE any ablation exclusion (M1); every per-model value sits in ``result.per_model.*`` and every
    excluded member in ``result.family_excluded`` so a gates block declares ``agg``/``over``/``excluding``
    (M2); the family interval is the interval of the per-replicate minimum on the shared resample (M3,
    M9). A receipt whose item count differs from the others cannot share the stream; the sheet then
    fits it alone and says the family interval is not paired across models.
    """
    recs = list(recs)
    if not recs:
        raise HorizonRefusal("fit_zoo: no receipts")
    names = [r["model"] for r in recs]
    if len(set(names)) != len(names):
        raise HorizonRefusal(f"fit_zoo: duplicate model names {sorted(n for n in names if names.count(n) > 1)}")
    B = int(n_resamples if n_resamples is not None else max(r["config"]["n_resamples"] for r in recs))
    ns = {min(r["cells"][f][d]["n"] for f in FAMILIES for d in r["config"]["grid"]) for r in recs}
    pairable = all(r["config"]["pairing"] == "items" for r in recs) and len(ns) == 1
    shared = _resample_matrix(next(iter(ns)), B, seed) if pairable else None
    rows = [fit_model(r, idx=shared, n_resamples=B, seed=seed) for r in recs]

    per_model: Dict[str, Dict[str, Optional[float]]] = {"dissoc": {}, "dissoc_ci_lo": {}, "dissoc_ci_hi": {},
                                                        "ratio_min_compare": {}, "ratio_min_recall": {},
                                                        "ablated_max_norm_acc": {}}
    excluded: List[str] = []
    reasons: Dict[str, str] = {}
    screened: List[dict] = []
    for row in rows:
        m = row["model"]
        per_model["dissoc"][m] = _round6(row["dissociation"]["point"])
        per_model["dissoc_ci_lo"][m] = _round6(row["dissociation"]["ci_lo"])
        per_model["dissoc_ci_hi"][m] = _round6(row["dissociation"]["ci_hi"])
        per_model["ratio_min_compare"][m] = _round6(row["ratio_min"]["compare"]["point"])
        per_model["ratio_min_recall"][m] = _round6(row["ratio_min"]["recall"]["point"])
        abl = row["controls"]["ablated"]
        per_model["ablated_max_norm_acc"][m] = _round6(abl["max_norm_acc"]) if abl else None
        fam = (row["arch_family"] or {}).get("family")
        if m in exclude:
            reasons[m] = "excluded by name (a copied or v3 row)"
        elif fam != family:
            reasons[m] = f"architecture family {fam or 'unasserted'} is not {family}"
        elif not row["screen"]["pass"]:
            reasons[m] = "SCREEN_FAIL: " + "; ".join(row["screen"]["reasons"])
        else:
            screened.append(row)
            if abl is None:
                reasons[m] = "no ablated control in the receipt"
            elif abl["reads_priors"]:
                reasons[m] = f"PROBE_READS_PRIORS: ablated max {abl['max_norm_acc']:+.3f} > {ABLATED_BAR}"
            elif per_model["dissoc"][m] is None:
                reasons[m] = "dissociation metric not estimable"
        if m in reasons:
            excluded.append(m)
    family_set = [r for r in screened if r["model"] not in reasons]
    abl_vals = [r["controls"]["ablated"]["max_norm_acc"] for r in screened if r["controls"]["ablated"]]
    result: Dict[str, Any] = {
        "family": family, "n_models": len(rows), "n_attention_screen_pass": len(family_set),
        "family_set": [r["model"] for r in family_set], "family_excluded": excluded,
        "family_excluded_reasons": reasons, "per_model": per_model,
        "family_set_ablated_max_norm_acc": _round6(max(abl_vals)) if abl_vals else None,
        "family_set_ablated_note": "max over the screened set BEFORE ablation exclusion (red team M1)",
        "shared_resample": bool(pairable), "n_resamples": B, "seed": seed,
    }
    if family_set:
        pts = [per_model["dissoc"][r["model"]] for r in family_set]
        result["attention_family_min_dissoc"] = min(pts)
        result["attention_family_min_dissoc_ci_lo_per_model_min"] = min(per_model["dissoc_ci_lo"][r["model"]] for r in family_set)
        result["attention_family_min_dissoc_ci_hi_per_model_min"] = min(per_model["dissoc_ci_hi"][r["model"]] for r in family_set)
        if pairable:
            mat = np.stack([r["_rep"]["dissoc"] for r in family_set], axis=1)
            fam_min = np.nanmin(mat, axis=1) if np.any(np.isfinite(mat)) else np.full(mat.shape[0], np.nan)
            fam_min[~np.all(np.isfinite(mat), axis=1)] = np.nan
            lo, hi, fnon = _percentiles(fam_min)
            result["attention_family_min_dissoc_ci"] = {"lo": lo, "hi": hi, "f_nonfinite": fnon,
                                                        "method": "per-replicate minimum over the family set on "
                                                                  "one shared item resample"}
        else:
            result["attention_family_min_dissoc_ci"] = {"lo": None, "hi": None, "f_nonfinite": None,
                                                        "method": "not paired across models: item counts or "
                                                                  "pairing declarations differ"}
    else:
        result["attention_family_min_dissoc"] = None
        result["attention_family_min_dissoc_ci"] = None
    return {"rows": rows, "result": result, "exclude": list(exclude)}


# ---------------------------------------------------------------------------------------------------
# the architecture family: a fact of the config and the weights, never a name
# ---------------------------------------------------------------------------------------------------

def _load_config(config) -> dict:
    if isinstance(config, (str, os.PathLike)):
        p = str(config)
        if os.path.isdir(p):
            p = os.path.join(p, "config.json")
        if not os.path.exists(p):
            raise HorizonRefusal(f"{p!r} is not a config.json or a snapshot directory on disk; a model name is "
                                 f"not a fact about its recurrence")
        try:
            with open(p, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            raise HorizonRefusal(f"cannot read config {p}: {e}") from e
        if not isinstance(cfg, dict):
            raise HorizonRefusal(f"{p}: a config is a JSON object")
        return cfg
    if isinstance(config, dict):
        return config
    raise HorizonRefusal("assert_family takes a config dict or a path to config.json; a model name is "
                         "not a fact about its recurrence")


def assert_family(config, A=None, required_reach: int = REQUIRED_REACH) -> dict:
    """DECAY or ATTN from ``config.json`` plus, for a state-space model, its A values.

    * ``model_type == "mamba"``: ``A`` is required — per-layer arrays of A (or ``{"A_log": [...]}``,
      ``A = -exp(A_log)``); every value must be real, finite and strictly negative, else refusal.
    * an attention ``model_type``: every layer's reach (``max_position_embeddings``, and
      ``sliding_window`` where the config applies one) must be >= ``required_reach`` tokens.
    * anything else, or a bare name: refused. A hybrid or an unknown type is not silently a family.
    """
    cfg = _load_config(config)
    mt = cfg.get("model_type")
    if not isinstance(mt, str) or not mt:
        raise HorizonRefusal("config has no model_type; a family is asserted from the config and the weights, "
                             "never from a name")
    if mt in DECAY_TYPES:
        if A is None:
            raise HorizonRefusal(f"model_type {mt!r} names a state-space architecture; the family fact needs "
                                 f"its A values (A = -exp(A_log) from the weights), which were not given")
        layers = _layers_of_A(A)
        per_layer_max, per_layer_min = [], []
        for i, arr in enumerate(layers):
            if np.iscomplexobj(arr):
                raise HorizonRefusal(f"layer {i}: A is complex-valued; this is not a pure-decay recurrence")
            arr = np.asarray(arr, dtype=float).ravel()
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                raise HorizonRefusal(f"layer {i}: A has no finite values")
            if not np.all(arr < 0.0):
                raise HorizonRefusal(f"layer {i}: A is not strictly negative (max {float(arr.max()):+.4g}); "
                                     f"the DECAY fact does not hold")
            per_layer_max.append(float(arr.max()))
            per_layer_min.append(float(arr.min()))
        return {"family": "DECAY", "model_type": mt,
                "fact": {"A_real_and_strictly_negative_every_layer": True, "layers": len(layers),
                         "max_A": max(per_layer_max), "min_A": min(per_layer_min),
                         "n_hidden_layers": cfg.get("num_hidden_layers", cfg.get("n_layer"))}}
    if mt in ATTN_TYPES:
        mpe = cfg.get("max_position_embeddings")
        reach = float("inf") if mpe is None else float(mpe)
        sw = cfg.get("sliding_window")
        applies = sw is not None and (cfg.get("use_sliding_window", True) is not False)
        if applies:
            reach = min(reach, float(sw))
        if not reach >= required_reach:
            raise HorizonRefusal(f"model_type {mt!r}: an attention layer reaches {reach:g} tokens, under the "
                                 f"{required_reach} the grid's longest prompt needs; not an ATTN row of this probe")
        return {"family": "ATTN", "model_type": mt,
                "fact": {"reach_tokens": None if reach == float("inf") else int(reach),
                         "max_position_embeddings": mpe, "sliding_window": sw if applies else None,
                         "required_reach": int(required_reach),
                         "n_hidden_layers": cfg.get("num_hidden_layers", cfg.get("n_layer"))}}
    raise HorizonRefusal(f"model_type {mt!r} is neither a known pure-decay nor a known attention type; the "
                         f"instrument refuses to guess a family")


def _layers_of_A(A) -> List[np.ndarray]:
    if isinstance(A, dict):
        if "A_log" in A:
            logs = A["A_log"]
            layers = logs if isinstance(logs, (list, tuple)) and logs and isinstance(logs[0], (list, tuple, np.ndarray)) else [logs]
            return [-np.exp(np.asarray(l, dtype=float)) for l in layers]
        if "A" in A:
            return _layers_of_A(A["A"])
        raise HorizonRefusal("A must be per-layer arrays, or {'A_log': ...} / {'A': ...}")
    if isinstance(A, np.ndarray):
        return [A] if A.ndim <= 1 else [a for a in A]
    if isinstance(A, (list, tuple)):
        if not A:
            raise HorizonRefusal("A is empty")
        if isinstance(A[0], (list, tuple, np.ndarray)):
            return [np.asarray(a) for a in A]
        return [np.asarray(A)]
    raise HorizonRefusal(f"A must be an array or a list of per-layer arrays, got {type(A).__name__}")


_ST_DTYPES = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2), "F64": ("<f8", 8)}


def mamba_A_from_safetensors(paths: Iterable[str], suffix: str = ".A_log") -> dict:
    """Read every ``*.A_log`` tensor from safetensors files WITHOUT torch and without loading a model:
    the header names the byte range of each tensor, and only those ranges are read.

    Returns ``{"layers": n, "names": [...], "A": [per-layer arrays of A = -exp(A_log)], "files": [...]}``.
    """
    paths = sorted(set(str(p) for p in paths))
    if not paths:
        raise HorizonRefusal("no safetensors files given")
    found: List[Tuple[str, np.ndarray]] = []
    for p in paths:
        try:
            with open(p, "rb") as fh:
                head = fh.read(8)
                if len(head) < 8:
                    raise HorizonRefusal(f"{p}: not a safetensors file (short header)")
                (hlen,) = struct.unpack("<Q", head)
                if hlen > 100 * 1024 * 1024:
                    raise HorizonRefusal(f"{p}: header length {hlen} is not credible")
                header = json.loads(fh.read(hlen).decode("utf-8"))
                base = 8 + hlen
                for name, meta in header.items():
                    if name == "__metadata__" or not name.endswith(suffix):
                        continue
                    dt = meta.get("dtype")
                    if dt not in _ST_DTYPES:
                        raise HorizonRefusal(f"{p}: {name} has dtype {dt!r}, which this reader does not decode")
                    npdt, _ = _ST_DTYPES[dt]
                    s, e = meta["data_offsets"]
                    fh.seek(base + s)
                    buf = fh.read(e - s)
                    arr = np.frombuffer(buf, dtype=npdt)
                    if dt == "BF16":
                        arr = (arr.astype(np.uint32) << 16).view(np.float32)
                    arr = arr.astype(np.float64).reshape(meta["shape"])
                    found.append((name, -np.exp(arr)))
        except OSError as e:
            raise HorizonRefusal(f"cannot read {p}: {e}") from e
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, ValueError, struct.error) as e:
            raise HorizonRefusal(f"{p}: malformed safetensors header: {e}") from e
    if not found:
        raise HorizonRefusal(f"no tensor named *{suffix} in {paths}")
    found.sort(key=lambda t: _layer_index(t[0]))
    return {"layers": len(found), "names": [n for n, _ in found], "A": [a for _, a in found], "files": paths}


def _layer_index(name: str) -> Tuple[int, str]:
    for part in name.split("."):
        if part.isdigit():
            return (int(part), name)
    return (10 ** 9, name)


# ---------------------------------------------------------------------------------------------------
# synthetic bits and the planted-crossing self-test
# ---------------------------------------------------------------------------------------------------

def synthetic_cell(family: str, c_norm: float, n: int, rng: np.random.Generator,
                   constant: Optional[Any] = None) -> dict:
    """Per-item bits for one cell with EXACT balance and a known normalised accuracy.

    ``constant`` builds a constant responder instead (compare: always that answer; recall: always
    that slot). Compare items carry log-odds whose sign is the prediction and whose magnitude is
    random, so the threshold-free curve tracks the thresholded one.
    """
    if family not in FAMILIES:
        raise HorizonRefusal(f"family must be one of {FAMILIES}")
    chance = CHANCE[family]
    if family == "compare":
        if n % 2:
            raise HorizonRefusal("a balanced compare cell needs an even n")
        labels = np.array([True, False] * (n // 2))
        rng.shuffle(labels)
    else:
        if n % 3:
            raise HorizonRefusal("a balanced recall cell needs n divisible by 3")
        labels = np.array([0, 1, 2] * (n // 3))
        rng.shuffle(labels)
    if constant is not None:
        preds = np.full(n, constant, dtype=labels.dtype)
    else:
        c_norm = max(-1.0, min(1.0, float(c_norm)))
        raw = chance + c_norm * (1.0 - chance)
        n_correct = int(round(raw * n))
        correct = np.zeros(n, dtype=bool)
        correct[rng.choice(n, size=n_correct, replace=False)] = True
        if family == "compare":
            preds = np.where(correct, labels, ~labels)
        else:
            wrong = (labels + rng.integers(1, 3, size=n)) % 3
            preds = np.where(correct, labels, wrong)
    spec = {"label": labels.tolist(), "prediction": preds.tolist()}
    if family == "compare":
        # a constant responder carries one constant log-odds, so its AUROC is exactly 0.5
        mag = np.full(n, 1.0) if constant is not None else rng.uniform(0.1, 3.0, size=n)
        spec["logodds"] = np.where(preds, mag, -mag).tolist()
    return spec


def _planted_curve(h_star: float, c0: float = 0.9, frac: float = FRAC, grid: Sequence[int] = GRID) -> List[float]:
    """A curve linear in log2 D through (D0, c0) that reaches frac * c0 exactly at h_star."""
    lg = np.log2(np.asarray(grid, float))
    slope = (frac * c0 - c0) / (math.log2(h_star) - lg[0])
    return [float(max(-1.0, min(1.0, c0 + slope * (x - lg[0])))) for x in lg]


def synthetic_receipt(model: str, compare_curve: Sequence[float], recall_curve: Sequence[float], n: int,
                      seed: int, grid: Sequence[int] = GRID, arch_family: Optional[str] = None,
                      constant_compare: Optional[bool] = None, constant_recall: Optional[int] = None,
                      controls: Optional[dict] = None, n_resamples: int = N_BOOT) -> dict:
    """A receipt dict in the module's own schema with cells of known normalised accuracy."""
    rng = np.random.default_rng(seed)
    per_item: Dict[str, Dict[str, dict]] = {"compare": {}, "recall": {}}
    for d, cc, rc in zip(grid, compare_curve, recall_curve):
        per_item["compare"][str(d)] = synthetic_cell("compare", cc, n, rng, constant_compare)
        per_item["recall"][str(d)] = synthetic_cell("recall", rc, n, rng, constant_recall)
    rec = {"schema": RECEIPT_SCHEMA, "model": model,
           "config": {"distances": list(grid), "D0": int(grid[0]), "screen_bar": SCREEN_BAR,
                      "bootstrap": {"n_resamples": n_resamples, "seed": SEED_BOOT}, "pairing": "items",
                      "d0_split": True, "notes": "synthetic bits; no model was run"},
           "per_item": per_item}
    if arch_family:
        rec["arch_family"] = {"family": arch_family, "fact": {"synthetic": True}}
    if controls:
        rec["controls"] = {}
        for name, spec in controls.items():
            rec["controls"][name] = {}
            for fam, by_d in spec.items():
                rec["controls"][name][fam] = {str(d): synthetic_cell(fam, c, n if cnt is None else cnt, rng)
                                              for d, (c, cnt) in by_d.items()}
    return rec


_SELFTEST_PLANTS = (32.0, 128.0, 203.0, 600.0)
_SELFTEST_TOL = (0.8, 1.25)


def selftest(n: int = 2400, n_resamples: int = 400, seed: int = SEED_BOOT) -> dict:
    """The estimator's positive control on synthetic bits. Refused datasheets cite its checks.

    * planted crossings at {32, 128, 203, 600} are recovered within [0.8, 1.25] x the plant, the
      interval covers the plant, and the recovered order is the planted order;
    * a curve with no crossing is CENSORED with f_cens above 0.975;
    * a constant responder under exact balance is exactly 0 at every distance in both families and
      reads SCREEN_FAIL with the guard in the reason;
    * compare crossing with recall censored reads COMPARE_ONLY_HORIZON;
    * two fits under one seed agree to the last digit.

    The design's illustration planted "128" with c = (0.9, 0.7, 0.4, 0.2); under the fitter's own rule
    those four values cross at about 203, so the plants here are built FROM the rule and the planted
    number and the fitter's model agree by construction.
    """
    checks: List[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    recovered = []
    for k, h in enumerate(_SELFTEST_PLANTS):
        rec = load_receipt(synthetic_receipt(f"plant-{int(h)}", _planted_curve(h), [0.9, 0.88, 0.86, 0.84],
                                             n, seed + k, n_resamples=n_resamples))
        row = fit_model(rec)
        hz = row["horizon"]["compare"]
        pt = hz["point"]
        ok = (hz["status"] == "FINITE" and pt is not None and _SELFTEST_TOL[0] * h <= pt <= _SELFTEST_TOL[1] * h
              and hz["ci_lo"] is not None and hz["ci_lo"] <= h <= (hz["ci_hi"] if hz["ci_hi"] is not None else float("inf")))
        recovered.append(pt if pt is not None else float("nan"))
        check(f"planted_{int(h)}", ok,
              f"planted {h:g}, recovered {pt if pt is None else round(pt, 1)} [{hz['ci_lo'] and round(hz['ci_lo'], 1)}, "
              f"{hz['ci_hi'] and round(hz['ci_hi'], 1)}] f_cens {hz['f_cens']}, status {hz['status']}")
    check("planted_order", all(np.isfinite(recovered)) and all(np.diff(recovered) > 0),
          f"recovered in planted order: {[None if not np.isfinite(x) else round(x, 1) for x in recovered]}")

    rec = load_receipt(synthetic_receipt("no-crossing", [0.9, 0.85, 0.8, 0.75], [0.9, 0.88, 0.86, 0.84], n, seed + 10,
                                         n_resamples=n_resamples))
    row = fit_model(rec)
    hz = row["horizon"]["compare"]
    check("censored", hz["status"] == "CENSORED" and hz["f_cens"] is not None and hz["f_cens"] > 0.975
          and "NO_HORIZON_INSIDE_RANGE" in row["flags"],
          f"status {hz['status']}, f_cens {hz['f_cens']}, flags {row['flags'][:2]}")

    rec = load_receipt(synthetic_receipt("constant", [0, 0, 0, 0], [0, 0, 0, 0], n, seed + 20,
                                         constant_compare=True, constant_recall=0, n_resamples=n_resamples))
    row = fit_model(rec)
    zero = all(x == 0.0 for x in row["curves"]["compare"]) and all(x == 0.0 for x in row["curves"]["recall"])
    scr = row["screen"]
    check("constant_responder", zero and not scr["pass"] and any("guard failed" in r for r in scr["reasons"])
          and row["flags"][0] == "SCREEN_FAIL" and row["horizon"]["compare"]["status"] == "NOT_ESTIMABLE",
          f"curves {row['curves']}, screen {scr['pass']}, reasons {scr['reasons']}")

    rec = load_receipt(synthetic_receipt("dissoc", _planted_curve(128.0), [0.9, 0.88, 0.86, 0.84], n, seed + 30,
                                         n_resamples=n_resamples))
    row = fit_model(rec)
    check("compare_only_flag", "COMPARE_ONLY_HORIZON" in row["flags"] and row["horizon"]["recall"]["status"] == "CENSORED",
          f"flags {row['flags']}")

    rec = load_receipt(synthetic_receipt("determinism", _planted_curve(203.0), [0.9, 0.88, 0.86, 0.84], n, seed + 40,
                                         n_resamples=n_resamples))
    a, b = fit_model(rec), fit_model(rec)
    same = (a["horizon"]["compare"]["point"] == b["horizon"]["compare"]["point"]
            and a["horizon"]["compare"]["ci_lo"] == b["horizon"]["compare"]["ci_lo"]
            and a["horizon"]["compare"]["ci_hi"] == b["horizon"]["compare"]["ci_hi"]
            and a["dissociation"] == b["dissociation"])
    check("deterministic_under_seed", same, "two fits under one seed agree")

    return {"pass": all(c["ok"] for c in checks), "checks": checks, "n": n, "n_resamples": n_resamples,
            "seed": seed, "plants": list(_SELFTEST_PLANTS), "tolerance": list(_SELFTEST_TOL)}


# ---------------------------------------------------------------------------------------------------
# the datasheet
# ---------------------------------------------------------------------------------------------------

LIMITS = (
    "this sheet describes curves under one frozen probe, one filler corpus and one few-shot format; it "
    "licenses no sentence about rotation, honesty, causation or ranking",
    "a horizon is read off a four-point grid: the bracketing grid pair is the measurement, the "
    "interpolated point is a convention, and the interval carries sampling noise only",
    "SCREEN_FAIL reads: the model did not perform this format at D0; not 'cannot compare'",
    "'> Dmax' means no crossing inside the tested range, which ends at the last grid point and is under "
    "1% of context for some models; the sheet says nothing about the rest of the context",
    "a chance-normalised 0 is a constant responder or chance, not an absence of knowledge; the guard "
    "and the AUROC per cell are in the receipt",
    "FORMAT_DISTANCE_NOT_EXCLUDED: compare is scored at a fixed threshold with exemplars at distance "
    ">= D; without a near-fact control a compare-only drop may be format decay, not binding",
    "nominal D is in NeoX tokens; the native token count per cell is printed where the receipt records it",
)


def _num(x, width: int = 6) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--".rjust(width)
    s = f"{x:+.2f}".replace("+0.", "+.").replace("-0.", "-.")
    return s.rjust(width)


def _h_text(h: dict, native: bool = True) -> str:
    st = h.get("status")
    if st == "NOT_RECORDED":
        return "not recorded"
    if st == "NOT_ESTIMABLE":
        return "not estimable"
    gmax = h.get("grid_max", GRID[-1])
    if st == "CENSORED" or h.get("print_censored"):
        s = f"> {gmax} (none inside the tested range"
        if h.get("f_cens") is not None:
            s += f"; f_cens {h['f_cens']:.2f}"
        return s + ")"
    br = h.get("bracket")
    s = f"{br[0]}..{br[1]} | ~{h['point']:.0f}"
    if h.get("ci_lo") is not None:
        hi = f"> {gmax}" if h.get("ci_hi_censored") else (f"{h['ci_hi']:.0f}" if h.get("ci_hi") is not None else "--")
        s += f" [{h['ci_lo']:.0f}, {hi}]"
        if h.get("f_cens"):
            s += f" f_cens {h['f_cens']:.2f}"
    if native and h.get("native_point") is not None:
        s += f" | ~{h['native_point']:.0f} native"
    return s


def _row_lines(row: dict) -> List[str]:
    grid = row["config"]["grid"]
    cfg = row["config"]
    fam = (row["arch_family"] or {}).get("family") or "--"
    params = cfg.get("params") or "--"
    dtype = cfg.get("dtype") or "--"
    ctx = cfg.get("ctx")
    ctx_s = ("--" if ctx is None else str(ctx)) + ("*" if cfg.get("ctx_is_training_length") else "")
    scr = "PASS" if row["screen"]["pass"] else "FAIL"
    copied = f" [{cfg['copied_from']}]" if cfg.get("copied_from") else ""
    head = f"{row['model'][:22]:<22} {fam:<6} {str(params)[:7]:<7} {str(dtype)[:6]:<6} {ctx_s:<8} {scr:<6}"
    pad = " " * len(head)
    lines = []
    cc = "".join(_num(row['curves']['compare'][j]) for j in range(len(grid)))
    rc = "".join(_num(row['curves']['recall'][j]) for j in range(len(grid)))
    ac = "".join(_num(row['curve_auc'][j]) for j in range(len(grid)))
    lines.append(f"{head} compare {cc}   H_compare: {_h_text(row['horizon']['compare'])}")
    lines.append(f"{pad} c_auc   {ac}   H_c_auc:   {_h_text(row['horizon']['compare_auc'], native=False)}")
    lines.append(f"{pad} recall  {rc}   H_recall:  {_h_text(row['horizon']['recall'])}")
    rm = row["ratio_min"]
    ds = row["dissociation"]

    def iv(d):
        if d.get("point") is None:
            return "not estimable"
        lo = "--" if d.get("ci_lo") is None else f"{d['ci_lo']:+.2f}"
        hi = "--" if d.get("ci_hi") is None else f"{d['ci_hi']:+.2f}"
        return f"{d['point']:+.2f} [{lo}, {hi}]"

    lines.append(f"{pad} ratio_min compare {iv(rm['compare'])}   recall {iv(rm['recall'])}   "
                 f"compare - recall {iv(ds)}")
    lines.append(f"{pad} flags: {' '.join(row['flags'])}{copied}")
    ctrl = row["controls"]
    bits = []
    if ctrl.get("ablated"):
        bits.append(f"ablated max {ctrl['ablated']['max_norm_acc']:+.2f} (bar {ABLATED_BAR})")
    if ctrl.get("near"):
        r = ctrl["near"]["ratio_to_D0"]
        bits.append(f"near@{ctrl['near']['distance']} {ctrl['near']['norm_acc']:+.2f}"
                    + ("" if r is None else f" ({r:.2f} of D0)"))
    if ctrl.get("overwrite"):
        bits.append(f"overwrite@{ctrl['overwrite']['distance']} {ctrl['overwrite']['norm_acc']:+.2f}")
    if bits:
        lines.append(f"{pad} controls: {'; '.join(bits)}")
    if row.get("native_D"):
        nat = "; ".join(f"{d}->{v.get('min', '?')}/{v.get('mean', '?')}/{v.get('max', '?')}"
                        for d, v in sorted(row["native_D"].items()) if isinstance(v, dict))
        lines.append(f"{pad} native D (min/mean/max): {nat}")
    lines.append(f"{pad} {row['split_note']}; pairing: {row['pairing']}; B = {row['n_resamples']}, seed {row['seed']}")
    return lines


def datasheet(sheet: dict, selftest_result: Optional[dict] = None, prereg_verdict: Optional[dict] = None) -> str:
    """The fixed-width text block. Runs :func:`selftest` unless a result is passed; REFUSES to print a
    datasheet when the self-test fails, because a fitter that cannot recover a planted crossing has no
    business printing a horizon."""
    st = selftest_result if selftest_result is not None else selftest()
    if not st.get("pass"):
        failed = [c["name"] + ": " + c["detail"] for c in st.get("checks", []) if not c.get("ok")]
        raise HorizonRefusal("the estimator self-test failed; no datasheet is printed. " + " | ".join(failed))
    rows = sheet["rows"] if "rows" in sheet else [sheet]
    grid = rows[0]["config"]["grid"]
    items = {r["config"].get("items_sha256") for r in rows}
    items_s = next(iter(items)) if len(items) == 1 and next(iter(items)) else "per receipt"
    gates = prereg_verdict.get("gates_sha256") if prereg_verdict else "none (unscored)"
    n_plants = sum(1 for c in st["checks"] if c["name"].startswith("planted_") and c["name"] != "planted_order")
    out = [f"styxx.horizon datasheet      items: {items_s}      gates sha256: {gates}      "
           f"selftest: PASS ({n_plants} planted horizons recovered, {len(st['checks'])} checks)",
           f"{'model':<22} {'family':<6} {'params':<7} {'dtype':<6} {'ctx':<8} {'screen':<6} {'op':<7} "
           + "".join(f"{d:>6}" for d in grid) + "   horizon (bracket | point [interval] | native)"]
    for r in rows:
        out.extend(_row_lines(r))
        out.append("")
    res = sheet.get("result")
    if res:
        mind = res.get("attention_family_min_dissoc")
        ci = res.get("attention_family_min_dissoc_ci") or {}
        ci_s = ("--" if ci.get("lo") is None else f"[{ci['lo']:+.2f}, {ci['hi']:+.2f}]")
        out.append(f"{res['family']} family set ({res['n_attention_screen_pass']}): {', '.join(res['family_set']) or 'none'}"
                   f"   excluded: {', '.join(res['family_excluded']) or 'none'}")
        out.append(f"family_set_ablated_max_norm_acc: "
                   f"{'--' if res.get('family_set_ablated_max_norm_acc') is None else format(res['family_set_ablated_max_norm_acc'], '+.3f')}"
                   f" ({res['family_set_ablated_note']})   min over the set of (ratio_min compare - ratio_min recall): "
                   f"{'--' if mind is None else format(mind, '+.3f')} {ci_s}"
                   + ("" if res.get("shared_resample") else "   (not paired across models)"))
        out.append("verdict: " + (f"{prereg_verdict['verdict']} (styxx.protocol, prereg {prereg_verdict['prereg_commit'][:8]})"
                                  if prereg_verdict else "unscored: no committed prereg"))
        out.append(f"reported numbers on this sheet: {len(rows) * 3} horizons and {len(rows)} dissociation intervals "
                   f"beside the one family number")
    out.append("")
    out.append("each cell: chance-normalised accuracy; 95% paired item-bootstrap intervals, guard and AUROC per cell in the "
               "receipt; * = trained context, no positional limit")
    out.extend("limit: " + l for l in LIMITS)
    text = "\n".join(out)
    for w in _HOUSE_WORDS:
        if re.search(r"\b" + re.escape(w) + r"s?\b", text, re.I):
            raise HorizonRefusal(f"the datasheet would print a word the lab never prints: {w!r}")
    return text


# ---------------------------------------------------------------------------------------------------
# JSON and CLI
# ---------------------------------------------------------------------------------------------------

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def _write_json(path: str, obj) -> None:
    data = json.dumps(_jsonable(obj), indent=1, ensure_ascii=False).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)
    with open(path, "rb") as fh:
        if fh.read() != data:
            raise HorizonRefusal(f"wrote {path} and read back different bytes")


def _score_prereg(prereg: str, result: dict) -> dict:
    from .protocol import Experiment
    exp = Experiment(prereg)
    v = exp.score(result)
    return {"verdict": v.verdict, "gates": v.gates, "gates_sha256": v.gates_sha256, "prereg_commit": v.prereg_commit}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.horizon",
                                 description="The horizon datasheet: per-item bits in, horizons with censored intervals out. No GPU.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest", help="the planted-crossing positive control of the estimator")
    p.add_argument("--json", default=None, help="write the self-test record here")
    p.add_argument("--n", type=int, default=2400)
    p.add_argument("--n-resamples", type=int, default=400)
    p = sub.add_parser("fit", help="one model's row from a receipt of per-item bits")
    p.add_argument("--receipt", required=True)
    p.add_argument("--arm", default=None, help="read per_item[arm] of a bridge-v3 receipt")
    p.add_argument("--n-resamples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", default=None, help="write the row here")
    p = sub.add_parser("sheet", help="several receipts on one shared item resample, with the family aggregates")
    p.add_argument("--receipts", nargs="+", required=True, help="receipt files, or directories of horizon_*.json")
    p.add_argument("--exclude", nargs="*", default=[], help="model names printed but never counted (v3's arms)")
    p.add_argument("--family", default="ATTN", choices=["ATTN", "DECAY"])
    p.add_argument("--n-resamples", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED_BOOT)
    p.add_argument("--prereg", default=None, help="a COMMITTED prereg; the sheet is then scored through styxx.protocol")
    p.add_argument("--json", default=None, help="write the sheet here")
    p = sub.add_parser("family", help="DECAY or ATTN from a config.json and, for mamba, its A_log tensors")
    p.add_argument("--config", required=True)
    p.add_argument("--safetensors", nargs="*", default=[], help="safetensors files (only *.A_log is read)")
    p.add_argument("--required-reach", type=int, default=REQUIRED_REACH)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "selftest":
            st = selftest(n=a.n, n_resamples=a.n_resamples)
            for c in st["checks"]:
                print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
            print("estimator self-test:", "PASS" if st["pass"] else "FAIL")
            if a.json:
                _write_json(a.json, st)
            return 0 if st["pass"] else 1
        if a.cmd == "fit":
            rec = load_receipt(a.receipt, arm=a.arm)
            row = fit_model(rec, n_resamples=a.n_resamples, seed=a.seed)
            st = selftest()
            print(datasheet(row, selftest_result=st))
            if a.json:
                row["estimator_selftest"] = st
                _write_json(a.json, row)
            return 0
        if a.cmd == "sheet":
            paths: List[str] = []
            for r in a.receipts:
                if os.path.isdir(r):
                    paths.extend(sorted(glob.glob(os.path.join(r, "horizon_*.json"))))
                else:
                    paths.append(r)
            if not paths:
                raise HorizonRefusal("no receipts found")
            recs = [load_receipt(p) for p in paths]
            sheet = fit_zoo(recs, exclude=a.exclude, n_resamples=a.n_resamples, seed=a.seed, family=a.family)
            st = selftest()
            verdict = _score_prereg(a.prereg, {"result": sheet["result"]}) if a.prereg else None
            print(datasheet(sheet, selftest_result=st, prereg_verdict=verdict))
            if a.json:
                sheet["estimator_selftest"] = st
                sheet["verdict"] = verdict
                _write_json(a.json, sheet)
            return 0
        if a.cmd == "family":
            A = None
            if a.safetensors:
                files: List[str] = []
                for pat in a.safetensors:
                    files.extend(glob.glob(pat) or [pat])
                A = mamba_A_from_safetensors(files)["A"]
            fact = assert_family(a.config, A=A, required_reach=a.required_reach)
            print(json.dumps(_jsonable(fact), indent=1))
            return 0
    except HorizonRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
