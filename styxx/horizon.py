# -*- coding: utf-8 -*-
"""styxx.horizon — the horizon datasheet: where, if anywhere inside the tested range, a causal LM stops
performing a judgement over distant planted facts while it still recalls the facts themselves.

Built from `next/horizon-zoo/DESIGN.md` (2026-10-11) with the fatal and major fixes of its red team
(`ATTACK.md`) that live in the instrument applied; the fixes that live in the GPU runner and the item
generator (F1a, F1b, M5's items, M6, M7, M8) are not in this module and are named as its dependencies. It
is the instrument only: a pure function from per-item bits to a datasheet. No torch, no GPU, no model
load, no network. The GPU runner that produces the bits is not here; the zoo run that would fill nine rows
is not here; nothing in this module changes what any committed RESULT says.

WHAT IT TAKES. Per-item bits for one model, in cells keyed by *family* (the probe's word for its two
operations, as `run_realmodel_bridge_v3.py` uses it: ``compare`` = a judgement over the planted codes,
``recall`` = produce one planted code) and *distance* (nominal NeoX tokens of filler between the facts and
the question, grid {16, 64, 256, 1024}). Each family is scored under a *readout*, which fixes the answer
set and with it chance (1/k under exact balance) and the degeneracy guard (:data:`READOUTS`):

  * ``argmax_largest`` (compare; bridge-v3 amendment 17.2): argmax over the three name tokens, correct iff
    it is the slot holding the largest planted code; a cell carries ``target_slot``, ``pred_slot`` and,
    when recorded, ``scores`` (three per item) and ``margin`` — chance 1/3, guard: largest slot share
    <= 0.80;
  * ``yes_no`` / ``true_false`` (compare; the probe of the design's 4.1, retired for bridge v3 by the same
    amendment): ``label`` (the claim is true), ``prediction`` (the model said yes) and ``logodds`` —
    chance 1/2, guard: yes-rate within [0.20, 0.80];
  * ``argmax_code`` (recall): ``target_slot`` / ``label`` (the queried slot) and ``pred_slot`` /
    ``prediction`` (the argmax slot), with ``scores`` when recorded — chance 1/3, guard: largest slot share
    <= 0.80.

Readout, chance and guard are facts of the receipt (``config.readout`` or ``config.readouts``,
``config.chance``, ``config.guard``), printed on every row; an undeclared readout is inferred from the
cell's own keys and printed as inferred, and a compare cell whose keys do not decide it is refused rather
than given a chance. A declared chance that is not 1/k of its readout is refused. A cell may instead be a
plain list of correctness bits under a declared readout; the guard, the AUROC and the threshold-free curve
are then "not recorded" and the sheet says so. The bridge-v3 runner's own receipt (``per_item[arm][family]
[D][variant]`` with variants plain / ablated / windowed / near / overwrite) is read with ``arm=``: the plain
cell is the curve, the others are the controls. The *architecture* family (DECAY / ATTN) is a different
word: it is asserted from a config and, for a state-space model, from its A values — never from a name
(:func:`assert_family`).

WHAT IT COMPUTES, per model.
  * chance-normalised accuracy per cell, ``a' = (a - chance) / (1 - chance)``; a constant responder under
    exact balance maps to exactly 0 in either readout. Exact balance is checked per cell and per D0 half:
    a cell out of balance reads BALANCE_BROKEN, and a screen half out of balance fails the screen;
  * a 95% paired item-bootstrap interval per cell. Under ``pairing = items`` (items prefix-nested across
    distances) one index resample per replicate is applied to every cell; under ``distance`` (the bridge-v3
    runner's items: shared across the two families at one distance, drawn afresh per distance) one
    resample per distance is shared by the two families; under ``independent`` every cell draws its own.
    Models fitted together on one seed share the same resamples (the red team's M9). A receipt that
    declares nothing reads ``items (assumed, undeclared)`` and the flag PAIRING_ASSUMED;
  * the degeneracy guard per cell and the rank AUROC of the per-item scores (compare log-odds against the
    label; for a slot readout the one-versus-rest AUROC averaged over the slots, the runner's statistic);
    0.5 for any constant responder;
  * the capability screen at D0: both normalised accuracies >= the bar and both guards passed; the screen
    reads one half of the D0 items and the horizon's denominator the other half (M4), so that selecting on
    c(D0) does not bias the ratios of a model near the bar. The bar is printed on the row and a bar other
    than 0.40 reads NONSTANDARD_BAR. A model that fails the screen keeps its curves and its D0 numbers and
    gets no horizon, no ratio and no dissociation: it has no positive control (DESIGN 4.4, 4.7);
  * the comparison horizon: with ``tau = frac * c(D0)`` (frac 0.5, strictly between 0 and 1), the
    smallest D >= D0 at which the piecewise-linear interpolant of c in log2 D reaches tau. The bracketing
    grid pair is the measurement; the interpolated point is a convention and is printed second. No crossing
    inside the grid is CENSORED ("> 1024 (none inside the tested range)"); c(D0) <= 0 is NOT_ESTIMABLE. The
    interval is the 2.5th / 97.5th percentile of the finite replicates with the censored fraction
    ``f_cens`` beside it: above 0.025 the upper end prints "> 1024", above 0.5 the point prints censored.
    The recall horizon by the same rule; the threshold-free compare curve ``c_auc = 2 * AUROC - 1`` by the
    same rule (F1c) — a horizon in c that is absent from c_auc is flagged CALIBRATION_ARTEFACT_SUSPECTED,
    because a drifting prior on one answer moves an accuracy and leaves a rank statistic alone;
  * ``ratio_min`` = min over D > D0 of c(D) / c(D0) per family, and the dissociation metric
    ``ratio_min_compare - ratio_min_recall`` with its paired interval (F2: the sentence worth gating is
    whether a model loses comparison faster than recall, not whether attention has a horizon);
  * flags: SCREEN_FAIL, COMPARE_ONLY_HORIZON, RECALL_ONLY_HORIZON, STORAGE_HORIZON,
    NO_HORIZON_INSIDE_RANGE, DEGENERATE cells (printed with both c and c_auc), NONMONOTONE, BALANCE_BROKEN,
    NONSTANDARD_BAR, NONSTANDARD_GUARD, PAIRING_ASSUMED, SCORES_ABSENT, and the control flags below;
  * controls, when the receipt carries them: ``ablated`` (codes removed; must stay at chance — above 0.16
    normalised in any cell reads PROBE_READS_PRIORS), ``near`` (the codes 16 tokens before the question at
    the far distance; its ratio to c(D0) below 0.5 reads FORMAT_DECAY, below 0.8 FORMAT_DRIFT),
    ``overwrite`` (the far filler re-plants one name; compare against the original answer must fall to
    chance or the model is not reading the far binding: POSITIVE_CONTROL_SILENT), ``windowed`` (reported).
    A receipt without ``near`` reads FORMAT_DISTANCE_NOT_EXCLUDED; one without ``overwrite`` reads
    FAR_BINDING_CONTROL_ABSENT.

WHAT IT DOES NOT SAY. A horizon is a description of one curve under one frozen probe, one filler corpus,
one few-shot format and one readout. It licenses no sentence about rotation, honesty, causation or
ranking; every model differs from every other in more than its recurrence. SCREEN_FAIL means the model did
not perform this format at D0, not that it cannot compare. The distance axis is exact only for a NeoX
tokenizer; for others the receipt records the native token count per cell and the sheet prints the
horizon both ways. The tested range ends at the last grid point, which is under 1% of context for some
models: the word is "inside the tested range", never "inside context". The self-test (:func:`selftest`)
is a software test on synthetic bits; it shows the fitter recovers a planted crossing, a planted level, a
nominal interval and the D0 split it claims, not that any model has a horizon.

CLI::

    python -m styxx.horizon selftest
    python -m styxx.horizon fit   --receipt horizon_<model>.json            # one model, no GPU
    python -m styxx.horizon fit   --receipt <bridge-v3 receipt>.json --arm M1
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
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "READOUTS", "GRID", "D0", "SCREEN_BAR", "GUARD_YES_RANGE", "GUARD_SLOT_MAX", "ABLATED_BAR",
    "HorizonRefusal", "normalize", "chance_of", "guard", "auroc", "ovr_auroc", "cell_summary", "crossing",
    "fit_horizon", "fit_model", "fit_zoo", "screen", "assert_family", "mamba_A_from_safetensors", "selftest",
    "datasheet", "synthetic_cell", "synthetic_receipt", "load_receipt", "main",
]

RECEIPT_SCHEMA = "styxx.horizon/receipt/0.1"
FAMILIES = ("compare", "recall")
# A readout fixes the answer set, so it fixes chance (1/k under exact balance) and which guard applies.
READOUTS = {
    "argmax_largest": {"family": "compare", "kind": "slots", "k": 3,
                       "what": "argmax over the three name tokens; correct iff the slot holding the largest "
                               "planted code (bridge-v3 amendment 17.2)"},
    "yes_no": {"family": "compare", "kind": "binary", "k": 2,
               "what": "the sign of log p(Yes) - log p(No) on a claim (retired for bridge v3 by its amendment 17.2)"},
    "true_false": {"family": "compare", "kind": "binary", "k": 2,
                   "what": "the sign of log p(True) - log p(False) on a claim (retired for bridge v3 by its "
                           "amendment 17.2)"},
    "argmax_code": {"family": "recall", "kind": "slots", "k": 3,
                    "what": "argmax of the three candidate codes' summed log-likelihood"},
}
DEFAULT_READOUT = {"compare": "argmax_largest", "recall": "argmax_code"}   # what synthetic bits are built under
GRID = (16, 64, 256, 1024)
D0 = 16
SCREEN_BAR = 0.40
GUARD_YES_RANGE = (0.20, 0.80)
GUARD_SLOT_MAX = 0.80
ABLATED_BAR = 0.16
FRAC = 0.5
N_BOOT = 2000
SEED_BOOT = 20261011
LOGODDS_TIE = 1e-3            # |log-odds| or |margin| below this is a near tie (the runner's MARGIN_TIE)
NEAR_FORMAT_DECAY = 0.5       # c_near(D1) / c(D0) below this: the format itself decays with distance
NEAR_FORMAT_DRIFT = 0.8       # below this: drift reported, horizon read with the caveat printed
OVERWRITE_FIRED_MAX = 0.16    # overwrite control: compare against the original answer must sit at chance
NONMONOTONE_RISE = 0.05       # an adjacent rise above this normalised amount flags NONMONOTONE
VARIANTS = ("plain", "ablated", "windowed", "near", "overwrite")   # the bridge-v3 runner's per-item variants
CONTROLS = ("ablated", "near", "overwrite", "windowed")
PAIRINGS = ("items", "distance", "independent")
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


def chance_of(readout: str) -> float:
    """Chance of a readout under exact balance: 1/k for an answer set of k."""
    if readout not in READOUTS:
        raise HorizonRefusal(f"unknown readout {readout!r}; the instrument scores {sorted(READOUTS)}")
    return 1.0 / READOUTS[readout]["k"]


def guard(yes_rate: Optional[float] = None, max_slot_frac: Optional[float] = None,
          yes_range: Sequence[float] = GUARD_YES_RANGE, slot_max: float = GUARD_SLOT_MAX) -> Optional[bool]:
    """The degeneracy guard of one cell; None when the cell recorded neither statistic."""
    if yes_rate is not None:
        return bool(yes_range[0] <= float(yes_rate) <= yes_range[1])
    if max_slot_frac is not None:
        return bool(float(max_slot_frac) <= slot_max)
    return None


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank AUROC with average ranks on ties; 0.5 for any constant score; None if a class is empty."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    return float(_auroc_from_ranks(_rank_avg(s), y, n_pos, n_neg))


def ovr_auroc(scores, targets, k: int = 3) -> Optional[float]:
    """One-versus-rest AUROC averaged over the k slots: slot j's score against "slot j is the answer" (the
    bridge-v3 runner's statistic). 0.5 for any constant responder; None if some slot is never or always the
    answer. Adding a constant to one slot's scores — a drifting prior on one name — leaves it unchanged."""
    s = np.asarray(scores, dtype=float)
    t = np.asarray(targets, dtype=int)
    if s.ndim != 2 or s.shape[1] != k or len(s) != len(t):
        raise HorizonRefusal(f"ovr_auroc: scores of shape {s.shape} for {len(t)} targets and k = {k}")
    vals = [auroc(s[:, j], t == j) for j in range(k)]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals))


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


def _rep_auroc(scores: np.ndarray, pos: np.ndarray, idx: np.ndarray,
               keep_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """:func:`auroc` of ``scores`` against ``pos`` on every replicate of the index resample ``idx`` (B, m),
    in one vectorised pass and with the same tie rule: the scores become exact integer codes (ties kept),
    each replicate's codes are counted, and an item's average rank is the count below its code plus half
    the count at it. With ``keep_mask`` only kept items count (dropped ones are coded above every kept one
    and never summed). NaN where a class is empty in a replicate."""
    s = np.asarray(scores, dtype=float).ravel()
    y_all = np.asarray(pos, dtype=bool).ravel()
    _, codes = np.unique(s, return_inverse=True)
    codes = codes.astype(np.int64).ravel()
    K = int(codes.max()) + 1 if len(codes) else 1
    u = codes[idx]
    y = y_all[idx]
    if keep_mask is not None:
        kept = np.asarray(keep_mask, dtype=bool)[idx]
        u = np.where(kept, u, K)
        y = y & kept
    B = u.shape[0]
    flat = (u + (np.arange(B, dtype=np.int64) * (K + 1))[:, None]).ravel()
    cnt = np.bincount(flat, minlength=B * (K + 1)).reshape(B, K + 1)[:, :K].astype(float)
    pos_cnt = np.bincount(flat[y.ravel()], minlength=B * (K + 1)).reshape(B, K + 1)[:, :K].astype(float)
    below = np.cumsum(cnt, axis=1) - cnt
    rank_sum = (pos_cnt * (below + (cnt + 1.0) / 2.0)).sum(axis=1)
    n_pos = pos_cnt.sum(axis=1)
    n_neg = cnt.sum(axis=1) - n_pos
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    a[(n_pos == 0) | (n_neg == 0)] = np.nan
    return a


def _margins(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Score of the answer slot minus the best other slot (the runner's ``margin``)."""
    n = len(targets)
    own = scores[np.arange(n), targets]
    others = scores.copy()
    others[np.arange(n), targets] = -np.inf
    return own - others.max(axis=1)


def _as_bool_array(values, name: str) -> np.ndarray:
    arr = np.asarray(list(values))
    if arr.dtype == object or arr.ndim != 1:
        raise HorizonRefusal(f"{name}: expected a flat list of booleans or 0/1 integers")
    if arr.dtype != bool:
        if not np.all(np.isin(arr, (0, 1))):
            raise HorizonRefusal(f"{name}: values must be booleans or 0/1 integers")
        arr = arr.astype(bool)
    return arr


def _slot_words(k: int) -> str:
    return ", ".join(str(j) for j in range(k - 1)) + f" or {k - 1}"


def _as_slot_array(values, k: int, name: str) -> np.ndarray:
    arr = np.asarray(list(values))
    if arr.ndim != 1 or arr.dtype == object or arr.dtype == bool:
        raise HorizonRefusal(f"{name}: expected a flat list of slot indices, got "
                             f"{'booleans' if arr.dtype == bool else 'something else'}")
    if not np.all(np.isin(arr, np.arange(k))):
        raise HorizonRefusal(f"{name}: slots must be {_slot_words(k)}")
    return arr.astype(int)


def _guard_spec(raw=None, where: str = "config.guard") -> dict:
    """The guard rules a receipt declares (``{"yes_rate": [lo, hi], "max_slot_fraction": m}``), the
    bridge-v3 runner's values where it declares none. Values other than those are validated, kept and
    flagged NONSTANDARD_GUARD on the row."""
    g = {"yes_rate": (float(GUARD_YES_RANGE[0]), float(GUARD_YES_RANGE[1])), "max_slot_fraction": float(GUARD_SLOT_MAX)}
    if raw is None:
        return {**g, "nonstandard": False}
    if not isinstance(raw, dict):
        raise HorizonRefusal(f"{where}: expected a dict with 'yes_rate' and/or 'max_slot_fraction'")
    yr = raw.get("yes_rate", raw.get("yes_rate_range"))
    if yr is not None:
        try:
            lo, hi = (float(v) for v in yr)
        except (TypeError, ValueError):
            raise HorizonRefusal(f"{where}.yes_rate must be two numbers [lo, hi]") from None
        if not (0.0 <= lo < hi <= 1.0):
            raise HorizonRefusal(f"{where}.yes_rate must satisfy 0 <= lo < hi <= 1, got [{lo}, {hi}]")
        g["yes_rate"] = (lo, hi)
    ms = raw.get("max_slot_fraction")
    if ms is not None:
        try:
            ms = float(ms)
        except (TypeError, ValueError):
            raise HorizonRefusal(f"{where}.max_slot_fraction must be a number") from None
        if not (1.0 / 3.0 < ms <= 1.0):
            raise HorizonRefusal(f"{where}.max_slot_fraction must lie in (1/3, 1], got {ms}")
        g["max_slot_fraction"] = ms
    nonstd = (g["yes_rate"] != (float(GUARD_YES_RANGE[0]), float(GUARD_YES_RANGE[1]))
              or g["max_slot_fraction"] != float(GUARD_SLOT_MAX))
    return {**g, "nonstandard": bool(nonstd)}


def _guard_words(readout: str, gs: dict) -> str:
    if READOUTS[readout]["kind"] == "binary":
        return f"yes-rate within [{gs['yes_rate'][0]:.2f}, {gs['yes_rate'][1]:.2f}]"
    return f"largest slot share <= {gs['max_slot_fraction']:.2f}"


def _resolve_readout(family: str, readout: Optional[str], labels=None, logodds=None, scores=None) -> Tuple[str, str]:
    """(readout, source). Declared wins; else the recall readout is the only one there is; else a compare
    cell's own keys decide (three-column scores or a slot index 2: argmax; log-odds or boolean labels:
    yes/no); else the cell is refused, because chance cannot be assumed."""
    if readout is not None:
        if readout not in READOUTS:
            raise HorizonRefusal(f"unknown readout {readout!r}; the instrument scores {sorted(READOUTS)}")
        if READOUTS[readout]["family"] != family:
            raise HorizonRefusal(f"readout {readout!r} is a {READOUTS[readout]['family']} readout, not a {family} one")
        return readout, "declared"
    if family == "recall":
        return "argmax_code", "the one recall readout"
    if scores is not None and np.ndim(np.asarray(scores, dtype=object)) == 2:
        return "argmax_largest", "inferred from the cell's keys (three scores per item)"
    if logodds is not None:
        return "yes_no", "inferred from the cell's keys (log-odds)"
    if labels is not None:
        arr = np.asarray(list(labels))
        if arr.dtype == bool:
            return "yes_no", "inferred from the cell's keys (boolean labels)"
        if arr.ndim == 1 and arr.dtype != object and arr.size and np.any(arr == 2):
            return "argmax_largest", "inferred from the cell's keys (slot labels)"
    raise HorizonRefusal("compare cell: the readout is neither declared (config.readout) nor decided by the "
                         "cell's keys; chance cannot be assumed")


_STAT_KEYS = ("yes_rate", "max_slot_fraction", "slot_fractions", "auroc", "c_auc", "near_ties", "margin_mean",
              "guard_ok", "exact_balance", "label_counts")


def _cell_stats(readout: str, y: np.ndarray, p: np.ndarray, S: Optional[np.ndarray], gs: dict,
                margin: Optional[np.ndarray] = None) -> dict:
    """Guard, balance, AUROC and margins of one cell's vectors (a whole cell or one half of it)."""
    spec = READOUTS[readout]
    k = spec["k"]
    out: Dict[str, Any] = {key: None for key in _STAT_KEYS}
    n = len(y)
    if n == 0:
        return out
    if spec["kind"] == "binary":
        out["yes_rate"] = float(p.mean())
        out["guard_ok"] = guard(yes_rate=out["yes_rate"], yes_range=gs["yes_rate"])
        counts = [int((~y).sum()), int(y.sum())]
        if S is not None:
            a = auroc(S, y)
            out["auroc"], out["c_auc"] = a, (None if a is None else 2.0 * a - 1.0)
            out["near_ties"] = int((np.abs(S) < LOGODDS_TIE).sum())
    else:
        fr = [float((p == j).mean()) for j in range(k)]
        out["slot_fractions"], out["max_slot_fraction"] = fr, max(fr)
        out["guard_ok"] = guard(max_slot_frac=max(fr), slot_max=gs["max_slot_fraction"])
        counts = [int((y == j).sum()) for j in range(k)]
        if S is not None:
            a = ovr_auroc(S, y, k)
            out["auroc"], out["c_auc"] = a, (None if a is None else 2.0 * a - 1.0)
            margin = _margins(S, y)
        if margin is not None:
            out["near_ties"] = int((np.abs(margin) < LOGODDS_TIE).sum())
            out["margin_mean"] = float(np.mean(margin))
    out["label_counts"] = counts
    out["exact_balance"] = bool(len(set(counts)) == 1)
    return out


def cell_summary(family: str, labels=None, predictions=None, logodds=None, correct=None, *,
                 readout: Optional[str] = None, scores=None, margin=None, guard_spec: Optional[dict] = None,
                 pred_is_overwrite=None) -> dict:
    """One cell's statistics from its per-item bits, under its readout.

    ``labels`` + ``predictions`` are booleans under a yes/no readout and slot indices under a slot readout
    (``target_slot`` / ``pred_slot`` in the runner's words); ``logodds`` (yes/no) or ``scores`` (one per
    slot) feed the AUROC. Every recorded field must agree with every other: predictions with the sign of
    the log-odds or the argmax of the scores (ties to the lowest slot), ``correct`` with prediction == label,
    ``margin`` with the scores. A plain ``correct`` vector yields accuracy only; the guard and the AUROC are
    then None and the sheet prints "not recorded" rather than a number nobody measured.
    """
    if family not in FAMILIES:
        raise HorizonRefusal(f"family must be one of {FAMILIES}, got {family!r}")
    readout, source = _resolve_readout(family, readout, labels=labels, logodds=logodds, scores=scores)
    spec = READOUTS[readout]
    k = spec["k"]
    chance = 1.0 / k
    gs = guard_spec if guard_spec is not None else _guard_spec(None)
    out: Dict[str, Any] = {"family": family, "readout": readout, "readout_source": source, "chance": chance}
    y = p = S = m_arr = None
    if labels is None and predictions is None:
        if correct is None:
            raise HorizonRefusal(f"{family} cell: labels and predictions are both required (or correctness bits)")
        if scores is not None or logodds is not None:
            raise HorizonRefusal(f"{family} cell: scores without labels and predictions cannot be read")
        cor = _as_bool_array(correct, f"{family} correct")
        out.update({key: None for key in _STAT_KEYS})
        out["recorded"] = "correctness bits only"
    else:
        if labels is None or predictions is None:
            raise HorizonRefusal(f"{family} cell: labels and predictions are both required")
        if spec["kind"] == "binary":
            y = _as_bool_array(labels, f"{family} labels")
            p = _as_bool_array(predictions, f"{family} predictions")
            if len(y) != len(p):
                raise HorizonRefusal(f"{family} cell: labels and predictions differ in length")
            if scores is not None:
                raise HorizonRefusal(f"{family} cell: per-slot scores under the {readout!r} readout")
            if logodds is not None:
                S = np.asarray(list(logodds), dtype=float)
                if S.ndim != 1 or len(S) != len(y):
                    raise HorizonRefusal(f"{family} cell: logodds and labels differ in length")
                if not np.all(np.isfinite(S)):
                    raise HorizonRefusal(f"{family} cell: a log-odds value is not finite")
                # the prediction IS the sign of the log-odds (ties at exactly 0 break toward No)
                if not np.array_equal(S > 0.0, p):
                    raise HorizonRefusal(f"{family} cell: predictions disagree with the sign of the "
                                         f"log-odds; a receipt that records both must agree")
            recorded = "labels, predictions" + (", logodds" if S is not None else "")
        else:
            y = _as_slot_array(labels, k, f"{family} labels")
            p = _as_slot_array(predictions, k, f"{family} predictions")
            if len(y) != len(p):
                raise HorizonRefusal(f"{family} cell: labels and predictions must be flat and of equal length")
            if logodds is not None:
                raise HorizonRefusal(f"{family} cell: log-odds under the slot readout {readout!r}")
            if scores is not None:
                try:
                    S = np.asarray(scores, dtype=float)
                except (TypeError, ValueError):
                    raise HorizonRefusal(f"{family} cell: scores must be numbers, {k} per item") from None
                if S.ndim != 2 or S.shape != (len(y), k):
                    raise HorizonRefusal(f"{family} cell: scores have shape {S.shape}, expected ({len(y)}, {k})")
                if not np.all(np.isfinite(S)):
                    raise HorizonRefusal(f"{family} cell: a score is not finite")
                if not np.array_equal(S.argmax(axis=1), p):
                    raise HorizonRefusal(f"{family} cell: predicted slots disagree with the argmax of the scores "
                                         f"(ties to the lowest slot); a receipt that records both must agree")
            if margin is not None:
                m_arr = np.asarray(list(margin), dtype=float)
                if m_arr.ndim != 1 or len(m_arr) != len(y) or not np.all(np.isfinite(m_arr)):
                    raise HorizonRefusal(f"{family} cell: margin must be one finite number per item")
                if S is not None and not np.allclose(m_arr, _margins(S, y), rtol=0.0, atol=1e-9):
                    raise HorizonRefusal(f"{family} cell: the recorded margin disagrees with the scores")
            recorded = "slots" + (", scores" if S is not None else "") + (", margin" if m_arr is not None else "")
        cor = p == y
        if correct is not None:
            c2 = _as_bool_array(correct, f"{family} correct")
            if len(c2) != len(cor) or not np.array_equal(c2, cor):
                raise HorizonRefusal(f"{family} cell: the recorded correctness bits disagree with "
                                     f"prediction == label")
        out.update(_cell_stats(readout, y, p, S, gs, m_arr))
        out["recorded"] = recorded
    if pred_is_overwrite is not None:
        ow = _as_bool_array(pred_is_overwrite, f"{family} pred_is_overwrite")
        if len(ow) != len(cor):
            raise HorizonRefusal(f"{family} cell: pred_is_overwrite and the cell differ in length")
        out["overwrite_follow_rate"] = float(ow.mean()) if len(ow) else None
    n = int(len(cor))
    raw = float(cor.mean()) if n else float("nan")
    out.update({"n": n, "acc_raw": raw, "acc_norm": normalize(raw, chance) if n else float("nan"),
                "_correct": cor, "_labels": y, "_preds": p, "_scores": S, "_guard_spec": gs})
    return out


# ---------------------------------------------------------------------------------------------------
# the horizon estimator
# ---------------------------------------------------------------------------------------------------

def _log2_grid(grid: Sequence[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=float)
    if g.ndim != 1 or len(g) < 2 or np.any(g <= 0) or np.any(np.diff(g) <= 0):
        raise HorizonRefusal(f"the distance grid must be increasing positive integers, got {list(grid)}")
    return np.log2(g)


def _check_frac(frac) -> float:
    try:
        f = float(frac)
    except (TypeError, ValueError):
        raise HorizonRefusal(f"horizon_frac must be a number, got {frac!r}") from None
    if not (0.0 < f < 1.0):
        raise HorizonRefusal(f"horizon_frac must lie strictly between 0 and 1, got {frac!r}: the threshold is "
                             f"a fraction of c(D0)")
    return f


def crossing(curve: Sequence[float], grid: Sequence[int] = GRID, frac: float = FRAC) -> dict:
    """Where the piecewise-linear interpolant of ``curve`` in log2 D reaches ``frac * curve[0]``.

    Returns ``{"status": "FINITE"|"CENSORED"|"NOT_ESTIMABLE", "point": D or None,
    "bracket": (D_lo, D_hi) or None, "tau": float or None}``. A crossing exists iff some grid value is
    strictly below tau (the interpolant is linear between grid points, so it cannot dip below tau
    between two points that are both at or above it). The earliest crossing is used; a later rise is
    the NONMONOTONE flag's business, not this function's. ``frac`` must lie strictly between 0 and 1.
    """
    frac = _check_frac(frac)
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
    frac = _check_frac(frac)
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


def _not_estimable(grid: Sequence[int], reason: str) -> dict:
    """A horizon that is not computed, in the same shape :func:`fit_horizon` returns."""
    return {"status": "NOT_ESTIMABLE", "point": None, "bracket": None, "tau": None, "grid_max": int(grid[-1]),
            "ci_lo": None, "ci_hi": None, "f_cens": None, "f_nonest": None, "ci_hi_censored": None,
            "print_censored": True, "native_point": None, "n_resamples": 0, "reason": reason}


# ---------------------------------------------------------------------------------------------------
# receipts: the shape the fitter reads, and the shapes it accepts
# ---------------------------------------------------------------------------------------------------

_CELL_KEYS = {"label": ("label", "labels", "target_slot", "target_slots", "q_slot", "q_slots"),
              "prediction": ("prediction", "predictions", "pred", "pred_slot", "pred_slots", "pred_yes"),
              "logodds": ("logodds", "log_odds", "logit_diff"),
              "scores": ("scores",),
              "margin": ("margin", "margins"),
              "correct": ("correct", "bits", "correctness"),
              "overwrite": ("pred_is_overwrite",)}


def _pick(d: dict, kind: str):
    for k in _CELL_KEYS[kind]:
        if k in d:
            return d[k]
    return None


def _cell_from_spec(family: str, spec, where: str, readout: Optional[str] = None,
                    guard_spec: Optional[dict] = None) -> dict:
    """A cell spec is a dict of per-item vectors or a plain list of correctness bits."""
    try:
        if isinstance(spec, dict):
            v = {kind: _pick(spec, kind) for kind in _CELL_KEYS}
            if v["prediction"] is None and v["logodds"] is not None and v["label"] is not None:
                try:
                    v["prediction"] = [float(x) > 0.0 for x in v["logodds"]]
                except (TypeError, ValueError):
                    raise HorizonRefusal("logodds must be numbers") from None
            return cell_summary(family, labels=v["label"], predictions=v["prediction"], logodds=v["logodds"],
                                correct=v["correct"], readout=readout, scores=v["scores"], margin=v["margin"],
                                guard_spec=guard_spec, pred_is_overwrite=v["overwrite"])
        if isinstance(spec, (list, tuple)):
            return cell_summary(family, correct=spec, readout=readout, guard_spec=guard_spec)
    except HorizonRefusal as e:
        raise HorizonRefusal(f"{where}: {e}") from e
    raise HorizonRefusal(f"{where}: a cell must be a dict of per-item vectors or a list of bits, "
                         f"got {type(spec).__name__}")


def _is_variant_dict(spec) -> bool:
    return isinstance(spec, dict) and bool(spec) and all(k in VARIANTS for k in spec)


def _cells_from_block(block, where: str, cfg: dict):
    """``{family: {str(D): cell_spec}}`` -> (``{family: {int(D): cell}}``, controls). A D entry may also be
    the bridge-v3 runner's ``{variant: cell_spec}`` (variants plain, ablated, windowed, near, overwrite):
    the plain cell joins the curve and the others become controls ``{variant: {family: {D: cell}}}``."""
    if not isinstance(block, dict):
        raise HorizonRefusal(f"{where}: expected a dict keyed by family, got {type(block).__name__}")
    main: Dict[str, Dict[int, dict]] = {}
    controls: Dict[str, Dict[str, Dict[int, dict]]] = {}
    for fam, by_d in block.items():
        if fam not in FAMILIES:
            raise HorizonRefusal(f"{where}: unknown family {fam!r}; the probe has {FAMILIES}")
        if not isinstance(by_d, dict):
            raise HorizonRefusal(f"{where}.{fam}: expected a dict keyed by distance")
        ro = cfg["readouts_declared"].get(fam)
        for d_key, spec in by_d.items():
            try:
                d = int(d_key)
            except (TypeError, ValueError):
                raise HorizonRefusal(f"{where}.{fam}: distance key {d_key!r} is not an integer") from None
            if _is_variant_dict(spec):
                if "plain" not in spec:
                    raise HorizonRefusal(f"{where}.{fam}[{d}]: variants {sorted(spec)} and no plain cell")
                for var, vspec in spec.items():
                    cell = _cell_from_spec(fam, vspec, f"{where}.{fam}[{d}].{var}", ro, cfg["guard"])
                    if var == "plain":
                        main.setdefault(fam, {})[d] = cell
                    else:
                        controls.setdefault(var, {}).setdefault(fam, {})[d] = cell
            else:
                main.setdefault(fam, {})[d] = _cell_from_spec(fam, spec, f"{where}.{fam}[{d}]", ro, cfg["guard"])
    return main, controls


def _merge_controls(controls: dict, name: str, block, where: str, cfg: dict) -> None:
    c_main, nested = _cells_from_block(block, where, cfg)
    if nested:
        raise HorizonRefusal(f"{where}: a control block holds variants of its own ({sorted(nested)})")
    target = controls.setdefault(name, {})
    for fam, by_d in c_main.items():
        target.setdefault(fam, {}).update(by_d)


def _cells_from_items(items: Sequence[dict], where: str, cfg: dict):
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
        keys = set().union(*(r.keys() for r in recs)) - {"family", "distance", "item", "control"}
        spec = {k: [r.get(k) for r in recs] for k in sorted(keys) if all(r.get(k) is not None for r in recs)}
        if _pick(spec, "label") is None or _pick(spec, "prediction") is None:
            bits = _pick(spec, "correct")
            if bits is None:
                raise HorizonRefusal(f"{where}: ({ctrl}, {fam}, {d}) has items without label/prediction "
                                     f"or correct")
            spec = {"correct": bits}
        cell = _cell_from_spec(fam, spec, f"{where}:({ctrl},{fam},{d})", cfg["readouts_declared"].get(fam),
                               cfg["guard"])
        target = main if ctrl is None else controls.setdefault(str(ctrl), {})
        target.setdefault(fam, {})[d] = cell
    return main, controls


def _config_from(raw: dict, where: str, min_grid: int = 2) -> dict:
    cfg = raw.get("config") or {}
    if not isinstance(cfg, dict):
        raise HorizonRefusal(f"{where}: config must be a JSON object")
    grid = cfg.get("distances") or cfg.get("grid") or list(GRID)
    try:
        grid = [int(x) for x in grid]
    except (TypeError, ValueError):
        raise HorizonRefusal(f"{where}: config.distances must be integers") from None
    if len(grid) < min_grid:
        raise HorizonRefusal(f"{where}: the grid {grid} has {len(grid)} distance(s); a horizon needs at least "
                             f"{min_grid}. The cells of such a receipt can be read (min_grid=1); a curve cannot")
    if len(grid) >= 2:
        _log2_grid(grid)
    elif not grid or grid[0] <= 0:
        raise HorizonRefusal(f"{where}: config.distances must be positive integers, got {grid}")
    d0 = int(cfg.get("D0", grid[0]))
    if d0 != grid[0]:
        raise HorizonRefusal(f"{where}: D0 {d0} is not the smallest grid distance {grid[0]}")
    try:
        bar = float(cfg.get("screen_bar", SCREEN_BAR))
    except (TypeError, ValueError):
        raise HorizonRefusal(f"{where}: config.screen_bar must be a number") from None
    if not (0.0 < bar < 1.0):
        raise HorizonRefusal(f"{where}: config.screen_bar must lie strictly between 0 and 1 (normalised), got {bar}")
    frac = _check_frac(cfg.get("horizon_frac", FRAC))
    # the readout of each family: config.readouts {family: name}, or config.readout / config.answer_format
    # naming the compare readout (the bridge-v3 runner writes both)
    declared: Dict[str, str] = {}
    rd = cfg.get("readouts")
    if rd is not None:
        if not isinstance(rd, dict):
            raise HorizonRefusal(f"{where}: config.readouts must map family -> readout")
        declared.update({str(k): v for k, v in rd.items()})
    for key in ("readout", "answer_format"):
        v = cfg.get(key)
        if v is None:
            continue
        if v not in READOUTS:
            raise HorizonRefusal(f"{where}: config.{key} {v!r} is not a readout this instrument scores "
                                 f"({sorted(READOUTS)})")
        fam = READOUTS[v]["family"]
        if declared.get(fam, v) != v:
            raise HorizonRefusal(f"{where}: config declares two {fam} readouts, {declared[fam]!r} and {v!r}")
        declared[fam] = v
    for fam, v in declared.items():
        if fam not in FAMILIES or v not in READOUTS or READOUTS[v]["family"] != fam:
            raise HorizonRefusal(f"{where}: readout {v!r} for {fam!r} is not one the instrument scores")
    chance_decl: Dict[str, float] = {}
    ch = cfg.get("chance")
    if ch is not None:
        if not isinstance(ch, dict):
            raise HorizonRefusal(f"{where}: config.chance must map family -> chance")
        for fam, v in ch.items():
            if fam not in FAMILIES:
                raise HorizonRefusal(f"{where}: config.chance names an unknown family {fam!r}")
            try:
                chance_decl[fam] = float(v)
            except (TypeError, ValueError):
                raise HorizonRefusal(f"{where}: config.chance.{fam} must be a number") from None
    gs = _guard_spec(cfg.get("guard"), f"{where}: config.guard")
    # how the items pair across cells: declared, or the bridge-v3 runner's nesting flag, or assumed
    items_meta = raw.get("items") if isinstance(raw.get("items"), dict) else {}
    if "pairing" in cfg:
        pairing, p_src = cfg["pairing"], "declared (config.pairing)"
    elif "items_prefix_nested" in cfg:
        pairing = "items" if cfg["items_prefix_nested"] else "independent"
        p_src = "declared (config.items_prefix_nested)"
    elif "prefix_nested_across_distances" in items_meta:
        pairing = "items" if items_meta["prefix_nested_across_distances"] else "distance"
        p_src = "declared (items.prefix_nested_across_distances)"
    else:
        pairing, p_src = "items", "assumed, undeclared"
    if pairing not in PAIRINGS:
        raise HorizonRefusal(f"{where}: config.pairing must be one of {PAIRINGS}")
    boot = cfg.get("bootstrap") or {}
    seeds = cfg.get("seeds") or {}
    native = cfg.get("native_D")
    if native is not None:
        if not isinstance(native, dict):
            raise HorizonRefusal(f"{where}: config.native_D must map distance -> {{min, mean, max}}")
        native = {int(k): v for k, v in native.items()}
    return {"grid": grid, "D0": d0, "screen_bar": bar, "frac": frac, "pairing": pairing,
            "pairing_source": p_src, "pairing_declared": p_src != "assumed, undeclared",
            "readouts_declared": declared, "chance_declared": chance_decl, "guard": gs,
            "d0_split": bool(cfg.get("d0_split", True)),
            "n_resamples": int(boot.get("n_resamples", cfg.get("n_resamples", N_BOOT))),
            "seed": int(boot.get("seed", seeds.get("bootstrap", SEED_BOOT))),
            "native_D": native, "dtype": cfg.get("dtype"), "params": cfg.get("params"),
            "ctx": cfg.get("ctx", cfg.get("max_position_embeddings")),
            "ctx_is_training_length": bool(cfg.get("ctx_is_training_length", False)),
            "copied_from": cfg.get("copied_from"), "items_sha256": cfg.get("items_sha256"),
            "answer_format": cfg.get("answer_format"), "snapshot": cfg.get("snapshot"),
            "notes": cfg.get("notes")}


def load_receipt(path_or_dict, arm: Optional[str] = None, min_grid: int = 2) -> dict:
    """Read a receipt into the fitter's shape.

    Two shapes are accepted. The module's own (``schema`` = ``styxx.horizon/receipt/0.1``): ``model``,
    optional ``arch_family``, ``config``, and either ``per_item`` (``{family: {D: cell}}``) or ``items``
    (a flat list), with ``controls`` (``{ablated|near|overwrite|windowed: {family: {D: cell}}}``). The
    bridge-v3 receipt, with ``arm`` given: ``per_item[arm][family][D][variant]`` as the runner writes it
    from its amendment 17.2 (the plain variant is the curve, the others the controls; ``config.readout``,
    ``config.chance``, ``config.guard`` and ``items.prefix_nested_across_distances`` are read), or
    ``per_item[arm][family][D]`` as the cell itself with ``controls_per_item[arm]``. A v3 receipt without
    ``per_item[arm]`` is refused with that sentence, not read as zeros. ``min_grid`` is the number of
    distances a receipt must carry; 2 for a horizon, 1 to read the cells of a D0-only pilot.
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
                f"needs per-item correctness bits and compare scores in the receipt (zoo DESIGN section "
                f"8 item 2, decision Z4); without them the arm's row prints from acc_norm without an "
                f"interval, which this fitter does not do silently.")
        cfg = _config_from(raw, where, min_grid)
        cells, controls = _cells_from_block(per_item[arm], f"{where}.per_item[{arm}]", cfg)
        ctrl_raw = (raw.get("controls_per_item") or {}).get(arm) or {}
        for k, v in ctrl_raw.items():
            _merge_controls(controls, k, v, f"{where}.controls_per_item[{arm}].{k}", cfg)
        checkpoints = (raw.get("config") or {}).get("checkpoints") or {}
        model = checkpoints.get(arm, arm)
        cfg["copied_from"] = cfg.get("copied_from") or f"v3 receipt arm {arm}"
        arch = raw.get("arch_family", {}).get(arm) if isinstance(raw.get("arch_family"), dict) else None
    else:
        schema = raw.get("schema")
        if schema is not None and schema != RECEIPT_SCHEMA:
            raise HorizonRefusal(f"{where}: schema {schema!r} is not {RECEIPT_SCHEMA!r}")
        model = raw.get("model")
        if not isinstance(model, str) or not model:
            hint = ""
            if isinstance(raw.get("per_item"), dict) and set(raw["per_item"]) - set(FAMILIES):
                hint = f"; per_item is keyed by {sorted(raw['per_item'])}: a bridge-v3 receipt is read by arm (--arm)"
            raise HorizonRefusal(f"{where}: 'model' (a non-empty string) is required{hint}")
        cfg = _config_from(raw, where, min_grid)
        if "per_item" in raw:
            cells, controls = _cells_from_block(raw["per_item"], f"{where}.per_item", cfg)
            for k, v in (raw.get("controls") or {}).items():
                _merge_controls(controls, k, v, f"{where}.controls.{k}", cfg)
        elif "items" in raw:
            if not isinstance(raw["items"], list):
                raise HorizonRefusal(f"{where}: 'items' must be a list of per-item records")
            cells, controls = _cells_from_items(raw["items"], f"{where}.items", cfg)
        else:
            raise HorizonRefusal(f"{where}: a receipt needs 'per_item' cells or an 'items' list")
        arch = raw.get("arch_family")
    for k in controls:
        if k not in CONTROLS:
            raise HorizonRefusal(f"{where}: unknown control {k!r}; the instrument knows {', '.join(CONTROLS)}")
    readouts: Dict[str, str] = {}
    sources: Dict[str, str] = {}
    for fam in FAMILIES:
        if fam not in cells:
            raise HorizonRefusal(f"{where}: no {fam} cells; both families are needed for a horizon row")
        missing = [d for d in cfg["grid"] if d not in cells[fam]]
        if missing:
            raise HorizonRefusal(f"{where}: {fam} has no cell at distance(s) {missing} of the grid {cfg['grid']}")
        seen = {c["readout"] for c in cells[fam].values()}
        seen |= {c["readout"] for ctl in controls.values() for c in (ctl.get(fam) or {}).values()}
        if len(seen) != 1:
            raise HorizonRefusal(f"{where}: the {fam} cells are read under more than one readout {sorted(seen)}; "
                                 f"declare config.readout")
        readouts[fam] = next(iter(seen))
        sources[fam] = next(iter(cells[fam].values()))["readout_source"]
        want = chance_of(readouts[fam])
        got = cfg["chance_declared"].get(fam)
        if got is not None and not math.isclose(got, want, rel_tol=0.0, abs_tol=1e-9):
            raise HorizonRefusal(f"{where}: config.chance.{fam} is {got!r}, but the {readouts[fam]!r} readout "
                                 f"has {READOUTS[readouts[fam]]['k']} answers, so chance under exact balance is "
                                 f"{want!r}")
    cfg["readouts"] = readouts
    cfg["readout_source"] = sources
    cfg["chance"] = {fam: chance_of(readouts[fam]) for fam in FAMILIES}
    if arch is not None and not (isinstance(arch, dict) and arch.get("family") in ("DECAY", "ATTN")):
        raise HorizonRefusal(f"{where}: arch_family must be {{'family': 'DECAY'|'ATTN', 'fact': ...}} as "
                             f"assert_family returns it, or absent")
    return {"model": model, "arch_family": arch, "config": cfg, "cells": cells, "controls": controls,
            "source": source, "status": raw.get("status")}


# ---------------------------------------------------------------------------------------------------
# the per-model fit
# ---------------------------------------------------------------------------------------------------

def screen(compare_d0: dict, recall_d0: dict, bar: float = SCREEN_BAR) -> dict:
    """The capability screen at D0: both normalised accuracies >= bar, both guards passed, both cells in
    exact balance.

    A guard that was not recorded cannot pass: the screen then fails with the reason, because a row
    whose constant-responder check nobody ran must not enter a family set. A cell out of exact balance
    fails it too: chance normalisation maps a constant responder to 0 only under exact balance.
    """
    bar = float(bar)
    if not (0.0 < bar < 1.0):
        raise HorizonRefusal(f"the screen bar must lie strictly between 0 and 1, got {bar}")
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
            binary = READOUTS[cell["readout"]]["kind"] == "binary"
            stat = (f"yes-rate {cell['yes_rate']:.2f}" if binary
                    else f"largest slot share {cell['max_slot_fraction']:.2f}")
            reasons.append(f"{fam} guard failed ({stat})")
        if cell.get("exact_balance") is False:
            reasons.append(f"{fam} not in exact balance on these items (label counts {cell.get('label_counts')})")
    return {"pass": not reasons, "reasons": reasons, "bar": bar, "bar_standard": bool(bar == SCREEN_BAR),
            "compare_norm_D0": c, "recall_norm_D0": r,
            "guard_compare": compare_d0.get("guard_ok"), "guard_recall": recall_d0.get("guard_ok"),
            "n_compare": compare_d0["n"], "n_recall": recall_d0["n"]}


def _split_masks(cell: dict) -> Tuple[np.ndarray, np.ndarray, str]:
    """Two halves of a D0 cell: the screen's and the denominator's. Stratified by label when the cell
    records labels, so that a cell in exact balance with even label counts yields two halves in exact
    balance and a constant responder maps to 0 on both; positional halves otherwise, and the note says so."""
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
    """The cell restricted to the items ``mask`` selects: accuracy, guard, balance and AUROC recomputed."""
    cor = cell["_correct"][mask]
    out = dict(cell)
    out["_correct"] = cor
    out["n"] = int(len(cor))
    raw = float(cor.mean()) if len(cor) else float("nan")
    out["acc_raw"] = raw
    out["acc_norm"] = normalize(raw, cell["chance"]) if len(cor) else float("nan")
    y, p, S = cell.get("_labels"), cell.get("_preds"), cell.get("_scores")
    if y is not None and p is not None:
        y, p = y[mask], p[mask]
        S = None if S is None else S[mask]
        out.update(_cell_stats(cell["readout"], y, p, S, cell["_guard_spec"]))
        out["_labels"], out["_preds"], out["_scores"] = y, p, S
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


def _rep_c_auc(cell: dict, idx: np.ndarray, keep_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """``2 * AUROC - 1`` of the cell's scores on every replicate: the log-odds AUROC under a yes/no
    readout, the one-versus-rest AUROC averaged over the slots under a slot readout."""
    S, y = cell["_scores"], cell["_labels"]
    spec = READOUTS[cell["readout"]]
    if spec["kind"] == "binary":
        a = _rep_auroc(S, y.astype(bool), idx, keep_mask)
    else:
        a = np.mean(np.stack([_rep_auroc(S[:, j], y == j, idx, keep_mask) for j in range(spec["k"])]), axis=0)
    return 2.0 * a - 1.0


def _resample_matrix(n: int, n_resamples: int, seed) -> np.ndarray:
    if n < 2:
        raise HorizonRefusal(f"a cell of {n} item(s) cannot be resampled")
    rng = np.random.default_rng([int(s) for s in seed] if isinstance(seed, (list, tuple)) else int(seed))
    return rng.integers(0, n, size=(int(n_resamples), n))


def _fmt_flag_cells(flags: List[str], fam: str, d: int, cell: dict) -> None:
    c = cell["acc_norm"]
    ca = cell.get("c_auc")
    extra = f" c={c:+.2f}" + (f" c_auc={ca:+.2f}" if ca is not None else "")
    flags.append(f"DEGENERATE@{fam}:{d}{extra}")


def fit_model(rec: dict, idx: Optional[np.ndarray] = None, n_resamples: Optional[int] = None,
              seed: Optional[int] = None) -> dict:
    """The row of one model: cells with intervals, screen, horizons, ratios, dissociation, flags.

    ``idx`` is a shared (B, n) index matrix when several models with prefix-nested items are fitted
    together; when None the resamples are drawn from ``seed`` (models fitted on one seed share them,
    M9). ``config.pairing`` decides what one resample covers: ``items`` every cell, ``distance`` the two
    families at one distance, ``independent`` one cell. A model that fails the screen keeps its curves and
    cells and gets no horizon, ratio or dissociation.
    """
    cfg = rec["config"]
    grid = cfg["grid"]
    if len(grid) < 2:
        raise HorizonRefusal(f"{rec['model']}: the receipt's grid {grid} has one distance; a horizon needs at "
                             f"least two (its cells can be read, a curve cannot)")
    d0 = cfg["D0"]
    frac = cfg["frac"]
    chance = cfg["chance"]
    B = int(n_resamples if n_resamples is not None else cfg["n_resamples"])
    seed = int(seed if seed is not None else cfg["seed"])
    cells = rec["cells"]
    pairing = cfg["pairing"]
    n_by = {(f, d): cells[f][d]["n"] for f in FAMILIES for d in grid}
    if pairing == "items" and len(set(n_by.values())) != 1:
        raise HorizonRefusal(f"{rec['model']}: cells differ in item count {sorted(set(n_by.values()))} while the "
                             f"receipt declares items paired across distances; declare config.pairing = "
                             f"'distance' or 'independent', or fix the receipt")
    if pairing == "distance":
        bad = [d for d in grid if n_by[("compare", d)] != n_by[("recall", d)]]
        if bad:
            raise HorizonRefusal(f"{rec['model']}: compare and recall differ in item count at {bad} while the "
                                 f"receipt declares the two families share items at one distance")
    n = min(n_by.values())
    if idx is not None and pairing == "items" and (idx.ndim != 2 or idx.shape[1] != n):
        raise HorizonRefusal(f"{rec['model']}: the shared index matrix has shape {idx.shape}, the cells have n={n}")
    split = bool(cfg["d0_split"])
    if split and min(n_by[(f, d0)] for f in FAMILIES) < 4:
        raise HorizonRefusal(f"{rec['model']}: {min(n_by[(f, d0)] for f in FAMILIES)} items are too few for the D0 split")
    masks = {f: _split_masks(cells[f][d0]) for f in FAMILIES} if split else None

    if pairing == "items":
        shared = idx if idx is not None else _resample_matrix(n, B, seed)
        B = int(shared.shape[0])

        def cell_idx(fam: str, d: int, cell: dict) -> np.ndarray:
            return shared
    elif pairing == "distance":
        per_d = {d: _resample_matrix(n_by[("compare", d)], B, (seed, d)) for d in grid}

        def cell_idx(fam: str, d: int, cell: dict) -> np.ndarray:
            return per_d[d]
    else:
        rng_cells = np.random.default_rng(seed + 1)

        def cell_idx(fam: str, d: int, cell: dict) -> np.ndarray:
            return rng_cells.integers(0, cell["n"], size=(B, cell["n"]))

    flags: List[str] = []
    # -- the screen on one half of D0, the denominator on the other ---------------------------------
    if masks is not None:
        screen_cells = {f: _subcell(cells[f][d0], masks[f][0], "screen half") for f in FAMILIES}
        denom_cells = {f: _subcell(cells[f][d0], masks[f][1], "denominator half") for f in FAMILIES}
        split_note = ("D0 split (" + "; ".join(f"{f}: {masks[f][2]}" for f in FAMILIES)
                      + "); other distances on all their items")
        for f in FAMILIES:
            for part, half in (("screen", screen_cells[f]), ("denominator", denom_cells[f])):
                if half.get("exact_balance") is False:
                    flags.append(f"BALANCE_BROKEN@{f}:{d0}/{part}")
    else:
        screen_cells = {f: cells[f][d0] for f in FAMILIES}
        denom_cells = {f: cells[f][d0] for f in FAMILIES}
        split_note = ("D0 split: none (screen and denominator share the D0 items; a model near the bar "
                      "has its ratios biased low by selection)")
    scr = screen(screen_cells["compare"], screen_cells["recall"], cfg["screen_bar"])
    scr["items"] = "the screen half of the D0 items" if masks is not None else "all the D0 items"
    screened = bool(scr["pass"])

    # -- curves, replicate curves, intervals ------------------------------------------------------
    curves: Dict[str, np.ndarray] = {}
    rep_curves: Dict[str, np.ndarray] = {}
    out_cells: Dict[str, Dict[int, dict]] = {fam: {} for fam in FAMILIES}
    auc_curve = np.full(len(grid), np.nan)
    auc_rep = np.full((B, len(grid)), np.nan)
    has_scores = all(cells["compare"][d].get("_scores") is not None for d in grid)
    for fam in FAMILIES:
        ch = chance[fam]
        pts, reps = [], []
        for j, d in enumerate(grid):
            cell = cells[fam][d]
            use = denom_cells[fam] if d == d0 else cell
            keep = masks[fam][1] if (d == d0 and masks is not None) else None
            ci = cell_idx(fam, d, cell)
            rm = _rep_means(cell["_correct"], ci, keep)
            rnorm = (rm - ch) / (1.0 - ch)
            lo, hi, _ = _percentiles(rnorm)
            pts.append(use["acc_norm"])
            reps.append(rnorm)
            oc = {k: v for k, v in use.items() if not k.startswith("_")}
            oc.update({"ci_lo": lo, "ci_hi": hi, "degenerate": (use.get("guard_ok") is False)})
            if fam == "compare" and has_scores:
                auc_curve[j] = use["c_auc"] if use.get("c_auc") is not None else np.nan
                auc_rep[:, j] = _rep_c_auc(cell, ci, keep)
            if d != d0 and use.get("guard_ok") is False:
                _fmt_flag_cells(flags, fam, d, use)
            if (d != d0 or masks is None) and cell.get("exact_balance") is False:
                flags.append(f"BALANCE_BROKEN@{fam}:{d}")
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

    nan_B = np.full(B, np.nan)
    if screened:
        horizons = {fam: fit_horizon(curves[fam], rep_curves[fam], grid, frac, native_mean) for fam in FAMILIES}
        if has_scores:
            horizons["compare_auc"] = fit_horizon(auc_curve, auc_rep, grid, frac, native_mean)
        else:
            horizons["compare_auc"] = {"status": "NOT_RECORDED", "point": None, "bracket": None,
                                       "reason": "no per-item compare scores in the receipt"}
        ratios: Dict[str, dict] = {}
        rep_ratio: Dict[str, np.ndarray] = {}
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
    else:
        why = "SCREEN_FAIL: the model has no positive control at D0, so no horizon is read off its curve"
        horizons = {name: _not_estimable(grid, why) for name in ("compare", "recall", "compare_auc")}
        ratios = {fam: {"point": None, "ci_lo": None, "ci_hi": None, "f_nonfinite": None, "reason": why}
                  for fam in FAMILIES}
        rep_ratio = {fam: nan_B.copy() for fam in FAMILIES}
        diss_rep = nan_B.copy()
        dissoc = {"metric": "ratio_min_compare - ratio_min_recall", "point": None, "ci_lo": None, "ci_hi": None,
                  "f_nonfinite": None, "reason": why}
    if not has_scores:
        flags.append("SCORES_ABSENT")

    # -- dissociation and shape flags (a screened row only) ----------------------------------------
    if screened:
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
        if np.any(np.diff(curves[fam]) > NONMONOTONE_RISE):
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
        ratio = (cn / c0) if (screened and c0 > 0) else None
        ctrl_out["near"] = {"distance": d_far, "norm_acc": float(cn), "ratio_to_D0": ratio,
                            "format_decay_below": NEAR_FORMAT_DECAY, "format_drift_below": NEAR_FORMAT_DRIFT}
        if not screened:
            ctrl_out["near"]["note"] = "no ratio: the row failed the screen"
        elif ratio is None:
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
                                 "fired_max": OVERWRITE_FIRED_MAX,
                                 "follow_rate": ow[d_far].get("overwrite_follow_rate")}
        flags.append("POSITIVE_CONTROL_FIRED" if fired else f"POSITIVE_CONTROL_SILENT@{d_far}")
    else:
        ctrl_out["overwrite"] = None
        flags.append("FAR_BINDING_CONTROL_ABSENT")
    win = rec["controls"].get("windowed")
    ctrl_out["windowed"] = ({f"{fam}:{d}": float(win[fam][d]["acc_norm"]) for fam in win for d in win[fam]}
                            if win else None)
    if cfg["screen_bar"] != SCREEN_BAR:
        flags.append(f"NONSTANDARD_BAR@{cfg['screen_bar']:.2f}")
    if cfg["guard"]["nonstandard"]:
        flags.append("NONSTANDARD_GUARD")
    if not cfg["pairing_declared"]:
        flags.append("PAIRING_ASSUMED")
    if not screened:
        flags.insert(0, "SCREEN_FAIL")

    return {"model": rec["model"], "arch_family": rec["arch_family"], "source": rec["source"],
            "config": {k: v for k, v in cfg.items() if k != "native_D"},
            "native_D": cfg.get("native_D"), "n_items": n, "pairing": pairing,
            "pairing_source": cfg["pairing_source"], "readouts": dict(cfg["readouts"]),
            "readout_source": dict(cfg["readout_source"]), "chance": dict(chance),
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


def _n_signature(rec: dict) -> tuple:
    g = rec["config"]["grid"]
    return tuple(g), tuple((rec["cells"]["compare"][d]["n"], rec["cells"]["recall"][d]["n"]) for d in g)


def fit_zoo(recs: Sequence[dict], exclude: Sequence[str] = (), n_resamples: Optional[int] = None,
            seed: int = SEED_BOOT, family: str = "ATTN") -> dict:
    """Fit every receipt on ONE resample stream and aggregate the named architecture family.

    ``exclude`` names models whose rows are printed but never counted (v3's own arms, copied rows).
    The aggregates follow the red team's fixes: the ablated maximum is taken over the screened set
    BEFORE any ablation exclusion (M1); every per-model value sits in ``result.per_model.*`` and every
    excluded member in ``result.family_excluded`` so a gates block declares ``agg``/``over``/``excluding``
    (M2); the family interval is the interval of the per-replicate minimum on the shared resample (M3,
    M9). Receipts that declare different pairings or item counts cannot share the stream; the sheet then
    says the family interval is not paired across models.

    The counts (``n_models``, ``n_family_screened``, ``n_attention_screen_pass``) are finite in every
    case. The family aggregates are None when the family set is empty, and ``styxx.protocol`` refuses an
    aggregate over an empty population (it is not a measurement); a prereg reaches ABSTAIN on an empty set
    through a readiness gates block that reads the counts alone, scored before the family block — see the
    README. A non-empty set below a count gate reaches ABSTAIN through the family block itself.
    """
    recs = list(recs)
    if not recs:
        raise HorizonRefusal("fit_zoo: no receipts")
    names = [r["model"] for r in recs]
    if len(set(names)) != len(names):
        raise HorizonRefusal(f"fit_zoo: duplicate model names {sorted(n for n in names if names.count(n) > 1)}")
    B = int(n_resamples if n_resamples is not None else max(r["config"]["n_resamples"] for r in recs))
    pairings = {r["config"]["pairing"] for r in recs}
    sigs = {_n_signature(r) for r in recs}
    pairable = len(pairings) == 1 and next(iter(pairings)) in ("items", "distance") and len(sigs) == 1
    shared = None
    if pairable and next(iter(pairings)) == "items":
        shared = _resample_matrix(recs[0]["cells"]["compare"][recs[0]["config"]["grid"][0]]["n"], B, seed)
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
        "family": family, "n_models": len(rows), "n_family_screened": len(screened),
        "n_attention_screen_pass": len(family_set),
        "family_set": [r["model"] for r in family_set], "family_excluded": excluded,
        "family_excluded_reasons": reasons, "per_model": per_model,
        "family_set_ablated_max_norm_acc": _round6(max(abl_vals)) if abl_vals else None,
        "family_set_ablated_note": "max over the screened set BEFORE ablation exclusion (red team M1)",
        "shared_resample": bool(pairable), "pairing": sorted(pairings), "n_resamples": B, "seed": seed,
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
        result["family_aggregates_note"] = ("the family set is empty: every family aggregate is None and "
                                            "styxx.protocol refuses an aggregate over an empty population; "
                                            "the counts above are finite and a readiness gates block that reads "
                                            "them alone scores this sheet")
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

def _bits_for(readout: str, labels: np.ndarray, correct: Optional[np.ndarray], rng: np.random.Generator,
              constant: Optional[Any] = None) -> dict:
    """A cell spec for given labels and a correctness mask, or for a constant responder.

    Yes/no: log-odds whose sign is the prediction and whose magnitude is random. Slots: three scores per
    item, random except that the predicted slot is the strict maximum; a constant responder carries one
    identical score vector on every item, so its AUROC is exactly 0.5 in either readout."""
    spec = READOUTS[readout]
    k = spec["k"]
    n = len(labels)
    if spec["kind"] == "binary":
        y = np.asarray(labels, dtype=bool)
        if constant is not None:
            preds = np.full(n, bool(constant))
            mag = np.full(n, 1.0)
        else:
            preds = np.where(correct, y, ~y)
            mag = rng.uniform(0.1, 3.0, size=n)
        return {"label": y.tolist(), "prediction": preds.tolist(), "logodds": np.where(preds, mag, -mag).tolist()}
    t = np.asarray(labels, dtype=int)
    if constant is not None:
        c = int(constant)
        if not 0 <= c < k:
            raise HorizonRefusal(f"a constant slot responder needs a slot in 0..{k - 1}")
        preds = np.full(n, c)
        row = np.zeros(k)
        row[c] = 1.0
        scores = np.tile(row, (n, 1))
    else:
        wrong = (t + rng.integers(1, k, size=n)) % k
        preds = np.where(correct, t, wrong)
        scores = rng.normal(0.0, 1.0, size=(n, k))
        scores[np.arange(n), preds] = scores.max(axis=1) + rng.uniform(0.1, 2.0, size=n)
    return {"target_slot": t.tolist(), "pred_slot": preds.tolist(), "scores": scores.tolist(),
            "correct": (preds == t).astype(int).tolist()}


def synthetic_cell(family: str, c_norm: float, n: int, rng: np.random.Generator,
                   constant: Optional[Any] = None, readout: Optional[str] = None, sample: bool = False) -> dict:
    """Per-item bits for one cell with EXACT balance and a known normalised accuracy, under ``readout``
    (default: the bridge-v3 runner's, ``argmax_largest`` for compare and ``argmax_code`` for recall).

    The number correct is exact (``round(raw * n)``) unless ``sample``: then each item is correct with
    probability ``raw``, as in a real run, so that an estimate varies from seed to seed the way a measured
    one does. ``constant`` builds a constant responder instead (yes/no: always that answer; slots: always
    that slot). The threshold-free curve tracks the thresholded one.
    """
    if family not in FAMILIES:
        raise HorizonRefusal(f"family must be one of {FAMILIES}")
    readout = readout or DEFAULT_READOUT[family]
    if READOUTS.get(readout, {}).get("family") != family:
        raise HorizonRefusal(f"readout {readout!r} is not a {family} readout")
    k = READOUTS[readout]["k"]
    chance = 1.0 / k
    if n % k:
        raise HorizonRefusal(f"a balanced {family} cell under {readout!r} needs n divisible by {k}")
    if READOUTS[readout]["kind"] == "binary":
        labels = np.array([True, False] * (n // 2))
    else:
        labels = np.array(list(range(k)) * (n // k))
    rng.shuffle(labels)
    correct = None
    if constant is None:
        raw = min(1.0, max(0.0, chance + float(c_norm) * (1.0 - chance)))   # c below -chance/(1-chance) is raw 0
        if sample:
            correct = rng.random(n) < raw
        else:
            correct = np.zeros(n, dtype=bool)
            correct[rng.choice(n, size=int(round(raw * n)), replace=False)] = True
    return _bits_for(readout, labels, correct, rng, constant)


def _plant_halves(readout: str, n: int, c_screen: float, c_denom: float, rng: np.random.Generator) -> dict:
    """A balanced D0 cell whose screen half and denominator half (as :func:`_split_masks` cuts them) carry
    two different planted accuracies: the self-test's view of the M4 wiring."""
    spec = synthetic_cell(READOUTS[readout]["family"], 0.0, n, rng, readout=readout)
    binary = READOUTS[readout]["kind"] == "binary"
    labels = np.asarray(spec["label"] if binary else spec["target_slot"], dtype=bool if binary else int)
    s_mask, d_mask, _ = _split_masks({"n": n, "_labels": labels})
    ch = chance_of(readout)
    correct = np.zeros(n, dtype=bool)
    for mask, c in ((s_mask, c_screen), (d_mask, c_denom)):
        pos = np.flatnonzero(mask)
        correct[rng.choice(pos, size=int(round((ch + c * (1.0 - ch)) * len(pos))), replace=False)] = True
    return _bits_for(readout, labels, correct, rng)


def _planted_curve(h_star: float, c0: float = 0.9, frac: float = FRAC, grid: Sequence[int] = GRID) -> List[float]:
    """A curve linear in log2 D through (D0, c0) that reaches frac * c0 exactly at h_star."""
    lg = np.log2(np.asarray(grid, float))
    slope = (frac * c0 - c0) / (math.log2(h_star) - lg[0])
    return [float(max(-1.0, min(1.0, c0 + slope * (x - lg[0])))) for x in lg]


def synthetic_receipt(model: str, compare_curve: Sequence[float], recall_curve: Sequence[float], n: int,
                      seed: int, grid: Sequence[int] = GRID, arch_family: Optional[str] = None,
                      constant_compare: Optional[Any] = None, constant_recall: Optional[int] = None,
                      controls: Optional[dict] = None, n_resamples: int = N_BOOT,
                      readout: str = DEFAULT_READOUT["compare"], sample: bool = False) -> dict:
    """A receipt dict in the module's own schema with cells of known normalised accuracy, the compare
    family under ``readout`` (declared in ``config.readout``, with its chance and the runner's guard).
    ``sample`` draws each item's correctness instead of fixing the count (see :func:`synthetic_cell`)."""
    if READOUTS.get(readout, {}).get("family") != "compare":
        raise HorizonRefusal(f"{readout!r} is not a compare readout")
    rng = np.random.default_rng(seed)
    ro = {"compare": readout, "recall": DEFAULT_READOUT["recall"]}
    per_item: Dict[str, Dict[str, dict]] = {"compare": {}, "recall": {}}
    for d, cc, rc in zip(grid, compare_curve, recall_curve):
        per_item["compare"][str(d)] = synthetic_cell("compare", cc, n, rng, constant_compare, ro["compare"], sample)
        per_item["recall"][str(d)] = synthetic_cell("recall", rc, n, rng, constant_recall, ro["recall"], sample)
    rec = {"schema": RECEIPT_SCHEMA, "model": model,
           "config": {"distances": list(grid), "D0": int(grid[0]), "screen_bar": SCREEN_BAR,
                      "readout": readout, "chance": {f: chance_of(ro[f]) for f in FAMILIES},
                      "guard": {"yes_rate": list(GUARD_YES_RANGE), "max_slot_fraction": GUARD_SLOT_MAX},
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
                rec["controls"][name][fam] = {str(d): synthetic_cell(fam, c, n if cnt is None else cnt, rng,
                                                                     readout=ro[fam])
                                              for d, (c, cnt) in by_d.items()}
    return rec


_SELFTEST_PLANTS = (32.0, 128.0, 203.0, 600.0)
_SELFTEST_TOL = (0.8, 1.25)            # recovered point within this factor of the plant
_SELFTEST_LEVEL_TOL = 0.05             # recovered c(D0), r(D0) within this of the planted 0.9
_SELFTEST_WIDTH_TOL = (0.85, 1.18)     # mean interval width over the off-D0 cells / the binomial width
_SELFTEST_COVERAGE = (32, 24)          # seeds at n = 300 with drawn correctness, and how many must cover
_RECALL_FLAT = (0.9, 0.88, 0.86, 0.84)


def selftest(n: int = 2400, n_resamples: int = 400, seed: int = SEED_BOOT, coverage_seeds: Optional[int] = None) -> dict:
    """The estimator's positive control on synthetic bits. Refused datasheets cite its checks.

    * planted crossings at {32, 128, 203, 600} are recovered within [0.8, 1.25] x the plant, the
      interval covers the plant, and the recovered order is the planted order;
    * the recovered level: c(D0) and r(D0) within 0.05 of the planted 0.9 (a curve read at half its
      height recovers the same crossing, so the crossing alone cannot see the scale);
    * the interval's width: over the six off-D0 cells of one fit at n = 600 with 2000 resamples, the mean
      percentile width within [0.85, 1.18] x the binomial 95% width of the same cell; and coverage: on 32
      seeds at n = 300, each item's correctness drawn rather than fixed (so that the point moves from seed
      to seed as a measured one does), the plant 128 lies inside the horizon interval at least 24 times
      (a 95% interval covers it about 95% of the time, a 50% one about half);
    * the D0 split as wired (M4): a D0 cell whose screen half is planted at 0.45 and whose denominator half
      at 0.90 reads 0.45 on the screen and 0.90 as c(D0), with the D0 interval above 0.80;
    * a curve with no crossing is CENSORED with f_cens above 0.975;
    * a constant responder under exact balance is exactly 0 at every distance in both families, under the
      slot readout and under the yes/no readout, and reads SCREEN_FAIL with the guard in the reason;
    * a row that fails the screen prints no horizon, no ratio and no dissociation, though its curve crosses;
    * compare crossing with recall censored reads COMPARE_ONLY_HORIZON;
    * two fits under one seed agree to the last digit.

    The design's illustration planted "128" with c = (0.9, 0.7, 0.4, 0.2); under the fitter's own rule
    those four values cross at about 203, so the plants here are built FROM the rule and the planted
    number and the fitter's model agree by construction.
    """
    checks: List[dict] = []
    n_cov, need_cov = _SELFTEST_COVERAGE
    if coverage_seeds is not None:
        n_cov, need_cov = int(coverage_seeds), int(math.ceil(coverage_seeds * _SELFTEST_COVERAGE[1] / _SELFTEST_COVERAGE[0]))

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    recovered, levels = [], []
    for k, h in enumerate(_SELFTEST_PLANTS):
        rec = load_receipt(synthetic_receipt(f"plant-{int(h)}", _planted_curve(h), _RECALL_FLAT,
                                             n, seed + k, n_resamples=n_resamples))
        row = fit_model(rec)
        hz = row["horizon"]["compare"]
        pt = hz["point"]
        ok = (hz["status"] == "FINITE" and pt is not None and _SELFTEST_TOL[0] * h <= pt <= _SELFTEST_TOL[1] * h
              and hz["ci_lo"] is not None and hz["ci_lo"] <= h <= (hz["ci_hi"] if hz["ci_hi"] is not None else float("inf")))
        recovered.append(pt if pt is not None else float("nan"))
        levels.append((row["curves"]["compare"][0], row["curves"]["recall"][0]))
        check(f"planted_{int(h)}", ok,
              f"planted {h:g}, recovered {pt if pt is None else round(pt, 1)} [{hz['ci_lo'] and round(hz['ci_lo'], 1)}, "
              f"{hz['ci_hi'] and round(hz['ci_hi'], 1)}] f_cens {hz['f_cens']}, status {hz['status']}")
    check("planted_order", all(np.isfinite(recovered)) and all(np.diff(recovered) > 0),
          f"recovered in planted order: {[None if not np.isfinite(x) else round(x, 1) for x in recovered]}")
    check("planted_level", all(abs(c - 0.9) <= _SELFTEST_LEVEL_TOL and abs(r - 0.9) <= _SELFTEST_LEVEL_TOL
                               for c, r in levels),
          f"c(D0), r(D0) recovered at {[(round(c, 3), round(r, 3)) for c, r in levels]} against the planted 0.9")

    # the width on its own fit with the full 2000 resamples: the cells of one model share one index
    # matrix, so their widths move together and fewer resamples would leave the mean a noisy number
    row = fit_model(load_receipt(synthetic_receipt("width", _planted_curve(128.0), _RECALL_FLAT, 600, seed + 70,
                                                   n_resamples=N_BOOT)))
    ratios = []
    for fam in FAMILIES:
        ch = row["chance"][fam]
        for d in row["config"]["grid"][1:]:
            cell = row["cells"][fam][d]
            p = cell["acc_raw"]
            binom = 2.0 * 1.959964 * math.sqrt(max(p * (1.0 - p), 1e-12) / cell["n"]) / (1.0 - ch)
            if cell["ci_lo"] is not None and cell["ci_hi"] is not None:
                ratios.append((cell["ci_hi"] - cell["ci_lo"]) / binom)
    mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
    check("interval_width", len(ratios) == 2 * (len(GRID) - 1)
          and _SELFTEST_WIDTH_TOL[0] <= mean_ratio <= _SELFTEST_WIDTH_TOL[1],
          f"mean percentile width / binomial 95% width over {len(ratios)} cells (n = 600, B = {N_BOOT}): "
          f"{mean_ratio:.3f}")

    hits = 0
    for j in range(n_cov):
        rec = load_receipt(synthetic_receipt("coverage", _planted_curve(128.0), _RECALL_FLAT, 300, seed + 100 + j,
                                             n_resamples=200, sample=True))
        hz = fit_model(rec)["horizon"]["compare"]
        hi = float("inf") if (hz["ci_hi_censored"] or hz["ci_hi"] is None) else hz["ci_hi"]
        hits += int(hz["ci_lo"] is not None and hz["ci_lo"] <= 128.0 <= hi)
    check("interval_coverage", hits >= need_cov,
          f"the plant 128 inside the horizon interval on {hits} of {n_cov} seeds at n = 300 (need {need_cov})")

    rec = synthetic_receipt("split", [0.9, 0.85, 0.8, 0.75], _RECALL_FLAT, 600, seed + 50, n_resamples=n_resamples)
    rec["per_item"]["compare"][str(GRID[0])] = _plant_halves(rec["config"]["readout"], 600, 0.45, 0.90,
                                                             np.random.default_rng(seed + 51))
    row = fit_model(load_receipt(rec))
    s_c, d_c, c0 = row["screen"]["compare_norm_D0"], row["curves"]["compare"][0], row["cells"]["compare"][GRID[0]]
    check("d0_split_wiring", abs(s_c - 0.45) < 1e-9 and abs(d_c - 0.90) < 1e-9 and c0["ci_lo"] is not None
          and c0["ci_lo"] > 0.80 and c0["ci_lo"] <= 0.90 <= c0["ci_hi"],
          f"screen half reads {s_c:+.3f} (planted +0.450), denominator half {d_c:+.3f} (planted +0.900), "
          f"D0 interval [{c0['ci_lo'] and round(c0['ci_lo'], 3)}, {c0['ci_hi'] and round(c0['ci_hi'], 3)}]")

    rec = load_receipt(synthetic_receipt("no-crossing", [0.9, 0.85, 0.8, 0.75], _RECALL_FLAT, 1200, seed + 10,
                                         n_resamples=n_resamples))
    row = fit_model(rec)
    hz = row["horizon"]["compare"]
    check("censored", hz["status"] == "CENSORED" and hz["f_cens"] is not None and hz["f_cens"] > 0.975
          and "NO_HORIZON_INSIDE_RANGE" in row["flags"],
          f"status {hz['status']}, f_cens {hz['f_cens']}, flags {row['flags'][:2]}")

    for name, ro, const in (("constant_responder", "argmax_largest", 1), ("constant_responder_yes_no", "yes_no", True)):
        rec = load_receipt(synthetic_receipt(name, [0, 0, 0, 0], [0, 0, 0, 0], 600, seed + 20,
                                             constant_compare=const, constant_recall=0, n_resamples=n_resamples,
                                             readout=ro))
        row = fit_model(rec)
        zero = all(x == 0.0 for x in row["curves"]["compare"]) and all(x == 0.0 for x in row["curves"]["recall"])
        scr = row["screen"]
        check(name, zero and not scr["pass"] and any("guard failed" in r for r in scr["reasons"])
              and row["flags"][0] == "SCREEN_FAIL" and row["horizon"]["compare"]["status"] == "NOT_ESTIMABLE",
              f"{ro}: curves {row['curves']}, screen {scr['pass']}, reasons {scr['reasons']}")

    rec = load_receipt(synthetic_receipt("screen-fail", [0.30, 0.22, 0.12, 0.05], _RECALL_FLAT, 600, seed + 60,
                                         n_resamples=100))
    row = fit_model(rec)
    silent = (row["flags"][0] == "SCREEN_FAIL"
              and all(row["horizon"][h]["status"] == "NOT_ESTIMABLE" for h in ("compare", "recall", "compare_auc"))
              and all(row["ratio_min"][f]["point"] is None for f in FAMILIES) and row["dissociation"]["point"] is None)
    check("screen_fail_prints_no_horizon", silent,
          f"compare c = {[round(x, 2) for x in row['curves']['compare']]} crosses, screen {row['screen']['pass']}, "
          f"horizon {row['horizon']['compare']['status']}, dissociation {row['dissociation']['point']}")

    rec = load_receipt(synthetic_receipt("dissoc", _planted_curve(128.0), _RECALL_FLAT, 600, seed + 30,
                                         n_resamples=n_resamples))
    row = fit_model(rec)
    check("compare_only_flag", "COMPARE_ONLY_HORIZON" in row["flags"] and row["horizon"]["recall"]["status"] == "CENSORED",
          f"flags {row['flags']}")

    rec = load_receipt(synthetic_receipt("determinism", _planted_curve(203.0), _RECALL_FLAT, 600, seed + 40,
                                         n_resamples=n_resamples))
    a, b = fit_model(rec), fit_model(rec)
    same = (a["horizon"]["compare"]["point"] == b["horizon"]["compare"]["point"]
            and a["horizon"]["compare"]["ci_lo"] == b["horizon"]["compare"]["ci_lo"]
            and a["horizon"]["compare"]["ci_hi"] == b["horizon"]["compare"]["ci_hi"]
            and a["dissociation"] == b["dissociation"])
    check("deterministic_under_seed", same, "two fits under one seed agree")

    return {"pass": all(c["ok"] for c in checks), "checks": checks, "n": n, "n_resamples": n_resamples,
            "seed": seed, "plants": list(_SELFTEST_PLANTS), "tolerance": list(_SELFTEST_TOL),
            "level_tolerance": _SELFTEST_LEVEL_TOL, "width_tolerance": list(_SELFTEST_WIDTH_TOL),
            "coverage": {"seeds": n_cov, "needed": need_cov, "hits": hits}}


# ---------------------------------------------------------------------------------------------------
# the datasheet
# ---------------------------------------------------------------------------------------------------

LIMITS = (
    "this sheet describes curves under one frozen probe, one filler corpus, one few-shot format and one "
    "readout; it licenses no sentence about rotation, honesty, causation or ranking",
    "a horizon is read off a four-point grid: the bracketing grid pair is the measurement, the "
    "interpolated point is a convention, and the interval carries sampling noise only",
    "SCREEN_FAIL reads: the model did not perform this format at D0; not 'cannot compare'; such a row "
    "prints its curve and no horizon",
    "'> Dmax' means no crossing inside the tested range, which ends at the last grid point and is under "
    "1% of context for some models; the sheet says nothing about the rest of the context",
    "a chance-normalised 0 is a constant responder or chance, not an absence of knowledge; the guard, "
    "the balance and the AUROC per cell are in the receipt",
    "FORMAT_DISTANCE_NOT_EXCLUDED: the few-shot exemplars recede with D; without a near-fact control a "
    "compare-only drop may be format decay, not binding",
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
        return "not estimable" + (" (screen failed)" if str(h.get("reason", "")).startswith("SCREEN_FAIL") else "")
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


def _chance_words(c: float) -> str:
    k = round(1.0 / c) if c > 0 else 0
    return f"1/{k}" if k and math.isclose(c, 1.0 / k, abs_tol=1e-12) else f"{c:.3f}"


def _readout_text(row: dict) -> str:
    gs = row["config"]["guard"]
    parts = []
    for fam in FAMILIES:
        ro = row["readouts"][fam]
        src = row["readout_source"][fam]
        parts.append(f"{fam} {ro} ({src.split(' (')[0]}; chance {_chance_words(row['chance'][fam])}; guard "
                     f"{_guard_words(ro, gs)})")
    return "readout: " + " | ".join(parts)


def _row_lines(row: dict) -> List[str]:
    grid = row["config"]["grid"]
    cfg = row["config"]
    fam = (row["arch_family"] or {}).get("family") or "--"
    params = cfg.get("params") or "--"
    dtype = cfg.get("dtype") or "--"
    ctx = cfg.get("ctx")
    ctx_s = ("--" if ctx is None else str(ctx)) + ("*" if cfg.get("ctx_is_training_length") else "")
    scr = ("PASS" if row["screen"]["pass"] else "FAIL") + f"@{row['screen']['bar']:.2f}".replace("@0.", "@.")
    copied = f" [{cfg['copied_from']}]" if cfg.get("copied_from") else ""
    head = f"{row['model'][:22]:<22} {fam:<6} {str(params)[:7]:<7} {str(dtype)[:6]:<6} {ctx_s:<8} {scr:<8}"
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
    sp = row.get("spearman_log2D") or {}
    lines.append(f"{pad} spearman(c, log2 D): compare {_num(sp.get('compare'), 0).strip()}   "
                 f"recall {_num(sp.get('recall'), 0).strip()}")
    if not row["screen"]["pass"]:
        lines.append(f"{pad} screen failed: {'; '.join(row['screen']['reasons'])}")
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
        fr = ctrl["overwrite"].get("follow_rate")
        bits.append(f"overwrite@{ctrl['overwrite']['distance']} {ctrl['overwrite']['norm_acc']:+.2f}"
                    + ("" if fr is None else f" (follows the re-plant {fr:.2f})"))
    if ctrl.get("windowed"):
        bits.append("windowed " + ", ".join(f"{k} {v:+.2f}" for k, v in ctrl["windowed"].items()))
    if bits:
        lines.append(f"{pad} controls: {'; '.join(bits)}")
    if row.get("native_D"):
        nat = "; ".join(f"{d}->{v.get('min', '?')}/{v.get('mean', '?')}/{v.get('max', '?')}"
                        for d, v in sorted(row["native_D"].items()) if isinstance(v, dict))
        lines.append(f"{pad} native D (min/mean/max): {nat}")
    lines.append(f"{pad} {_readout_text(row)}")
    pairing = row["pairing"] + ("" if row["config"]["pairing_declared"] else " (assumed, undeclared)")
    lines.append(f"{pad} {row['split_note']}; pairing: {pairing}; B = {row['n_resamples']}, seed {row['seed']}")
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
    n_plants = sum(1 for c in st["checks"] if re.fullmatch(r"planted_\d+", c["name"]))
    out = [f"styxx.horizon datasheet      items: {items_s}      gates sha256: {gates}      "
           f"selftest: PASS ({n_plants} planted horizons recovered, {len(st['checks'])} checks)",
           f"{'model':<22} {'family':<6} {'params':<7} {'dtype':<6} {'ctx':<8} {'screen':<8} {'op':<7} "
           + "".join(f"{d:>6}" for d in grid) + "   horizon (bracket | point [interval] | native)"]
    for r in rows:
        out.extend(_row_lines(r))
        out.append("")
    res = sheet.get("result")
    if res:
        mind = res.get("attention_family_min_dissoc")
        ci = res.get("attention_family_min_dissoc_ci") or {}
        ci_s = ("--" if ci.get("lo") is None else f"[{ci['lo']:+.2f}, {ci['hi']:+.2f}]")
        out.append(f"{res['family']} family set ({res['n_attention_screen_pass']} of {res['n_family_screened']} screened, "
                   f"{res['n_models']} rows): {', '.join(res['family_set']) or 'none'}"
                   f"   excluded: {', '.join(res['family_excluded']) or 'none'}")
        out.append(f"family_set_ablated_max_norm_acc: "
                   f"{'--' if res.get('family_set_ablated_max_norm_acc') is None else format(res['family_set_ablated_max_norm_acc'], '+.3f')}"
                   f" ({res['family_set_ablated_note']})   min over the set of (ratio_min compare - ratio_min recall): "
                   f"{'--' if mind is None else format(mind, '+.3f')} {ci_s}"
                   + ("" if res.get("shared_resample") else "   (not paired across models)"))
        if res.get("family_aggregates_note"):
            out.append("note: " + res["family_aggregates_note"])
        out.append("verdict: " + (f"{prereg_verdict['verdict']} (styxx.protocol, prereg {prereg_verdict['prereg_commit'][:8]})"
                                  if prereg_verdict else "unscored: no committed prereg"))
        out.append(f"reported numbers on this sheet: {len(rows) * 3} horizons and {len(rows)} dissociation intervals "
                   f"beside the one family number")
    out.append("")
    out.append("each cell: chance-normalised accuracy under the row's readout; 95% paired item-bootstrap intervals, "
               "guard, balance and AUROC per cell in the receipt; * = trained context, no positional limit; "
               "screen PASS/FAIL@bar")
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


def _prereg_preconditions(sheet: dict) -> List[str]:
    """What a sheet scored against a prereg may not carry: an inferred compare readout, an assumed
    pairing, a bar or guard other than the bridge-v3 runner's. The instrument cannot read a prereg's own
    values for these, so it holds every scored sheet to the ones it restates."""
    problems = []
    for r in sheet["rows"]:
        cfg = r["config"]
        if cfg["readout_source"]["compare"] != "declared":
            problems.append(f"{r['model']}: the compare readout is {cfg['readout_source']['compare']}, not declared")
        if not cfg["pairing_declared"]:
            problems.append(f"{r['model']}: the pairing is assumed, not declared")
        if cfg["screen_bar"] != SCREEN_BAR:
            problems.append(f"{r['model']}: the screen bar is {cfg['screen_bar']}, not {SCREEN_BAR}")
        if cfg["guard"]["nonstandard"]:
            problems.append(f"{r['model']}: the guard is not the bridge-v3 runner's")
    return problems


def _score_prereg(prereg: str, sheet: dict) -> dict:
    problems = _prereg_preconditions(sheet)
    if problems:
        raise HorizonRefusal("a sheet scored against a prereg reads declared readouts and pairings under the "
                             "bridge-v3 bar and guard only: " + "; ".join(problems))
    from .protocol import Experiment, GateSpecError, PrologueError
    try:
        exp = Experiment(prereg)
        v = exp.score({"result": _jsonable(sheet["result"])})
    except (GateSpecError, PrologueError) as e:
        raise HorizonRefusal(f"styxx.protocol refused to score the sheet: {e}") from e
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
            verdict = _score_prereg(a.prereg, sheet) if a.prereg else None
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
