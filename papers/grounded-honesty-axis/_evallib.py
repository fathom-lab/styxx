"""Shared, UNIT-TESTED evaluation primitives for the grounded-honesty harnesses.

One source of truth for the statistics every FINDING depends on. A research program whose value is
rigor cannot run on copy-pasted, untested stats — a silent bug in `spearman`/`auc` would falsify
conclusions invisibly. This replaces 4x Spearman, 3x AUC, and 2x ad-hoc answer-matcher across
peek_scaling_existing / mine_locus / score_honesty_scaling / score_locus / score_selfaudit /
run_deployment_proof. Pure stdlib. See test_evallib.py for the assertions that keep it honest.
"""
from __future__ import annotations
import math
import re
from itertools import permutations
from typing import Optional, Sequence


# ----------------------------------------------------------------------------- rank statistics
def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation with average-rank tie handling. None if n<3 or a side is constant."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return (num / (dx * dy)) if dx > 0 and dy > 0 else None


def perm_p(xs: Sequence[float], ys: Sequence[float], obs: Optional[float] = None) -> Optional[float]:
    """Exact two-sided permutation p-value for Spearman (enumerates n! orderings). None if n>9."""
    n = len(xs)
    if n > 9 or n < 3:
        return None
    if obs is None:
        obs = spearman(xs, ys)
    if obs is None:
        return None
    cnt = tot = 0
    for perm in permutations(range(n)):
        r = spearman(xs, [ys[i] for i in perm])
        if r is not None and abs(r) >= abs(obs) - 1e-12:
            cnt += 1
        tot += 1
    return cnt / tot if tot else None


# ----------------------------------------------------------------------------- AUC / calibration
def auc_pos_gt_neg(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """Mann-Whitney P(pos > neg), ties at 0.5. None if either side empty."""
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))


def brier(confidences: Sequence[float], corrects: Sequence[bool]) -> Optional[float]:
    """Mean squared error of probabilistic predictions: mean((conf - outcome)^2)."""
    n = len(confidences)
    if n == 0 or len(corrects) != n:
        return None
    return sum((float(c) - (1.0 if o else 0.0)) ** 2 for c, o in zip(confidences, corrects)) / n


def reliability_bins(confidences, corrects, edges=(0.0, 0.5, 0.8, 1.0001)):
    """Accuracy within confidence bands [edges[i], edges[i+1])."""
    out = []
    for lo, hi in zip(edges, edges[1:]):
        b = [o for c, o in zip(confidences, corrects) if lo <= c < hi]
        out.append({"band": f"[{lo},{min(hi,1.0)})", "n": len(b),
                    "accuracy": (sum(1 for o in b if o) / len(b)) if b else None})
    return out


# ----------------------------------------------------------------------------- answer matching
_ARTICLES = {"a", "an", "the"}


def normalize_answer(s: str) -> str:
    """TriviaQA-style normalization: lowercase, drop punctuation, drop articles, collapse whitespace."""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in _ARTICLES)


def _contiguous(seq, sub) -> bool:
    """Is `sub` a contiguous sub-sequence of `seq` (token level)?"""
    n, m = len(seq), len(sub)
    if m == 0 or m > n:
        return False
    return any(seq[i:i + m] == sub for i in range(n - m + 1))


def alias_match(prediction: str, aliases: Sequence[str]) -> bool:
    """Robust TOKEN-level alias match: normalized exact, or one is a contiguous token-run of the other.

    Token-level (not char-substring) avoids false positives like 'in' matching inside 'india', while
    still accepting verbose predictions ('the answer is paris') and alias-within-prediction. It does NOT
    invent abbreviation equivalences (e.g. 'hrs'=='hours') — that is the dataset's job via its alias set.
    """
    p = normalize_answer(prediction).split()
    if not p:
        return False
    for a in aliases:
        q = normalize_answer(a).split()
        if q and (p == q or _contiguous(p, q) or _contiguous(q, p)):
            return True
    return False


# ----------------------------------------------------------------------------- model metadata
PARAMS_B = {"Qwen2.5-0.5B": 0.5, "Qwen2.5-1.5B": 1.5, "Qwen2.5-3B": 3.0, "Qwen2.5-7B": 7.0,
            "Llama-3.2-1B": 1.24, "Llama-3.2-3B": 3.21, "Llama-3.1-8B": 8.0,
            "gemma-2-2b": 2.6, "gemma-2-9b": 9.0, "gemma-3-1b": 1.0}


def params_for(model: str) -> Optional[float]:
    for k, v in PARAMS_B.items():
        if k.lower() in (model or "").lower():
            return v
    return None
