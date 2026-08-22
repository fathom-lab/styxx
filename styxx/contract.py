# -*- coding: utf-8 -*-
"""styxx.contract — catch a confident answer computed from nothing, at call time.

    @measures(min_n=1)
    def forecast(trajectories, n_tokens=None) -> ForecastResult: ...

    forecast({"entropy": [], "logprob": []})     # recorded violation (or a raise)

MEASURED YIELD, AND THE CRITERION THIS FAILED
─────────────────────────────────────────────
A kill criterion was fixed and published before this module was written: replayed
against the 5 **SP-6** cases in `benchmarks/silent_pass`, catch **>= 4**, or the
idea dies and the number gets published.

**It scored 3 of 5. The criterion was not met.** See
`papers/RESULT_contract_sp6_2026_08_21.md` and
`scripts/contract_sp6_replay_real.py`, which replays the real pre-fix functions
extracted from git at `<fix_commit>~1`.

So this is **not** "the SP-6 fix". It is a guard with a measured yield and a
structural blind spot, and both are stated here because a tool that hides its
blind spot is the defect class this module was written to remove.

WHAT IT CATCHES — boundary-degenerate: 3 of 3
    Nothing arrives; something confident leaves.
    SP-2026-0008 (empty trajectories -> confidence 0.695),
    SP-2026-0012 (empty trajectory -> 'steady'),
    SP-2026-0016 (both arms empty -> suspected=False).

WHAT IT CANNOT CATCH — interior-degenerate: 0 of 2
    A **well-formed** argument arrives and the emptiness is manufactured
    *inside* the function. No boundary test reaches these, at any tuning.
    SP-2026-0011: a 20-token response, scoring never completed, gate stayed
        'pending', and the test was `gate != "fail"`.
    SP-2026-0020: four valid Japanese strings; the `[a-z0-9]+` tokenizer
        emptied them internally.

    Corollary from SP-2026-0020, worth its own line: this module can only judge
    a return value that **carries its own name**. `looks_confident(-0.0)` is
    None because polarity is unknowable for a bare float; `{"entropy": -0.0}`
    is flagged. If you return a naked scalar, no polarity heuristic can help
    you -- pass `confident_when=`.

WHY A BOUNDARY TEST AT ALL
──────────────────────────
Static screens fare worse on this subtype: `styxx.absence` catches 0 of 5,
`absence` + `loops` together catch 1 of 5. The defect is *a guard that was never
written*, and no pass over source can flag code that does not exist. You cannot
see the missing guard; you can sometimes see the call where nothing went in and
a confident value came out.

    "was there anything to measure?"   <- inspect the arguments
    "did it claim something anyway?"   <- inspect the return
    the CONJUNCTION is the finding.

Default is RECORD, not raise
────────────────────────────
A decorator that raises in production is removed by the first on-call engineer,
and a removed guard protects nobody. So violations are recorded and warned by
default; `strict=True` raises for tests and CI, where breaking loudly is the
whole point.
"""
from __future__ import annotations

import functools
import inspect
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = ["measures", "ContractViolation", "violations", "clear_violations",
           "is_degenerate", "looks_confident", "contract_violations"]


class ContractViolation(RuntimeError):
    """A confident value was returned from an input with nothing in it."""


@dataclass
class Violation:
    function: str
    why_degenerate: str
    what_was_returned: str
    args_summary: str

    def as_dict(self) -> Dict[str, Any]:
        return {"function": self.function, "why_degenerate": self.why_degenerate,
                "returned": self.what_was_returned, "args": self.args_summary}


_VIOLATIONS: List[Violation] = []


def violations() -> List[Violation]:
    """Everything recorded this process. A record nobody reads is not a guard,
    so this is the surface a caller is expected to check."""
    return list(_VIOLATIONS)


def clear_violations() -> int:
    n = len(_VIOLATIONS)
    _VIOLATIONS.clear()
    return n


# ── "was there anything to measure?" ───────────────────────────────────────

def _all_nonfinite(seq) -> bool:
    vals = [v for v in seq if isinstance(v, (int, float))]
    return bool(vals) and all(
        isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in vals)


def _zero_variance(seq) -> bool:
    vals = [float(v) for v in seq if isinstance(v, (int, float))
            and not (isinstance(v, float) and math.isnan(v))]
    return len(vals) >= 2 and max(vals) == min(vals)


def is_degenerate(value: Any, *, min_n: int = 1) -> Optional[str]:
    """Why this argument carries nothing to measure, or None if it does.

    Deliberately conservative: it names a reason rather than returning a bare
    bool, because a guard that cannot say why it fired is the shape this whole
    program exists to remove.
    """
    if value is None:
        return "None"
    if isinstance(value, str):
        return "empty string" if not value.strip() else None
    if isinstance(value, dict):
        if not value:
            return "empty dict"
        # a dict of sequences (the trajectories shape) is degenerate when every
        # sequence in it is
        seqs = [v for v in value.values() if isinstance(v, (list, tuple))]
        if seqs and all(len(s) < min_n for s in seqs):
            return f"every sequence shorter than min_n={min_n}"
        if seqs and all(_all_nonfinite(s) for s in seqs if s):
            return "every sequence is all-NaN"
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        if len(value) < min_n:
            return f"length {len(value)} < min_n={min_n}"
        if _all_nonfinite(value):
            return "all values non-finite"
        if _zero_variance(value):
            return "zero variance"
        # a sequence whose every element is itself empty (the tokenizer case:
        # a list of answers that all tokenized to nothing)
        inner = [v for v in value if isinstance(v, (list, tuple, set, frozenset, str, dict))]
        if inner and len(inner) == len(value) and all(len(v) == 0 for v in inner):
            return "every element is empty"
        return None
    if hasattr(value, "__len__"):
        try:
            if len(value) < min_n:
                return f"length {len(value)} < min_n={min_n}"
        except TypeError:
            return None
    return None


# ── "did it claim something anyway?" ───────────────────────────────────────

_HIGH_IS_CONFIDENT = ("confidence", "conf", "trust", "score", "grounded",
                      "reliability", "accuracy", "auc", "coherence", "agreement",
                      "stability", "r2")
_LOW_IS_CONFIDENT = ("risk", "error", "drift", "divergence", "entropy",
                     "deception", "hallucination", "violation", "uncertainty")
_HEALTHY_STRINGS = {"pass", "ok", "healthy", "valid", "clean", "safe", "steady",
                    "low", "verified", "sealed"}


def _confident_scalar(name: str, v: Any) -> Optional[str]:
    if isinstance(v, bool):
        return f"{name}={v}" if v else None
    if isinstance(v, (int, float)):
        if isinstance(v, float) and math.isnan(v):
            return None                      # NaN is an honest refusal
        low = name.lower()
        if any(k in low for k in _HIGH_IS_CONFIDENT) and float(v) >= 0.5:
            return f"{name}={v}"
        if any(k in low for k in _LOW_IS_CONFIDENT) and float(v) <= 0.5:
            return f"{name}={v}"
        return None
    if isinstance(v, str) and v.strip().lower() in _HEALTHY_STRINGS:
        return f"{name}={v!r}"
    return None


def looks_confident(result: Any) -> Optional[str]:
    """Describe what this return claims, or None if it claims nothing.

    Polarity-aware, because the flattering end depends on the metric: a trust of
    1.0 and a risk of 0.0 are the same statement.
    """
    if result is None:
        return None
    # a Measured that knows it is unmeasured is exactly right, never a violation
    if getattr(result, "measured", None) is False:
        return None
    if isinstance(result, (bool, int, float, str)):
        return _confident_scalar("return", result)
    if isinstance(result, dict):
        for k, v in result.items():
            hit = _confident_scalar(str(k), v)
            if hit:
                return hit
        return None
    for attr in dir(result):
        if attr.startswith("_"):
            continue
        try:
            v = getattr(result, attr)
        except Exception:
            continue
        if callable(v):
            continue
        hit = _confident_scalar(attr, v)
        if hit:
            return hit
    return None


# ── the contract ───────────────────────────────────────────────────────────

def measures(
    *,
    inputs: Optional[Sequence[str]] = None,
    min_n: int = 1,
    strict: bool = False,
    confident_when: Optional[Callable[[Any], bool]] = None,
):
    """Assert that this function does not claim more than its input supports.

    Args:
        inputs: parameter names carrying the thing being measured. Default: every
                positional parameter except ``self``/``cls``.
        min_n:  the minimum a sequence needs before a measurement is meaningful.
                This is the guard the author would otherwise have to remember;
                declaring it here makes forgetting it visible.
        strict: raise instead of record. Off by default -- a decorator that
                raises in production gets deleted, and a deleted guard protects
                nobody.
        confident_when: override the built-in polarity heuristic.
    """
    def deco(fn: Callable) -> Callable:
        sig = inspect.signature(fn)
        names = list(inputs) if inputs else [
            p.name for p in sig.parameters.values()
            if p.name not in ("self", "cls")
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]

        @functools.wraps(fn)
        def wrapper(*a, **k):
            result = fn(*a, **k)
            try:
                bound = sig.bind_partial(*a, **k)
                bound.apply_defaults()
            except TypeError:
                return result

            reasons = []
            for n in names:
                if n not in bound.arguments:
                    continue
                why = is_degenerate(bound.arguments[n], min_n=min_n)
                if why:
                    reasons.append(f"{n}: {why}")
            if not reasons:
                return result

            claim = ("return (custom predicate)" if confident_when and confident_when(result)
                     else None) if confident_when else looks_confident(result)
            if not claim:
                return result           # degenerate in, nothing claimed out: correct

            v = Violation(function=getattr(fn, "__qualname__", str(fn)),
                          why_degenerate="; ".join(reasons),
                          what_was_returned=str(claim)[:120],
                          args_summary=", ".join(names))
            _VIOLATIONS.append(v)
            msg = (f"styxx.contract: {v.function} returned {v.what_was_returned} "
                   f"from an input with nothing to measure ({v.why_degenerate}). "
                   f"A confident value computed from nothing is indistinguishable "
                   f"from one that was earned.")
            if strict:
                raise ContractViolation(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            return result

        wrapper.__styxx_contract__ = {"inputs": names, "min_n": min_n, "strict": strict}
        return wrapper

    return deco


# exported under a qualified name so `from styxx import *` cannot shadow a
# caller's own `violations`
contract_violations = violations
