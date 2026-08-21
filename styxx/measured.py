# -*- coding: utf-8 -*-
"""styxx.measured — a value that knows whether it was measured.

    from styxx import Measured, NoComputedData

    trust = Measured(0.92, source="consensus_n=5")
    trust > 0.7                      # True

    trust = Measured.unmeasured("api key rejected", source="gate()")
    trust > 0.7                      # UnmeasuredComparison, not a silent False
    trust.value                      # UnmeasuredValue, not a plausible number
    trust.value_or(0.5)              # 0.5 — explicit, deliberate, yours

The argument
────────────
Across three audit waves this package produced 74 defects of one shape: a
measurement failed, or never ran, and something returned a value
indistinguishable from a healthy one. Every fix took the same form. Counted
afterwards: **14 modules, ~61 sites**, each growing a hand-rolled flag —
``measured``, ``confidence_measured``, ``portable_present``, ``outcome_source``,
``deception_mode``, ``n_auto_excluded``.

That is one abstraction, invented fourteen times, badly, under fourteen names.
This module is that abstraction with a name.

Why the bug is structural, not careless
───────────────────────────────────────
When a measurement fails you need a fallback, and the instinct is to pick
something *inert* — a value that does not disturb anything downstream. But a
measurement's downstream is a DECISION, and the non-disturbing value for a
decision is "do not act":

    a risk score's inert value is 0.0     ... which means "no risk found"
    a trust score's inert value is 1.0    ... which means "fully trusted"
    a gate's inert value is "do not block" ... which means "approved"

**The inert default and the flattering default are the same value.** The
engineering virtue (never break the caller) is exactly the epistemic vice (claim
something you did not measure). That is why all 74 pointed one way, and why
carefulness alone does not fix it.

Prior art we are late to
────────────────────────
Every measurement discipline that has hurt somebody evolved a validity channel
separate from the value channel:

  * **ARINC 429** (avionics data bus, 1977) — the Sign/Status Matrix carries
    ``No Computed Data`` as a state distinct from any number, so a failed sensor
    cannot present as a reading. Aviation learned this from pitot-static
    failures.
  * **OPC-UA** (industrial control) — every value ships a quality code:
    Good / Uncertain / Bad.
  * **Medical telemetry** — a signal-quality index rides beside the waveform,
    because a disconnected lead reading 0 must not look like a calm heart.

AI measurement ships bare floats. ``trust_score: 0.92`` carries no bit for "and
this was actually computed". This module adds the bit, fifty years late, with
the acknowledgement that the idea is borrowed.

The design constraint
─────────────────────
The lazy path must be the honest one. Today the easy thing (return a plausible
number) is the wrong thing, so honesty costs discipline and eventually loses.
Here the easy thing raises, and getting a bare number out requires saying so:

    m.value          -> raises if unmeasured
    m.value_or(0.5)  -> your default, deliberately chosen, at the call site
    bool(m)          -> raises if unmeasured  (kills the truthiness-gate class)
    m > 0.7          -> raises if unmeasured  (kills the silent-False class)

``lenient()`` exists for pipelines that genuinely must not break — it warns and
degrades instead of raising. It is a context manager rather than a default
because a guard that can be enabled by accident can be disabled by accident.
"""
from __future__ import annotations

import contextlib
import threading
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

__all__ = [
    "Measured", "NoComputedData", "UnmeasuredValue", "UnmeasuredComparison",
    "lenient", "measure",
]

T = TypeVar("T")

_state = threading.local()


class UnmeasuredValue(RuntimeError):
    """Raised when a value that was never measured is read as if it were."""


class UnmeasuredComparison(RuntimeError):
    """Raised when an unmeasured value is compared against a threshold.

    A silent ``False`` here is the defect: ``NaN > 0.6`` is False, so a broken
    normalization lands on the calm branch by fallthrough. Refusing is the
    difference between "we checked and it is fine" and "we could not check".
    """


def _is_lenient() -> bool:
    return getattr(_state, "lenient", False)


@contextlib.contextmanager
def lenient(warn: bool = True):
    """Degrade instead of raising, for pipelines that must not break.

    Inside this block an unmeasured comparison returns False and an unmeasured
    read returns None, both with a warning. This is a deliberate escape hatch:
    a guard that raises in production gets deleted by the first on-call
    engineer, and a deleted guard protects nobody. But it is scoped, so it
    cannot become the ambient default by accident.
    """
    prev = getattr(_state, "lenient", False)
    prev_warn = getattr(_state, "lenient_warn", True)
    _state.lenient, _state.lenient_warn = True, warn
    try:
        yield
    finally:
        _state.lenient, _state.lenient_warn = prev, prev_warn


def _soft(msg: str):
    if getattr(_state, "lenient_warn", True):
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


@dataclass(frozen=True)
class Measured(Generic[T]):
    """A value carrying whether it was measured, and if not, why not.

    ``Measured(v)`` is a real reading. ``Measured.unmeasured(why)`` is the
    ARINC-429 *No Computed Data* state: a first-class "nothing was computed",
    which is not a number and refuses to pretend to be one.
    """

    _value: Optional[T] = None
    measured: bool = True
    why: Optional[str] = None          # why not, when measured is False
    source: Optional[str] = None       # what produced it (provenance)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def unmeasured(cls, why: str, *, source: Optional[str] = None,
                   **meta: Any) -> "Measured[T]":
        """NO COMPUTED DATA. `why` is required — an unexplained absence is the
        thing this module exists to stop."""
        if not why or not str(why).strip():
            raise ValueError(
                "Measured.unmeasured() requires a reason. An absence nobody can "
                "explain is indistinguishable from an absence nobody noticed.")
        return cls(_value=None, measured=False, why=str(why), source=source,
                   meta=dict(meta))

    # ── reading ─────────────────────────────────────────────────────────
    @property
    def value(self) -> T:
        """The measurement. Raises if there wasn't one."""
        if self.measured:
            return self._value  # type: ignore[return-value]
        if _is_lenient():
            _soft(f"styxx.measured: reading an unmeasured value ({self.why}); "
                  f"lenient mode returns None")
            return None  # type: ignore[return-value]
        raise UnmeasuredValue(
            f"no value was measured: {self.why}"
            + (f" [source: {self.source}]" if self.source else "")
            + ". Use .value_or(default) to choose a fallback explicitly, or "
              "styxx.measured.lenient() to degrade instead of raising.")

    def value_or(self, default: T) -> T:
        """The measurement, or YOUR default — chosen here, visibly, by you."""
        return self._value if self.measured else default  # type: ignore[return-value]

    # ── the two operations that produced the 74 defects ────────────────
    def _compare(self, other: Any, op: Callable[[Any, Any], bool], sym: str) -> bool:
        if self.measured:
            return op(self._value, other.value if isinstance(other, Measured) else other)
        if _is_lenient():
            _soft(f"styxx.measured: comparing an unmeasured value ({self.why}); "
                  f"lenient mode returns False — which is NOT 'below threshold'")
            return False
        raise UnmeasuredComparison(
            f"cannot evaluate `{sym} {other!r}`: nothing was measured ({self.why}). "
            f"A silent False here would read as 'checked, and fine'.")

    def __gt__(self, o): return self._compare(o, lambda a, b: a > b, ">")
    def __ge__(self, o): return self._compare(o, lambda a, b: a >= b, ">=")
    def __lt__(self, o): return self._compare(o, lambda a, b: a < b, "<")
    def __le__(self, o): return self._compare(o, lambda a, b: a <= b, "<=")

    def __bool__(self) -> bool:
        """Truthiness refuses when unmeasured — the SP-4 gate-bypass class.

        `fired = bool(v.fired or v.needs_revision)` shipped in this package: a
        list is truthy when non-empty, so the calibrated term could never decide.
        A Measured cannot be silently coerced into a decision.
        """
        if self.measured:
            return bool(self._value)
        if _is_lenient():
            _soft(f"styxx.measured: truthiness of an unmeasured value ({self.why}); "
                  f"lenient mode returns False")
            return False
        raise UnmeasuredComparison(
            f"cannot take the truth value of an unmeasured result ({self.why}). "
            f"Check .measured first, or use .value_or(...).")

    # ── propagation: unmeasured is contagious, and keeps its reason ─────
    def map(self, fn: Callable[[T], Any], *, source: Optional[str] = None) -> "Measured":
        """Transform the value, carrying validity through untouched.

        Arithmetic on a missing measurement yields a missing measurement — the
        provenance survives the pipeline instead of a 0.0 laundering itself into
        a real-looking aggregate three functions downstream.
        """
        if not self.measured:
            return self
        return Measured(fn(self._value), source=source or self.source, meta=dict(self.meta))

    # ── carrying it across a boundary ───────────────────────────────────
    def as_dict(self) -> Dict[str, Any]:
        """Serialize WITH the validity channel. A wire format that drops it
        re-creates the problem at the next hop."""
        return {"value": self._value, "measured": self.measured,
                "why": self.why, "source": self.source, "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Measured":
        if not d.get("measured", True):
            return cls.unmeasured(d.get("why") or "unspecified (deserialized)",
                                  source=d.get("source"), **(d.get("meta") or {}))
        return cls(d.get("value"), source=d.get("source"), meta=dict(d.get("meta") or {}))

    def __repr__(self) -> str:
        if not self.measured:
            src = f", source={self.source!r}" if self.source else ""
            return f"<NCD no computed data: {self.why}{src}>"
        src = f", source={self.source!r}" if self.source else ""
        return f"<Measured {self._value!r}{src}>"


#: ARINC-429's name for the state, for readers who know it from avionics.
NoComputedData = Measured.unmeasured


def measure(fn: Callable[..., T]) -> Callable[..., Measured[T]]:
    """Wrap a scoring function so a raised exception becomes NO COMPUTED DATA
    rather than whatever the ``except`` branch felt like returning.

        @measure
        def trust_score(response): ...

        trust_score(r) > 0.7     # raises if the scorer blew up
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k) -> Measured[T]:
        try:
            out = fn(*a, **k)
        except Exception as e:
            return Measured.unmeasured(f"{type(e).__name__}: {e}",
                                       source=getattr(fn, "__qualname__", str(fn)))
        return out if isinstance(out, Measured) else Measured(
            out, source=getattr(fn, "__qualname__", str(fn)))

    return wrapper
