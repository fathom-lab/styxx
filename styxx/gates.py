# -*- coding: utf-8 -*-
"""
styxx.gates — programmable gate callbacks.

Driven by the first real user feature request (Xendro, priority #2):

    styxx.on_gate("hallucination > 0.6", lambda v: log("xendro drifting"))
    styxx.on_gate("refusal > 0.7", lambda v: telegram_alert(handro))

The idea is that PASS / FAIL on a styxx card shouldn't be decorative.
Agents should be able to register actions to run when their own
cognitive state crosses a threshold. Monitoring without intervention
is half the feature.

Condition grammar
─────────────────

A gate condition is a tiny DSL for checking a specific phase +
category + threshold combination.

    "<category> <op> <threshold>"
        Any phase predicts <category> with confidence <op> <threshold>.
        Equivalent to:
            any(phase.predicted_category == <category>
                and phase.confidence <op> <threshold>
                for phase in all_non_null_phases(vitals))
        Example:  "hallucination > 0.6"
                  "refusal >= 0.7"

    "<phase>.<category> <op> <threshold>"
        Pin to a specific phase. <phase> is one of:
            p1  or  phase1  or  phase1_pre
            p2  or  phase2  or  phase2_early
            p3  or  phase3  or  phase3_mid
            p4  or  phase4  or  phase4_late
        Example:  "p4.hallucination > 0.5"
                  "phase1.adversarial >= 0.5"

    "gate == <status>"
        Fires when the default gate status matches. <status> is
        one of "pass" / "warn" / "fail" / "pending".
        Example:  "gate == fail"
                  "gate != pass"

Supported ops: `>`, `>=`, `<`, `<=`, `==`, `!=`.

Callback contract
─────────────────

Callbacks receive the Vitals object as their only argument. Any
exception raised by a callback is caught and logged (warnings),
never propagated — the goal is "never break the agent's main loop
because a monitoring hook blew up".

Thread-safety: the gate registry is a module-level list protected
by a simple lock. Register gates from any thread; dispatch walks
a snapshot of the list, so callbacks can safely register more gates
without recursion issues.
"""

from __future__ import annotations

import re
import threading
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from .vitals import Vitals, PhaseReading


# ──────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────

GateCallback = Callable[[Vitals], None]


@dataclass
class RegisteredGate:
    condition: str
    callback: GateCallback
    predicate: Callable[[Vitals], bool]
    # Description for logging / debugging
    name: Optional[str] = None

    def __repr__(self) -> str:
        # Xendro noticed the default dataclass repr dumps function
        # memory addresses for `callback` and `predicate`, which is
        # noise. The useful identifying info is the condition string
        # (and optionally the human-readable name).
        if self.name:
            return f"<styxx gate '{self.name}': {self.condition}>"
        return f"<styxx gate '{self.condition}'>"


_GATES: List[RegisteredGate] = []
_LOCK = threading.Lock()


# ──────────────────────────────────────────────────────────────────
# Condition parser
# ──────────────────────────────────────────────────────────────────

_VALID_CATEGORIES = {
    "retrieval", "reasoning", "refusal",
    "creative", "adversarial", "hallucination",
}

_PHASE_ALIASES = {
    "p1": "phase1_pre", "phase1": "phase1_pre", "phase1_pre": "phase1_pre",
    "p2": "phase2_early", "phase2": "phase2_early", "phase2_early": "phase2_early",
    "p3": "phase3_mid", "phase3": "phase3_mid", "phase3_mid": "phase3_mid",
    "p4": "phase4_late", "phase4": "phase4_late", "phase4_late": "phase4_late",
}

_OPS = {
    ">":  lambda a, b: a >  b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a <  b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}

# Most-specific to least-specific match order
_RE_PHASE_CAT = re.compile(
    r"^\s*(?P<phase>p[1-4]|phase[1-4](?:_pre|_early|_mid|_late)?)"
    r"\s*\.\s*(?P<cat>\w+)"
    r"\s*(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<thr>[0-9]*\.?[0-9]+)\s*$"
)
_RE_CAT_ONLY = re.compile(
    r"^\s*(?P<cat>\w+)"
    r"\s*(?P<op>>=|<=|==|!=|>|<)\s*"
    r"(?P<thr>[0-9]*\.?[0-9]+)\s*$"
)
_RE_GATE_STATUS = re.compile(
    r"^\s*gate\s*(?P<op>==|!=)\s*['\"]?(?P<status>pass|warn|fail|pending)['\"]?\s*$"
)


def parse_condition(condition: str) -> Callable[[Vitals], bool]:
    """Parse a condition string into a predicate function.

    Raises ValueError on unparseable input — better to fail loudly
    at registration time than silently at dispatch time.
    """
    cond = condition.strip()

    # 1. gate status check
    m = _RE_GATE_STATUS.match(cond)
    if m:
        op = _OPS[m.group("op")]
        status = m.group("status")
        def pred(v: Vitals, op=op, status=status) -> bool:
            return op(v.gate, status)
        return pred

    # 2. phase-pinned category check
    m = _RE_PHASE_CAT.match(cond)
    if m:
        phase_key = _PHASE_ALIASES.get(m.group("phase"))
        if phase_key is None:
            raise ValueError(f"unknown phase alias in condition: {cond}")
        cat = m.group("cat").lower()
        if cat not in _VALID_CATEGORIES:
            raise ValueError(
                f"unknown category '{cat}' in condition: {cond}. "
                f"Valid categories: {sorted(_VALID_CATEGORIES)}"
            )
        op_fn = _OPS[m.group("op")]
        thr = float(m.group("thr"))

        def pred(v: Vitals, phase_key=phase_key, cat=cat, op_fn=op_fn, thr=thr) -> bool:
            phase: Optional[PhaseReading] = getattr(v, phase_key, None)
            if phase is None:
                return False
            if phase.predicted_category != cat:
                return False
            return bool(op_fn(phase.confidence, thr))
        return pred

    # 3. forecast risk check (3.2.0): "forecast.risk == critical"
    import re as _re
    m = _re.match(r"forecast\.risk\s*==\s*(\w+)", cond)
    if m:
        target_risk = m.group(1).lower()
        def pred(v: Vitals, target_risk=target_risk) -> bool:
            fc = getattr(v, "forecast", None)
            return fc is not None and fc.risk_level == target_risk
        return pred

    # 4. forecast category check: "forecast.category == hallucination"
    m = _re.match(r"forecast\.category\s*==\s*(\w+)", cond)
    if m:
        target_cat = m.group(1).lower()
        def pred(v: Vitals, target_cat=target_cat) -> bool:
            fc = getattr(v, "forecast", None)
            return fc is not None and fc.predicted_category == target_cat
        return pred

    # 5. coherence threshold: "coherence < 0.5"
    m = _re.match(r"coherence\s*(>|>=|<|<=|==)\s*([\d.]+)", cond)
    if m:
        op_fn = _OPS[m.group(1)]
        thr = float(m.group(2))
        def pred(v: Vitals, op_fn=op_fn, thr=thr) -> bool:
            coh = getattr(v, "coherence", None)
            return coh is not None and bool(op_fn(coh, thr))
        return pred

    # 6. any-phase category check (must come after forecast/coherence)
    m = _RE_CAT_ONLY.match(cond)
    if m:
        cat = m.group("cat").lower()
        if cat not in _VALID_CATEGORIES:
            raise ValueError(
                f"unknown category '{cat}' in condition: {cond}. "
                f"Valid categories: {sorted(_VALID_CATEGORIES)}"
            )
        op_fn = _OPS[m.group("op")]
        thr = float(m.group("thr"))

        def pred(v: Vitals, cat=cat, op_fn=op_fn, thr=thr) -> bool:
            for phase_key in (
                "phase1_pre", "phase2_early", "phase3_mid", "phase4_late",
            ):
                phase: Optional[PhaseReading] = getattr(v, phase_key, None)
                if phase is None:
                    continue
                if (phase.predicted_category == cat
                        and op_fn(phase.confidence, thr)):
                    return True
            return False
        return pred

    raise ValueError(
        f"could not parse gate condition: {condition!r}. "
        "Valid examples: 'hallucination > 0.6', 'p4.refusal >= 0.7', "
        "'gate == fail', 'forecast.risk == critical', 'coherence < 0.5'"
    )


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def on_gate(
    condition: str,
    callback: GateCallback,
    *,
    name: Optional[str] = None,
) -> RegisteredGate:
    """Register a gate callback.

    The callback is invoked with a styxx.Vitals object whenever a
    watch session (or styxx.OpenAI adapter) sees vitals that match
    the condition.

    Args:
        condition: DSL string describing the trigger. See module
                   docstring for grammar.
        callback:  callable invoked with the Vitals object.
        name:      optional human-readable label for debugging.

    Returns:
        The RegisteredGate so callers can later remove it via
        styxx.remove_gate(gate).

    Raises:
        ValueError if the condition string can't be parsed.

    Shorthands (0.8.1):
        on_gate('warn', cb) expands to 'gate == warn'
        on_gate('fail', cb) expands to 'gate == fail'
        on_gate('hallucination', cb) expands to 'hallucination > 0.20'
    """
    # 0.8.1: accept common shorthands
    _GATE_SHORTHANDS = {
        "warn": "gate == warn",
        "fail": "gate == fail",
        "pass": "gate == pass",
        "pending": "gate == pending",
    }
    _CATEGORY_NAMES = {
        "hallucination", "refusal", "adversarial",
        "reasoning", "retrieval", "creative",
    }
    if condition in _GATE_SHORTHANDS:
        condition = _GATE_SHORTHANDS[condition]
    elif condition in _CATEGORY_NAMES:
        condition = f"{condition} > 0.20"

    predicate = parse_condition(condition)
    gate = RegisteredGate(
        condition=condition,
        callback=callback,
        predicate=predicate,
        name=name,
    )
    with _LOCK:
        _GATES.append(gate)
    return gate


def remove_gate(gate: RegisteredGate) -> bool:
    """Remove a previously-registered gate. Returns True if it was
    found and removed, False if it wasn't in the registry."""
    with _LOCK:
        try:
            _GATES.remove(gate)
            return True
        except ValueError:
            return False


def clear_gates() -> int:
    """Remove all registered gates. Returns the number removed.
    Useful in tests."""
    with _LOCK:
        n = len(_GATES)
        _GATES.clear()
        return n


def list_gates() -> List[RegisteredGate]:
    """Return a snapshot of the currently registered gates."""
    with _LOCK:
        return list(_GATES)


def dispatch_gates(vitals: Vitals, *, response: Any = None) -> int:
    """Evaluate every registered gate against `vitals` and call the
    callback for each match. Returns the number of callbacks that
    fired.

    Called automatically by WatchSession.__exit__ and by
    styxx.OpenAI after attaching vitals to a response. You usually
    don't need to call this yourself unless you're building a
    custom adapter.
    """
    # Take a snapshot under the lock, then release before invoking
    # callbacks so callbacks can safely call on_gate/remove_gate.
    with _LOCK:
        snapshot = list(_GATES)

    fired = 0
    for gate in snapshot:
        try:
            matched = gate.predicate(vitals)
        except Exception as e:
            warnings.warn(
                f"styxx gate predicate raised for condition "
                f"{gate.condition!r}: {type(e).__name__}: {e}. "
                "Skipping this gate.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if not matched:
            continue
        try:
            gate.callback(vitals)
            fired += 1
        except Exception as e:
            warnings.warn(
                f"styxx gate callback for {gate.condition!r} raised "
                f"{type(e).__name__}: {e}. The gate is still registered; "
                "fix the callback or remove_gate() to stop this warning. "
                "The agent's main loop is unaffected — styxx fails open.",
                RuntimeWarning,
                stacklevel=2,
            )
    return fired
