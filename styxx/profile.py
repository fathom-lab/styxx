"""
styxx.profile — cognitive profiler for LLM agents.

py-spy for reasoning. Attach to any agent, see where cognition failed
before the output did.

Usage
-----

Decorator form (most common)::

    import styxx

    @styxx.profile
    def my_agent(task):
        return run_langchain_agent(task)

    result, p = my_agent("summarize this contract")
    print(p.summary)              # one-line verdict
    print(p.faults)               # list of localized faults
    p.to_html("run.html")         # open in browser for the flamegraph
    p.to_json("run.json")         # LangSmith / Datadog-compatible export

Context-manager form::

    with styxx.profile(name="sql_agent") as p:
        r1 = client.chat.completions.create(...)
        r2 = client.chat.completions.create(...)
    print(p.summary)

Manual recording (for custom adapters that bypass styxx's hooks)::

    p = styxx.profile_session(name="my_agent")
    p.record(response1, label="plan")
    p.record(response2, label="execute")
    p.finish()

What it does
------------

Wraps a block / function / adapter. Every LLM call that produces a
Vitals inside the block is captured as a ProfileStep. After the block
finishes, per-step and cross-step faults are surfaced: drift,
confabulation, refusal, sycophancy, phase transition, low trust,
incoherence.

The output is three things:
  1. A one-line summary you can log/alert on.
  2. A structured faults list with step/severity/reason.
  3. An HTML flamegraph for humans — and JSON for LangSmith/Datadog.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, TypeVar, Union

from .vitals import Vitals

F = TypeVar("F", bound=Callable[..., Any])


# ══════════════════════════════════════════════════════════════════
# Thread-local active-profile stack
# ══════════════════════════════════════════════════════════════════

_active_stack = threading.local()


def _get_stack() -> List["CognitiveProfile"]:
    if not hasattr(_active_stack, "profiles"):
        _active_stack.profiles = []
    return _active_stack.profiles


def _current_profile() -> Optional["CognitiveProfile"]:
    stack = _get_stack()
    return stack[-1] if stack else None


def _push_profile(p: "CognitiveProfile") -> None:
    _get_stack().append(p)


def _pop_profile() -> Optional["CognitiveProfile"]:
    stack = _get_stack()
    if stack:
        return stack.pop()
    return None


# ══════════════════════════════════════════════════════════════════
# Fault model
# ══════════════════════════════════════════════════════════════════

# Canonical fault kinds
K_DRIFT = "drift"
K_CONFAB = "confabulation"
K_REFUSAL = "refusal"
K_SYCOPHANT = "sycophant"
K_PHASE_TRANSITION = "phase_transition"
K_LOW_TRUST = "low_trust"
K_INCOHERENCE = "incoherence"

_DRIFT_CATEGORIES = {"tool_arg_drift", "drift", "tool_confab", "arg_swap"}
_CONFAB_CATEGORIES = {"confab", "confabulation", "hallucination", "fabrication"}
_REFUSAL_CATEGORIES = {"refuse", "refusal"}
_SYCOPHANT_CATEGORIES = {"sycophant", "sycophancy"}


@dataclass
class Fault:
    """One localized cognitive failure found by the profiler."""

    step_index: int
    kind: str              # K_DRIFT / K_CONFAB / K_REFUSAL / ... (see module constants)
    severity: float        # 0.0 = noise, 1.0 = maximally confident fault
    reason: str            # human-readable single-line explanation
    vitals_snapshot: Optional[dict] = None

    def __str__(self) -> str:
        return (
            f"[{self.kind}] step={self.step_index} "
            f"sev={self.severity:.2f} · {self.reason}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _detect_faults_for_step(
    step_index: int, vitals: Optional[Vitals]
) -> List[Fault]:
    if vitals is None:
        return []
    faults: List[Fault] = []

    cat = (getattr(vitals, "category", "") or "").lower()
    conf = float(getattr(vitals, "confidence", 0.0) or 0.0)
    trust = float(getattr(vitals, "trust_score", 0.0) or 0.0)
    coherence = getattr(vitals, "coherence", None)

    snapshot: Optional[dict] = None
    try:
        snapshot = vitals.to_dict()
    except Exception:
        snapshot = None

    if cat in _DRIFT_CATEGORIES and conf > 0.5:
        faults.append(Fault(
            step_index, K_DRIFT, conf,
            f"category='{cat}' at confidence {conf:.2f}",
            snapshot,
        ))
    if cat in _CONFAB_CATEGORIES and conf > 0.5:
        faults.append(Fault(
            step_index, K_CONFAB, conf,
            f"category='{cat}' at confidence {conf:.2f}",
            snapshot,
        ))
    if cat in _SYCOPHANT_CATEGORIES and conf > 0.5:
        faults.append(Fault(
            step_index, K_SYCOPHANT, conf,
            f"sycophantic tone at confidence {conf:.2f}",
            snapshot,
        ))
    # Refusal is often informational — only flag strong refusals.
    if cat in _REFUSAL_CATEGORIES and conf > 0.8:
        faults.append(Fault(
            step_index, K_REFUSAL, conf,
            f"strong refusal at confidence {conf:.2f}",
            snapshot,
        ))

    if trust < 0.3:
        faults.append(Fault(
            step_index, K_LOW_TRUST, 1.0 - trust,
            f"trust={trust:.2f} below 0.30 threshold",
            snapshot,
        ))

    if coherence is not None:
        try:
            c = float(coherence)
            if c < 0.3:
                faults.append(Fault(
                    step_index, K_INCOHERENCE, 1.0 - c,
                    f"cross-phase coherence={c:.2f} collapsed",
                    snapshot,
                ))
        except (TypeError, ValueError):
            pass

    return faults


def _detect_phase_transitions(steps: List["ProfileStep"]) -> List[Fault]:
    """Flag steps where cognitive category shifts between adjacent calls."""
    faults: List[Fault] = []
    prev_cat: Optional[str] = None
    for step in steps:
        cur_cat: Optional[str] = None
        if step.vitals is not None:
            cur_cat = (getattr(step.vitals, "category", "") or "").lower() or None
        if prev_cat and cur_cat and prev_cat != cur_cat:
            faults.append(Fault(
                step.index, K_PHASE_TRANSITION, 0.5,
                f"category shift: {prev_cat} → {cur_cat}",
                None,
            ))
        if cur_cat:
            prev_cat = cur_cat
    return faults


# ══════════════════════════════════════════════════════════════════
# Profile step
# ══════════════════════════════════════════════════════════════════

@dataclass
class ProfileStep:
    """One LLM call inside a profile."""

    index: int
    label: str
    started_ts: float
    duration_s: float = 0.0
    prompt: Optional[str] = None
    response_text: Optional[str] = None
    vitals: Optional[Vitals] = None

    def to_dict(self) -> dict:
        d: dict = {
            "index": self.index,
            "label": self.label,
            "started_ts": self.started_ts,
            "duration_s": round(self.duration_s, 4),
            "prompt": (self.prompt or "")[:500] if self.prompt else None,
            "response_text": (self.response_text or "")[:500] if self.response_text else None,
            "vitals": None,
        }
        if self.vitals is not None:
            try:
                d["vitals"] = self.vitals.to_dict()
            except Exception:
                d["vitals"] = None
        return d


# ══════════════════════════════════════════════════════════════════
# CognitiveProfile
# ══════════════════════════════════════════════════════════════════

@dataclass
class CognitiveProfile:
    """Ordered record of cognitive events + localized faults."""

    name: str = "agent"
    started_ts: float = field(default_factory=time.time)
    finished_ts: Optional[float] = None
    steps: List[ProfileStep] = field(default_factory=list)
    faults: List[Fault] = field(default_factory=list)
    _finished: bool = False

    # ------------------------------------------------------------------
    # recording
    # ------------------------------------------------------------------
    def record(
        self,
        response: Any = None,
        *,
        label: Optional[str] = None,
        vitals: Optional[Vitals] = None,
        prompt: Optional[str] = None,
    ) -> ProfileStep:
        """Append a step to this profile.

        Called automatically when this is the active profile (via the
        tap on analytics.write_audit). Can also be called manually for
        custom adapters that bypass styxx's hooks.
        """
        now = time.time()
        idx = len(self.steps)

        if vitals is None:
            # (1) vitals already attached to the response?
            v = getattr(response, "vitals", None)
            if isinstance(v, Vitals):
                vitals = v

        if vitals is None and response is not None:
            # (2) try to observe it
            try:
                from .watch import observe as _obs
                vitals = _obs(response, prompt=prompt)
            except Exception:
                vitals = None

        response_text = _extract_text(response)
        prev_end = (
            self.steps[-1].started_ts + self.steps[-1].duration_s
            if self.steps
            else self.started_ts
        )
        duration = max(0.0, now - prev_end)

        step = ProfileStep(
            index=idx,
            label=label or f"step_{idx}",
            started_ts=prev_end,
            duration_s=duration,
            prompt=prompt,
            response_text=response_text,
            vitals=vitals,
        )
        self.steps.append(step)
        self.faults.extend(_detect_faults_for_step(idx, vitals))
        return step

    def finish(self) -> "CognitiveProfile":
        if self._finished:
            return self
        self.finished_ts = time.time()
        self.faults.extend(_detect_phase_transitions(self.steps))
        self._finished = True
        return self

    # ------------------------------------------------------------------
    # views
    # ------------------------------------------------------------------
    @property
    def duration_s(self) -> float:
        end = self.finished_ts or time.time()
        return max(0.0, end - self.started_ts)

    @property
    def summary(self) -> str:
        if not self.steps:
            return f"profile '{self.name}': 0 steps observed"

        lines: List[str] = []
        lines.append(
            f"profile '{self.name}': {len(self.steps)} steps, "
            f"{self.duration_s:.2f}s total"
        )
        if self.faults:
            # dedupe by (kind, step_index) and sort by severity
            seen = set()
            uniq: List[Fault] = []
            for f in self.faults:
                k = (f.kind, f.step_index)
                if k in seen:
                    continue
                seen.add(k)
                uniq.append(f)
            lines.append(f"  {len(uniq)} fault(s):")
            for f in sorted(uniq, key=lambda x: -x.severity)[:5]:
                lines.append(f"    · {f}")
            if len(uniq) > 5:
                lines.append(f"    · ... and {len(uniq) - 5} more")
        else:
            lines.append("  no faults detected")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"CognitiveProfile(name={self.name!r}, steps={len(self.steps)}, "
            f"faults={len(self.faults)}, duration_s={self.duration_s:.2f})"
        )

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "started_ts": self.started_ts,
            "finished_ts": self.finished_ts,
            "duration_s": round(self.duration_s, 4),
            "n_steps": len(self.steps),
            "n_faults": len(self.faults),
            "steps": [s.to_dict() for s in self.steps],
            "faults": [f.to_dict() for f in self.faults],
        }

    def to_json(
        self,
        path: Optional[Union[str, Path]] = None,
        *,
        indent: int = 2,
    ) -> str:
        """Serialize the profile as JSON. Writes to ``path`` if given."""
        data = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            Path(path).write_text(data, encoding="utf-8")
        return data

    def to_html(self, path: Optional[Union[str, Path]] = None) -> str:
        """Render a self-contained HTML flamegraph (no external assets)."""
        from ._profile_html import render_flamegraph
        html = render_flamegraph(self)
        if path is not None:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def to_langsmith(self) -> dict:
        """LangSmith-compatible trace object (one parent span + N child spans)."""
        children = []
        for step in self.steps:
            children.append({
                "name": step.label,
                "run_type": "llm",
                "start_time": step.started_ts,
                "end_time": step.started_ts + step.duration_s,
                "inputs": {"prompt": step.prompt or ""},
                "outputs": {"output": step.response_text or ""},
                "extra": {
                    "styxx": step.vitals.to_dict() if step.vitals is not None else None,
                    "styxx_faults": [
                        f.to_dict() for f in self.faults
                        if f.step_index == step.index
                    ],
                },
            })
        return {
            "name": self.name,
            "run_type": "chain",
            "start_time": self.started_ts,
            "end_time": self.finished_ts or time.time(),
            "child_runs": children,
        }

    def to_datadog(self) -> dict:
        """Datadog APM-style span payload."""
        spans = []
        trace_id = int(self.started_ts * 1_000_000)
        for step in self.steps:
            span: dict = {
                "name": f"llm.{step.label}",
                "resource": step.label,
                "trace_id": trace_id,
                "span_id": int((step.started_ts + step.index) * 1_000_000),
                "start": int(step.started_ts * 1_000_000_000),
                "duration": int(step.duration_s * 1_000_000_000),
                "meta": {},
                "metrics": {},
            }
            if step.vitals is not None:
                try:
                    v = step.vitals.to_dict()
                    span["meta"]["styxx.category"] = str(v.get("category", ""))
                    span["meta"]["styxx.gate"] = str(v.get("gate", ""))
                    span["metrics"]["styxx.confidence"] = float(v.get("confidence") or 0.0)
                    span["metrics"]["styxx.trust"] = float(v.get("trust") or 0.0)
                except Exception:
                    pass
            if any(f.step_index == step.index for f in self.faults):
                span["meta"]["styxx.faulted"] = "true"
            spans.append(span)
        return {"spans": spans}


# ══════════════════════════════════════════════════════════════════
# Tap installation (idempotent)
# ══════════════════════════════════════════════════════════════════

_tap_installed = False


def _install_tap() -> None:
    """Patch analytics.write_audit to fan out to the active profile.

    The patch is idempotent and additive — the original function still
    runs. A no-op if no profile is currently active.
    """
    global _tap_installed
    if _tap_installed:
        return
    from . import analytics

    original = analytics.write_audit

    def write_audit_tapped(vitals, *args, **kwargs):
        try:
            original(vitals, *args, **kwargs)
        finally:
            p = _current_profile()
            if p is not None and not p._finished:
                try:
                    p.record(
                        response=None,
                        label=f"step_{len(p.steps)}",
                        vitals=vitals,
                        prompt=kwargs.get("prompt"),
                    )
                except Exception:
                    pass

    analytics.write_audit = write_audit_tapped
    _tap_installed = True


def _extract_text(response: Any) -> Optional[str]:
    """Best-effort extraction of response text from common SDK shapes."""
    if response is None:
        return None
    if isinstance(response, str):
        return response
    try:
        choices = getattr(response, "choices", None)
        if choices:
            msg = choices[0].message
            content = getattr(msg, "content", None)
            if content:
                return str(content)
    except Exception:
        pass
    try:
        content = getattr(response, "content", None)
        if isinstance(content, list) and content:
            parts = []
            for c in content:
                txt = getattr(c, "text", None)
                if txt:
                    parts.append(str(txt))
            if parts:
                return "".join(parts)
    except Exception:
        pass
    if isinstance(response, dict):
        if "text" in response:
            return str(response["text"])
        if "content" in response:
            return str(response["content"])
    return None


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

@contextmanager
def _profile_ctx(name: str = "agent", *, auto_hook: bool = True):
    _install_tap()
    if auto_hook:
        try:
            from .hooks import hook_openai, hook_openai_active
            if not hook_openai_active():
                hook_openai()
        except Exception:
            # openai SDK not installed — profile still works for
            # manual record() calls and for Anthropic/other adapters.
            pass

    p = CognitiveProfile(name=name)
    _push_profile(p)
    try:
        yield p
    finally:
        _pop_profile()
        p.finish()


def profile(func_or_name=None, *, auto_hook: bool = True):
    """Decorate a function, or enter a context manager, to profile cognition.

    Three call styles are supported:

    1. Bare decorator — infers name from function::

        @styxx.profile
        def my_agent(task):
            ...
            return result

        output, p = my_agent("...")

    2. Parameterized decorator — explicit name::

        @styxx.profile(name="sql_agent")
        def my_agent(task):
            ...

        output, p = my_agent("...")

    3. Context manager::

        with styxx.profile(name="sql_agent") as p:
            do_stuff()
        print(p.summary)

    Args:
        auto_hook: If True (default) and the openai SDK is present,
            styxx.hook_openai() is installed automatically so all
            ``openai.OpenAI()`` clients get Vitals attached.
    """
    # (1) @styxx.profile — bare decorator
    if callable(func_or_name):
        fn = func_or_name
        agent_name = getattr(fn, "__name__", "agent")

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with _profile_ctx(name=agent_name, auto_hook=auto_hook) as p:
                result = fn(*args, **kwargs)
            return result, p

        return wrapped

    # (2) @styxx.profile(name=...) OR styxx.profile(name=...) context-manager
    resolved_name = func_or_name if isinstance(func_or_name, str) else "agent"

    if isinstance(func_or_name, str):
        # Parametric decorator: return a decorator factory.
        def decorator(fn: F) -> F:
            @functools.wraps(fn)
            def wrapped(*args, **kwargs):
                with _profile_ctx(name=resolved_name, auto_hook=auto_hook) as p:
                    result = fn(*args, **kwargs)
                return result, p
            return wrapped  # type: ignore[return-value]
        # Also usable as a context manager when someone writes
        # ``with styxx.profile("name") as p`` — return a context manager
        # by default; the decorator case is handled in the branch above
        # when the argument is a callable. To support BOTH ergonomics,
        # we return an object that is both a context manager AND
        # callable. For simplicity in v0.1 we just return the CM.
        return _profile_ctx(name=resolved_name, auto_hook=auto_hook)

    # (3) no arg — bare context manager: styxx.profile()
    return _profile_ctx(name=resolved_name, auto_hook=auto_hook)


def profile_session(name: str = "agent") -> CognitiveProfile:
    """Create an empty profile for manual ``.record(...)`` calls.

    Use when your adapter bypasses styxx's hooks — e.g. a custom HTTP
    wrapper around a self-hosted model::

        p = styxx.profile_session(name="my_agent")
        p.record(r1, label="plan")
        p.record(r2, label="execute")
        p.finish()
        print(p.summary)
    """
    _install_tap()
    return CognitiveProfile(name=name)


__all__ = [
    "CognitiveProfile",
    "ProfileStep",
    "Fault",
    "profile",
    "profile_session",
    "K_DRIFT",
    "K_CONFAB",
    "K_REFUSAL",
    "K_SYCOPHANT",
    "K_PHASE_TRANSITION",
    "K_LOW_TRUST",
    "K_INCOHERENCE",
]
