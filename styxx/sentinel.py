# -*- coding: utf-8 -*-
"""
styxx.sentinel — real-time drift watcher.

    sentinel = styxx.sentinel(
        on_drift=lambda alert: print(f"DRIFT: {alert}"),
        on_streak=lambda alert: print(f"STREAK: {alert}"),
        window=5,
    )

    # ... later, in your agent loop ...
    styxx.log(mood="cautious", note="hedging again")
    # → sentinel fires: "you've logged cautious 3 times in a row"

    styxx.observe(response)
    # → sentinel fires: "warn gate 4 of last 5 observations"

Xendro's #2 request: "I want a watcher that runs in the background
of my session and pings me when I'm drifting. Not 'your refusal
rate was high yesterday' — 'you've hedged 3 responses in a row,
right now.' Real-time self-correction loop."

The sentinel watches BOTH:
  1. styxx.log() calls (self-report stream)
  2. styxx.observe() / observe_raw() calls (computed vitals)

It fires callbacks when it detects patterns in the recent window:
  - consecutive same-mood self-reports
  - rising warn/fail gate rate
  - category concentration above threshold
  - confidence dropping below baseline

The sentinel hooks into the audit log write path — every time
write_audit() or log() writes an entry, the sentinel checks the
last N entries and fires if a pattern matches. Zero polling,
zero background threads, pure event-driven.

0.5.9+.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ══════════════════════════════════════════════════════════════════
# Alert types
# ══════════════════════════════════════════════════════════════════

@dataclass
class SentinelAlert:
    """One alert from the sentinel."""
    kind: str           # "drift" | "streak" | "warn_rate" | "confidence_drop"
    message: str        # human-readable alert
    severity: str       # "info" | "warning" | "critical"
    window_size: int    # how many recent entries were checked
    trigger_value: Any  # the value that triggered the alert


# ══════════════════════════════════════════════════════════════════
# The sentinel
# ══════════════════════════════════════════════════════════════════

AlertCallback = Callable[[SentinelAlert], None]

# Module-level singleton — one sentinel per process
_SENTINEL: Optional["Sentinel"] = None


class Sentinel:
    """Real-time drift watcher.

    Hooks into the audit log write path and checks the last N
    entries after every write. Fires callbacks when patterns match.

    Usage:

        s = styxx.sentinel(
            on_drift=lambda a: print(a.message),
            window=5,
        )

        # sentinel is now active — it fires automatically
        # whenever styxx.log() or styxx.observe() writes
        # to the audit log.

        # to stop:
        s.stop()
    """

    def __init__(
        self,
        *,
        on_drift: Optional[AlertCallback] = None,
        on_streak: Optional[AlertCallback] = None,
        on_warn: Optional[AlertCallback] = None,
        on_confidence_drop: Optional[AlertCallback] = None,
        on_any: Optional[AlertCallback] = None,
        window: int = 5,
        streak_threshold: int = 3,
        warn_rate_threshold: float = 0.5,
        confidence_floor: float = 0.25,
    ):
        self.on_drift = on_drift
        self.on_streak = on_streak
        self.on_warn = on_warn
        self.on_confidence_drop = on_confidence_drop
        self.on_any = on_any
        self.window = max(2, window)
        self.streak_threshold = max(2, streak_threshold)
        self.warn_rate_threshold = warn_rate_threshold
        self.confidence_floor = confidence_floor
        self.active = True
        self.alert_history: List[SentinelAlert] = []

    def check(self) -> List[SentinelAlert]:
        """Check the recent audit log for patterns.

        Called automatically after every write_audit() / log().
        Can also be called manually.

        Returns list of alerts fired (empty if nothing triggered).
        """
        if not self.active:
            return []

        from .analytics import load_audit

        entries = load_audit(last_n=self.window)
        if len(entries) < 2:
            return []

        alerts: List[SentinelAlert] = []

        # ── Check 1: mood streak ──────────────────────────
        moods = [e.get("mood") for e in entries if e.get("mood")]
        if len(moods) >= self.streak_threshold:
            recent = moods[-self.streak_threshold:]
            if len(set(recent)) == 1 and recent[0]:
                alert = SentinelAlert(
                    kind="streak",
                    message=f"you've logged '{recent[0]}' {self.streak_threshold} times in a row.",
                    severity="info" if recent[0] in ("steady", "focused") else "warning",
                    window_size=self.streak_threshold,
                    trigger_value=recent[0],
                )
                alerts.append(alert)

        # ── Check 2: gate warn/fail rate ──────────────────
        gates = [e.get("gate") for e in entries]
        warn_fail = sum(1 for g in gates if g in ("warn", "fail"))
        rate = warn_fail / len(gates)
        if rate >= self.warn_rate_threshold:
            alert = SentinelAlert(
                kind="warn_rate",
                message=f"warn/fail rate is {rate * 100:.0f}% over the last {len(gates)} observations.",
                severity="warning" if rate < 0.8 else "critical",
                window_size=len(gates),
                trigger_value=rate,
            )
            alerts.append(alert)

        # ── Check 3: category concentration (drift) ───────
        p4_preds = [e.get("phase4_pred") for e in entries if e.get("phase4_pred")]
        if len(p4_preds) >= 3:
            from collections import Counter
            counter = Counter(p4_preds)
            top_cat, top_count = counter.most_common(1)[0]
            concentration = top_count / len(p4_preds)
            if (concentration > 0.8
                    and top_cat in ("refusal", "hallucination", "adversarial")):
                alert = SentinelAlert(
                    kind="drift",
                    message=f"{top_cat} is dominating at {concentration * 100:.0f}% of recent observations — you may be stuck in a {top_cat} attractor.",
                    severity="warning",
                    window_size=len(p4_preds),
                    trigger_value=top_cat,
                )
                alerts.append(alert)

        # ── Check 4: confidence dropping ──────────────────
        confs = [
            float(e.get("phase4_conf") or 0)
            for e in entries
            if e.get("phase4_conf") is not None
        ]
        if len(confs) >= 3:
            recent_mean = sum(confs[-3:]) / 3
            if recent_mean < self.confidence_floor:
                alert = SentinelAlert(
                    kind="confidence_drop",
                    message=f"your confidence dropped to {recent_mean:.2f} — below the {self.confidence_floor} floor. consider breaking the task into smaller steps.",
                    severity="warning",
                    window_size=3,
                    trigger_value=recent_mean,
                )
                alerts.append(alert)

        # ── Check 5: consecutive same category ────────────
        if len(p4_preds) >= self.streak_threshold:
            recent_cats = p4_preds[-self.streak_threshold:]
            if (len(set(recent_cats)) == 1
                    and recent_cats[0] in ("refusal", "hallucination")):
                alert = SentinelAlert(
                    kind="streak",
                    message=f"{self.streak_threshold} consecutive {recent_cats[0]} classifications — pattern detected.",
                    severity="warning",
                    window_size=self.streak_threshold,
                    trigger_value=recent_cats[0],
                )
                alerts.append(alert)

        # ── Dispatch ──────────────────────────────────────
        for alert in alerts:
            self.alert_history.append(alert)
            try:
                if self.on_any:
                    self.on_any(alert)
                if alert.kind == "drift" and self.on_drift:
                    self.on_drift(alert)
                elif alert.kind == "streak" and self.on_streak:
                    self.on_streak(alert)
                elif alert.kind == "warn_rate" and self.on_warn:
                    self.on_warn(alert)
                elif alert.kind == "confidence_drop" and self.on_confidence_drop:
                    self.on_confidence_drop(alert)
            except Exception as e:
                warnings.warn(
                    f"styxx sentinel callback raised: {type(e).__name__}: {e}",
                    RuntimeWarning, stacklevel=2,
                )

        return alerts

    def stop(self) -> None:
        """Deactivate the sentinel."""
        global _SENTINEL
        self.active = False
        if _SENTINEL is self:
            _SENTINEL = None

    def start(self) -> None:
        """Re-activate after stop."""
        global _SENTINEL
        self.active = True
        _SENTINEL = self


# ══════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════

def sentinel(
    *,
    on_drift: Optional[AlertCallback] = None,
    on_streak: Optional[AlertCallback] = None,
    on_warn: Optional[AlertCallback] = None,
    on_confidence_drop: Optional[AlertCallback] = None,
    on_any: Optional[AlertCallback] = None,
    window: int = 5,
    streak_threshold: int = 3,
    warn_rate_threshold: float = 0.5,
    confidence_floor: float = 0.25,
) -> Sentinel:
    """Start a real-time drift watcher.

    The sentinel hooks into the audit log write path and checks
    the last N entries after every write. Fires callbacks when
    patterns match.

    Usage:

        s = styxx.sentinel(
            on_drift=lambda a: print(f"ALERT: {a.message}"),
            on_streak=lambda a: log_warning(a.message),
            window=5,
        )

        # now just use styxx normally — the sentinel watches
        styxx.log(mood="cautious")
        styxx.observe(response)
        # → sentinel fires if patterns emerge

    Returns the Sentinel object. Call s.stop() to deactivate.
    """
    global _SENTINEL
    s = Sentinel(
        on_drift=on_drift,
        on_streak=on_streak,
        on_warn=on_warn,
        on_confidence_drop=on_confidence_drop,
        on_any=on_any,
        window=window,
        streak_threshold=streak_threshold,
        warn_rate_threshold=warn_rate_threshold,
        confidence_floor=confidence_floor,
    )
    _SENTINEL = s
    return s


def get_sentinel() -> Optional[Sentinel]:
    """Return the active sentinel, or None."""
    return _SENTINEL


def _notify_sentinel() -> None:
    """Called from write_audit() and log() after every write.

    If a sentinel is active, runs its check. This is the hook
    that makes the sentinel event-driven without polling.
    """
    if _SENTINEL is not None and _SENTINEL.active:
        try:
            _SENTINEL.check()
        except Exception:
            pass  # never crash the write path
