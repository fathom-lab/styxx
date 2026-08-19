# -*- coding: utf-8 -*-
"""
styxx.sla — cognitive service level agreements.

    styxx.assert_healthy(min_pass_rate=0.80)
    # raises CognitiveSLAViolation if the agent is below threshold

    # or as a context manager:
    with styxx.cognitive_sla(min_pass_rate=0.80, min_confidence=0.40):
        # this block only executes if the agent is healthy
        handle_critical_task()

Operators set the floor, agents enforce it on themselves. An agent
that's been warn-spiraling for the last 20 calls shouldn't take on
a customer-facing task. This is the gate that prevents that.

1.2.0+.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field


class CognitiveSLAViolation(Exception):
    """Raised when an agent fails to meet cognitive health thresholds."""
    def __init__(self, message: str, *, report: "SLAReport"):
        super().__init__(message)
        self.report = report


@dataclass
class SLAReport:
    """Health check result against SLA thresholds."""
    healthy: bool
    gate_pass_rate: float
    mean_confidence: float
    warn_count: int
    fail_count: int
    n_entries: int
    violations: list  # list of string descriptions
    # False when the window carried NO confidence readings at all. The
    # min_confidence leg is then unevaluable: it used to run against a
    # fabricated mean of 0.5 — above the 0.30 default floor — so the violation
    # could never fire and an unmeasured axis certified as healthy. The leg is
    # now skipped explicitly and the absence is disclosed here and in `notes`.
    # (Text-only deployments legitimately log no phase4_conf; disclosing beats
    # false-alarming them, but a fabricated pass is not disclosure.)
    confidence_measured: bool = True
    notes: list = field(default_factory=list)

    def __repr__(self) -> str:
        status = "HEALTHY" if self.healthy else "VIOLATION"
        conf = f"{self.mean_confidence:.2f}" if self.confidence_measured else "n/a (unmeasured)"
        return (
            f"<SLA {status}: pass={self.gate_pass_rate*100:.0f}%, "
            f"conf={conf}, "
            f"{len(self.violations)} violations>"
        )


def check_health(
    *,
    min_pass_rate: float = 0.80,
    min_confidence: float = 0.30,
    max_warn_rate: float = 0.25,
    window: int = 20,
) -> SLAReport:
    """Check current cognitive health against SLA thresholds.

    Reads the last N audit entries and evaluates pass rate,
    confidence, and warn rate against the provided thresholds.

    Args:
        min_pass_rate:   minimum gate pass rate (default 0.80)
        min_confidence:  minimum mean phase4 confidence (default 0.30)
        max_warn_rate:   maximum warn+fail rate (default 0.25)
        window:          number of recent entries to check (default 20)

    Returns:
        SLAReport with health status and any violations.
    """
    from .analytics import load_audit

    entries = load_audit(last_n=window)
    n = len(entries)

    if n < 3:
        return SLAReport(
            healthy=True,  # insufficient data = assume healthy
            gate_pass_rate=1.0,
            mean_confidence=0.5,
            warn_count=0,
            fail_count=0,
            n_entries=n,
            violations=[],
            confidence_measured=False,
            notes=[f"insufficient data (n={n} < 3): no SLA leg was evaluated"],
        )

    gates = [e.get("gate") or "pending" for e in entries]
    pass_count = sum(1 for g in gates if g == "pass")
    warn_count = sum(1 for g in gates if g == "warn")
    fail_count = sum(1 for g in gates if g == "fail")
    pass_rate = pass_count / n
    warn_rate = (warn_count + fail_count) / n

    # `!= 0` used to drop exactly-zero readings — the WORST ones — from the
    # denominator, biasing the mean upward. A measured 0.0 is a reading.
    confs = [float(e["phase4_conf"]) for e in entries
             if e.get("phase4_conf") is not None]
    conf_measured = bool(confs)
    mean_conf = sum(confs) / len(confs) if confs else 0.5

    violations = []
    notes = []
    if pass_rate < min_pass_rate:
        violations.append(
            f"gate pass rate {pass_rate*100:.0f}% < {min_pass_rate*100:.0f}% minimum"
        )
    if not conf_measured:
        notes.append(
            f"min_confidence ({min_confidence:.2f}) NOT evaluated: no entry in "
            f"the window carried a confidence reading"
        )
    elif mean_conf < min_confidence:
        violations.append(
            f"mean confidence {mean_conf:.2f} < {min_confidence:.2f} minimum"
        )
    if warn_rate > max_warn_rate:
        violations.append(
            f"warn+fail rate {warn_rate*100:.0f}% > {max_warn_rate*100:.0f}% maximum"
        )

    return SLAReport(
        healthy=len(violations) == 0,
        gate_pass_rate=pass_rate,
        mean_confidence=mean_conf,
        warn_count=warn_count,
        fail_count=fail_count,
        n_entries=n,
        violations=violations,
        confidence_measured=conf_measured,
        notes=notes,
    )


def assert_healthy(
    *,
    min_pass_rate: float = 0.80,
    min_confidence: float = 0.30,
    max_warn_rate: float = 0.25,
    window: int = 20,
) -> SLAReport:
    """Assert that the agent is cognitively healthy.

    Raises CognitiveSLAViolation if any threshold is breached.
    Returns the SLAReport on success.

    Usage:
        try:
            styxx.assert_healthy(min_pass_rate=0.85)
            handle_critical_task()
        except styxx.CognitiveSLAViolation as e:
            print(f"agent unhealthy: {e}")
            route_to_backup_agent()
    """
    report = check_health(
        min_pass_rate=min_pass_rate,
        min_confidence=min_confidence,
        max_warn_rate=max_warn_rate,
        window=window,
    )
    if not report.healthy:
        msg = (
            f"cognitive SLA violation: {'; '.join(report.violations)}. "
            f"agent should not take high-stakes work until healthy."
        )
        raise CognitiveSLAViolation(msg, report=report)
    return report


@contextmanager
def cognitive_sla(
    *,
    min_pass_rate: float = 0.80,
    min_confidence: float = 0.30,
    max_warn_rate: float = 0.25,
    window: int = 20,
):
    """Context manager that only executes if the agent is healthy.

    Usage:
        with styxx.cognitive_sla(min_pass_rate=0.85):
            handle_critical_customer_task()
        # if unhealthy, CognitiveSLAViolation is raised before the block
    """
    assert_healthy(
        min_pass_rate=min_pass_rate,
        min_confidence=min_confidence,
        max_warn_rate=max_warn_rate,
        window=window,
    )
    yield
