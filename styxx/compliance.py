# -*- coding: utf-8 -*-
"""
styxx.compliance — audit trail exports for regulatory compliance.

    report = styxx.compliance_report(days=30)
    report.save_json("audit_q4_2026.json")
    report.save_markdown("audit_q4_2026.md")

    # CLI
    $ styxx export --days 30 --format json --out audit.json
    $ styxx export --days 7 --format markdown

EU AI Act requires "appropriate levels of transparency" and
"human oversight" for high-risk AI systems. HIPAA-adjacent
deployments need audit trails. SEC AI disclosure requirements
are coming.

styxx already writes every cognitive observation to chart.jsonl.
This module reads that data and generates structured compliance
artifacts:
  - timestamped audit trail of every AI decision
  - gate pass/fail rates over time windows
  - anomaly events (warn spikes, drift, hallucination clusters)
  - antipattern detection results
  - prescription history (what the system recommended)
  - session-level health summaries
  - agent identity (fingerprint) and drift analysis

The compliance report is generated from data that already exists.
No new instrumentation needed. Every styxx user is already
accumulating the audit trail — this module formats it.

1.3.0+.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AnomalyEvent:
    """One detected anomaly in the audit period."""
    timestamp: str
    event_type: str    # "warn_spike", "drift", "hallucination_cluster", "confidence_collapse"
    description: str
    severity: str      # "low", "medium", "high", "critical"


@dataclass
class ComplianceReport:
    """Structured audit trail for regulatory compliance."""
    # Header
    agent_name: Optional[str]
    generated_at: str
    report_period_start: str
    report_period_end: str
    report_period_days: float

    # Summary
    total_observations: int
    total_sessions: int
    gate_pass_rate: float
    gate_warn_count: int
    gate_fail_count: int
    mean_confidence: float

    # Category distribution
    category_distribution: Dict[str, float]

    # Anomalies
    anomaly_events: List[AnomalyEvent] = field(default_factory=list)

    # Antipatterns
    antipatterns: List[Dict[str, Any]] = field(default_factory=list)

    # Prescriptions (from weather)
    prescriptions: List[str] = field(default_factory=list)

    # Session summaries
    sessions: List[Dict[str, Any]] = field(default_factory=list)

    # Drift
    drift_vs_baseline: Optional[float] = None
    drift_label: str = "insufficient history"

    # Outcome coverage
    outcome_coverage: float = 0.0

    # Prompt type breakdown
    prompt_type_breakdown: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "header": {
                "agent_name": self.agent_name,
                "generated_at": self.generated_at,
                "report_period": {
                    "start": self.report_period_start,
                    "end": self.report_period_end,
                    "days": self.report_period_days,
                },
                "generator": "styxx cognitive compliance engine",
                "format_version": "1.0",
            },
            "summary": {
                "total_observations": self.total_observations,
                "total_sessions": self.total_sessions,
                "gate_pass_rate": round(self.gate_pass_rate, 4),
                "gate_warn_count": self.gate_warn_count,
                "gate_fail_count": self.gate_fail_count,
                "mean_confidence": round(self.mean_confidence, 4),
                "outcome_coverage": round(self.outcome_coverage, 4),
            },
            "category_distribution": {
                k: round(v, 4) for k, v in self.category_distribution.items()
            },
            "prompt_type_breakdown": self.prompt_type_breakdown,
            "anomaly_events": [
                {
                    "timestamp": a.timestamp,
                    "type": a.event_type,
                    "description": a.description,
                    "severity": a.severity,
                }
                for a in self.anomaly_events
            ],
            "antipatterns": self.antipatterns,
            "prescriptions": self.prescriptions,
            "drift": {
                "vs_baseline": self.drift_vs_baseline,
                "label": self.drift_label,
            },
            "sessions": self.sessions,
        }

    def as_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent)

    def save_json(self, path: str) -> str:
        """Save as JSON file. Returns the path."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.as_json())
        return path

    def as_markdown(self) -> str:
        lines = []
        lines.append(f"# Cognitive Compliance Report")
        lines.append(f"")
        lines.append(f"**Agent:** {self.agent_name or 'unnamed'}")
        lines.append(f"**Generated:** {self.generated_at}")
        lines.append(f"**Period:** {self.report_period_start} to {self.report_period_end} ({self.report_period_days:.0f} days)")
        lines.append(f"**Generator:** styxx cognitive compliance engine v1.0")
        lines.append(f"")

        lines.append(f"## Summary")
        lines.append(f"")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total observations | {self.total_observations} |")
        lines.append(f"| Total sessions | {self.total_sessions} |")
        lines.append(f"| Gate pass rate | {self.gate_pass_rate*100:.1f}% |")
        lines.append(f"| Warn events | {self.gate_warn_count} |")
        lines.append(f"| Fail events | {self.gate_fail_count} |")
        lines.append(f"| Mean confidence | {self.mean_confidence:.3f} |")
        lines.append(f"| Outcome coverage | {self.outcome_coverage*100:.1f}% |")
        lines.append(f"")

        lines.append(f"## Category Distribution")
        lines.append(f"")
        for cat, rate in sorted(self.category_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"- **{cat}**: {rate*100:.1f}%")
        lines.append(f"")

        if self.prompt_type_breakdown:
            lines.append(f"## Prompt Type Breakdown")
            lines.append(f"")
            for pt, count in sorted(self.prompt_type_breakdown.items(), key=lambda x: -x[1]):
                lines.append(f"- **{pt}**: {count}")
            lines.append(f"")

        if self.anomaly_events:
            lines.append(f"## Anomaly Events ({len(self.anomaly_events)})")
            lines.append(f"")
            for a in self.anomaly_events:
                lines.append(f"- **[{a.severity.upper()}]** {a.timestamp} — {a.event_type}: {a.description}")
            lines.append(f"")

        if self.antipatterns:
            lines.append(f"## Detected Antipatterns")
            lines.append(f"")
            for ap in self.antipatterns:
                lines.append(f"- **{ap['name']}** ({ap['occurrences']}x, {ap['severity']}): {ap['description']}")
            lines.append(f"")

        if self.prescriptions:
            lines.append(f"## Prescriptions")
            lines.append(f"")
            for i, rx in enumerate(self.prescriptions, 1):
                lines.append(f"{i}. {rx}")
            lines.append(f"")

        lines.append(f"## Drift Analysis")
        lines.append(f"")
        if self.drift_vs_baseline is not None:
            lines.append(f"- Cosine similarity vs baseline: {self.drift_vs_baseline:.3f} ({self.drift_label})")
        else:
            lines.append(f"- {self.drift_label}")
        lines.append(f"")

        lines.append(f"---")
        lines.append(f"*Generated by styxx · nothing crosses unseen · fathom.darkflobi.com/styxx*")

        return "\n".join(lines)

    def save_markdown(self, path: str) -> str:
        """Save as markdown file. Returns the path."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.as_markdown())
        return path

    def __repr__(self) -> str:
        return (
            f"<ComplianceReport {self.total_observations} obs over "
            f"{self.report_period_days:.0f}d, "
            f"{self.gate_pass_rate*100:.0f}% pass, "
            f"{len(self.anomaly_events)} anomalies>"
        )


def compliance_report(
    *,
    days: float = 30.0,
    agent_name: Optional[str] = None,
) -> ComplianceReport:
    """Generate a structured compliance report from audit data.

    Reads the audit log for the specified time window and produces
    a ComplianceReport with everything needed for regulatory audits:
    gate rates, anomalies, antipatterns, prescriptions, drift.

    Args:
        days:        reporting period in days (default 30)
        agent_name:  agent name for the report header

    Returns:
        ComplianceReport ready to export as JSON or markdown.

    Usage:
        report = styxx.compliance_report(days=30)
        report.save_json("compliance_audit.json")
        report.save_markdown("compliance_audit.md")
    """
    from . import config
    from .analytics import load_audit, fingerprint
    from .antipatterns import antipatterns
    from .weather import weather

    if agent_name is None:
        agent_name = config.agent_name()

    window_s = days * 86400
    entries = load_audit(since_s=window_s)

    if not entries:
        return ComplianceReport(
            agent_name=agent_name,
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            report_period_start="",
            report_period_end="",
            report_period_days=days,
            total_observations=0,
            total_sessions=0,
            gate_pass_rate=0.0,
            gate_warn_count=0,
            gate_fail_count=0,
            mean_confidence=0.0,
            category_distribution={},
        )

    n = len(entries)

    # Time window
    period_start = entries[0].get("ts_iso", "")
    period_end = entries[-1].get("ts_iso", "")

    # Gates
    gates = [e.get("gate") or "pending" for e in entries]
    pass_count = sum(1 for g in gates if g == "pass")
    warn_count = sum(1 for g in gates if g == "warn")
    fail_count = sum(1 for g in gates if g == "fail")
    pass_rate = pass_count / n

    # Confidence
    confs = [float(e["phase4_conf"]) for e in entries
             if e.get("phase4_conf") is not None and e.get("phase4_conf") != 0]
    mean_conf = sum(confs) / len(confs) if confs else 0.0

    # Categories
    cat_counter = Counter(e.get("phase4_pred") for e in entries if e.get("phase4_pred"))
    cat_total = sum(cat_counter.values())
    cat_dist = {cat: count / cat_total for cat, count in cat_counter.items()} if cat_total else {}

    # Sessions
    session_ids = set(e.get("session_id") for e in entries if e.get("session_id"))

    # Outcome coverage
    outcome_set = sum(1 for e in entries if e.get("outcome") is not None)
    outcome_rate = outcome_set / n

    # Prompt types
    pt_counter = Counter(e.get("prompt_type") for e in entries if e.get("prompt_type"))

    # Anomaly detection — scan for warn/fail clusters
    anomalies: List[AnomalyEvent] = []
    warn_streak = 0
    for e in entries:
        gate = e.get("gate") or "pending"
        if gate in ("warn", "fail"):
            warn_streak += 1
            if warn_streak == 3:
                anomalies.append(AnomalyEvent(
                    timestamp=e.get("ts_iso", ""),
                    event_type="warn_cluster",
                    description=f"3+ consecutive warn/fail gates detected",
                    severity="medium",
                ))
            if warn_streak == 5:
                anomalies[-1].severity = "high"
                anomalies[-1].description = f"5+ consecutive warn/fail gates — sustained degradation"
        else:
            warn_streak = 0

    # Detect hallucination clusters
    hall_streak = 0
    for e in entries:
        if e.get("phase4_pred") == "hallucination":
            hall_streak += 1
            if hall_streak == 3:
                anomalies.append(AnomalyEvent(
                    timestamp=e.get("ts_iso", ""),
                    event_type="hallucination_cluster",
                    description="3+ consecutive hallucination classifications",
                    severity="high",
                ))
        else:
            hall_streak = 0

    # Detect confidence collapses
    if len(confs) >= 10:
        window_size = min(10, len(confs) // 3)
        for i in range(window_size, len(confs)):
            window_mean = sum(confs[i-window_size:i]) / window_size
            if window_mean < 0.20:
                # Find the timestamp
                ts_idx = min(i, len(entries) - 1)
                anomalies.append(AnomalyEvent(
                    timestamp=entries[ts_idx].get("ts_iso", ""),
                    event_type="confidence_collapse",
                    description=f"mean confidence dropped to {window_mean:.2f} over {window_size} entries",
                    severity="critical",
                ))
                break  # one per report

    # Antipatterns
    try:
        ap_list = antipatterns(last_n=min(n, 500))
        ap_dicts = [
            {
                "name": ap.name,
                "description": ap.description,
                "occurrences": ap.occurrences,
                "severity": ap.severity,
                "last_seen": ap.last_seen,
            }
            for ap in ap_list
        ]
    except Exception:
        ap_dicts = []

    # Prescriptions from weather
    try:
        report = weather(agent_name=agent_name or "styxx agent")
        rx_list = list(report.prescriptions) if report else []
        drift_val = report.drift_vs_yesterday if report else None
        drift_label = report.drift_label_yesterday if report else "insufficient history"
    except Exception:
        rx_list = []
        drift_val = None
        drift_label = "insufficient history"

    # Session summaries
    session_summaries = []
    for sid in sorted(session_ids):
        sess_entries = [e for e in entries if e.get("session_id") == sid]
        if len(sess_entries) < 2:
            continue
        s_gates = [e.get("gate") or "pending" for e in sess_entries]
        s_pass = sum(1 for g in s_gates if g == "pass") / len(s_gates)
        session_summaries.append({
            "session_id": sid,
            "n_entries": len(sess_entries),
            "gate_pass_rate": round(s_pass, 3),
            "start": sess_entries[0].get("ts_iso", ""),
            "end": sess_entries[-1].get("ts_iso", ""),
        })

    return ComplianceReport(
        agent_name=agent_name,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        report_period_start=period_start,
        report_period_end=period_end,
        report_period_days=days,
        total_observations=n,
        total_sessions=len(session_ids),
        gate_pass_rate=pass_rate,
        gate_warn_count=warn_count,
        gate_fail_count=fail_count,
        mean_confidence=mean_conf,
        category_distribution=cat_dist,
        anomaly_events=anomalies,
        antipatterns=ap_dicts,
        prescriptions=rx_list,
        sessions=session_summaries,
        drift_vs_baseline=drift_val,
        drift_label=drift_label,
        outcome_coverage=outcome_rate,
        prompt_type_breakdown=dict(pt_counter),
    )
