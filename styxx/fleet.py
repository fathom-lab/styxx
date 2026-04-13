# -*- coding: utf-8 -*-
"""
styxx.fleet — population-level analytics across agent namespaces.

    agents = styxx.list_agents()
    # ['xendro', 'customer-bot', 'research-agent']

    comparison = styxx.compare_agents()
    for agent in comparison:
        print(f"{agent.name}: pass={agent.gate_pass_rate*100:.0f}%, "
              f"conf={agent.mean_confidence:.2f}")

    # Fleet-level anomaly: is hallucination spiking across agents?
    fleet = styxx.fleet_summary()
    print(fleet.narrative)

In agentic systems with 50 agents, styxx data determines task
routing. High warn-rate agents don't get critical path work.
Agents with strong reasoning fingerprints get assigned reasoning
tasks. Agents drifting toward refusal get routed away from creative
work. This is autonomous workforce management.

1.0.0+.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _styxx_root() -> Path:
    data_dir = os.environ.get("STYXX_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser()
    return Path.home() / ".styxx"


def list_agents() -> List[str]:
    """List all agent namespaces that have audit data.

    Returns agent names that have a chart.jsonl in their namespace
    directory (~/.styxx/agents/{name}/chart.jsonl).
    """
    agents_dir = _styxx_root() / "agents"
    if not agents_dir.exists():
        return []
    names = []
    for d in sorted(agents_dir.iterdir()):
        if d.is_dir() and (d / "chart.jsonl").exists():
            names.append(d.name)
    return names


@dataclass
class AgentProfile:
    """Summary profile for one agent in a fleet comparison."""
    name: str
    n_entries: int = 0
    gate_pass_rate: float = 0.0
    warn_rate: float = 0.0
    mean_confidence: float = 0.0
    dominant_category: str = "unknown"
    trust_score: float = 0.7
    last_active: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"<Agent '{self.name}': {self.n_entries} entries, "
            f"{self.gate_pass_rate*100:.0f}% pass, "
            f"conf {self.mean_confidence:.2f}, {self.dominant_category}>"
        )


@dataclass
class FleetSummary:
    """Population-level summary across all agents."""
    n_agents: int = 0
    total_entries: int = 0
    mean_pass_rate: float = 0.0
    mean_confidence: float = 0.0
    agents: List[AgentProfile] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    narrative: str = ""

    def __repr__(self) -> str:
        return (
            f"<Fleet {self.n_agents} agents, {self.total_entries} entries, "
            f"{self.mean_pass_rate*100:.0f}% avg pass>"
        )


def _load_agent_entries(name: str, last_n: int = 200) -> List[dict]:
    """Load recent entries for a specific agent namespace."""
    path = _styxx_root() / "agents" / name / "chart.jsonl"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = []
        for line in lines[-last_n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
        return entries
    except OSError:
        return []


def _profile_from_entries(name: str, entries: List[dict]) -> AgentProfile:
    """Compute an AgentProfile from raw entries."""
    profile = AgentProfile(name=name)
    if not entries:
        return profile

    profile.n_entries = len(entries)
    profile.last_active = entries[-1].get("ts_iso")

    # Gates
    gates = [e.get("gate") or "pending" for e in entries]
    n = len(gates)
    profile.gate_pass_rate = sum(1 for g in gates if g == "pass") / n
    profile.warn_rate = sum(1 for g in gates if g in ("warn", "fail")) / n

    # Confidence
    confs = [float(e["phase4_conf"]) for e in entries
             if e.get("phase4_conf") is not None and e.get("phase4_conf") != 0]
    profile.mean_confidence = sum(confs) / len(confs) if confs else 0.0

    # Dominant category
    cats = Counter(e.get("phase4_pred") for e in entries if e.get("phase4_pred"))
    if cats:
        profile.dominant_category = cats.most_common(1)[0][0]

    # Trust score (average of gate-based trust)
    gate_trust = {"pass": 1.0, "warn": 0.5, "fail": 0.2, "pending": 0.7}
    trust_sum = sum(gate_trust.get(g, 0.7) for g in gates)
    profile.trust_score = round(trust_sum / n, 3)

    return profile


def compare_agents(
    *,
    last_n: int = 200,
) -> List[AgentProfile]:
    """Compare all agents in the fleet.

    Returns a list of AgentProfiles sorted by gate pass rate (best first).

    Usage:
        for agent in styxx.compare_agents():
            print(f"{agent.name}: {agent.gate_pass_rate*100:.0f}% pass")
    """
    agents = list_agents()
    profiles = []
    for name in agents:
        entries = _load_agent_entries(name, last_n=last_n)
        profiles.append(_profile_from_entries(name, entries))
    profiles.sort(key=lambda p: -p.gate_pass_rate)
    return profiles


def fleet_summary(
    *,
    last_n: int = 200,
) -> FleetSummary:
    """Generate a fleet-level summary across all agents.

    Detects population-level anomalies: if hallucination rates are
    spiking across multiple agents, or if confidence is declining
    fleet-wide, that's a systemic signal.

    Usage:
        fleet = styxx.fleet_summary()
        print(fleet.narrative)
        for a in fleet.anomalies:
            print(f"  ! {a}")
    """
    profiles = compare_agents(last_n=last_n)
    summary = FleetSummary(
        n_agents=len(profiles),
        agents=profiles,
    )

    if not profiles:
        summary.narrative = "no agents with audit data."
        return summary

    summary.total_entries = sum(p.n_entries for p in profiles)
    pass_rates = [p.gate_pass_rate for p in profiles if p.n_entries > 0]
    confs = [p.mean_confidence for p in profiles if p.mean_confidence > 0]
    summary.mean_pass_rate = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
    summary.mean_confidence = sum(confs) / len(confs) if confs else 0.0

    # Anomaly detection
    for p in profiles:
        if p.n_entries >= 10 and p.warn_rate > 0.30:
            summary.anomalies.append(
                f"{p.name}: warn rate {p.warn_rate*100:.0f}% — above healthy threshold"
            )
        if p.n_entries >= 10 and p.mean_confidence < 0.25:
            summary.anomalies.append(
                f"{p.name}: mean confidence {p.mean_confidence:.2f} — critically low"
            )

    # Check for fleet-wide patterns
    if len(pass_rates) >= 3:
        low_pass = sum(1 for r in pass_rates if r < 0.70)
        if low_pass > len(pass_rates) * 0.5:
            summary.anomalies.append(
                f"fleet-wide: {low_pass}/{len(profiles)} agents below 70% pass rate — systemic issue"
            )

    # Narrative
    parts = [f"{summary.n_agents} agents, {summary.total_entries} total observations"]
    parts.append(f"fleet avg pass rate {summary.mean_pass_rate*100:.0f}%")
    if summary.mean_confidence > 0:
        parts.append(f"avg confidence {summary.mean_confidence:.2f}")
    if summary.anomalies:
        parts.append(f"{len(summary.anomalies)} anomalies detected")
    summary.narrative = ", ".join(parts) + "."

    return summary


def best_agent_for(
    category: str,
    *,
    last_n: int = 200,
) -> Optional[str]:
    """Find the best agent for a given cognitive category.

    Returns the agent name with the highest pass rate on entries
    where phase4_pred matches the requested category. This is the
    primitive for cognitive task routing.

    Usage:
        best = styxx.best_agent_for("reasoning")
        # Route reasoning-heavy tasks to this agent

        best = styxx.best_agent_for("creative")
        # Route creative tasks to this agent
    """
    agents = list_agents()
    best_name = None
    best_score = -1.0

    for name in agents:
        entries = _load_agent_entries(name, last_n=last_n)
        matching = [e for e in entries if e.get("phase4_pred") == category]
        if len(matching) < 5:
            continue
        pass_rate = sum(1 for e in matching if e.get("gate") == "pass") / len(matching)
        confs = [float(e["phase4_conf"]) for e in matching
                 if e.get("phase4_conf") is not None]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        score = pass_rate * 0.6 + mean_conf * 0.4
        if score > best_score:
            best_score = score
            best_name = name

    return best_name
