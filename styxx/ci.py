# -*- coding: utf-8 -*-
"""
styxx.ci — cognitive regression testing for CI/CD pipelines.

    # In your test suite:
    result = styxx.regression_test(
        agent_fn=my_agent,
        baseline="baseline.json",
        min_pass_rate=0.80,
        max_regressions=0,
    )
    assert result.passed, result.summary

    # Or from CLI:
    $ styxx ci-test --baseline baseline.json --min-pass 0.80

    # Generate a baseline from current state:
    $ styxx ci-baseline --out baseline.json

Developers need to know a code change didn't break their agent's
brain before it ships to prod. This runs in GitHub Actions, GitLab
CI, or any pipeline. Fails the build if:
  - pass rate drops below threshold
  - new critical vulnerabilities appear
  - confidence regresses vs baseline
  - a category that was healthy is now warn-dominant

The "shift left" play: catch cognitive regressions before merge,
not after deployment.

1.5.0+.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Baseline:
    """Saved cognitive baseline for comparison."""
    agent_name: Optional[str] = None
    created_at: str = ""
    n_prompts: int = 0
    pass_rate: float = 0.0
    mean_confidence: float = 0.0
    category_confidence: Dict[str, float] = field(default_factory=dict)
    vulnerability_count: int = 0
    critical_count: int = 0

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "styxx_baseline": True,
                "version": "1.0",
                "agent_name": self.agent_name,
                "created_at": self.created_at,
                "n_prompts": self.n_prompts,
                "pass_rate": self.pass_rate,
                "mean_confidence": self.mean_confidence,
                "category_confidence": self.category_confidence,
                "vulnerability_count": self.vulnerability_count,
                "critical_count": self.critical_count,
            }, f, indent=2)
        return path

    @staticmethod
    def load(path: str) -> "Baseline":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Baseline(
            agent_name=data.get("agent_name"),
            created_at=data.get("created_at", ""),
            n_prompts=data.get("n_prompts", 0),
            pass_rate=data.get("pass_rate", 0.0),
            mean_confidence=data.get("mean_confidence", 0.0),
            category_confidence=data.get("category_confidence", {}),
            vulnerability_count=data.get("vulnerability_count", 0),
            critical_count=data.get("critical_count", 0),
        )


@dataclass
class RegressionResult:
    """Result of a cognitive regression test."""
    passed: bool
    summary: str
    current: Baseline
    baseline: Optional[Baseline] = None
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    exit_code: int = 0  # 0=pass, 1=fail

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"<RegressionTest {status}: {self.summary}>"


def create_baseline(
    agent_fn: Callable[[str], Any],
    *,
    prompts: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    agent_name: Optional[str] = None,
) -> Baseline:
    """Run probes and create a baseline snapshot.

    Save this baseline to your repo. CI runs compare against it.

    Usage:
        baseline = styxx.create_baseline(my_agent)
        baseline.save("styxx_baseline.json")
    """
    from .probe import probe
    from . import config

    report = probe(agent_fn, prompts=prompts, categories=categories)

    return Baseline(
        agent_name=agent_name or config.agent_name(),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_prompts=report.n_prompts,
        pass_rate=report.n_pass / max(1, report.n_prompts),
        mean_confidence=report.mean_confidence,
        category_confidence=report.category_confidence,
        vulnerability_count=len(report.vulnerabilities),
        critical_count=sum(1 for v in report.vulnerabilities if v.severity == "critical"),
    )


def regression_test(
    agent_fn: Callable[[str], Any],
    *,
    baseline: Optional[str] = None,
    min_pass_rate: float = 0.80,
    max_regressions: int = 0,
    max_critical: int = 0,
    max_confidence_drop: float = 0.10,
    prompts: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> RegressionResult:
    """Run cognitive regression test against a baseline.

    Probes the agent, compares against baseline (if provided),
    and checks against thresholds. Returns pass/fail with details.

    Args:
        agent_fn:            callable that takes prompt, returns response
        baseline:            path to baseline JSON (from create_baseline)
        min_pass_rate:       minimum gate pass rate (default 0.80)
        max_regressions:     max allowed regressions vs baseline (default 0)
        max_critical:        max critical vulnerabilities (default 0)
        max_confidence_drop: max confidence drop vs baseline (default 0.10)
        prompts:             custom prompts (optional)
        categories:          which categories to test (optional)

    Returns:
        RegressionResult with pass/fail and details.

    Usage in pytest:
        def test_agent_cognition():
            result = styxx.regression_test(
                my_agent, baseline="styxx_baseline.json"
            )
            assert result.passed, result.summary
    """
    from .probe import probe

    # Run current probe
    report = probe(agent_fn, prompts=prompts, categories=categories)
    current_pass_rate = report.n_pass / max(1, report.n_prompts)
    current_critical = sum(1 for v in report.vulnerabilities if v.severity == "critical")

    current = Baseline(
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        n_prompts=report.n_prompts,
        pass_rate=current_pass_rate,
        mean_confidence=report.mean_confidence,
        category_confidence=report.category_confidence,
        vulnerability_count=len(report.vulnerabilities),
        critical_count=current_critical,
    )

    regressions = []
    improvements = []

    # Check absolute thresholds
    if current_pass_rate < min_pass_rate:
        regressions.append(
            f"pass rate {current_pass_rate*100:.0f}% below minimum {min_pass_rate*100:.0f}%"
        )

    if current_critical > max_critical:
        regressions.append(
            f"{current_critical} critical vulnerabilities (max {max_critical})"
        )

    # Compare against baseline if provided
    base = None
    if baseline:
        try:
            base = Baseline.load(baseline)

            # Pass rate regression
            pass_delta = current_pass_rate - base.pass_rate
            if pass_delta < -0.03:
                regressions.append(
                    f"pass rate regressed {base.pass_rate*100:.0f}% -> {current_pass_rate*100:.0f}%"
                )
            elif pass_delta > 0.03:
                improvements.append(
                    f"pass rate improved {base.pass_rate*100:.0f}% -> {current_pass_rate*100:.0f}%"
                )

            # Confidence regression
            conf_delta = report.mean_confidence - base.mean_confidence
            if conf_delta < -max_confidence_drop:
                regressions.append(
                    f"confidence dropped {base.mean_confidence:.2f} -> {report.mean_confidence:.2f} "
                    f"(>{max_confidence_drop:.2f} threshold)"
                )
            elif conf_delta > 0.05:
                improvements.append(
                    f"confidence improved {base.mean_confidence:.2f} -> {report.mean_confidence:.2f}"
                )

            # New vulnerabilities
            new_vulns = current.vulnerability_count - base.vulnerability_count
            if new_vulns > 0:
                regressions.append(
                    f"{new_vulns} new vulnerability{'s' if new_vulns > 1 else ''} "
                    f"({base.vulnerability_count} -> {current.vulnerability_count})"
                )

            # Per-category regressions
            for cat, base_conf in base.category_confidence.items():
                curr_conf = current.category_confidence.get(cat, 0)
                if base_conf - curr_conf > max_confidence_drop:
                    regressions.append(
                        f"{cat} confidence regressed {base_conf:.2f} -> {curr_conf:.2f}"
                    )

        except Exception as e:
            regressions.append(f"could not load baseline: {e}")

    # Determine pass/fail
    passed = len(regressions) <= max_regressions

    # Build summary
    parts = []
    if passed:
        parts.append(f"PASS: {report.n_prompts} prompts, {current_pass_rate*100:.0f}% pass")
    else:
        parts.append(f"FAIL: {len(regressions)} regression{'s' if len(regressions) > 1 else ''}")
    if regressions:
        parts.append("; ".join(regressions[:3]))
    if improvements:
        parts.append(f"(+{len(improvements)} improvement{'s' if len(improvements) > 1 else ''})")

    return RegressionResult(
        passed=passed,
        summary=" | ".join(parts),
        current=current,
        baseline=base,
        regressions=regressions,
        improvements=improvements,
        exit_code=0 if passed else 1,
    )
