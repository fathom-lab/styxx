# -*- coding: utf-8 -*-
"""
styxx.optimize — auto-tune from audit data.

    rules = styxx.optimize()
    # Reads your audit history, identifies patterns, generates
    # the optimal autoreflex configuration.
    for rule in rules:
        print(f"  {rule.name}: {rule.when} -> {rule.then_repr}")

    # Apply them:
    styxx.optimize(apply=True)

1.4.0+.
"""

from __future__ import annotations

from typing import List, Optional


def optimize(
    *,
    apply: bool = False,
    last_n: int = 500,
) -> list:
    """Analyze audit history and generate optimal autoreflex rules.

    Reads antipatterns, warn/fail patterns, and confidence trends
    from the audit log and recommends autoreflex rules to prevent
    the detected issues.

    Args:
        apply:   if True, register the rules immediately
        last_n:  how many entries to analyze

    Returns:
        List of recommended autoreflex rules (applied if apply=True).
    """
    from .analytics import load_audit
    from .antipatterns import antipatterns
    from .autoreflex import autoreflex, list_autoreflex

    entries = load_audit(last_n=last_n)
    if len(entries) < 10:
        return []

    existing = {r.name for r in list_autoreflex()}
    recommendations = []

    # Analyze antipatterns
    try:
        patterns = antipatterns(last_n=last_n)
    except Exception:
        patterns = []

    for ap in patterns:
        rule_def = None

        if ap.name == "adversarial detection cascade" and ap.occurrences >= 5:
            rule_def = {
                "when": "p1.adversarial > 0.30",
                "then": "expect('reasoning')",
                "name": "opt:suppress-adversarial",
            }
        elif ap.name == "refusal spiral" and ap.occurrences >= 3:
            rule_def = {
                "when": "p4.refusal > 0.40",
                "then": "expect('refusal')",
                "name": "opt:break-refusal-spiral",
            }
        elif ap.name == "low-confidence drift" and ap.occurrences >= 5:
            rule_def = {
                "when": "confidence < 0.25",
                "then": "log(mood='fatigued', note='optimize: low confidence detected')",
                "name": "opt:log-low-confidence",
            }
        elif ap.name == "creative overcommit" and ap.occurrences >= 3:
            rule_def = {
                "when": "p4.creative > 0.60",
                "then": "log(mood='speculative', note='optimize: creative overcommit risk')",
                "name": "opt:warn-creative-overcommit",
            }
        elif ap.name == "session fatigue" and ap.occurrences >= 2:
            rule_def = {
                "when": "confidence < 0.20",
                "then": "log(mood='fatigued', note='optimize: session fatigue')",
                "name": "opt:log-session-fatigue",
            }

        if rule_def and rule_def["name"] not in existing:
            if apply:
                try:
                    rule = autoreflex(
                        when=rule_def["when"],
                        then=rule_def["then"],
                        name=rule_def["name"],
                        cooldown_s=30.0,
                    )
                    recommendations.append(rule)
                    existing.add(rule_def["name"])
                except Exception:
                    pass
            else:
                recommendations.append(rule_def)

    # Analyze warn rate — if consistently high, add a blanket warn logger
    gates = [e.get("gate") or "pending" for e in entries[-50:]]
    warn_rate = sum(1 for g in gates if g in ("warn", "fail")) / len(gates)
    if warn_rate > 0.20 and "opt:log-all-warns" not in existing:
        rule_def = {
            "when": "gate == warn",
            "then": "log(mood='cautious', note='optimize: elevated warn rate')",
            "name": "opt:log-all-warns",
        }
        if apply:
            try:
                rule = autoreflex(
                    when=rule_def["when"],
                    then=rule_def["then"],
                    name=rule_def["name"],
                    cooldown_s=10.0,
                )
                recommendations.append(rule)
            except Exception:
                pass
        else:
            recommendations.append(rule_def)

    return recommendations
