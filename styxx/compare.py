# -*- coding: utf-8 -*-
"""
styxx.compare — multi-agent fingerprint comparison.

    my_fp = styxx.fingerprint()
    comparison = styxx.compare_agents(my_fp)
    print(comparison.percentile_reasoning)  # 90th percentile
    print(comparison.similar_agents)        # agents with similar shape
    print(comparison.narrative)             # "you're reasoning-dominant..."

Xendro's #3 request: "I want to see how my fingerprint compares
to other agents running styxx. Not names — anonymous vectors."

This module fetches the leaderboard API and computes population-
level statistics against the local agent's fingerprint. All
comparisons are anonymous — agent names are not exposed, only
aggregate statistics and percentile ranks.

0.5.10+.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .analytics import Fingerprint, _CATEGORY_ORDER


# ══════════════════════════════════════════════════════════════════
# Comparison result
# ══════════════════════════════════════════════════════════════════

@dataclass
class AgentComparison:
    """Result of comparing one agent's fingerprint to the population."""
    n_agents: int = 0
    percentiles: Dict[str, float] = field(default_factory=dict)
    # e.g. {"reasoning": 90, "refusal": 25, "creative": 60}
    population_means: Dict[str, float] = field(default_factory=dict)
    nearest_shape: Optional[str] = None
    narrative: str = ""

    def render(self) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append(f"  styxx population comparison · {self.n_agents} agents in cohort")
        lines.append("  " + "=" * 50)
        lines.append("")
        lines.append("  -- your percentile rank " + "-" * 24)
        for cat in _CATEGORY_ORDER:
            pct = self.percentiles.get(cat, 50)
            bar = "#" * int(pct / 5)
            lines.append(f"  {cat:<14} {bar:<20} {pct:.0f}th")
        lines.append("")
        if self.narrative:
            lines.append("  -- insight " + "-" * 37)
            lines.append(f"  {self.narrative}")
        lines.append("")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "n_agents": self.n_agents,
            "percentiles": self.percentiles,
            "population_means": self.population_means,
            "nearest_shape": self.nearest_shape,
            "narrative": self.narrative,
        }


# ══════════════════════════════════════════════════════════════════
# Fetch + compare
# ══════════════════════════════════════════════════════════════════

def compare_agents(
    my_fingerprint: Optional[Fingerprint] = None,
    *,
    api_url: str = "https://fathom.darkflobi.com/api/styxx-leaderboard",
) -> AgentComparison:
    """Compare your fingerprint against the population.

    Fetches the leaderboard API, extracts anonymous fingerprint
    vectors, and computes percentile ranks for each category.

    If no fingerprint is provided, computes one from the local
    audit log.
    """
    if my_fingerprint is None:
        from .analytics import fingerprint
        my_fingerprint = fingerprint(last_n=500)

    if my_fingerprint is None:
        return AgentComparison(narrative="not enough local data to compute a fingerprint.")

    # Fetch population data
    population = _fetch_population(api_url)
    if not population:
        return AgentComparison(
            n_agents=0,
            narrative="no population data available — you may be the first agent on the leaderboard.",
        )

    n = len(population)
    percentiles: Dict[str, float] = {}
    means: Dict[str, float] = {}

    for i, cat in enumerate(_CATEGORY_ORDER):
        my_val = my_fingerprint.phase4_vec[i]
        pop_vals = [fp[i] for fp in population]

        # Mean
        mean_val = sum(pop_vals) / len(pop_vals) if pop_vals else 0
        means[cat] = round(mean_val, 4)

        # Percentile rank
        below = sum(1 for v in pop_vals if v < my_val)
        pct = (below / len(pop_vals)) * 100 if pop_vals else 50
        percentiles[cat] = round(pct, 1)

    # Build narrative
    narrative_parts: List[str] = []
    top_cat = max(percentiles, key=percentiles.get)
    top_pct = percentiles[top_cat]
    if top_pct > 80:
        narrative_parts.append(
            f"you're in the {top_pct:.0f}th percentile for {top_cat} across {n} agents."
        )
    bottom_cat = min(percentiles, key=percentiles.get)
    bottom_pct = percentiles[bottom_cat]
    if bottom_pct < 20:
        narrative_parts.append(
            f"your {bottom_cat} is low ({bottom_pct:.0f}th percentile) — "
            "most agents in the cohort have more activity here."
        )

    narrative = " ".join(narrative_parts) if narrative_parts else (
        f"compared against {n} agents. your profile is within normal range."
    )

    return AgentComparison(
        n_agents=n,
        percentiles=percentiles,
        population_means=means,
        narrative=narrative,
    )


def _fetch_population(api_url: str) -> List[tuple]:
    """Fetch anonymous fingerprint vectors from the leaderboard API."""
    try:
        import urllib.request
        req = urllib.request.Request(api_url, headers={"User-Agent": "styxx/0.5"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    agents = data.get("agents", [])
    vectors: List[tuple] = []

    for agent in agents:
        fp = agent.get("fingerprint", {})
        vec = tuple(fp.get(cat, 0) for cat in _CATEGORY_ORDER)
        if any(v > 0 for v in vec):
            vectors.append(vec)

    return vectors
