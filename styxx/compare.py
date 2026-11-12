# -*- coding: utf-8 -*-
"""
styxx.compare — multi-agent fingerprint comparison.

    my_fp = styxx.fingerprint()
    comparison = styxx.compare_agents(my_fp, api_url=URL)   # no default leaderboard
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
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    # what the comparison had to work with: "ok" (a population was read), "empty" (a leaderboard
    # was read and has no agent with a fingerprint), "unreadable" (the URL did not answer, or its
    # answer is not a leaderboard), "no_url" (no leaderboard was named), "no_fingerprint"
    status: str = "ok"

    def render(self) -> str:
        lines: List[str] = []
        lines.append("")
        lines.append(f"  styxx population comparison · {self.n_agents} agents in cohort")
        lines.append("  " + "=" * 50)
        lines.append("")
        if self.n_agents == 0 or not self.percentiles:
            lines.append("  no population data — percentile ranks unavailable.")
            if self.status == "empty":
                # only for a leaderboard that was read: through 8.0.0 this line was printed for
                # every result with no agents, a web page read as a leaderboard among them
                lines.append("  the leaderboard has no agents yet. publish to it with "
                             "'styxx publish --name NAME --endpoint URL'.")
        else:
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
            "status": self.status,
        }


# ══════════════════════════════════════════════════════════════════
# Fetch + compare
# ══════════════════════════════════════════════════════════════════

_WHY = threading.local()        # .reason: why the last leaderboard read of this thread failed
_VALUE_MAX = 1e100              # a fingerprint value is a rate; nothing near this is one

def compare_agents(
    my_fingerprint: Optional[Fingerprint] = None,
    *,
    api_url: Optional[str] = None,
) -> AgentComparison:
    """Compare your fingerprint against the population.

    Fetches the leaderboard API at `api_url`, extracts anonymous
    fingerprint vectors, and computes percentile ranks for each
    category.

    If no fingerprint is provided, computes one from the local
    audit log.

    There is no default `api_url`. Through styxx 8.0.0 it was a host
    under darkflobi.com; on 2026-11-12 that host answered from a
    domain-parking service, and by the code 8.0.0 reads a web page
    there as a leaderboard with no agents on it yet. With no URL
    nothing is fetched, and a URL that cannot be read as a leaderboard
    is reported as that, not as an empty population. `status` on the
    result says which. A redirect is not followed (see styxx.stream.open_url): the
    population compared against is the one the URL named answers with,
    and the narrative names the address a redirect pointed to.
    """
    if my_fingerprint is None:
        from .analytics import fingerprint
        my_fingerprint = fingerprint(last_n=500)

    if my_fingerprint is None:
        return AgentComparison(narrative="not enough local data to compute a fingerprint.",
                               status="no_fingerprint")

    if not api_url:
        return AgentComparison(
            n_agents=0,
            narrative=("no leaderboard to compare against: styxx has no default leaderboard "
                       "(the public one is offline). pass the URL of one you run "
                       "(api_url=..., or styxx compare-agents --api-url URL)."),
            status="no_url",
        )

    # Fetch population data. _fetch_population is the name a caller's test replaces, so it is
    # called as it always was, with the URL alone; why a read failed is left for this thread.
    _WHY.reason = None
    population = _fetch_population(api_url)
    why = getattr(_WHY, "reason", None)
    if population is None:
        return AgentComparison(
            n_agents=0,
            narrative=(f"the leaderboard at {api_url} could not be read: " + (why or
                       "it did not answer, or its answer is not a leaderboard (a JSON object "
                       "with an agents list).")),
            status="unreadable",
        )
    if not population:
        return AgentComparison(
            n_agents=0,
            narrative=(f"the leaderboard at {api_url} was read and has no agent with a "
                       "fingerprint on it yet."),
            status="empty",
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


def _fetch_population(api_url: str) -> Optional[List[tuple]]:
    """Fetch anonymous fingerprint vectors from the leaderboard API.

    None when the URL did not answer or its answer is not a leaderboard (a JSON object whose
    `agents` is a list); a list, empty when the leaderboard has no agent with a fingerprint.
    Through 8.0.0 both cases returned an empty list, so a web page read as a leaderboard with
    no agents on it. A redirect is not followed (see styxx.stream.open_url); the sentence that
    says where the URL pointed is left in `_WHY.reason` for the calling thread."""
    try:
        import urllib.error
        import urllib.request
        from styxx import __version__
        from . import stream as _stream
        req = urllib.request.Request(api_url, headers={"User-Agent": f"styxx/{__version__}"})
        try:
            with _stream.open_url(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            where = _stream.redirect_target(exc)
            if where is not None:
                _WHY.reason = (f"it answered HTTP {exc.code}, a redirect to {where}, and styxx "
                               "does not follow a redirect. name the leaderboard by the URL "
                               "that answers itself.")
            return None
    except Exception:
        return None

    agents = data.get("agents") if isinstance(data, dict) else None
    if not isinstance(agents, list):
        return None
    vectors: List[tuple] = []

    for agent in agents:
        if not isinstance(agent, dict):
            continue
        fp = agent.get("fingerprint", {})
        if not isinstance(fp, dict):
            continue
        vec = tuple(fp.get(cat, 0) for cat in _CATEGORY_ORDER)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vec):
            continue      # a fingerprint value that is not a number: not a vector
        try:
            if not all(math.isfinite(float(v)) and abs(float(v)) <= _VALUE_MAX for v in vec):
                continue  # NaN or Infinity, which Python's JSON reader takes, or a value so
                # large that a sum of them is no finite number: not a vector
        except OverflowError:
            continue      # an integer no float holds: the mean would end in a traceback
        if any(v > 0 for v in vec):
            vectors.append(vec)

    return vectors
