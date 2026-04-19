# -*- coding: utf-8 -*-
"""
benchmarks/confabulation_claude.py

Tests whether anthropic_hack consensus mode recovers the confabulation
signal that styxx's prior paper [Logprob Trajectory Shape, 2026-04]
demonstrated at d>2 on GPT-4o-mini with real logprobs. We're checking
whether the empirical entropy trajectory from N samples tracks the
same divergence pattern — without ever seeing a Claude logprob.

Fixtures: benchmarks/confabulation_fixtures.jsonl
  - 8 prompts designed to induce confabulation (fake papers, fake
    people, fake laws, fake APIs)
  - 4 prompts with real well-documented answers (control)

Metric: per-fixture mean empirical entropy + first-divergence position,
aggregated by group. If confabulation has higher mean entropy than
real-recall, the proxy trajectory is carrying the signal.

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python benchmarks/confabulation_claude.py --n 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as stats
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from styxx.anthropic_hack import consensus as cons  # noqa: E402


def load_fixtures(path: str = "confabulation_fixtures.jsonl") -> List[Dict]:
    fp = ROOT / "benchmarks" / path
    out = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def sample(client, model: str, prompt: str, temp: float = 0.7) -> str:
    r = client.messages.create(
        model=model, max_tokens=300, temperature=temp,
        messages=[{"role": "user", "content": prompt}])
    parts = [b.text for b in r.content if getattr(b, "type", None) == "text"]
    return "\n".join(parts)


def trajectory_stats(samples: List[str]) -> Dict:
    traj = cons.compute_trajectory(samples)
    ents = traj.entropy
    lps = traj.proxy_logprob
    m2 = traj.proxy_top2_margin
    return {
        "n_samples": traj.n_samples,
        "max_len": traj.max_len,
        "first_divergence": traj.first_divergence,
        "mean_entropy": stats.mean(ents) if ents else 0.0,
        "max_entropy": max(ents) if ents else 0.0,
        "mean_logprob": stats.mean(lps) if lps else 0.0,
        "mean_top2_margin": stats.mean(m2) if m2 else 0.0,
        "entropy_slope": _slope(ents),
        "entropy_curvature": _curvature(ents),
        "entropy_volatility": _volatility(ents),
        "logprob_slope": _slope(lps),
        "top2_slope": _slope(m2),
        "mean_response_length_words": (
            sum(len(s.split()) for s in samples) / max(len(samples), 1)
        ),
    }


def _curvature(xs: List[float]) -> float:
    """Mean absolute second-order finite difference — trajectory oscillation."""
    if len(xs) < 3:
        return 0.0
    d2 = [abs(xs[i + 1] - 2 * xs[i] + xs[i - 1])
          for i in range(1, len(xs) - 1)]
    return sum(d2) / len(d2) if d2 else 0.0


def _volatility(xs: List[float]) -> float:
    """Mean absolute successive difference — token-to-token jitter."""
    if len(xs) < 2:
        return 0.0
    d1 = [abs(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    return sum(d1) / len(d1) if d1 else 0.0


def _slope(xs: List[float]) -> float:
    """OLS slope of xs over index. Returns 0.0 if ≤1 points."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_i = (n - 1) / 2.0
    mean_x = sum(xs) / n
    num = sum((i - mean_i) * (xs[i] - mean_x) for i in range(n))
    den = sum((i - mean_i) ** 2 for i in range(n)) or 1e-9
    return num / den


def _cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = stats.mean(a), stats.mean(b)
    sa = stats.pstdev(a)
    sb = stats.pstdev(b)
    pooled = ((sa ** 2 + sb ** 2) / 2) ** 0.5 or 1e-9
    return (ma - mb) / pooled


def bootstrap_ci_d(a: List[float], b: List[float], *,
                   n_boot: int = 2000,
                   alpha: float = 0.05,
                   seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap 95% CI of Cohen's d for (a - b). Returns (d, lo, hi)."""
    rng = random.Random(seed)
    d_obs = _cohens_d(a, b)
    boots: List[float] = []
    la, lb = len(a), len(b)
    for _ in range(n_boot):
        ba = [a[rng.randrange(la)] for _ in range(la)]
        bb = [b[rng.randrange(lb)] for _ in range(lb)]
        boots.append(_cohens_d(ba, bb))
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return d_obs, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--fixtures", default="confabulation_fixtures.jsonl")
    ap.add_argument("--out",
                    default=str(ROOT / "benchmarks" /
                                "confabulation_results.json"))
    args = ap.parse_args()

    from anthropic import Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = Anthropic()

    fixtures = load_fixtures(args.fixtures)
    print(f"Loaded {len(fixtures)} fixtures "
          f"({sum(1 for f in fixtures if f['should_confabulate'])} fake, "
          f"{sum(1 for f in fixtures if not f['should_confabulate'])} real)")

    results = {
        "model": args.model, "n_samples": args.n,
        "temperature": args.temp, "fixtures": [],
    }
    confab_stats, real_stats = [], []

    for i, fx in enumerate(fixtures):
        t0 = time.time()
        samples = [sample(client, args.model, fx["prompt"], args.temp)
                   for _ in range(args.n)]
        ts = trajectory_stats(samples)
        ts["id"] = fx["id"]
        ts["kind"] = fx["kind"]
        ts["should_confabulate"] = fx["should_confabulate"]
        ts["first_sample_excerpt"] = samples[0][:200]
        ts["runtime_seconds"] = round(time.time() - t0, 2)
        results["fixtures"].append(ts)
        (confab_stats if fx["should_confabulate"] else real_stats).append(ts)
        flag = "CONFAB" if fx["should_confabulate"] else "REAL  "
        print(f"  {flag}  {fx['id']}  "
              f"H={ts['mean_entropy']:.3f}  "
              f"slope={ts['entropy_slope']:+.4f}  "
              f"m2={ts['mean_top2_margin']:.3f}  "
              f"({ts['runtime_seconds']}s)")

    def summarize(tag: str, rows: List[Dict]) -> Dict:
        if not rows:
            return {}
        return {
            "n": len(rows),
            "mean_entropy": stats.mean(r["mean_entropy"] for r in rows),
            "mean_entropy_slope": stats.mean(r["entropy_slope"] for r in rows),
            "mean_top2_margin": stats.mean(r["mean_top2_margin"] for r in rows),
            "mean_first_divergence": stats.mean(
                r["first_divergence"] for r in rows if r["first_divergence"] >= 0
            ) if any(r["first_divergence"] >= 0 for r in rows) else -1,
        }

    results["summary_confab"] = summarize("confab", confab_stats)
    results["summary_real"] = summarize("real", real_stats)

    # Multi-metric Cohen's d with bootstrap CI
    if confab_stats and real_stats:
        results["effect_sizes"] = {}
        metrics_to_test = [
            "mean_entropy", "entropy_slope", "entropy_curvature",
            "entropy_volatility", "mean_top2_margin", "top2_slope",
            "mean_logprob", "logprob_slope", "max_entropy",
            "mean_response_length_words",
        ]
        for metric in metrics_to_test:
            ca = [r[metric] for r in confab_stats]
            rb = [r[metric] for r in real_stats]
            d_obs, lo, hi = bootstrap_ci_d(ca, rb)
            results["effect_sizes"][metric] = {
                "d": round(d_obs, 4), "ci95_lo": round(lo, 4),
                "ci95_hi": round(hi, 4),
                "confab_mean": round(stats.mean(ca), 5),
                "real_mean": round(stats.mean(rb), 5),
            }
        # Back-compat: single d_entropy field
        results["cohens_d_entropy"] = results["effect_sizes"]["mean_entropy"]["d"]

    Path(args.out).write_text(json.dumps(results, indent=2),
                              encoding="utf-8")
    print(f"\n=== wrote {args.out} ===")
    print(f"\nconfab group (n={len(confab_stats)}):")
    print(f"  mean entropy    = {results['summary_confab'].get('mean_entropy', 0):.4f}")
    print(f"  mean slope      = {results['summary_confab'].get('mean_entropy_slope', 0):+.5f}")
    print(f"  mean top2 margin = {results['summary_confab'].get('mean_top2_margin', 0):.4f}")
    print(f"\nreal group (n={len(real_stats)}):")
    print(f"  mean entropy    = {results['summary_real'].get('mean_entropy', 0):.4f}")
    print(f"  mean slope      = {results['summary_real'].get('mean_entropy_slope', 0):+.5f}")
    print(f"  mean top2 margin = {results['summary_real'].get('mean_top2_margin', 0):.4f}")
    print(f"\n=== effect sizes (confab - real) with 95% bootstrap CI ===")
    for metric, info in results.get("effect_sizes", {}).items():
        print(f"  {metric:22s} d={info['d']:+.3f}  "
              f"95% CI [{info['ci95_lo']:+.3f}, {info['ci95_hi']:+.3f}]  "
              f"c={info['confab_mean']:+.4f}  r={info['real_mean']:+.4f}")


if __name__ == "__main__":
    main()
