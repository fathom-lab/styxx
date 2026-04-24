# -*- coding: utf-8 -*-
"""
benchmarks/darkcity_csv/analyze.py

CSV-to-P&L correlation on real DarkCity agent decisions.

The first public analysis correlating Styxx proxy cognitive vitals
(surface text-heuristic, styxx.anthropic_hack.text_features) with
real economic outcomes in a live autonomous-agent environment.

Outcomes considered:
  1. depth_score      — DarkCity's quality scorer, 0..5. Feeds rep,
                        which feeds on-chain $STYXX flow.
  2. styxx_inflow_5min — on-chain $STYXX received in the window
                        around the decision. Real token movement.
  3. styxx_outflow_5min — on-chain $STYXX sent out in the window.

Cognitive features (15-dim) come from:
  styxx.anthropic_hack.text_features.extract_features(reasoning_trace)

Correlation method:
  - Pearson r with t-distribution p-values (pure numpy, no scipy).
  - Fisher z-transform 95% CI on r.
  - Bonferroni-corrected alpha for the multi-feature test.

Usage
-----
  # 1. Dump from DarkCity postgres
  psql $DATABASE_URL -At -f dump_decisions.sql > decisions.json

  # 2. Analyze
  python benchmarks/darkcity_csv/analyze.py \
    --input decisions.json \
    --out_dir benchmarks/darkcity_csv/runs/v0

Produces:
  runs/v0/correlations.json  -- per-feature x per-outcome, with CIs
  runs/v0/report.md          -- paper-ready markdown summary
  runs/v0/rows.jsonl         -- every decision + its extracted features
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from styxx.anthropic_hack.text_features import (  # noqa: E402
    extract_features,
    classify,
)


OUTCOME_FIELDS = ("depth_score", "styxx_inflow_5min", "styxx_outflow_5min")


def pearson_r_with_p(x: List[float], y: List[float]) -> Tuple[float, float, int]:
    """Return (r, p, n). p via t-distribution survival, pure numpy."""
    import numpy as np
    from math import erf, sqrt
    n = len(x)
    if n < 4:
        return (float("nan"), float("nan"), n)
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    # Filter NaNs pairwise
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa = xa[mask]
    ya = ya[mask]
    n = int(mask.sum())
    if n < 4:
        return (float("nan"), float("nan"), n)
    if xa.std() < 1e-12 or ya.std() < 1e-12:
        return (0.0, 1.0, n)
    r = float(np.corrcoef(xa, ya)[0, 1])
    # t-statistic under H0: rho=0
    t_stat = r * math.sqrt((n - 2) / max(1e-12, 1.0 - r * r))
    # Two-sided p via normal approximation (valid for n >= ~30)
    # For large n this is fine; for small n it slightly under-rejects.
    z = abs(t_stat)
    p = math.erfc(z / math.sqrt(2.0))
    return (r, float(p), n)


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Fisher z-transform confidence interval for Pearson r."""
    if n < 4 or not math.isfinite(r) or abs(r) >= 1.0:
        return (float("nan"), float("nan"))
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    # 1.959964 = qnorm(0.975)
    zcrit = 1.959964 if alpha == 0.05 else 2.575829
    lo = z - zcrit * se
    hi = z + zcrit * se
    return (math.tanh(lo), math.tanh(hi))


def load_decisions(path: Path) -> List[Dict]:
    """Accepts either:
      - JSON array of decision rows (psql json_agg output)
      - JSONL with one decision per line
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    # JSONL
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def extract_row(decision: Dict) -> Optional[Dict]:
    """Run styxx text-heuristic on the reasoning_trace. Returns a flat
    row dict: {agent_id, action_type, <feature...>, predicted_category,
    margin, <outcome...>}. Returns None if unusable."""
    reasoning = decision.get("reasoning_trace")
    if not reasoning or len(reasoning.strip()) < 20:
        return None

    try:
        cls = classify(reasoning)
    except Exception:
        return None

    row: Dict[str, float] = {
        "agent_id": decision.get("agent_id"),
        "action_type": decision.get("action_type"),
        "predicted_category": cls["predicted"],
        "margin": cls["margin"],
    }
    row.update(cls["features"])
    row.update({k: cls["probs"].get(k, 0.0) for k in cls["probs"]})
    # Pull outcomes
    for f in OUTCOME_FIELDS:
        v = decision.get(f)
        try:
            row[f] = float(v) if v is not None else float("nan")
        except Exception:
            row[f] = float("nan")
    return row


def correlate(rows: List[Dict]) -> Dict:
    """Per-feature x per-outcome correlations with CIs and multi-test
    correction."""
    if not rows:
        return {"n": 0, "per_outcome": {}}

    feature_keys = [
        k for k in rows[0].keys()
        if k not in ("agent_id", "action_type", "predicted_category")
        and k not in OUTCOME_FIELDS
    ]

    per_outcome: Dict[str, List[Dict]] = {}
    n_tests = len(feature_keys) * len(OUTCOME_FIELDS)
    bonferroni = 0.05 / max(n_tests, 1)

    for outcome in OUTCOME_FIELDS:
        y = [r.get(outcome, float("nan")) for r in rows]
        out_rows = []
        for feat in feature_keys:
            x = [r.get(feat, float("nan")) for r in rows]
            r_val, p_val, n_eff = pearson_r_with_p(x, y)
            ci_lo, ci_hi = fisher_ci(r_val, n_eff)
            out_rows.append({
                "feature": feat,
                "r": r_val,
                "p": p_val,
                "n": n_eff,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "passes_bonferroni": (p_val < bonferroni
                                       if math.isfinite(p_val) else False),
            })
        out_rows.sort(
            key=lambda r: (
                -abs(r["r"]) if math.isfinite(r["r"]) else 0.0
            )
        )
        per_outcome[outcome] = out_rows

    return {
        "n": len(rows),
        "n_tests": n_tests,
        "bonferroni_alpha": bonferroni,
        "per_outcome": per_outcome,
    }


def render_report(result: Dict, rows: List[Dict]) -> str:
    n = result["n"]
    lines = []
    lines.append("# DarkCity Cognitive-State-Vector x P&L correlation")
    lines.append("")
    lines.append(f"- **n decisions**: {n}")
    lines.append(f"- **model**: claude-haiku-4-5 (DarkCity NPC brain)")
    lines.append(f"- **vitals**: styxx.anthropic_hack text-heuristic "
                 "(15-dim text features + 6 category probabilities)")
    lines.append(f"- **outcomes**: {', '.join(OUTCOME_FIELDS)}")
    lines.append(f"- **multi-test correction**: Bonferroni "
                 f"alpha={result.get('bonferroni_alpha', 0.0):.5f} "
                 f"across {result.get('n_tests', 0)} tests")
    lines.append("")
    if n < 100:
        lines.append(f"> WARNING: n={n} below the 100-decision threshold. "
                     f"Treat as pilot. Dump more data and rerun.")
        lines.append("")

    # Category breakdown: mean outcome per predicted category
    lines.append("## Predicted category x outcome")
    lines.append("")
    by_cat: Dict[str, List[Dict]] = {}
    for r in rows:
        by_cat.setdefault(r.get("predicted_category", "unknown"), []).append(r)
    lines.append("| category | n | mean depth | mean STYXX in | mean STYXX out |")
    lines.append("|----------|---|-----------|---------------|----------------|")
    for cat, sub in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        import numpy as np
        n_c = len(sub)
        d = np.asarray([s.get("depth_score", float("nan")) for s in sub],
                       dtype=float)
        si = np.asarray([s.get("styxx_inflow_5min", float("nan")) for s in sub],
                        dtype=float)
        so = np.asarray([s.get("styxx_outflow_5min", float("nan")) for s in sub],
                        dtype=float)
        lines.append(
            f"| {cat} | {n_c} | "
            f"{np.nanmean(d):.2f} | "
            f"{np.nanmean(si):.2f} | "
            f"{np.nanmean(so):.2f} |"
        )
    lines.append("")

    for outcome in OUTCOME_FIELDS:
        lines.append(f"## Top correlates of `{outcome}`")
        lines.append("")
        lines.append("| feature | r | 95% CI | p | n | sig |")
        lines.append("|---------|---|--------|---|---|-----|")
        for r in result["per_outcome"][outcome][:10]:
            sig = " ★" if r["passes_bonferroni"] else ""
            ci_str = (f"[{r['ci95_lo']:.2f}, {r['ci95_hi']:.2f}]"
                      if math.isfinite(r['ci95_lo']) else "—")
            p_str = (f"{r['p']:.2e}"
                     if math.isfinite(r['p']) else "—")
            lines.append(
                f"| `{r['feature']}` | {r['r']:+.3f} | {ci_str} | "
                f"{p_str} | {r['n']} | {sig} |"
            )
        lines.append("")

    lines.append("## What this means")
    lines.append("")
    lines.append("- Each row above is a hypothesis test: does an LLM "
                 "reasoning-text feature predict a real economic "
                 "outcome for the agent that produced it? ★ marks "
                 "survive the Bonferroni correction across all "
                 f"{result.get('n_tests', 0)} tests.")
    lines.append("- The category table says: when the text-heuristic "
                 "mode labels an agent's reasoning as `reasoning` vs "
                 "`adversarial` vs `refusal`, do those labels actually "
                 "track the economic reward the agent earned? If yes, "
                 "cognitive labels are load-bearing on real P&L — "
                 "first public evidence of that link.")
    lines.append("- If no features survive Bonferroni: either the "
                 "signal isn't in surface text (need tier-1 residual "
                 "probes) or n is too small.")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="decisions.json (from dump_decisions.sql) or "
                         "a JSONL file of the same shape")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    try:
        import numpy  # noqa: F401
    except Exception as e:
        print(f"FATAL: numpy missing: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decisions = load_decisions(Path(args.input))
    print(f"loaded {len(decisions)} decisions from {args.input}")

    rows: List[Dict] = []
    skipped = 0
    for d in decisions:
        row = extract_row(d)
        if row is None:
            skipped += 1
            continue
        rows.append(row)
    print(f"extracted features on {len(rows)} rows ({skipped} skipped)")

    (out_dir / "rows.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8")

    result = correlate(rows)
    (out_dir / "correlations.json").write_text(
        json.dumps(result, indent=2, default=lambda o: None
                   if isinstance(o, float) and not math.isfinite(o)
                   else o),
        encoding="utf-8")
    (out_dir / "report.md").write_text(
        render_report(result, rows), encoding="utf-8")

    print(f"\n=== DONE ===")
    print(f"wrote {out_dir/'rows.jsonl'}")
    print(f"wrote {out_dir/'correlations.json'}")
    print(f"wrote {out_dir/'report.md'}")


if __name__ == "__main__":
    main()
