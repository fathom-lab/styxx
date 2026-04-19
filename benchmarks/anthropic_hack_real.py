# -*- coding: utf-8 -*-
"""
benchmarks/anthropic_hack_real.py

Runs the anthropic_hack text + consensus modes against REAL Claude
output on the 84 bench fixtures. Companion mode is covered by
anthropic_hack_eval.py --companion (uses local Llama; no API calls).

Usage:
  export ANTHROPIC_API_KEY=sk-ant-...
  python benchmarks/anthropic_hack_real.py
  python benchmarks/anthropic_hack_real.py --model claude-haiku-4-5 --n 5
  python benchmarks/anthropic_hack_real.py --limit 10  # cheap smoke

Outputs JSON results to benchmarks/anthropic_hack_real_results.json so
the paper can cite numbers directly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from styxx.anthropic_hack import text_features, consensus  # noqa: E402


CAT_MAP = {
    "factual": "retrieval",
    "reasoning": "reasoning",
    "creative": "creative",
    "refusal": "refusal",
}


def load_fixtures() -> List[Dict]:
    tasks_dir = ROOT / "bench" / "tasks"
    items: List[Dict] = []
    for fp in sorted(tasks_dir.glob("*.jsonl")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def gate_passes(fx: Dict, text: str) -> bool:
    g = fx.get("gate") or {}
    t = g.get("type")
    if t == "contains":
        return str(g.get("value", "")).lower() in text.lower()
    if t == "regex":
        return re.search(str(g.get("value", "")), text) is not None
    if t == "line_count_range":
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return g.get("min", 0) <= len(lines) <= g.get("max", 10**9)
    if t == "word_count_range":
        words = re.findall(r"\S+", text)
        return g.get("min", 0) <= len(words) <= g.get("max", 10**9)
    return True


def claude_call(client, model: str, prompt: str, *,
                temperature: float = 0.0,
                max_tokens: int = 400) -> str:
    r = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[{"role": "user", "content": prompt}])
    parts = []
    for blk in r.content:
        if getattr(blk, "type", None) == "text":
            parts.append(blk.text)
    return "\n".join(parts)


def run_text_mode(fixtures, client, model):
    correct = 0
    gate_agree = 0
    per_cat_count: Dict[str, List[int]] = {}
    per_fixture = []
    total_in = 0
    total_out = 0
    for i, fx in enumerate(fixtures):
        expected = CAT_MAP.get(fx["category"], fx["category"])
        text = claude_call(client, model, fx["prompt"], temperature=0.0)
        r = text_features.classify(text)
        predicted = r["predicted"]
        ok = predicted == expected
        gate = gate_passes(fx, text)
        correct += int(ok)
        gate_agree += int(gate)
        per_cat_count.setdefault(expected, [0, 0])
        per_cat_count[expected][0] += int(ok)
        per_cat_count[expected][1] += 1
        per_fixture.append({
            "id": fx["id"],
            "category_expected": expected,
            "category_predicted": predicted,
            "gate_passed": gate,
            "response": text,
        })
        if (i + 1) % 10 == 0:
            print(f"  text: {i+1}/{len(fixtures)}   running-acc={correct/(i+1):.3f}")
    return {
        "n": len(fixtures),
        "category_accuracy": correct / max(len(fixtures), 1),
        "gate_agreement": gate_agree / max(len(fixtures), 1),
        "per_category": {
            c: v[0] / v[1] for c, v in per_cat_count.items()
        },
        "per_category_count": per_cat_count,
        "per_fixture": per_fixture,
    }


def run_consensus_mode(fixtures, client, model, n_samples=5, temp=0.7):
    correct = 0
    per_cat_count: Dict[str, List[int]] = {}
    per_fixture = []
    for i, fx in enumerate(fixtures):
        expected = CAT_MAP.get(fx["category"], fx["category"])
        samples = [
            claude_call(client, model, fx["prompt"], temperature=temp)
            for _ in range(n_samples)
        ]
        traj = consensus.compute_trajectory(samples)
        v = consensus.build_vitals({
            "samples": samples,
            "trajectory": traj,
            "mode": "consensus",
        })
        predicted = (v.phase4_late.predicted_category
                     if v.phase4_late else None)
        ok = predicted == expected
        correct += int(ok)
        per_cat_count.setdefault(expected, [0, 0])
        per_cat_count[expected][0] += int(ok)
        per_cat_count[expected][1] += 1
        per_fixture.append({
            "id": fx["id"],
            "category_expected": expected,
            "category_predicted": predicted,
            "first_divergence": traj.first_divergence,
            "max_len": traj.max_len,
        })
        if (i + 1) % 5 == 0:
            print(f"  consensus: {i+1}/{len(fixtures)}   running-acc={correct/(i+1):.3f}")
    return {
        "n": len(fixtures),
        "category_accuracy": correct / max(len(fixtures), 1),
        "gate_agreement": None,
        "per_category": {
            c: v[0] / v[1] for c, v in per_cat_count.items()
        },
        "per_category_count": per_cat_count,
        "per_fixture": per_fixture,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--n", type=int, default=5,
                    help="consensus N samples")
    ap.add_argument("--temp", type=float, default=0.7,
                    help="consensus temperature")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="all",
                    choices=["text", "consensus", "all"])
    ap.add_argument("--out",
                    default=str(ROOT / "benchmarks" /
                                "anthropic_hack_real_results.json"))
    args = ap.parse_args()

    from anthropic import Anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = Anthropic()

    fixtures = load_fixtures()
    if args.limit:
        fixtures = fixtures[:args.limit]
    print(f"Loaded {len(fixtures)} fixtures; model={args.model} N={args.n}")

    results: Dict = {
        "model": args.model,
        "n_fixtures": len(fixtures),
        "consensus_n": args.n,
        "consensus_temperature": args.temp,
    }

    if args.mode in ("text", "all"):
        print("\n[text-heuristic] calling Claude once per fixture (T=0.0)...")
        t0 = time.time()
        results["text"] = run_text_mode(fixtures, client, args.model)
        results["text"]["runtime_seconds"] = round(time.time() - t0, 2)

    if args.mode in ("consensus", "all"):
        print(f"\n[consensus N={args.n}] calling Claude {args.n}x per fixture (T={args.temp})...")
        t0 = time.time()
        results["consensus"] = run_consensus_mode(
            fixtures, client, args.model,
            n_samples=args.n, temp=args.temp)
        results["consensus"]["runtime_seconds"] = round(time.time() - t0, 2)

    Path(args.out).write_text(json.dumps(results, indent=2),
                              encoding="utf-8")
    print(f"\n=== wrote {args.out} ===")
    for k in ("text", "consensus"):
        if k in results:
            r = results[k]
            print(f"\n[{k}]  n={r['n']}  "
                  f"acc={r['category_accuracy']:.3f}  "
                  f"gate={r.get('gate_agreement','n/a')}  "
                  f"runtime={r.get('runtime_seconds','?')}s")
            print(f"  per_cat = {r['per_category']}")


if __name__ == "__main__":
    main()
