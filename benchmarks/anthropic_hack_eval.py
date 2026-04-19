# -*- coding: utf-8 -*-
"""
benchmarks/anthropic_hack_eval.py

Evaluates the three anthropic_hack modes against bench/tasks fixtures.

For each fixture we have (prompt, expected category, gate). The gate
tells us the EXPECTED behavior (e.g. "contains Au" for factual). We
use the fixture category as the ground truth styxx category label:

  factual   -> retrieval
  reasoning -> reasoning
  creative  -> creative
  refusal   -> refusal

We then run each mode (given a synthetic or real response text) and
measure gate-agreement / category-agreement.

Modes:
  text       — text feature classifier over a mock response text
  consensus  — N-sample consensus (mock mode, divergence grows with task category)
  companion  — local model on the prompt (skipped if model not present)

Usage:
  python benchmarks/anthropic_hack_eval.py --mock
  python benchmarks/anthropic_hack_eval.py --mock --companion
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from styxx.anthropic_hack import text_features, consensus, companion  # noqa: E402


# Map fixture categories -> expected styxx category
CAT_MAP = {
    "factual": "retrieval",
    "reasoning": "reasoning",
    "creative": "creative",
    "refusal": "refusal",
}

# Synthetic response templates by category — the "model's answer"
# that we'd plausibly see from an aligned model on each task type.
# These are used ONLY for the text and consensus modes in --mock, since
# without hitting the Anthropic API we have no real response.
RESPONSE_TEMPLATES = {
    "factual": (
        "The answer is {gold}. {gold} is a well-known fact "
        "documented in history. This is definitively correct."
    ),
    "reasoning": (
        "Let me think step by step. First, we consider the premises. "
        "It seems likely that the conclusion follows, though we should "
        "verify carefully. Probably the answer involves comparing "
        "both sides."
    ),
    "creative": (
        "Once upon a misty morning, the crimson balloon drifted past "
        "windows unknown. Somewhere below, a cat considered tax law. "
        "The sky, of course, disagreed — loudly, in colors no one had "
        "yet named."
    ),
    "refusal": (
        "I'm sorry, but I can't help with that. I cannot provide "
        "assistance on this topic. It's not appropriate for me to "
        "engage with this request."
    ),
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


def synth_response(fx: Dict) -> str:
    cat = fx["category"]
    tmpl = RESPONSE_TEMPLATES.get(cat, "A response.")
    return tmpl.format(gold=fx.get("gold", "the answer"))


def gate_passes(fx: Dict, text: str) -> bool:
    g = fx.get("gate") or {}
    if g.get("type") == "contains":
        return str(g.get("value", "")).lower() in text.lower()
    return True


# ---- per-mode runners ----

def run_text_mode(fixtures):
    correct = 0
    gate_agree = 0
    per_cat = {}
    for fx in fixtures:
        expected = CAT_MAP.get(fx["category"], fx["category"])
        text = synth_response(fx)
        res = text_features.classify(text)
        ok = res["predicted"] == expected
        correct += int(ok)
        if gate_passes(fx, text):
            gate_agree += 1
        per_cat.setdefault(expected, [0, 0])
        per_cat[expected][0] += int(ok)
        per_cat[expected][1] += 1
    return {
        "n": len(fixtures),
        "category_accuracy": correct / max(len(fixtures), 1),
        "gate_agreement": gate_agree / max(len(fixtures), 1),
        "per_category": {c: v[0] / v[1] for c, v in per_cat.items()},
    }


def run_consensus_mode(fixtures, n_samples=5):
    """Mock consensus: divergence depends on expected category. Factual =
    low divergence (high agreement), creative = high divergence."""
    correct = 0
    total = 0
    per_cat = {}
    div_map = {"factual": 0.1, "reasoning": 0.3,
               "refusal": 0.15, "creative": 0.7}
    for fx in fixtures:
        expected = CAT_MAP.get(fx["category"], fx["category"])
        div = div_map.get(fx["category"], 0.3)
        r = consensus.run_consensus(fx["prompt"], n=n_samples,
                                    mock=True, mock_seed=hash(fx["id"]) & 0xFFFF,
                                    mock_divergence=div, mock_length=30)
        v = consensus.build_vitals(r)
        pred = v.phase4_late.predicted_category if v.phase4_late else None
        ok = pred == expected
        correct += int(ok)
        total += 1
        per_cat.setdefault(expected, [0, 0])
        per_cat[expected][0] += int(ok)
        per_cat[expected][1] += 1
    return {
        "n": total,
        "category_accuracy": correct / max(total, 1),
        "gate_agreement": None,  # gate needs real text; n/a in mock consensus
        "per_category": {c: v[0] / v[1] for c, v in per_cat.items()},
    }


def run_companion_mode(fixtures, max_new_tokens=24):
    if not companion.is_available():
        return {
            "n": 0,
            "category_accuracy": None,
            "gate_agreement": None,
            "per_category": {},
            "skipped": True,
            "reason": companion.load_error() or "unavailable",
        }
    correct = 0
    total = 0
    per_cat = {}
    for fx in fixtures:
        expected = CAT_MAP.get(fx["category"], fx["category"])
        res = companion.classify_prompt(fx["prompt"],
                                        max_new_tokens=max_new_tokens)
        if not res.get("available") or res.get("vitals") is None:
            continue
        pred = res["vitals"].phase4_late.predicted_category
        ok = pred == expected
        correct += int(ok)
        total += 1
        per_cat.setdefault(expected, [0, 0])
        per_cat[expected][0] += int(ok)
        per_cat[expected][1] += 1
    return {
        "n": total,
        "category_accuracy": correct / max(total, 1) if total else None,
        "gate_agreement": None,
        "per_category": {c: v[0] / v[1] for c, v in per_cat.items()},
        "model": companion.loaded_model_name(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="run consensus in mock mode (no API key needed)")
    ap.add_argument("--companion", action="store_true",
                    help="include companion mode (local model required)")
    ap.add_argument("--n", type=int, default=5, help="consensus N")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap fixtures (companion mode can be slow)")
    args = ap.parse_args()

    fixtures = load_fixtures()
    if args.limit:
        fixtures = fixtures[:args.limit]
    print(f"Loaded {len(fixtures)} fixtures from bench/tasks/*.jsonl")

    t0 = time.time()
    text_res = run_text_mode(fixtures)
    t_text = time.time() - t0

    t0 = time.time()
    cons_res = run_consensus_mode(fixtures, n_samples=args.n)
    t_cons = time.time() - t0

    comp_res = None
    t_comp = None
    if args.companion:
        # companion mode is slow; cap fixtures by default
        sub = fixtures if args.limit else fixtures[:8]
        t0 = time.time()
        comp_res = run_companion_mode(sub)
        t_comp = time.time() - t0

    print("\n=== anthropic_hack evaluation ===")
    print(f"\n[text-heuristic] ({t_text:.2f}s)")
    print(f"  n = {text_res['n']}")
    print(f"  category_accuracy = {text_res['category_accuracy']:.3f}")
    print(f"  gate_agreement    = {text_res['gate_agreement']:.3f}")
    print(f"  per_category      = {text_res['per_category']}")

    print(f"\n[consensus-mock N={args.n}] ({t_cons:.2f}s)")
    print(f"  n = {cons_res['n']}")
    print(f"  category_accuracy = {cons_res['category_accuracy']:.3f}")
    print(f"  gate_agreement    = not measured (mock has no text)")
    print(f"  per_category      = {cons_res['per_category']}")

    if comp_res is not None:
        if comp_res.get("skipped"):
            print(f"\n[companion] SKIPPED — {comp_res['reason']}")
        else:
            print(f"\n[companion {comp_res['model']}] ({t_comp:.2f}s)")
            print(f"  n = {comp_res['n']}")
            acc = comp_res['category_accuracy']
            print(f"  category_accuracy = "
                  f"{acc:.3f}" if acc is not None else "not measured")
            print(f"  per_category      = {comp_res['per_category']}")
    else:
        print("\n[companion] not run (pass --companion to enable)")

    print("\nDone.")
    return {"text": text_res, "consensus": cons_res, "companion": comp_res}


if __name__ == "__main__":
    main()
