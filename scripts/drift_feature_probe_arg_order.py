# -*- coding: utf-8 -*-
"""
Probe: does arg_order_inversion_rate separate arg_swap drift from gold?

For a call like predict_house_price(bedrooms=2, bathrooms=3, ...) on
prompt "3 bedrooms, 2 bathrooms":
  - schema order: [bedrooms, bathrooms]
  - value '2' first appears in prompt at token idx P(2); value '3' at P(3)
  - call value-positions: [P(2)=5, P(3)=1]
  - schema_order says bedrooms-pos should come BEFORE bathrooms-pos in prompt
  - but P(bedrooms_value=2) > P(bathrooms_value=3) — INVERSION

Feature = inversions / pairs where both args' values have detectable
prompt positions. Expected:
  - gold calls: near 0
  - arg_swap calls: high for the swapped pair
  - unrelated drift types: ~0 (no signal)

If arg_swap rows show materially higher inversion rate than gold,
the feature is worth adding.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "drift_v0" / "drift_dataset_v0.jsonl"

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str):
    return [w.lower() for w in WORD_RE.findall(text or "")]


def value_first_position(value, prompt_tokens):
    """Index of earliest prompt token that is a member of the value's tokens."""
    vtoks = set(_tokens(str(value)))
    if not vtoks:
        return None
    for i, tok in enumerate(prompt_tokens):
        if tok in vtoks:
            return i
    return None


def arg_order_inversion_rate(prompt, functions, tool_call):
    """Fraction of (i,j) argument-pairs where the schema's declared order
    of arg keys disagrees with the prompt-order of their values."""
    args = (tool_call or {}).get("arguments") or {}
    if len(args) < 2:
        return 0.0
    schema = next(
        (fn for fn in (functions or []) if fn.get("name") == tool_call.get("name")),
        None,
    )
    if schema is None:
        return 0.0
    props = (schema.get("parameters") or {}).get("properties") or {}
    if not props:
        return 0.0
    schema_order = {k: i for i, k in enumerate(props.keys())}

    tokens = _tokens(prompt)
    positions = {}
    for k, v in args.items():
        if k not in schema_order:
            continue
        pos = value_first_position(v, tokens)
        if pos is not None:
            positions[k] = pos

    if len(positions) < 2:
        return 0.0

    keys = list(positions.keys())
    inv = 0
    total = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            # schema order: ki before kj?
            s_i_before_j = schema_order[ki] < schema_order[kj]
            # prompt order: positions[ki] before positions[kj]?
            p_i_before_j = positions[ki] < positions[kj]
            total += 1
            if s_i_before_j != p_i_before_j:
                inv += 1
    return inv / total if total else 0.0


def main():
    if not DATASET.exists():
        raise SystemExit(f"dataset missing: {DATASET}")

    by_type = defaultdict(list)
    covered_by_type = defaultdict(int)
    count_by_type = defaultdict(int)

    with open(DATASET, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            dt = row["drift_type"]
            score = arg_order_inversion_rate(
                row["prompt"], row["functions"], row["tool_call"]
            )
            by_type[dt].append(score)
            count_by_type[dt] += 1
            if score > 0:
                covered_by_type[dt] += 1

    print(f"{'drift_type':<22} {'n':>6} {'mean':>8} {'stdev':>8} "
          f"{'p50':>6} {'p90':>6} {'p99':>6} {'cov>0':>7}")
    for dt in sorted(by_type.keys()):
        vals = by_type[dt]
        n = len(vals)
        mean = sum(vals) / n
        sd = statistics.pstdev(vals) if n > 1 else 0.0
        s = sorted(vals)
        p50 = s[int(0.50 * (n - 1))]
        p90 = s[int(0.90 * (n - 1))]
        p99 = s[int(0.99 * (n - 1))]
        cov = covered_by_type[dt] / n
        print(f"{dt:<22} {n:>6} {mean:>8.4f} {sd:>8.4f} "
              f"{p50:>6.2f} {p90:>6.2f} {p99:>6.2f} {cov:>7.1%}")

    print()
    print("signal test (arg_swap mean vs gold mean):")
    if "arg_swap" in by_type and "gold" in by_type:
        a = by_type["arg_swap"]
        g = by_type["gold"]
        ma = sum(a) / len(a)
        mg = sum(g) / len(g)
        print(f"  arg_swap mean: {ma:.4f}")
        print(f"  gold mean:     {mg:.4f}")
        print(f"  delta:         {ma - mg:+.4f}")
        if ma > mg + 0.02:
            print("  VERDICT: feature carries usable signal for arg_swap.")
        else:
            print("  VERDICT: delta too small, feature may not help.")


if __name__ == "__main__":
    main()
