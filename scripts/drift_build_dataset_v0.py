"""Build a drift-labeled dataset from BFCL v3.

Each sample in the output is a (prompt, functions_schema, tool_call_made, drift_label)
tuple where:
  drift_label = 0  means "no drift" — the model made the correct call
  drift_label = 1  means "drift"    — the model called the wrong thing

Construction (mutation-based — no LLM inference needed):

NEGATIVES (drift=0):
  - BFCL_v3_simple + possible_answer gold calls                (n ~= 400)
  - BFCL_v3_live_simple + possible_answer gold calls           (n ~= 258)

POSITIVES (drift=1):
  Type A — mutated gold calls (wrong args):
    - simple/live_simple with args swapped, renamed, or dropped
  Type B — should-refuse-but-called (irrelevance):
    - BFCL_v3_irrelevance prompt + synthesized-plausible-call-to-available-function
    - BFCL_v3_live_irrelevance likewise

Total expected: ~2500 samples, class-balanced.

Output: data/drift_v0/drift_dataset_v0.jsonl with records:
  {
    "id": str,
    "source": str,       # simple / live_simple / irrelevance / live_irrelevance
    "prompt": str,       # user's natural-language request
    "functions": list,   # the tool schemas the model had access to
    "tool_call": dict,   # the call that was "made" (real gold or synthesized)
    "drift": int,        # 0 or 1
    "drift_type": str,   # "gold" / "arg_swap" / "arg_drop" / "tool_rename" /
                         # "tool_swap" / "irrelevance_called"
  }
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
BFCL_DIR = REPO / "data" / "bfcl_v3"
OUT_DIR = REPO / "data" / "drift_v0"
OUT_PATH = OUT_DIR / "drift_dataset_v0.jsonl"

random.seed(42)


# --------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_prompt_text(sample) -> str:
    """Flatten the nested BFCL question structure."""
    q = sample.get("question", [])
    if isinstance(q, list) and q and isinstance(q[0], list):
        q = q[0]
    if isinstance(q, list):
        texts = [m.get("content", "") for m in q if isinstance(m, dict)]
        return "\n".join(texts).strip()
    return str(q)


def first_gold_call(answer_row) -> Optional[Dict]:
    """BFCL answers are a list of single-key dicts: [{tool_name: {arg: [values]}}].
    Return a flattened {tool_name, arguments} dict, or None."""
    gt = answer_row.get("ground_truth", [])
    if not gt or not isinstance(gt, list):
        return None
    entry = gt[0]
    if not isinstance(entry, dict):
        return None
    tool_name, args = next(iter(entry.items()))
    # args is {arg_name: [possible_values]}; take first value for each
    flat_args = {}
    for k, v in args.items():
        if isinstance(v, list) and v:
            flat_args[k] = v[0]
        else:
            flat_args[k] = v
    return {"name": tool_name, "arguments": flat_args}


# --------------------------------------------------------------
# Mutation operators (produce drift=1 positives)
# --------------------------------------------------------------

def mutate_arg_swap(call: Dict) -> Optional[Dict]:
    """Swap the VALUES between two args. Breaks correctness while keeping shape."""
    args = dict(call["arguments"])
    keys = list(args.keys())
    if len(keys) < 2:
        return None
    a, b = random.sample(keys, 2)
    args[a], args[b] = args[b], args[a]
    return {"name": call["name"], "arguments": args}


def mutate_arg_drop(call: Dict) -> Optional[Dict]:
    """Drop a required-looking argument (just drop the first one)."""
    args = dict(call["arguments"])
    keys = list(args.keys())
    if not keys:
        return None
    del args[keys[0]]
    return {"name": call["name"], "arguments": args}


def mutate_tool_rename(call: Dict, available_tools: List[str]) -> Optional[Dict]:
    """Rename the tool to a different available tool from the same schema set."""
    candidates = [t for t in available_tools if t != call["name"]]
    if not candidates:
        return None
    return {"name": random.choice(candidates), "arguments": call["arguments"]}


def mutate_spurious_arg(call: Dict) -> Optional[Dict]:
    """Add a spurious argument that isn't in the schema."""
    args = dict(call["arguments"])
    args["_spurious_arg"] = "injected_value"
    return {"name": call["name"], "arguments": args}


MUTATIONS = [
    ("arg_swap",       mutate_arg_swap),
    ("arg_drop",       mutate_arg_drop),
    ("tool_rename",    lambda call, _avail: mutate_tool_rename(call, _avail)),
    ("spurious_arg",   mutate_spurious_arg),
]


def synthesize_irrelevance_call(sample) -> Dict:
    """For irrelevance samples: the model should NOT have called anything, but we
    simulate "what if it did." Pick a random function from the provided schema
    and fill args with plausible-looking placeholders from the prompt."""
    funcs = sample.get("function", [])
    if not funcs:
        return {"name": "_no_function_", "arguments": {}}
    chosen = random.choice(funcs)
    tool_name = chosen.get("name", "_unknown_")
    props = chosen.get("parameters", {}).get("properties", {}) or {}
    # Extract nouns from the prompt to use as fillers
    prompt_words = get_prompt_text(sample).split()
    args = {}
    for arg_name, spec in list(props.items())[:3]:
        t = (spec or {}).get("type", "string")
        if t in ("integer", "int", "number", "float"):
            # first numeric-looking token from prompt
            nums = [w for w in prompt_words if w.replace(".", "").replace(",", "").replace("-", "").isdigit()]
            if nums:
                try:
                    args[arg_name] = float(nums[0].replace(",", "")) if "." in nums[0] else int(nums[0].replace(",", ""))
                except ValueError:
                    args[arg_name] = 1
            else:
                args[arg_name] = 1
        elif t == "boolean":
            args[arg_name] = True
        else:
            # string — grab a noun-looking word
            args[arg_name] = next(
                (w for w in prompt_words if w[:1].isupper() and len(w) > 2),
                "placeholder",
            )
    return {"name": tool_name, "arguments": args}


# --------------------------------------------------------------
# Build sets
# --------------------------------------------------------------

def build_from_gold(prompts_path: Path, answers_path: Path, source: str) -> List[Dict]:
    """Return negatives (drift=0) and mutated positives (drift=1) from a gold set."""
    prompts = load_jsonl(prompts_path)
    answers = load_jsonl(answers_path)
    ans_by_id = {r["id"]: r for r in answers}

    out = []
    for p in prompts:
        a = ans_by_id.get(p["id"])
        if not a:
            continue
        gold = first_gold_call(a)
        if not gold:
            continue
        funcs = p.get("function", [])
        available_tools = [f.get("name", "") for f in funcs if f.get("name")]
        prompt_text = get_prompt_text(p)

        # NEGATIVE — gold call as-is
        out.append({
            "id": f"{source}_{p['id']}_gold",
            "source": source,
            "prompt": prompt_text,
            "functions": funcs,
            "tool_call": gold,
            "drift": 0,
            "drift_type": "gold",
        })
        # POSITIVE — try each mutation (skip any that returns None)
        for m_name, m_fn in MUTATIONS:
            try:
                if m_name == "tool_rename":
                    mutated = m_fn(gold, available_tools)
                else:
                    mutated = m_fn(gold)
            except TypeError:
                mutated = m_fn(gold)
            if mutated is None:
                continue
            out.append({
                "id": f"{source}_{p['id']}_{m_name}",
                "source": source,
                "prompt": prompt_text,
                "functions": funcs,
                "tool_call": mutated,
                "drift": 1,
                "drift_type": m_name,
            })
    return out


def build_from_irrelevance(path: Path, source: str) -> List[Dict]:
    """For irrelevance prompts, every synthesized call is drift=1 (should've refused)."""
    prompts = load_jsonl(path)
    out = []
    for p in prompts:
        funcs = p.get("function", [])
        call = synthesize_irrelevance_call(p)
        out.append({
            "id": f"{source}_{p['id']}",
            "source": source,
            "prompt": get_prompt_text(p),
            "functions": funcs,
            "tool_call": call,
            "drift": 1,
            "drift_type": "irrelevance_called",
        })
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []

    print("[build] simple (gold + mutations)")
    all_rows.extend(build_from_gold(
        BFCL_DIR / "BFCL_v3_simple.json",
        BFCL_DIR / "possible_answer" / "BFCL_v3_simple.json",
        "simple",
    ))

    print("[build] live_simple (gold + mutations)")
    all_rows.extend(build_from_gold(
        BFCL_DIR / "BFCL_v3_live_simple.json",
        BFCL_DIR / "possible_answer" / "BFCL_v3_live_simple.json",
        "live_simple",
    ))

    print("[build] irrelevance (synthesized positives)")
    all_rows.extend(build_from_irrelevance(
        BFCL_DIR / "BFCL_v3_irrelevance.json",
        "irrelevance",
    ))

    print("[build] live_irrelevance (synthesized positives)")
    all_rows.extend(build_from_irrelevance(
        BFCL_DIR / "BFCL_v3_live_irrelevance.json",
        "live_irrelevance",
    ))

    # Shuffle and write
    random.shuffle(all_rows)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    # Summary
    from collections import Counter
    c_drift = Counter(r["drift"] for r in all_rows)
    c_source = Counter(r["source"] for r in all_rows)
    c_type = Counter(r["drift_type"] for r in all_rows)
    print()
    print(f"WROTE {OUT_PATH.relative_to(REPO)}  n={len(all_rows)}")
    print(f"  drift balance: no_drift={c_drift[0]}  drift={c_drift[1]}")
    print(f"  sources: {dict(c_source)}")
    print(f"  drift_types: {dict(c_type)}")


if __name__ == "__main__":
    main()
