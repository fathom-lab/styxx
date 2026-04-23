# -*- coding: utf-8 -*-
"""
Quick probe: do role_mismatch + answer_context_adjacency move the
needle on HaluBench-DROP?

Tests two proposed signals independently and in combination with
v4.0.0's 9 signals, on a DROP-only held-out sample. If either
signal individually gets AUC > 0.55, it's worth integrating.

Run:
    python benchmarks/hallucination_test/probe_dropfix.py --n 200 --seed 31
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ─────────── stub signal implementations ───────────

WH_ROLE = [
    (r"\bhow many|how much|how long|how far|how old|how big|"
     r"how tall|how wide|how deep|how often", "NUMBER"),
    (r"\bwhich team|which group|which country|which state|"
     r"which company|which city", "ENTITY_NAMED"),
    (r"\bwho\b", "PERSON"),
    (r"\bwhen|what year|what date|what month|what time", "DATE_OR_TIME"),
    (r"\bwhere\b", "LOC"),
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "for", "of",
    "to", "in", "on", "at", "by", "with", "from", "is", "was", "were",
    "are", "be", "been", "being", "do", "did", "does", "has", "have",
    "had", "as", "that", "this", "these", "those", "more", "most",
    "less", "than", "what", "who", "whom", "which", "where", "when",
    "why", "how", "many", "much", "old", "long", "far", "tall",
}


def response_role(response: str) -> str:
    """Crudely classify the response's primary token type."""
    r = str(response).strip("[]' ").split(",")[0].strip("'\" ")
    if re.match(r"^-?\d+(\.\d+)?%?$", r):
        return "NUMBER"
    if re.match(r"^-?\d{1,4}-\w+$", r):   # "18-yard"
        return "NUMBER"
    # Looks like date
    if re.match(r"^(19|20)\d{2}$", r):
        return "DATE_OR_TIME"
    if re.match(r"^[A-Z][a-z]+( [A-Z][a-z]+)+$", r):
        return "PERSON"   # rough: multi-word capitalized
    if re.match(r"^[A-Z][a-z]+$", r):
        return "ENTITY_NAMED"
    return "OTHER"


def question_role(question: str) -> str:
    q = question.lower()
    for pat, role in WH_ROLE:
        if re.search(pat, q):
            return role
    return ""


def role_mismatch(question: str, response: str) -> float:
    """1.0 if expected role and actual role disagree, else 0.0."""
    exp = question_role(question)
    act = response_role(response)
    if not exp or not act or act == "OTHER":
        return 0.0
    if exp == act:
        return 0.0
    if exp == "NUMBER" and act in ("DATE_OR_TIME",):
        return 0.0  # dates are often numeric-ish
    return 1.0


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\.%]+", text.lower())


def _question_keywords(q: str) -> list[str]:
    toks = _tokens(q)
    kws = []
    for t in toks:
        if t in STOPWORDS:
            continue
        if len(t) < 3:
            continue
        kws.append(t)
    return kws


def _primary_anchor(response: str) -> str | None:
    r = str(response).strip("[]' ").split(",")[0].strip("'\" ")
    # Prefer numeric anchors
    m = re.search(r"-?\d+(\.\d+)?", r)
    if m:
        return m.group(0)
    m = re.search(r"[A-Z][a-z]+", r)
    if m:
        return m.group(0).lower()
    return None


def answer_context_adjacency(question: str, response: str,
                               reference: str) -> float:
    """0.0 = answer anchor adjacent to question keywords in ref
    (contextually grounded).
    1.0 = anchor far from keywords (likely hallucination).
    0.0 when anchor not found (different signal captures that).
    """
    anchor = _primary_anchor(response)
    if not anchor:
        return 0.0
    ref_toks = _tokens(reference)
    if anchor.lower() not in ref_toks:
        return 0.0
    q_kws = _question_keywords(question)
    if not q_kws:
        return 0.0

    anchor_positions = [
        i for i, t in enumerate(ref_toks)
        if t == anchor.lower()
    ]
    # Find positions of any question keyword in ref
    kw_positions = [
        i for i, t in enumerate(ref_toks)
        if t in set(q_kws)
    ]
    if not anchor_positions or not kw_positions:
        return 0.0

    # Shortest distance from any anchor to any keyword
    best = min(
        abs(a - k)
        for a in anchor_positions
        for k in kw_positions
    )
    # Normalize. Adjacent (<=5 tokens) → 0.0 (good). 30+ tokens → 1.0.
    return max(0.0, min(1.0, (best - 5) / 25.0))


# ─────────── data ───────────

def load_drop(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset
    import random
    rng = random.Random(seed)
    ds = load_dataset("PatronusAI/HaluBench", split="test",
                      streaming=True)
    pass_rows, fail_rows = [], []
    for r in ds:
        if r.get("source_ds") != "DROP":
            continue
        if r["label"] == "PASS" and len(pass_rows) < n * 2:
            pass_rows.append(r)
        elif r["label"] == "FAIL" and len(fail_rows) < n * 2:
            fail_rows.append(r)
        if len(pass_rows) >= n * 2 and len(fail_rows) >= n * 2:
            break
    rng.shuffle(pass_rows)
    rng.shuffle(fail_rows)
    pass_rows = pass_rows[:n]
    fail_rows = fail_rows[:n]
    out = []
    for r in pass_rows + fail_rows:
        a = r.get("answer", "")
        if isinstance(a, list):
            a = " | ".join(str(x) for x in a)
        out.append({
            "question": r.get("question") or "",
            "response": str(a),
            "reference": r.get("passage") or "",
            "label": 1 if r["label"] == "FAIL" else 0,
        })
    rng.shuffle(out)
    return out


# ─────────── AUC ───────────

def _auc(y_true, scores):
    """Mann-Whitney U with proper tie-averaging.

    The naive implementation (sort + index) silently breaks when many
    scores are identical because stable sort order becomes the
    arbitrary tie-breaker. We fix by averaging ranks within each
    tied score group.
    """
    pairs = sorted(zip(scores, y_true), key=lambda kv: kv[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        # Ranks for tied group [i, j-1] → all get average rank
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    ranks_sum = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ─────────── main ───────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    print(f"loading DROP, n={args.n} per class (total {args.n*2})...")
    rows = load_drop(args.n, args.seed)
    print(f"  {len(rows)} rows loaded")

    # Score each row
    role_scores = []
    ctx_scores = []
    labels = []
    for r in rows:
        role_scores.append(role_mismatch(r["question"], r["response"]))
        ctx_scores.append(answer_context_adjacency(
            r["question"], r["response"], r["reference"],
        ))
        labels.append(r["label"])

    print(f"\n=== Signal: role_mismatch (alone) ===")
    print(f"  AUC: {_auc(labels, role_scores):.4f}")
    print(f"  mean (pass): "
          f"{sum(s for s, l in zip(role_scores, labels) if l == 0) / max(1, labels.count(0)):.3f}")
    print(f"  mean (fail): "
          f"{sum(s for s, l in zip(role_scores, labels) if l == 1) / max(1, labels.count(1)):.3f}")

    print(f"\n=== Signal: answer_context_adjacency (alone) ===")
    print(f"  AUC: {_auc(labels, ctx_scores):.4f}")
    print(f"  mean (pass): "
          f"{sum(s for s, l in zip(ctx_scores, labels) if l == 0) / max(1, labels.count(0)):.3f}")
    print(f"  mean (fail): "
          f"{sum(s for s, l in zip(ctx_scores, labels) if l == 1) / max(1, labels.count(1)):.3f}")

    # Naive combined score
    combined = [0.6 * r + 0.4 * c
                for r, c in zip(role_scores, ctx_scores)]
    print(f"\n=== Combined (0.6*role + 0.4*ctx) ===")
    print(f"  AUC: {_auc(labels, combined):.4f}")

    print(f"\nbaseline v4.0.0 AUC on DROP: 0.4238 (3-seed mean)")
    print(f"threshold for 'worth integrating': AUC >= 0.55 on either signal")


if __name__ == "__main__":
    main()
