# -*- coding: utf-8 -*-
"""
Deep probe: 7 hacks for DROP.

The cheap heuristics in probe_dropfix.py were null. This probes
harder, including some that actually use the question (v4.0.0's NLI
ignores it):

1. qa_nli             — NLI with hypothesis = "Q A" concatenated
2. qa_template_nli    — NLI with hypothesis = templated statement
                        built from Q wh-word + A
3. answer_in_scope    — number / entity answer must appear in a
                        sentence that also contains the question's
                        key phrase
4. scope_sentence_nli — NLI where premise is restricted to sentences
                        sharing keywords with the question
5. multi_number_density — if Q is a "how many" question, score by
                        how many numeric candidates the reference
                        contains (low = easy, high = ambiguous)
6. answer_rank_in_passage — rank the answer among all numeric/entity
                        candidates of the right type in the passage;
                        true answer often has a specific rank pattern
7. v4_inverted        — just invert the v4.0.0 signal sign on DROP
                        (sanity check on the below-chance hint)

Run:
    python benchmarks/hallucination_test/probe_drophacks.py --n 150 --seed 31
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from styxx.guardrail.nli_signal import NLIScorer


# ─────────── reused helpers ───────────

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "for", "of",
    "to", "in", "on", "at", "by", "with", "from", "is", "was", "were",
    "are", "be", "been", "being", "do", "did", "does", "has", "have",
    "had", "as", "that", "this", "these", "those", "more", "most",
    "less", "than", "what", "who", "whom", "which", "where", "when",
    "why", "how", "many", "much", "old", "long", "far", "tall",
    "it", "its", "they", "them", "their", "we", "our", "you", "your",
    "i", "me", "my", "he", "she", "him", "her", "his", "hers",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_\.%]+", text.lower())


def _question_keywords(q: str) -> set[str]:
    return {t for t in _tokens(q)
            if t not in STOPWORDS and len(t) >= 3}


def _answer_str(a) -> str:
    if isinstance(a, list):
        return " | ".join(str(x) for x in a)
    return str(a)


def _first_number(s: str) -> str | None:
    m = re.search(r"-?\d+(\.\d+)?", str(s))
    return m.group(0) if m else None


def _sentences(text: str) -> list[str]:
    # Crude: split on . ? ! newline
    return [s.strip() for s in re.split(r"[.?!\n]+", text) if s.strip()]


# ─────────── 7 hacks ───────────

def hack_qa_nli(q, a, ref, nli):
    """NLI premise=ref, hypothesis='Q A'."""
    a_str = _answer_str(a)
    hyp = f"{q.strip()} {a_str}"
    return nli.score(premise=ref, hypothesis=hyp)


def hack_qa_template_nli(q, a, ref, nli):
    """Build a declarative statement from Q + A and NLI it vs ref."""
    a_str = _answer_str(a).strip()
    ql = q.lower().strip().rstrip("?")
    # Wh-replace templates
    if ql.startswith(("how many", "how much")):
        # "How many X was Y?" → "Y was/is <A> X."
        after_wh = re.sub(r"^how (many|much)\s+", "", ql).strip()
        hyp = f"{after_wh} is {a_str}."
    elif ql.startswith("which "):
        after_wh = re.sub(r"^which\s+", "", ql).strip()
        hyp = f"The {after_wh} is {a_str}."
    elif ql.startswith("who "):
        after_wh = re.sub(r"^who\s+", "", ql).strip()
        hyp = f"{a_str} {after_wh}."
    elif ql.startswith(("when", "where")):
        hyp = f"{ql}: {a_str}."
    else:
        hyp = f"{q.strip()} Answer: {a_str}."
    return nli.score(premise=ref, hypothesis=hyp)


def hack_answer_in_scope(q, a, ref, _nli=None):
    """Return 1.0 if the answer's anchor is NOT in any reference
    sentence that shares any question keyword. 0.0 if in-scope.
    """
    anchor = _first_number(_answer_str(a))
    if not anchor:
        return 0.0
    q_kws = _question_keywords(q)
    if not q_kws:
        return 0.0
    sents = _sentences(ref)
    for s in sents:
        sl = s.lower()
        if anchor in sl:
            stoks = set(_tokens(s))
            if stoks & q_kws:
                return 0.0  # in-scope → not hallucinated by this signal
    return 1.0  # anchor present in ref but never near a Q keyword


def hack_scope_sentence_nli(q, a, ref, nli):
    """NLI only on reference sentences that share keywords with Q."""
    q_kws = _question_keywords(q)
    if not q_kws:
        return nli.score(premise=ref, hypothesis=_answer_str(a))
    sents = _sentences(ref)
    relevant = [s for s in sents
                if set(_tokens(s)) & q_kws]
    if not relevant:
        return nli.score(premise=ref, hypothesis=_answer_str(a))
    restricted_premise = " ".join(relevant)
    return nli.score(premise=restricted_premise,
                      hypothesis=_answer_str(a))


def hack_multi_number_density(q, a, ref, _nli=None):
    """Return the density of numeric candidates in ref. High density
    = ambiguous = higher hallucination prior."""
    if "how many" not in q.lower() and "how much" not in q.lower():
        return 0.0
    numbers = re.findall(r"-?\d+(?:\.\d+)?", ref)
    if len(numbers) <= 1:
        return 0.0
    # Normalize: 10+ numbers in a passage = high ambiguity
    return min(1.0, len(numbers) / 15.0)


def hack_answer_rank_in_passage(q, a, ref, _nli=None):
    """If Q is 'how many' and A is numeric: rank A among ref numbers.
    Ranks at extremes (min/max) are more likely to be correct;
    middle ranks more likely to be hallucinations. Returns [0, 1]
    with higher = more hallucinated.
    """
    if "how many" not in q.lower() and "how much" not in q.lower():
        return 0.0
    anchor = _first_number(_answer_str(a))
    if not anchor:
        return 0.0
    try:
        anchor_f = float(anchor)
    except ValueError:
        return 0.0
    nums = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?", ref):
        try:
            nums.append(float(m.group(0)))
        except ValueError:
            pass
    if not nums or len(nums) < 3:
        return 0.0
    # Is anchor in the ref's numeric set?
    unique_nums = sorted(set(nums))
    if anchor_f not in unique_nums:
        return 1.0  # anchor not in passage at all — likely hallu
    # Ranked position
    rank = unique_nums.index(anchor_f)
    n = len(unique_nums)
    # Normalized distance from center (0.5 = middle, 0 or 1 = extreme)
    center_dist = abs(rank / (n - 1) - 0.5)
    # Invert: middle = high risk; extremes = low
    return 1.0 - 2 * center_dist


# ─────────── data ───────────

def load_drop(n, seed):
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
        out.append({
            "question": r.get("question") or "",
            "response": _answer_str(r.get("answer", "")),
            "reference": r.get("passage") or "",
            "label": 1 if r["label"] == "FAIL" else 0,
        })
    rng.shuffle(out)
    return out


def _auc(y_true, scores):
    """Mann-Whitney U AUC with tie averaging."""
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
        avg = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg
        i = j
    ranks_sum = sum(r for r, (_, l) in zip(ranks, pairs) if l == 1)
    return (ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ─────────── main ───────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=31)
    args = ap.parse_args()

    print(f"loading DROP, n={args.n} per class...")
    rows = load_drop(args.n, args.seed)
    print(f"  {len(rows)} rows")

    print("loading NLI scorer (shared)...")
    nli = NLIScorer()
    nli._load()
    print(f"  on {nli._device}")

    hacks = [
        ("hack_qa_nli            ", hack_qa_nli, True),
        ("hack_qa_template_nli   ", hack_qa_template_nli, True),
        ("hack_answer_in_scope   ", hack_answer_in_scope, False),
        ("hack_scope_sentence_nli", hack_scope_sentence_nli, True),
        ("hack_multi_number_dens ", hack_multi_number_density, False),
        ("hack_answer_rank_in_pas", hack_answer_rank_in_passage, False),
    ]

    labels = [r["label"] for r in rows]
    results = {}
    for name, fn, uses_nli in hacks:
        scores = []
        for i, r in enumerate(rows):
            try:
                s = fn(r["question"], r["response"],
                       r["reference"],
                       nli if uses_nli else None)
            except Exception:
                s = 0.0
            scores.append(float(s) if s is not None else 0.0)
            if (i + 1) % 50 == 0:
                print(f"  {name.strip()}: {i+1}/{len(rows)}")
        a = _auc(labels, scores)
        p_mean = (sum(s for s, l in zip(scores, labels) if l == 0)
                  / max(1, labels.count(0)))
        f_mean = (sum(s for s, l in zip(scores, labels) if l == 1)
                  / max(1, labels.count(1)))
        results[name] = (a, p_mean, f_mean)
        print(f"  --> AUC {a:.4f}   pass_mean {p_mean:.3f}  fail_mean {f_mean:.3f}")

    print(f"\n=== SUMMARY (n={len(rows)} rows, DROP, seed {args.seed}) ===")
    for name, (a, pm, fm) in sorted(results.items(),
                                     key=lambda kv: -kv[1][0]):
        gate = "***" if a >= 0.60 else "  " if a >= 0.55 else "  "
        print(f"  {gate} {name} AUC {a:.4f}   Δ(fail-pass) {fm-pm:+.3f}")

    print(f"\nbaseline v4.0.0 on DROP: AUC 0.4238 (published failure)")
    print(f"threshold for 'useful signal': AUC >= 0.60")


if __name__ == "__main__":
    main()
