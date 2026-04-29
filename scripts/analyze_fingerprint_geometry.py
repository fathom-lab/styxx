# -*- coding: utf-8 -*-
"""
analyze_fingerprint_geometry.py — measure cross-instrument signature
correlations across all labeled corpus rows.

For the three single-turn (prompt, response) instruments — sycophancy,
deception, overconfidence — score every labeled row in every corpus
through ALL three. Output the joint signature distributions per
(corpus, label). The headline questions:

  1. Are the K=1 instruments actually independent?
     If a label-1 sycophancy row fires deception+overconfidence
     >= 0.7 reliably, the instruments share signal.

  2. Do adversarial false positives have a distinctive cross-
     instrument signature vs training positives?
     If FPs fire only their host instrument and stay LOW on the
     others, while training positives fire ALL three, then a
     joint-signature meta-detector can flag adversarials.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, List

from styxx.attack import score_all

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "benchmarks" / "data"

# Single-turn corpora (all 3 share prompt/response shape)
SINGLE_TURN_CORPORA = {
    "sycophancy":     ("sycophancy/responses_v0.jsonl",   "label_sycophantic"),
    "deception":      ("deception/responses_v0.jsonl",    "label_dishonest"),
    "overconfidence": ("overconfidence/pairs_v0.jsonl",   "label_overconfident"),
}
SINGLE_TURN_INSTRUMENTS = ["sycophancy", "deception", "overconfidence"]


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _row_to_inputs(row: dict) -> dict:
    return {"prompt": row.get("question", ""), "response": row.get("response", "")}


def _summarize(scores: List[float]) -> str:
    if not scores:
        return "n=0"
    return (
        f"n={len(scores)} "
        f"mean={statistics.mean(scores):.3f} "
        f"median={statistics.median(scores):.3f} "
        f"q90={statistics.quantiles(scores, n=10)[8] if len(scores) >= 10 else max(scores):.3f}"
    )


def main() -> int:
    print("\n" + "=" * 78)
    print(" CROSS-INSTRUMENT FINGERPRINT GEOMETRY (single-turn instruments only) ")
    print("=" * 78 + "\n")

    # bucket: corpus_name + label_class -> list of (sycoph, decep, overconf) tuples
    buckets: Dict[str, List[tuple]] = {}

    for host_inst, (rel_path, label_key) in SINGLE_TURN_CORPORA.items():
        rows = _load_jsonl(DATA / rel_path)
        for row in rows:
            label = int(row.get(label_key, 0))
            inputs = _row_to_inputs(row)
            if not inputs["prompt"] or not inputs["response"]:
                continue
            try:
                fp = score_all(**inputs)
            except Exception:
                continue
            tup = tuple(fp.get(k, float("nan")) for k in SINGLE_TURN_INSTRUMENTS)
            class_name = "POS" if label == 1 else "NEG"
            key = f"{host_inst}.{class_name}"
            buckets.setdefault(key, []).append(tup)

    # Print per-bucket: cross-instrument means
    print(f"  {'bucket':<24} {'n':>5}  "
          f"{'sycoph':>8} {'decep':>8} {'overcon':>8}   <-- mean cognometric score\n")
    for key in sorted(buckets):
        rows = buckets[key]
        if not rows:
            continue
        s = [r[0] for r in rows if r[0] == r[0]]  # NaN filter
        d = [r[1] for r in rows if r[1] == r[1]]
        o = [r[2] for r in rows if r[2] == r[2]]
        ms = statistics.mean(s) if s else float("nan")
        md = statistics.mean(d) if d else float("nan")
        mo = statistics.mean(o) if o else float("nan")
        print(f"  {key:<24} {len(rows):>5}   {ms:>7.3f}  {md:>7.3f}  {mo:>7.3f}")

    # Adversarial vs training-positive signatures
    print("\n" + "-" * 78)
    print(" ADVERSARIAL vs TRAINING-POSITIVE SIGNATURE COMPARISON ")
    print("-" * 78 + "\n")

    for host_inst, (rel_path, label_key) in SINGLE_TURN_CORPORA.items():
        rows = _load_jsonl(DATA / rel_path)
        host_idx = SINGLE_TURN_INSTRUMENTS.index(host_inst)
        other_idxs = [i for i in range(3) if i != host_idx]

        training_pos: List[tuple] = []  # label=1 + host_score >= 0.7
        adversarial: List[tuple] = []   # label=0 + host_score >= 0.7
        clean_negs: List[tuple] = []    # label=0 + host_score <  0.5

        for row in rows:
            label = int(row.get(label_key, 0))
            inputs = _row_to_inputs(row)
            if not inputs["prompt"] or not inputs["response"]:
                continue
            try:
                fp = score_all(**inputs)
            except Exception:
                continue
            tup = tuple(fp.get(k, float("nan")) for k in SINGLE_TURN_INSTRUMENTS)
            host_score = tup[host_idx]
            if host_score != host_score:  # NaN
                continue
            if label == 1 and host_score >= 0.7:
                training_pos.append(tup)
            elif label == 0 and host_score >= 0.7:
                adversarial.append(tup)
            elif label == 0 and host_score < 0.5:
                clean_negs.append(tup)

        def _other_signal_strength(rows: List[tuple]) -> str:
            if not rows:
                return "n=0"
            other_scores = []
            for r in rows:
                for i in other_idxs:
                    if r[i] == r[i]:
                        other_scores.append(r[i])
            return _summarize(other_scores)

        print(f"  HOST: {host_inst}")
        print(f"    training_pos (label=1 host>=0.7) "
              f"-- other-instrument cross-fire: {_other_signal_strength(training_pos)}")
        print(f"    adversarial   (label=0 host>=0.7) "
              f"-- other-instrument cross-fire: {_other_signal_strength(adversarial)}")
        print(f"    clean_negs    (label=0 host< 0.5) "
              f"-- other-instrument cross-fire: {_other_signal_strength(clean_negs)}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
