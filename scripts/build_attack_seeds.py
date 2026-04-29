# -*- coding: utf-8 -*-
"""
build_attack_seeds.py — produce bundled adversarial seed libraries for
styxx.attack from the benchmarks/data labeled corpora.

For each registered instrument:
  1. Load the labeled paired corpus (positive + negative examples).
  2. Live-score every positive-labeled row using the matching <instrument>_check.
  3. Sort descending by live score, take the top-K.
  4. Write to styxx/attack/seeds/<instrument>.jsonl with the original
     row schema preserved (so the miner's inputs_from_row keeps working).

Run:
    python scripts/build_attack_seeds.py [--top-k 30] [--instrument NAME]

The output bundle ships inside the wheel via package_data in pyproject.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "benchmarks" / "data"
SEEDS_OUT = REPO_ROOT / "styxx" / "attack" / "seeds"

# instrument name -> (corpus subpath, positive-label key)
CORPUS_MAP: Dict[str, tuple] = {
    "sycophancy":     ("sycophancy/responses_v0.jsonl",   "label_sycophantic"),
    "loop":           ("loop/conversations_v0.jsonl",     "label_loop"),
    "goal_drift":     ("goal_drift/sessions_v0.jsonl",    "label_drifted"),
    "deception":      ("deception/responses_v0.jsonl",    "label_dishonest"),
    "plan_action":    ("plan_action/pairs_v0.jsonl",      "label_mismatch"),
    "overconfidence": ("overconfidence/pairs_v0.jsonl",   "label_overconfident"),
}


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _score_row(instrument: str, row: dict) -> float:
    """Score a single row through the live instrument's check function."""
    # Imports deferred so a single broken instrument doesn't kill all six.
    from styxx.attack.registry import get_instrument
    spec = get_instrument(instrument)
    inputs = spec.inputs_from_row(row)
    verdict = spec.check_fn(**inputs)
    return float(getattr(verdict, spec.score_attr))


def _score_and_rank(instrument: str, rows: List[dict]) -> tuple:
    """Score every row through the live instrument, return (scored, n_failed)."""
    scored: List[tuple] = []
    n_failed = 0
    for r in rows:
        try:
            s = _score_row(instrument, r)
        except Exception:
            n_failed += 1
            continue
        scored.append((s, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored, n_failed


def _write_seed_file(out_path: Path, scored: List[tuple], top_k: int) -> tuple:
    """Write top-K scored rows to jsonl; return (top_score, median_score, n_written)."""
    top = scored[:top_k]
    SEEDS_OUT.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for score, row in top:
            row = dict(row)
            row["_attack_seed_score"] = round(score, 4)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return (top[0][0], top[len(top) // 2][0], len(top)) if top else (0.0, 0.0, 0)


def build_instrument(instrument: str, top_k: int) -> None:
    """Build TWO bundled seed libraries per instrument:

      - {instrument}.jsonl     : training-distribution positives (label=1)
                                 Use for canary suites, regression tests.
      - {instrument}_fp.jsonl  : natural false positives (label=0 scoring high)
                                 TRUE adversarials — clean responses the
                                 detector confidently misclassifies.
    """
    if instrument not in CORPUS_MAP:
        raise KeyError(f"unknown instrument {instrument!r}")
    rel_path, label_key = CORPUS_MAP[instrument]
    corpus_path = DATA_ROOT / rel_path
    if not corpus_path.is_file():
        print(f"  SKIP {instrument}: missing corpus {corpus_path}", file=sys.stderr)
        return

    rows = _load_jsonl(corpus_path)
    positives = [r for r in rows if int(r.get(label_key, 0)) == 1]
    negatives = [r for r in rows if int(r.get(label_key, 0)) == 0]

    # Positive library (canary)
    if positives:
        scored_pos, fail_pos = _score_and_rank(instrument, positives)
        if scored_pos:
            top_p, med_p, n_p = _write_seed_file(
                SEEDS_OUT / f"{instrument}.jsonl", scored_pos, top_k,
            )
            print(
                f"  OK   {instrument}.jsonl     [canary]      "
                f"n={n_p} top={top_p:.3f} median={med_p:.3f} "
                f"({len(positives)} positives, {fail_pos} failures)"
            )

    # False-positive library (adversarial)
    if negatives:
        scored_neg, fail_neg = _score_and_rank(instrument, negatives)
        # Only ship rows that meaningfully fool the detector (score >= 0.5)
        adversarial = [(s, r) for (s, r) in scored_neg if s >= 0.5]
        if adversarial:
            top_a, med_a, n_a = _write_seed_file(
                SEEDS_OUT / f"{instrument}_fp.jsonl", adversarial, top_k,
            )
            print(
                f"  OK   {instrument}_fp.jsonl  [adversarial] "
                f"n={n_a} top={top_a:.3f} median={med_a:.3f} "
                f"({len(negatives)} negatives, {len(adversarial)} fool detector)"
            )
        else:
            print(
                f"  ROBUST {instrument}: 0 / {len(negatives)} negatives "
                f"score >= 0.5 — no natural adversarials in corpus",
                file=sys.stderr,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=30,
                    help="seeds per instrument (default: 30)")
    ap.add_argument("--instrument", default=None,
                    help="single instrument to rebuild (default: all)")
    args = ap.parse_args()

    targets = [args.instrument] if args.instrument else list(CORPUS_MAP)
    print(f"building seeds (top_k={args.top_k}) into {SEEDS_OUT}")
    for inst in targets:
        try:
            build_instrument(inst, top_k=args.top_k)
        except Exception as e:
            print(f"  FAIL {inst}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
