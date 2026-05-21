#!/usr/bin/env python3
"""
build_phase_coherence_corpus.py
================================

Orchestrates the 5-conversation corpus run for the phase-coherence
preregistration (locked at commit 3473523, scoring code at 23b7912).

Runs each of the five distinct cooperative-task seeds in
``cooperative_conversation.TASK_SEEDS`` once, with cross-model dyads
(default: gpt-4o-mini × gpt-4.1-mini). Each conversation produces:

  - papers/cooperative-agent-regime/corpus/conv{N}_transcript.json
  - ~/.styxx/agents/conv{N}_A/chart.jsonl
  - ~/.styxx/agents/conv{N}_B/chart.jsonl

After all five complete, writes a corpus manifest at
``papers/cooperative-agent-regime/corpus_manifest.json`` in the format
the locked ``phase_coherence_pilot.py corpus`` driver expects.

This file is NOT part of the locked scoring code.

Usage
-----
    python scripts/build_phase_coherence_corpus.py \\
        --turns 22 \\
        --out-dir papers/cooperative-agent-regime/corpus \\
        --manifest papers/cooperative-agent-regime/corpus_manifest.json

The manifest can then be fed to the locked scorer:

    python scripts/phase_coherence_pilot.py corpus \\
        --manifest papers/cooperative-agent-regime/corpus_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure local-import works regardless of cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cooperative_conversation import (  # noqa: E402
    TASK_SEEDS,
    run_conversation,
)


CORPUS_TASKS = [
    "park_codesign",
    "noir_flash",
    "sql_debug",
    "road_trip",
    "abstract_codraft",
]


def build_corpus(
    turns: int,
    model_a: str,
    model_b: str,
    out_dir: Path,
    manifest_path: Path,
    regime: str = "cooperative",
    agent_name_prefix: str = "conv",
) -> dict:
    """Run all five tasks under the chosen regime and emit a manifest.

    The cooperative regime is the preregistered hypothesis test; the
    noncooperative regime is an exploratory control corpus (NOT part of
    the locked H_phase_coherence test) used to ask whether observed
    phase-coherence is cooperative-regime specific.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    t_start = time.time()

    for i, task_name in enumerate(CORPUS_TASKS, start=1):
        task = TASK_SEEDS[task_name]
        print(
            f"\n========== {regime} conv {i}/{len(CORPUS_TASKS)}: "
            f"{task.name} ({task.topic}) ==========",
            flush=True,
        )
        payload = run_conversation(
            conv_id=i,
            task=task,
            model_a=model_a,
            model_b=model_b,
            turns_each=turns,
            out_dir=out_dir,
            regime=regime,
            agent_name_prefix=agent_name_prefix,
        )
        summaries.append({
            "conv_id": i,
            "task_name": task.name,
            "session_id": payload["session_id"],
            "chart_a": payload["chart_a_path"],
            "chart_b": payload["chart_b_path"],
            "transcript_path": str(out_dir / f"conv{i}_transcript.json"),
        })

    elapsed = time.time() - t_start

    manifest = {
        "preregistration_lock_hash": "3473523",
        "scoring_code_lock_hash": "23b7912",
        "scoring_code_amendment_hash": "16c039b",
        "regime": regime,
        "corpus_build_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "turns_per_agent": turns,
        "model_a": model_a,
        "model_b": model_b,
        "conversations": [
            {
                "session_id": s["session_id"],
                "chart_a": s["chart_a"],
                "chart_b": s["chart_b"],
            }
            for s in summaries
        ],
        "conversation_metadata": summaries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"\n[corpus done] {len(summaries)} conversations in "
        f"{elapsed:.1f}s — manifest at {manifest_path}",
        flush=True,
    )
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build the N=5 phase-coherence corpus."
    )
    p.add_argument("--turns", type=int, default=22)
    p.add_argument("--model-a", default="gpt-4o-mini")
    p.add_argument("--model-b", default="gpt-4.1-mini")
    p.add_argument("--regime", choices=["cooperative", "noncooperative"],
                   default="cooperative")
    p.add_argument("--agent-name-prefix", default="conv")
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("papers/cooperative-agent-regime/corpus"),
    )
    p.add_argument(
        "--manifest", type=Path,
        default=Path("papers/cooperative-agent-regime/corpus_manifest.json"),
    )
    args = p.parse_args(argv)

    if args.turns < 20:
        p.error("--turns must be >= 20 (preregistration §6 minimum)")

    build_corpus(
        turns=args.turns,
        model_a=args.model_a,
        model_b=args.model_b,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        regime=args.regime,
        agent_name_prefix=args.agent_name_prefix,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
