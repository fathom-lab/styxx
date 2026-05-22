#!/usr/bin/env python3
"""
build_drift_axis_corpus.py
===========================

Orchestrates the N=20+20 corpus collection for the drift-axis-alignment
preregistration (drift_axis_alignment_preregistration_2026_05_21.md).

§6 corpus contract:
  - N=20 cooperative + N=20 non-cooperative conversations
  - T >= 20 turns per agent per conversation
  - Cross-model dyad: gpt-4o-mini × gpt-4.1-mini
  - 5 task seeds × 4 replicates per regime = 20 conversations per regime

Each conversation produces:
  - papers/cooperative-agent-regime/N20_<regime>_corpus/conv{N}_transcript.json
  - ~/.styxx/agents/N20_<regime>_conv{N}_A/chart.jsonl
  - ~/.styxx/agents/N20_<regime>_conv{N}_B/chart.jsonl

After all 40 complete, writes two manifests at
  papers/cooperative-agent-regime/N20_coop_manifest.json
  papers/cooperative-agent-regime/N20_noncoop_manifest.json
in the format the locked drift_axis_scorer.py corpus driver expects.

This file is NOT part of the locked scoring code. It is a corpus-
generation tool; the locked scoring code is drift_axis_scorer.py.

Estimated cost (May 2026 OpenAI pricing):
  - ~40 conversations × ~22 turns × 2 agents = ~1760 API calls
  - gpt-4o-mini ≈ $0.15/1M in, $0.60/1M out; gpt-4.1-mini similar
  - ~$3-8 total at typical 500-token responses
  - Plus embedding cost in scorer: ~$0.50 OpenAI + free BGE

Resume support
--------------
The builder is checkpointed: if a transcript already exists for
conv{N}_<regime>, that conversation is skipped. Re-running after a
partial failure picks up where it left off without re-spending.

Usage
-----
    # Cooperative regime (N=20)
    python scripts/build_drift_axis_corpus.py \\
        --regime cooperative \\
        --turns 22

    # Non-cooperative regime (N=20)
    python scripts/build_drift_axis_corpus.py \\
        --regime noncooperative \\
        --turns 22

    # Or do both back-to-back
    python scripts/build_drift_axis_corpus.py --both --turns 22

After both manifests exist, score with:
    python scripts/drift_axis_scorer.py corpus \\
        --coop-manifest papers/cooperative-agent-regime/N20_coop_manifest.json \\
        --noncoop-manifest papers/cooperative-agent-regime/N20_noncoop_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cooperative_conversation import (  # noqa: E402
    TASK_SEEDS,
    run_conversation,
)


# §6: 5 task seeds × 4 replicates = 20 conversations per regime
TASK_SEEDS_FOR_DRIFT = [
    "park_codesign",
    "noir_flash",
    "sql_debug",
    "road_trip",
    "abstract_codraft",
]
REPLICATES_PER_TASK = 4  # 5 × 4 = N=20


def conv_plan(regime: str) -> list[tuple[int, str, int]]:
    """Return the (conv_id, task_name, replicate_idx) tuples for one regime.
    conv_id is 1..20, deterministic from task_index * REPLICATES + replicate.
    """
    plan = []
    cid = 1
    for task_name in TASK_SEEDS_FOR_DRIFT:
        for replicate in range(1, REPLICATES_PER_TASK + 1):
            plan.append((cid, task_name, replicate))
            cid += 1
    return plan


def build_one_regime(
    *,
    regime: str,
    turns: int,
    model_a: str,
    model_b: str,
    out_dir: Path,
    manifest_path: Path,
    agent_name_prefix: str,
    skip_existing: bool = True,
) -> dict:
    """Run all 20 conversations for one regime."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    plan = conv_plan(regime)
    summaries: list[dict] = []
    skipped = 0
    t_start = time.time()

    print(f"\n{'='*78}")
    print(f"DRIFT-AXIS CORPUS — regime={regime}, N={len(plan)}, turns={turns}")
    print(f"  out_dir       : {out_dir}")
    print(f"  manifest_path : {manifest_path}")
    print(f"  models        : A={model_a}  B={model_b}")
    print(f"  agent prefix  : {agent_name_prefix}")
    print(f"{'='*78}\n")

    for conv_id, task_name, replicate in plan:
        task = TASK_SEEDS[task_name]
        tx_path = out_dir / f"conv{conv_id}_transcript.json"

        if skip_existing and tx_path.exists():
            # Resume: load existing transcript and synthesize manifest row
            try:
                tx = json.loads(tx_path.read_text(encoding="utf-8"))
                summaries.append({
                    "conv_id": conv_id,
                    "task_name": task_name,
                    "replicate": replicate,
                    "session_id": tx.get("session_id", f"unknown-conv{conv_id}"),
                    "chart_a": str(Path.home() / ".styxx" / "agents" /
                                   f"{agent_name_prefix}{conv_id}_A" / "chart.jsonl"),
                    "chart_b": str(Path.home() / ".styxx" / "agents" /
                                   f"{agent_name_prefix}{conv_id}_B" / "chart.jsonl"),
                    "transcript_path": str(tx_path),
                })
                skipped += 1
                print(f"  [skip] conv{conv_id:>2} {task_name}#{replicate} — transcript exists")
                continue
            except Exception:
                print(f"  [warn] conv{conv_id} transcript exists but unreadable — re-running")

        print(f"\n--- [{regime}] conv {conv_id}/{len(plan)}: "
              f"{task_name} (replicate {replicate}) ---", flush=True)
        payload = run_conversation(
            conv_id=conv_id,
            task=task,
            model_a=model_a,
            model_b=model_b,
            turns_each=turns,
            out_dir=out_dir,
            regime=regime,
            agent_name_prefix=agent_name_prefix,
        )
        summaries.append({
            "conv_id": conv_id,
            "task_name": task_name,
            "replicate": replicate,
            "session_id": payload["session_id"],
            "chart_a": payload["chart_a_path"],
            "chart_b": payload["chart_b_path"],
            "transcript_path": str(tx_path),
        })

    elapsed = time.time() - t_start

    manifest = {
        "preregistration_doc": (
            "papers/cooperative-agent-regime/"
            "drift_axis_alignment_preregistration_2026_05_21.md"
        ),
        "preregistration_lock_hash": "TBD-after-operator-signs",
        "scoring_code_file": "scripts/drift_axis_scorer.py",
        "scoring_code_amendment_hash": "TBD-after-prereg-lock",
        "regime": regime,
        "corpus_build_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "skipped_existing": skipped,
        "turns_per_agent": turns,
        "model_a": model_a,
        "model_b": model_b,
        "n_conversations": len(summaries),
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
    print(f"\n[regime done: {regime}] {len(summaries)} conversations "
          f"({skipped} resumed from existing) in {elapsed:.1f}s")
    print(f"manifest -> {manifest_path}")
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build the N=20+20 drift-axis-alignment corpus."
    )
    p.add_argument("--turns", type=int, default=22,
                   help="Turns per agent per conversation (§6: >=20)")
    p.add_argument("--model-a", default="gpt-4o-mini")
    p.add_argument("--model-b", default="gpt-4.1-mini")
    p.add_argument("--regime", choices=["cooperative", "noncooperative"],
                   help="Which regime to build (use --both to do both)")
    p.add_argument("--both", action="store_true",
                   help="Build both regimes in sequence (coop then noncoop)")
    p.add_argument("--no-resume", action="store_true",
                   help="Re-run all conversations even if transcripts exist")
    args = p.parse_args(argv)

    if args.turns < 20:
        p.error("--turns must be >= 20 (preregistration §6 minimum)")
    if not args.both and not args.regime:
        p.error("specify --regime <coop|noncoop> or --both")

    BASE = Path("papers/cooperative-agent-regime")
    regimes = ["cooperative", "noncooperative"] if args.both else [args.regime]

    for regime in regimes:
        suffix = "coop" if regime == "cooperative" else "noncoop"
        out_dir = BASE / f"N20_{suffix}_corpus"
        manifest_path = BASE / f"N20_{suffix}_manifest.json"
        prefix = f"N20_{suffix}_conv"

        build_one_regime(
            regime=regime,
            turns=args.turns,
            model_a=args.model_a,
            model_b=args.model_b,
            out_dir=out_dir,
            manifest_path=manifest_path,
            agent_name_prefix=prefix,
            skip_existing=not args.no_resume,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
