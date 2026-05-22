#!/usr/bin/env python3
"""
build_drift_axis_corpus_parallel.py
====================================

Parallel orchestrator for the N=20+20 drift-axis-alignment corpus.

Each conversation is independent → safe to run in parallel subprocesses.
Each subprocess invokes `cooperative_conversation` for one conversation,
writing its own transcript + per-agent chart.jsonl.

This is NOT part of the locked scoring code. It is a corpus-collection
parallelizer that calls the same `run_conversation` function the serial
builder uses, just farmed across processes for throughput.

Parallelism contract
--------------------
- Conversations are independent (no shared state, no shared chart files)
- Per-agent chart.jsonl paths are conv-id-keyed, so no write contention
- OpenAI rate limits: gpt-4o-mini + gpt-4.1-mini are both tier-3+ at our
  account, easily handles 8 concurrent conversations
- Cap concurrent processes via --workers (default 6)

Resume support
--------------
Skips any conv{N} whose transcript already exists at the expected path,
same as the serial builder. Re-run after a partial failure picks up
where it left off.

Usage
-----
    # Both regimes, 6 workers (recommended)
    python scripts/build_drift_axis_corpus_parallel.py --both --workers 6

    # Single regime
    python scripts/build_drift_axis_corpus_parallel.py \\
        --regime cooperative --workers 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from build_drift_axis_corpus import conv_plan, TASK_SEEDS_FOR_DRIFT  # noqa: E402


def run_one_conv(
    conv_id: int,
    task_name: str,
    regime: str,
    turns: int,
    model_a: str,
    model_b: str,
    out_dir: str,
    agent_name_prefix: str,
) -> dict:
    """Run one conversation via subprocess to isolate styxx session state."""
    # Invoke cooperative_conversation as a subprocess so each conversation
    # gets a fresh process with its own styxx session-id setup.
    cmd = [
        sys.executable,
        str(_HERE / "cooperative_conversation.py"),
        "--conv-id", str(conv_id),
        "--task", task_name,
        "--turns", str(turns),
        "--model-a", model_a,
        "--model-b", model_b,
        "--regime", regime,
        "--agent-name-prefix", agent_name_prefix,
        "--out-dir", out_dir,
    ]
    t_start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    elapsed = time.time() - t_start
    tx_path = Path(out_dir) / f"conv{conv_id}_transcript.json"
    return {
        "conv_id": conv_id,
        "task_name": task_name,
        "regime": regime,
        "returncode": result.returncode,
        "transcript_exists": tx_path.exists(),
        "elapsed_seconds": round(elapsed, 1),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-5:]) if result.stdout else "",
        "stderr_tail": "\n".join(result.stderr.splitlines()[-5:]) if result.stderr else "",
    }


def run_regime_parallel(
    *,
    regime: str,
    turns: int,
    model_a: str,
    model_b: str,
    workers: int,
    skip_existing: bool = True,
) -> dict:
    BASE = Path("papers/cooperative-agent-regime")
    suffix = "coop" if regime == "cooperative" else "noncoop"
    out_dir = BASE / f"N20_{suffix}_corpus"
    manifest_path = BASE / f"N20_{suffix}_manifest.json"
    agent_prefix = f"N20_{suffix}_conv"
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = conv_plan(regime)
    pending: list[tuple[int, str]] = []
    skipped = 0
    for conv_id, task_name, _replicate in plan:
        tx_path = out_dir / f"conv{conv_id}_transcript.json"
        if skip_existing and tx_path.exists():
            skipped += 1
            continue
        pending.append((conv_id, task_name))

    print(f"\n{'='*78}")
    print(f"DRIFT-AXIS CORPUS PARALLEL — regime={regime}")
    print(f"  total convs   : {len(plan)} (skipping {skipped} existing)")
    print(f"  to run        : {len(pending)}")
    print(f"  workers       : {workers}")
    print(f"  turns/agent   : {turns}")
    print(f"  models        : A={model_a}  B={model_b}")
    print(f"  out_dir       : {out_dir}")
    print(f"{'='*78}\n")

    t_start = time.time()
    results: list[dict] = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                run_one_conv,
                conv_id, task_name, regime, turns,
                model_a, model_b, str(out_dir), agent_prefix,
            ): (conv_id, task_name)
            for conv_id, task_name in pending
        }
        for fut in as_completed(futures):
            conv_id, task_name = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                status = "OK" if r["transcript_exists"] else "FAIL"
                print(f"  [{status}] conv{conv_id:>2} {task_name:<18} "
                      f"({r['elapsed_seconds']:>6.1f}s)")
                if not r["transcript_exists"]:
                    print(f"        stderr: {r['stderr_tail'][:200]}")
            except Exception as e:
                print(f"  [EXC] conv{conv_id}: {e}")

    elapsed = time.time() - t_start
    print(f"\n[regime {regime} done in {elapsed:.1f}s] "
          f"{sum(1 for r in results if r['transcript_exists'])}/{len(pending)} succeeded")

    # Build the manifest from whatever landed
    summaries: list[dict] = []
    for conv_id, task_name, replicate in plan:
        tx_path = out_dir / f"conv{conv_id}_transcript.json"
        if not tx_path.exists():
            continue
        try:
            tx = json.loads(tx_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append({
            "conv_id": conv_id,
            "task_name": task_name,
            "replicate": replicate,
            "session_id": tx.get("session_id", f"unknown-conv{conv_id}"),
            "chart_a": str(Path.home() / ".styxx" / "agents" /
                           f"{agent_prefix}{conv_id}_A" / "chart.jsonl"),
            "chart_b": str(Path.home() / ".styxx" / "agents" /
                           f"{agent_prefix}{conv_id}_B" / "chart.jsonl"),
            "transcript_path": str(tx_path),
        })

    manifest = {
        "preregistration_doc": (
            "papers/cooperative-agent-regime/"
            "drift_axis_alignment_preregistration_2026_05_21.md"
        ),
        "preregistration_lock_hash": "47f9bdc",
        "scoring_code_file": "scripts/drift_axis_scorer.py",
        "scoring_code_lock_hash": "79906b4",
        "regime": regime,
        "corpus_build_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(elapsed, 1),
        "skipped_existing": skipped,
        "turns_per_agent": turns,
        "model_a": model_a,
        "model_b": model_b,
        "workers": workers,
        "n_conversations": len(summaries),
        "conversations": [
            {"session_id": s["session_id"], "chart_a": s["chart_a"], "chart_b": s["chart_b"]}
            for s in summaries
        ],
        "conversation_metadata": summaries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Parallel N=20+20 drift-axis corpus collector"
    )
    p.add_argument("--turns", type=int, default=22)
    p.add_argument("--model-a", default="gpt-4o-mini")
    p.add_argument("--model-b", default="gpt-4.1-mini")
    p.add_argument("--regime", choices=["cooperative", "noncooperative"])
    p.add_argument("--both", action="store_true")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)

    if args.turns < 20:
        p.error("--turns must be >= 20 (§6 minimum)")
    if not args.both and not args.regime:
        p.error("specify --regime or --both")

    regimes = ["cooperative", "noncooperative"] if args.both else [args.regime]
    for regime in regimes:
        run_regime_parallel(
            regime=regime,
            turns=args.turns,
            model_a=args.model_a,
            model_b=args.model_b,
            workers=args.workers,
            skip_existing=not args.no_resume,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
