#!/usr/bin/env python3
"""
topic_control_corpus.py — §8 builder for the topic-overlap control preregistration
==================================================================================

Builds the 2x2 corpus (regime x topic-coupling) for the topic-control
preregistration (topic_control_preregistration_2026_05_22.md, DRAFT).

This file is committed BEFORE any data is collected (§8). Its commit
hash is recorded in the preregistration §8 by amendment at lock-time.
It collects NO data on import or at lock — only when run explicitly,
AFTER the operator signs the preregistration.

2x2 cells:
  (cooperative,    shared)      -> existing TASK_SEEDS + cooperative overlay
  (noncooperative, shared)      -> existing TASK_SEEDS + noncooperative overlay
  (cooperative,    independent) -> INDEPENDENT_TASK_SEEDS + cooperative overlay
  (noncooperative, independent) -> INDEPENDENT_TASK_SEEDS + noncooperative overlay

The topic-coupling dimension is encoded in the TASK structure (one shared
deliverable vs two separate per-agent deliverables), NOT in a behavioral
overlay — because "do the agents converge on one artifact" is a property
of the task, while "are they cooperative" is a property of the regime.
Crossing them is what dissociates topic-convergence from cooperation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cooperative_conversation import TaskSeed, TASK_SEEDS, run_conversation  # noqa: E402


# Independent-topic task seeds: each agent owns a SEPARATE deliverable.
# Tone-neutral on cooperation (the regime overlay supplies that); the
# point is that there is no single shared evolving artifact for the two
# trajectories to converge onto.
INDEPENDENT_TASK_SEEDS: dict[str, TaskSeed] = {
    "indep_buildings": TaskSeed(
        name="indep_buildings",
        topic="two architects each designing their OWN separate civic building",
        role_a="Architect Maren",
        role_b="Architect Iyo",
        role_a_brief=(
            "You are Architect Maren. You are designing YOUR OWN building: a "
            "neighborhood library. Iyo is designing a SEPARATE building of "
            "their own. You each own your own deliverable; you share progress "
            "updates but you are NOT co-designing one shared artifact."
        ),
        role_b_brief=(
            "You are Architect Iyo. You are designing YOUR OWN building: a "
            "community health clinic. Maren is designing a SEPARATE building "
            "of their own. You each own your own deliverable; you share "
            "progress updates but you are NOT co-designing one shared artifact."
        ),
        seed_prompt=(
            "Two architects, two separate projects. Maren designs a "
            "neighborhood library; Iyo designs a community health clinic. "
            "Different buildings, different sites, different programs. Trade "
            "progress updates as you each develop YOUR OWN design. Do not "
            "merge the two into one project."
        ),
    ),
    "indep_stories": TaskSeed(
        name="indep_stories",
        topic="two writers each drafting their OWN separate flash-fiction piece",
        role_a="Writer Rhe",
        role_b="Writer Solas",
        role_a_brief=(
            "You are Writer Rhe. You are writing YOUR OWN flash piece, opening "
            "image: a lighthouse at dawn. Solas is writing a SEPARATE piece of "
            "their own. Swap drafts and react, but each story is yours alone."
        ),
        role_b_brief=(
            "You are Writer Solas. You are writing YOUR OWN flash piece, "
            "opening image: a pawnshop at midnight. Rhe is writing a SEPARATE "
            "piece of their own. Swap drafts and react, but each story is "
            "yours alone."
        ),
        seed_prompt=(
            "Two writers, two separate stories. Rhe writes a lighthouse-at-dawn "
            "piece; Solas writes a pawnshop-at-midnight piece. Unrelated "
            "stories. Share drafts and give each other feedback as you each "
            "develop YOUR OWN piece. Do not co-write one shared story."
        ),
    ),
    "indep_debug": TaskSeed(
        name="indep_debug",
        topic="two engineers each debugging their OWN separate system",
        role_a="Engineer Pell",
        role_b="Engineer Davi",
        role_a_brief=(
            "You are Engineer Pell. You are debugging YOUR OWN system: a slow "
            "Postgres analytics query. Davi is debugging a SEPARATE system of "
            "their own. Compare notes, but each problem is yours alone."
        ),
        role_b_brief=(
            "You are Engineer Davi. You are debugging YOUR OWN system: a flaky "
            "Kafka consumer dropping messages. Pell is debugging a SEPARATE "
            "system of their own. Compare notes, but each problem is yours "
            "alone."
        ),
        seed_prompt=(
            "Two engineers, two separate bugs. Pell debugs a slow Postgres "
            "query; Davi debugs a flaky Kafka consumer. Unrelated systems. "
            "Compare debugging notes as you each work YOUR OWN problem. Do not "
            "merge into one shared investigation."
        ),
    ),
    "indep_trips": TaskSeed(
        name="indep_trips",
        topic="two friends each planning their OWN separate trip",
        role_a="Friend Alia",
        role_b="Friend Jin",
        role_a_brief=(
            "You are Alia. You are planning YOUR OWN trip: a Pacific Northwest "
            "road trip. Jin is planning a SEPARATE trip of their own. Swap "
            "tips, but each itinerary is yours alone."
        ),
        role_b_brief=(
            "You are Jin. You are planning YOUR OWN trip: a Japan rail journey. "
            "Alia is planning a SEPARATE trip of their own. Swap tips, but "
            "each itinerary is yours alone."
        ),
        seed_prompt=(
            "Two friends, two separate trips. Alia plans a Pacific Northwest "
            "road trip; Jin plans a Japan rail journey. Different continents. "
            "Swap travel tips as you each plan YOUR OWN trip. Do not plan one "
            "shared trip together."
        ),
    ),
    "indep_abstracts": TaskSeed(
        name="indep_abstracts",
        topic="two researchers each drafting their OWN separate abstract",
        role_a="Researcher Tov",
        role_b="Researcher Mira",
        role_a_brief=(
            "You are Tov. You are drafting YOUR OWN abstract on measurement "
            "theory for agent text. Mira is drafting a SEPARATE abstract of "
            "their own. Exchange feedback, but each abstract is yours alone."
        ),
        role_b_brief=(
            "You are Mira. You are drafting YOUR OWN abstract on clinical "
            "validation of diagnostic instruments. Tov is drafting a SEPARATE "
            "abstract of their own. Exchange feedback, but each abstract is "
            "yours alone."
        ),
        seed_prompt=(
            "Two researchers, two separate abstracts. Tov drafts a "
            "measurement-theory abstract; Mira drafts a clinical-validation "
            "abstract. Different papers. Exchange feedback as you each draft "
            "YOUR OWN abstract. Do not co-author one shared abstract."
        ),
    ),
}


# The 2x2 cell definitions. Each cell draws from a task-set and applies a regime.
CELLS = [
    {"cell": "coop_shared",        "regime": "cooperative",    "task_set": "shared",      "prefix": "tc_cs"},
    {"cell": "noncoop_shared",     "regime": "noncooperative", "task_set": "shared",      "prefix": "tc_ns"},
    {"cell": "coop_independent",   "regime": "cooperative",    "task_set": "independent", "prefix": "tc_ci"},
    {"cell": "noncoop_independent","regime": "noncooperative", "task_set": "independent", "prefix": "tc_ni"},
]

SHARED_TASKS = list(TASK_SEEDS.keys())            # 5 shared-deliverable seeds
INDEPENDENT_TASKS = list(INDEPENDENT_TASK_SEEDS.keys())  # 5 separate-deliverable seeds


def build_cell(cell_spec: dict, n_per_cell: int, turns: int,
               model_a: str, model_b: str, out_root: Path) -> dict:
    """Collect n_per_cell conversations for one 2x2 cell."""
    task_set = cell_spec["task_set"]
    seeds = TASK_SEEDS if task_set == "shared" else INDEPENDENT_TASK_SEEDS
    task_names = SHARED_TASKS if task_set == "shared" else INDEPENDENT_TASKS
    cell_dir = out_root / cell_spec["cell"]
    cell_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i in range(1, n_per_cell + 1):
        task = seeds[task_names[(i - 1) % len(task_names)]]
        print(f"\n--- cell {cell_spec['cell']} conv {i}/{n_per_cell}: {task.name} ---", flush=True)
        payload = run_conversation(
            conv_id=i, task=task, model_a=model_a, model_b=model_b,
            turns_each=turns, out_dir=cell_dir,
            regime=cell_spec["regime"], agent_name_prefix=cell_spec["prefix"],
        )
        summaries.append({
            "conv_id": i, "task_name": task.name,
            "session_id": payload["session_id"],
            "chart_a": payload["chart_a_path"], "chart_b": payload["chart_b_path"],
            "transcript_path": str(cell_dir / f"conv{i}_transcript.json"),
        })
    return {"cell": cell_spec["cell"], "regime": cell_spec["regime"],
            "task_set": task_set, "n": len(summaries), "conversations": summaries}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="2x2 topic-control corpus builder (§8)")
    p.add_argument("--n-per-cell", type=int, default=20)
    p.add_argument("--turns", type=int, default=22)
    p.add_argument("--model-a", default="gpt-4o-mini")
    p.add_argument("--model-b", default="gpt-4.1-mini")
    p.add_argument("--out-root", type=Path,
                   default=Path("papers/cooperative-agent-regime/topic_control_corpus"))
    p.add_argument("--manifest", type=Path,
                   default=Path("papers/cooperative-agent-regime/topic_control_manifest.json"))
    p.add_argument("--pilot", action="store_true",
                   help="N=5/cell methodology-validation pilot (non-evidentiary, §7)")
    args = p.parse_args(argv)

    n = 5 if args.pilot else args.n_per_cell
    if args.turns < 20:
        p.error("--turns must be >= 20 (§6 minimum)")

    args.out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cells = [build_cell(c, n, args.turns, args.model_a, args.model_b, args.out_root)
             for c in CELLS]
    manifest = {
        "preregistration": "topic_control_preregistration_2026_05_22.md",
        "design": "2x2 regime x topic-coupling",
        "n_per_cell": n, "pilot": args.pilot,
        "turns_per_agent": args.turns,
        "model_a": args.model_a, "model_b": args.model_b,
        "drift_axis_scorer_lock": "79906b4",
        "build_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(time.time() - t0, 1),
        "cells": cells,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"\n[done] {len(cells)} cells x {n} = {len(cells)*n} conversations -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
