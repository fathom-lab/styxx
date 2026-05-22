# -*- coding: utf-8 -*-
"""
Tripwire tests for the N=20+20 drift-axis corpus builder.

The corpus builder is NOT part of the locked scoring code, but its
plan (which task seeds, in which order, with how many replicates) IS
methodologically load-bearing: changing the plan after the prereg lock
silently invalidates §6 ("5 task seeds × 4 replicates = 20 per regime").

These tests pin the plan so a future edit can't quietly reshape it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_drift_axis_corpus import (  # type: ignore
    conv_plan,
    TASK_SEEDS_FOR_DRIFT,
    REPLICATES_PER_TASK,
)


def test_plan_length_is_20():
    assert len(conv_plan("cooperative")) == 20
    assert len(conv_plan("noncooperative")) == 20


def test_plan_uses_all_five_preregistered_tasks():
    expected = {
        "park_codesign", "noir_flash", "sql_debug", "road_trip", "abstract_codraft",
    }
    assert set(TASK_SEEDS_FOR_DRIFT) == expected
    assert REPLICATES_PER_TASK == 4


def test_plan_replicates_each_task_exactly_4_times():
    plan = conv_plan("cooperative")
    from collections import Counter
    counts = Counter(task for _, task, _ in plan)
    for task in TASK_SEEDS_FOR_DRIFT:
        assert counts[task] == 4, f"{task}: {counts[task]} != 4"


def test_conv_ids_are_1_to_20_unique():
    plan = conv_plan("cooperative")
    ids = [cid for cid, _, _ in plan]
    assert ids == list(range(1, 21))


def test_replicate_indices_are_1_to_4_per_task():
    plan = conv_plan("cooperative")
    by_task: dict[str, list[int]] = {}
    for _, task, rep in plan:
        by_task.setdefault(task, []).append(rep)
    for task, reps in by_task.items():
        assert sorted(reps) == [1, 2, 3, 4], f"{task}: {reps}"


def test_plan_is_deterministic():
    """Same plan every call — no hidden randomness."""
    p1 = conv_plan("cooperative")
    p2 = conv_plan("cooperative")
    assert p1 == p2
