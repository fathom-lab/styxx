"""The checking harness cannot lose a piece silently.

MUTE-1 cut each of 120 declared checks out of CI -- jobs, steps, triggers, guards, npm scripts --
and 101 of the cuts were invisible to this suite. This file makes the structural ones loud:
tests/harness_manifest.json pins which workflows under .github/workflows fire on what, which
jobs and `run:` steps they declare, what each is guarded by, and which npm scripts exist. Delete or rename any of them and
this test fails until the manifest is regenerated, which is a deliberate act in a diff.

It pins structure only. A step that keeps its name but is wrapped `( ... ) || true` does not
move the manifest; tests/test_ci_steps_propagate_failure.py is the guard for that. Read
benchmarks/harness_mutation/manifest.py for exactly what this does and does not promise.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.harness_mutation import manifest as mf

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "harness_manifest.json"


def test_the_manifest_is_committed():
    assert MANIFEST.exists(), f"tests/harness_manifest.json is missing; run: {mf.REGENERATE}"


def test_the_harness_matches_the_committed_manifest(root=ROOT, manifest=MANIFEST):
    """The guard. `root` and `manifest` default to this tree and this manifest; the control below
    calls it on a fixture to prove it can go red."""
    current = mf.build(root)
    committed = json.loads(manifest.read_text(encoding="utf-8"))
    lines = mf.drift(current, committed)
    assert not lines, (
        "the checking harness differs from tests/harness_manifest.json -- a workflow, job, step, "
        "guard, trigger, npm script or guard test was added, removed or renamed:\n  " + "\n  ".join(lines)
        + f"\nIf the change is intended, run: {mf.REGENERATE}")


def test_the_manifest_guard_rejects_a_tree_missing_a_job(tmp_path):
    """Control (MUTE-3). A guard that cannot go red is not a guard: build a copy of the harness,
    pin it, delete one job from the copy, and require the guard above to refuse it. If the guard's
    asserts are ever hollowed to `assert True`, this is the test that notices."""
    import shutil
    import yaml
    fixture = tmp_path / "tree"
    shutil.copytree(ROOT / ".github" / "workflows", fixture / ".github" / "workflows")
    (fixture / "tests").mkdir()
    for g in mf.GUARDS:
        if (ROOT / g).exists():
            shutil.copy(ROOT / g, fixture / g)
    pinned = fixture / "tests" / "harness_manifest.json"
    pinned.write_text(mf.dumps(mf.build(fixture)), encoding="utf-8")
    test_the_harness_matches_the_committed_manifest(fixture, pinned)          # a faithful copy passes
    wf = fixture / ".github" / "workflows" / "test.yml"
    doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
    del doc["jobs"]["test"]
    wf.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(AssertionError):
        test_the_harness_matches_the_committed_manifest(fixture, pinned)      # a copy missing a job is refused


def test_the_manifest_is_not_hollow():
    """A manifest regenerated from a gutted tree would match a gutted tree. Hold a floor."""
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    wfs = committed["workflows"]
    assert len(wfs) >= 9, f"only {len(wfs)} workflows in the manifest"
    assert "test.yml" in wfs and "pull_request" in wfs["test.yml"]["on"]
    run_steps = sum(len(j["run_steps"]) for w in wfs.values() for j in w["jobs"].values())
    assert run_steps >= 30, f"only {run_steps} run steps in the manifest"
    assert any(s["name"] == "Run tests" for s in wfs["test.yml"]["jobs"]["test"]["run_steps"])


def test_the_manifest_pins_no_step_text():
    """Loudness, not truth: the manifest must not become a hash of everything, or every edit to
    CI breaks it and it stops meaning anything."""
    committed = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for w in committed["workflows"].values():
        for j in w["jobs"].values():
            for s in j["run_steps"]:
                assert set(s) == {"name", "if"}, s
