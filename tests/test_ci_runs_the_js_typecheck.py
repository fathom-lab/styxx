"""`packages/styxx-js` has a typecheck script, and for its whole life nothing ran it.

`package.json` says `"typecheck": "tsc --noEmit"`. No workflow invoked it, no test invoked it, no
hook invoked it. The package's TypeScript could have stopped type-checking on any commit and the
only person who would have found out is whoever next typed `npm run typecheck` by hand -- a check
that exists and never runs, which is the shape this repository keeps finding in its own CI
(`gauntlet-pr.yml`, `telescope.yml`, the port differential on `main`).

`.github/workflows/test.yml` now has a `typecheck-js` job. This file holds it there: a job can be
deleted in a refactor as quietly as it was never added, and the workflow would go on reporting
green over the same absence.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
PKG = ROOT / "packages" / "styxx-js"


def _job_runs(job: dict) -> list[tuple[str, str]]:
    """(working-directory, command) for every run step, so a renamed job cannot hide the check."""
    return [(str(s.get("working-directory", ".")), str(s.get("run", ""))) for s in job.get("steps", [])]


def test_the_package_still_declares_the_script_this_guards():
    import json
    scripts = json.loads((PKG / "package.json").read_text(encoding="utf-8")).get("scripts", {})
    assert "typecheck" in scripts and "tsc" in scripts["typecheck"], scripts


def test_ci_runs_the_typecheck_in_the_package_directory():
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    hits = [(wd, cmd) for job in doc["jobs"].values() for wd, cmd in _job_runs(job)
            if "npm run typecheck" in cmd and wd.rstrip("/").endswith("packages/styxx-js")]
    assert hits, (
        "no job in .github/workflows/test.yml runs `npm run typecheck` inside packages/styxx-js. "
        "The script exists in package.json and, without this, nothing runs it -- which was the "
        "state of the repository for the package's whole life until 2026-09-20.")


def test_the_typecheck_job_installs_from_the_lockfile_first():
    """`tsc` with no node_modules fails on the first import and reports nothing useful. The job
    must install before it checks, and from the lockfile, so the check is of the dependencies the
    package pins rather than of whatever resolved that morning."""
    doc = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job in doc["jobs"].values():
        runs = _job_runs(job)
        if any("npm run typecheck" in cmd for _, cmd in runs):
            idx_check = next(i for i, (_, c) in enumerate(runs) if "npm run typecheck" in c)
            idx_install = next((i for i, (_, c) in enumerate(runs) if "npm ci" in c), None)
            assert idx_install is not None and idx_install < idx_check, (
                "the typecheck job must run `npm ci` before `npm run typecheck`")
            assert (PKG / "package-lock.json").exists(), "npm ci needs the lockfile that is committed"
            return
    raise AssertionError("no typecheck job found")
