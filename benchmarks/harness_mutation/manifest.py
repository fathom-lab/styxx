"""The harness manifest: what the checking apparatus is *declared* to consist of.

    python -m benchmarks.harness_mutation.manifest            # print the manifest of this tree
    python -m benchmarks.harness_mutation.manifest --check    # compare with tests/harness_manifest.json, exit 1 on drift
    python -m benchmarks.harness_mutation.manifest --write    # regenerate tests/harness_manifest.json

MUTE-1 (papers/harness/RESULT_mute1_harness_mutation_2026_09_20.md) cut each of 120 declared
checks out of this repository's CI and found that 101 of the cuts were invisible to the test
suite: a job could be deleted, a workflow switched to manual-only, a step's guard flipped to
false, and every pull request would go on showing green. The manifest is the cheapest possible
answer to that, and it is important to say exactly what kind of answer it is.

It pins STRUCTURE, not content: for every workflow, the events it fires on; for every job, its
name and its `if:`; for every `run:` step, its name and its `if:`; and the names of the npm
scripts. `tests/test_harness_manifest.py` fails when the tree's structure differs from the
committed manifest. So deleting a job, deleting a step, flipping a guard, removing a trigger or
dropping a script is LOUD: it cannot be done without also editing the manifest, which is a
deliberate act visible in a diff -- the same guarantee `papers/build_index.py` gives for arcs.

What it does NOT do: it does not know whether a step does anything. A step that still exists,
still has its name and still has its guard can be wrapped `( ... ) || true` and the manifest will
not move, because the text of a step is not in it. That case is the job of
`tests/test_ci_steps_propagate_failure.py`, which runs the step and checks that it can fail. The
two guards are deliberately separate: one says *the check is still declared*, the other says
*the check can still go red*. Neither says the check is correct.

Loudness is not truth. A manifest that pinned the sha256 of every step's text would kill every
mutant MUTE-1 can make and would say nothing at all, since any edit to CI would break it. This
one pins only what a deletion changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PACKAGE_JSON = ROOT / "packages" / "styxx-js" / "package.json"
MANIFEST = ROOT / "tests" / "harness_manifest.json"
REGENERATE = "python -m benchmarks.harness_mutation.manifest --write"


def _events(on) -> list[str]:
    if isinstance(on, dict):
        return sorted(str(k) for k in on)
    if isinstance(on, list):
        return sorted(str(k) for k in on)
    return [str(on)]


def build(root: Path = ROOT) -> dict:
    """The structural manifest of the tree at `root`. Deterministic; sorted; no content hashes."""
    import yaml
    out: dict = {"what": "structure of the checking harness: triggers, jobs, run steps, guards, npm scripts",
                 "regenerate": REGENERATE, "workflows": {}, "npm_scripts": {}}
    for wf in sorted((root / ".github" / "workflows").glob("*.yml")):
        doc = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
        on = doc.get(True, doc.get("on"))    # PyYAML reads the `on:` key as the boolean True
        jobs = {}
        for jid, job in (doc.get("jobs") or {}).items():
            steps = []
            for st in job.get("steps") or []:
                if "run" not in st:
                    continue
                steps.append({"name": st.get("name"), "if": st.get("if")})
            jobs[jid] = {"name": job.get("name"), "if": job.get("if"), "run_steps": steps}
        out["workflows"][wf.name] = {"name": doc.get("name"), "on": _events(on), "jobs": jobs}
    pkg = root / "packages" / "styxx-js" / "package.json"
    if pkg.exists():
        scripts = json.loads(pkg.read_text(encoding="utf-8")).get("scripts", {})
        out["npm_scripts"][pkg.relative_to(root).as_posix()] = sorted(scripts)
    return out


def _flatten(d, prefix=""):
    """Leaf paths of a nested dict/list, for a readable diff."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _flatten(v, f"{prefix}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, d


def drift(current: dict, committed: dict) -> list[str]:
    a = dict(_flatten(current))
    b = dict(_flatten(committed))
    lines = []
    for k in sorted(set(a) | set(b)):
        if k not in b:
            lines.append(f"+ {k} = {a[k]!r}   (in the tree, not in the manifest)")
        elif k not in a:
            lines.append(f"- {k} = {b[k]!r}   (in the manifest, not in the tree)")
        elif a[k] != b[k]:
            lines.append(f"~ {k}: manifest {b[k]!r} -> tree {a[k]!r}")
    return lines


def dumps(m: dict) -> str:
    return json.dumps(m, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    m = build(root)
    if a.write:
        (root / "tests" / "harness_manifest.json").write_text(dumps(m), encoding="utf-8")
        print(f"wrote tests/harness_manifest.json: {len(m['workflows'])} workflows, "
              f"{sum(len(w['jobs']) for w in m['workflows'].values())} jobs, "
              f"{sum(len(j['run_steps']) for w in m['workflows'].values() for j in w['jobs'].values())} run steps")
        return 0
    if a.check:
        path = root / "tests" / "harness_manifest.json"
        if not path.exists():
            print(f"no manifest at {path}; run: {REGENERATE}")
            return 1
        lines = drift(m, json.loads(path.read_text(encoding="utf-8")))
        if lines:
            print("the checking harness differs from tests/harness_manifest.json:")
            print(*lines, sep="\n")
            print(f"\nif the change is intended, run: {REGENERATE}")
            return 1
        print("harness matches the manifest")
        return 0
    sys.stdout.write(dumps(m))
    return 0


if __name__ == "__main__":
    sys.exit(main())
