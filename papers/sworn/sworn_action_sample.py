# -*- coding: utf-8 -*-
"""A sample run of the sworn action over a fixture repository, reproducible: the clock is pinned,
the git dates are pinned, every path the outputs carry is relative, and the command is a script
committed in the fixture's base commit. The population is this script.

Writes, beside itself, ``sworn_action_sample.<name>`` for the composed manifest, the two adapter
manifests, ``run.json``, ``summary.md`` and one verdict receipt per document. It refuses to
overwrite an existing sample: a sample cited by a sworn document is history, and a new sample is
a new prefix at a new commit. ``--check`` regenerates into a temporary directory and compares —
manifests, run.json and the summary byte for byte, receipts on their core (minus the verifier
block, the digest, the timestamp and coverage, which move with the verifier's build) — and exits
one on any difference without writing anything.

  python papers/sworn/sworn_action_sample.py            # write the sample (refuses if present)
  python papers/sworn/sworn_action_sample.py --check    # regenerate in memory; exit 1 on drift
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import styxx.sworn as sworn                     # noqa: E402

PREFIX = "sworn_action_sample"
CLOCK = "2026-09-05T00:00:00Z"
GIT_DATE = "2026-09-05T00:00:00 +0000"
RECEIPT_OUTSIDE_CORE = ("verifier", "digest", "timestamp", "coverage", "coverage_sha256")

JUNIT = (b'<?xml version="1.0" encoding="utf-8"?>\n'
         b'<testsuites><testsuite name="pytest" errors="0" failures="0" skipped="0" tests="3" time="0.012">'
         b'<testcase classname="tests.test_app" name="test_one" time="0.004" />'
         b'<testcase classname="tests.test_app" name="test_two" time="0.004" />'
         b'<testcase classname="tests.test_app" name="test_three" time="0.004" />'
         b'</testsuite></testsuites>\n')

WRITER = (b"import os, shutil, sys\n"
          b"shutil.copy(sys.argv[1], os.environ['SWORN_JUNIT'])\n"
          b"print('3 passed in 0.01s')\n")

BODY = ("Adds the notes.\r\n\r\n"
        "<sworn r=\"r2\" k=\"numeric\">The run resolved 0 failures.</sworn>\r\n")


def _docs(base_sha: str) -> Dict[str, bytes]:
    return {
        "HELD.md": ("# notes that hold\n\n"
                    '<sworn r="r1" k="numeric">The runner resolved 3 passed testcases.</sworn>\n'
                    '<sworn r="r2" k="numeric">It resolved 0 failures.</sworn>\n'
                    '<sworn r="r4#/outcome" k="quote">The reader\'s outcome over the report reads `PASSED`.</sworn>\n'
                    '<sworn r="r6" k="quote">The base sha is `%s`.</sworn>\n'
                    '<sworn r="r9#/pull_request/number" k="numeric">This is pull request 7.</sworn>\n'
                    % base_sha).encode(),
        "FAILED.md": (b"# notes that fail\n\n"
                      b'<sworn r="r1" k="numeric">The runner resolved 4 passed testcases.</sworn>\n'),
        "UNRESOLVED.md": (b"# notes the runner could not see\n\n"
                          b'<sworn r="r12" k="numeric">The manifest carries 12 receipts.</sworn>\n'),
        "plain.md": b"# plain\n\nNo sworn tag here.\n",
        "stale.sworn.json": (b'{"spec": "sworn/0.1", "commit": null, "document": {"name": "x", '
                             b'"sha256": "0"}, "text": "", "spans": []}\n'),
    }


def _git(repo: Path, *args: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="sample", GIT_AUTHOR_EMAIL="sample@example.invalid",
               GIT_COMMITTER_NAME="sample", GIT_COMMITTER_EMAIL="sample@example.invalid",
               GIT_AUTHOR_DATE=GIT_DATE, GIT_COMMITTER_DATE=GIT_DATE)
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, check=True).stdout.strip()


def _load_action():
    spec = importlib.util.spec_from_file_location("sworn_action_sample_target", ROOT / "sworn" / "sworn_action.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate(work: Path) -> Path:
    """Build the fixture under ``work``, run the action, return its output directory."""
    ws = work / "ws"
    ws.mkdir()
    _git(ws, "init", "-q")
    _git(ws, "config", "core.autocrlf", "false")
    (ws / "tools").mkdir()
    (ws / "tools" / "write_junit.py").write_bytes(WRITER)
    (ws / "tools" / "ok.xml").write_bytes(JUNIT)
    (ws / "README.md").write_bytes(b"# app\n")
    _git(ws, "add", ".")
    _git(ws, "commit", "-q", "-m", "base")
    base = _git(ws, "rev-parse", "HEAD")
    for name, data in _docs(base).items():
        (ws / name).write_bytes(data)
    _git(ws, "add", ".")
    _git(ws, "commit", "-q", "-m", "the turn")
    head = _git(ws, "rev-parse", "HEAD")
    event = {"action": "synchronize", "number": 7,
             "pull_request": {"number": 7, "title": "notes", "body": BODY,
                              "base": {"ref": "main", "sha": base, "repo": {"full_name": "lab/app", "fork": False}},
                              "head": {"ref": "notes", "sha": head, "repo": {"full_name": "lab/app", "fork": False}}},
             "repository": {"full_name": "lab/app"}}
    event_path = work / "event.json"
    with open(event_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(event, indent=1) + "\n")
    out = ws / ".sworn-action"
    env = {k: v for k, v in os.environ.items() if not (k.startswith("SWORN_") or k.startswith("GITHUB_"))}
    env.update({"GITHUB_EVENT_PATH": str(event_path), "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_WORKSPACE": str(ws), "SWORN_COMMAND": "python tools/write_junit.py tools/ok.xml",
                "SWORN_OUT_DIR": str(out), "SWORN_JUNIT": ".sworn-action/junit.xml"})
    saved_env, saved_now = dict(os.environ), sworn._now
    os.environ.clear()
    os.environ.update(env)
    sworn._now = lambda: CLOCK
    try:
        mod = _load_action()
        code = mod.main()
    finally:
        sworn._now = saved_now
        os.environ.clear()
        os.environ.update(saved_env)
    if code != 0:
        raise SystemExit("the action exited %d" % code)
    return out


def _outputs(out: Path) -> Dict[str, Path]:
    files = {"manifest.json": out / "sworn.manifest.json",
             "junit.manifest.json": out / "junit.manifest.json",
             "github.manifest.json": out / "github.manifest.json",
             "run.json": out / "run.json", "summary.md": out / "summary.md"}
    for p in sorted((out / "receipts").glob("*.sworn-receipt.json")):
        files[p.name.replace(".md.sworn-receipt.json", ".sworn-receipt.json")] = p
    return files


def _receipt_core(raw: bytes) -> dict:
    rec = json.loads(raw.decode("utf-8"))
    return {k: v for k, v in rec.items() if k not in RECEIPT_OUTSIDE_CORE}


def _run_json_stable(raw: bytes) -> dict:
    run = json.loads(raw.decode("utf-8"))
    for d in run.get("documents", []):
        d.pop("receipt_digest", None)             # covers the verifier block; moves with the build
    return run


def check() -> int:
    work = Path(tempfile.mkdtemp(prefix="sworn_sample_check_"))
    try:
        out = generate(work)
        drift: List[str] = []
        for name, fresh in _outputs(out).items():
            committed = HERE / ("%s.%s" % (PREFIX, name))
            if not committed.exists():
                drift.append("%s: not committed" % name)
                continue
            a, b = committed.read_bytes(), fresh.read_bytes()
            if name.endswith(".sworn-receipt.json"):
                same = _receipt_core(a) == _receipt_core(b)
            elif name == "run.json":
                same = _run_json_stable(a) == _run_json_stable(b)
            else:
                same = a == b
            if not same:
                drift.append("%s: differs from the committed sample" % name)
        produced = _outputs(out)
        for p in HERE.glob(PREFIX + ".*"):
            if p.suffix != ".py" and p.name[len(PREFIX) + 1:] not in produced:
                drift.append("%s: committed but not produced" % p.name)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if drift:
        sys.stderr.write("sworn action sample does not reproduce:\n  " + "\n  ".join(drift) + "\n")
        return 1
    print("sworn action sample reproduces (%d files)" % len(produced))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--check"]:
        return check()
    if argv:
        print(__doc__)
        return 2
    existing = [p for p in HERE.glob(PREFIX + ".*") if p.suffix != ".py"]
    if existing:
        sys.stderr.write("REFUSED: a sample is already committed under %s.* — a sample is history; "
                         "write a new prefix at a new commit\n" % PREFIX)
        return 1
    work = Path(tempfile.mkdtemp(prefix="sworn_sample_"))
    try:
        out = generate(work)
        for name, fresh in _outputs(out).items():
            (HERE / ("%s.%s" % (PREFIX, name))).write_bytes(fresh.read_bytes())
            print("wrote %s.%s" % (PREFIX, name))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
