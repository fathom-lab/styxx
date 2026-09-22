# -*- coding: utf-8 -*-
"""SWALLOW-12 -- the click, live: the Action's one-click suggestions posted by GitHub, committed with
GitHub's own button, and read back byte for byte.

SWALLOW-11 replayed every hidden check the gate has caught and counted a fix as one click away
when the smallest suggestion that reproduces the verified repair sits inside one hunk of the
change's diff -- the documented rule, never put to GitHub's API. This module builds a pull request
made for that: eighteen workflow files. Fifteen carry the suggestion shapes the replay found -- the
empty suggestion that deletes a `continue-on-error` (the step's, the job's, on a file's last line),
the strict shell from one line to three and over a block (on a file's last line, with and without
its final newline), the `|| echo` default removed, the guard, both edits, two suggestions in one
file. Three are controls: a repair whose line the change did not touch, a repair whose lines leave
the change's hunk, a check no repair family reaches. Every file triggers on `workflow_dispatch`
only: the gate reads it, nothing runs it.

  files    the base and head trees for the pull request: the fixtures, and the head's ci-audit.yml
           (the repository's own gate with `suggest: true` and `pull-requests: write`)
  plan     the Action run offline on that change exactly as GitHub will run it -- a depth-1 checkout
           of the test merge, the event, the token -- with the review API answered by the rule
           (201 when a suggestion's lines sit inside one hunk, 422 when not); then re-run on the
           same head; then run on the head with every placed suggestion applied. What it prints
           and posts is the prediction, line for line.
  receipt  what GitHub did, read back: the review comments, the check runs' annotations and
           outcomes, and the files at the commit GitHub's "Commit suggestions" made, against the
           plan. No name, no address: the comments' author is recorded as a type (Bot/User).

    python -m benchmarks.harness_mutation.live_click files --out <dir>
    python -m benchmarks.harness_mutation.live_click plan --out plan.json
    python -m benchmarks.harness_mutation.live_click receipt --plan plan.json --observed observed.json --clone <clone> --out receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from styxx.ciaudit import action as A          # noqa: E402  -- the product, read as it ships
from styxx.ciaudit import differential as D    # noqa: E402

SCHEMA = "styxx.harness-live-click/v1"
WF = ".github/workflows"
ENV = {"GIT_AUTHOR_NAME": "fixture", "GIT_AUTHOR_EMAIL": "fixture@example.invalid", "GIT_COMMITTER_NAME": "fixture",
       "GIT_COMMITTER_EMAIL": "fixture@example.invalid", "GIT_AUTHOR_DATE": "2026-09-22T00:00:00+00:00",
       "GIT_COMMITTER_DATE": "2026-09-22T00:00:00+00:00"}
HDR = "on:\n  workflow_dispatch:\n\npermissions:\n  contents: read\n\njobs:\n"

# name -> (what it is, base text, head text); the order is the files' order in the change
FIXTURES: dict[str, tuple[str, str, str]] = {
    's12-01-coe-step-acquired.yml': (
        "the step's continue-on-error added to an existing check (acquired): an empty suggestion",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: npm ci
      - name: Unit tests
        run: npm test
      - name: Build
        run: npm run build
""",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: npm ci
      - name: Unit tests
        continue-on-error: true
        run: npm test
      - name: Build
        run: npm run build
"""),
    's12-02-coe-step-born.yml': (
        "a new check written with the step's continue-on-error (born hidden), mid-file",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -e .
      - name: Unit tests
        run: python -m pytest tests/unit -q
      - name: Package
        run: python -m build
""",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -e .
      - name: Unit tests
        run: python -m pytest tests/unit -q
      - name: Integration tests
        run: python -m pytest tests/integration -q
        continue-on-error: true
      - name: Package
        run: python -m build
"""),
    's12-03-coe-job-born.yml': (
        "a new job written with the job's continue-on-error",
        HDR + """  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint
        run: npx eslint .
""",
        HDR + """  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Lint
        run: npx eslint .
  e2e:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - name: End-to-end tests
        run: npx playwright test
"""),
    's12-04-coe-step-eof.yml': (
        "the hiding line is the file's last line",
        HDR + """  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Format
        run: cargo fmt --check
""",
        HDR + """  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Format
        run: cargo fmt --check
      - name: Clippy
        run: cargo clippy -- -D warnings
        continue-on-error: true
"""),
    's12-05-strict-one-line.yml': (
        'a new one-line check with an or-true: the strict shell, one line to three',
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: cargo build --locked
      - name: Docs
        run: cargo doc --no-deps
""",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: cargo build --locked
      - name: Tests
        run: cargo test --locked || true
      - name: Docs
        run: cargo doc --no-deps
"""),
    's12-05b-strict-acquired.yml': (
        'an existing check given an or-true (acquired): the strict shell',
        HDR + """  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: npm ci
      - name: Lint
        run: npm run lint
      - name: Format
        run: npx prettier --check .
""",
        HDR + """  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: npm ci
      - name: Lint
        run: npm run lint || true
      - name: Format
        run: npx prettier --check .
"""),
    's12-06-strict-multi.yml': (
        'a block with `set +e` and an or-true: the strict shell over several lines',
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup
        run: go mod download
""",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup
        run: go mod download
      - name: Tests
        run: |
          set +e
          go vet ./...
          go test ./... -race -count=1 || true
          echo "tests finished"
"""),
    's12-06b-strict-ws-line.yml': (
        'a block ending in a whitespace-only line: the strict shell deletes it',
        HDR + """  load:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup k6
        run: sudo apt-get install -y k6
      - name: Report
        run: echo "done"
""",
        HDR + """  load:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup k6
        run: sudo apt-get install -y k6
      - name: Load test
        run: |
          k6 run --out json=k6-results.json load-test.js || true
          
      - name: Report
        run: echo "done"
"""),
    's12-07-strict-eof.yml': (
        'the strict shell on the last line of a file that ends in a newline',
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -e .[test]
""",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -e .[test]
      - name: Tests
        run: python -m pytest -q || true
"""),
    's12-08-strict-eof-no-newline.yml': (
        'the strict shell on the last line of a file with no final newline',
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: composer install""",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: composer install
      - name: PHPUnit
        run: vendor/bin/phpunit || true"""),
    's12-09-no-default.yml': (
        'a check whose `|| echo` default is the green path, one line',
        HDR + """  types:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: npm ci
""",
        HDR + """  types:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: npm ci
      - name: Typecheck
        run: npm run typecheck || echo "type errors, see the log"
      - name: Upload the log
        uses: actions/upload-artifact@v4
        with:
          name: logs
          path: logs/
"""),
    's12-09b-no-default-block.yml': (
        'the same default ending a continued command in a block',
        HDR + """  links:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup
        run: pip install -r docs/requirements.txt
""",
        HDR + """  links:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup
        run: pip install -r docs/requirements.txt
      - name: Check links
        run: |
          python scripts/check_links.py             --root docs             --strict || echo "::warning::some links need attention"
      - name: Upload the log
        uses: actions/upload-artifact@v4
        with:
          name: logs
          path: logs/
"""),
    's12-10-guard.yml': (
        'a guard whose failing tool reads as its green path',
        HDR + """  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Unit tests
        run: npm test
""",
        HDR + """  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Unit tests
        run: npm test
      - name: No focused tests
        run: |
          if grep -rn "it.only(" tests/; then
            echo "::error::a focused test was left in"
            exit 1
          fi
"""),
    's12-11-both.yml': (
        'continue-on-error and an or-true on one step: both edits',
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: yarn install --frozen-lockfile
""",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: yarn install --frozen-lockfile
      - name: Tests
        continue-on-error: true
        run: yarn test --ci || true
      - name: Upload the log
        uses: actions/upload-artifact@v4
        with:
          name: logs
          path: logs/
"""),
    's12-12-outside-diff.yml': (
        "control: the job was already continue-on-error; the repair's line is outside the change's diff",
        HDR + """  nightly:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - name: Toolchain
        run: rustup toolchain install nightly
      - name: Build
        run: cargo +nightly build
      - name: Docs
        run: cargo +nightly doc --no-deps
      - name: Bench build
        run: cargo +nightly bench --no-run
""",
        HDR + """  nightly:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - name: Toolchain
        run: rustup toolchain install nightly
      - name: Build
        run: cargo +nightly build
      - name: Docs
        run: cargo +nightly doc --no-deps
      - name: Bench build
        run: cargo +nightly bench --no-run
      - name: Tests
        run: cargo +nightly test
"""),
    's12-12b-span-leaves-hunk.yml': (
        "control: an or-true appended far below `run: |`; the repair's lines leave the hunk",
        HDR + """  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Integration
        run: |
          echo "integration tests"
          echo "database: $DATABASE_URL"
          echo "seed: $SEED"
          echo "shard: 1 of 1"
          npm run test:integration
      - name: Report
        if: always()
        run: echo "integration finished"
""",
        HDR + """  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Integration
        run: |
          echo "integration tests"
          echo "database: $DATABASE_URL"
          echo "seed: $SEED"
          echo "shard: 1 of 1"
          npm run test:integration || true
      - name: Report
        if: always()
        run: echo "integration finished"
"""),
    's12-13-no-repair.yml': (
        'control: a check hidden by `|| exit 0`, which no repair family reaches',
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -r requirements.txt
""",
        HDR + """  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -r requirements.txt
      - name: Tests
        run: python -m pytest -q || exit 0
"""),
    's12-14-two-in-one.yml': (
        'two checks in one file: two suggestions, committed in one batch',
        HDR + """  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -e .[dev]
      - name: Build wheel
        run: python -m build
""",
        HDR + """  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install
        run: pip install -e .[dev]
      - name: Type check
        run: mypy src || true
      - name: Build wheel
        run: python -m build
      - name: Unit tests
        continue-on-error: true
        run: python -m pytest -q
"""),
}

# The base's ci-audit.yml: the repository's own gate as this preregistration freezes it.
BASE_WORKFLOW = """name: ci-audit

# The pull request's gate (SWALLOW-7), run as the action this repository ships: ci-audit/action.yml.
# When a workflow, the action or the gate's own code changes, the action reads only the workflows
# that changed between GitHub's test merge and its first parent, and fails only when the change
# brings a check that hides its own failure -- a `|| true`, a `continue-on-error`, a guard whose
# failing tool is its green path -- marking the line and putting the verified repair in the job
# summary. A check that was already hidden is reported, not failed: it is not this change's doing.
# The checkout is the default one (depth 1, the test merge): the action fetches the one commit it
# needs, which is the path every repository that adopts it will take.
on:
  pull_request:
    paths:
      - ".github/workflows/**"
      - "ci-audit/**"
      - "styxx/ciaudit/**"

permissions:
  contents: read

jobs:
  differential:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout source
        uses: actions/checkout@v5
      - name: What this change hides that its base did not
        uses: ./ci-audit
"""

# The head's ci-audit.yml: the same gate with the suggestions turned on. It exists only on the live
# pull request's head branch and is never merged.
LIVE_WORKFLOW = """name: ci-audit

# SWALLOW-12's live run: this repository's own gate, with the Action's one-click suggestions on.
# This file exists only on the live pull request's head branch; it is never merged.
on:
  pull_request:
    paths:
      - ".github/workflows/**"
      - "ci-audit/**"
      - "styxx/ciaudit/**"

permissions:
  contents: read
  pull-requests: write

jobs:
  differential:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout source
        uses: actions/checkout@v5
      - name: What this change hides that its base did not
        uses: ./ci-audit
        with:
          suggest: true
"""
REFUSED = "Validation Failed: pull_request_review_thread.line must be part of the diff"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def instrument_sha256() -> str:
    return _sha(Path(__file__).read_bytes())


def side(which: str) -> dict[str, bytes]:
    """The pull request's workflow files on one side, {repository path: bytes}: the fixtures, and the
    repository's gate -- its own on the base, with the suggestions on at the head."""
    out = {f"{WF}/{n}": (b if which == "base" else h).encode("utf-8") for n, (_, b, h) in FIXTURES.items()}
    out[f"{WF}/ci-audit.yml"] = (BASE_WORKFLOW if which == "base" else LIVE_WORKFLOW).encode("utf-8")
    return out


def files(out: Path) -> dict:
    """The two trees to upload: <out>/base/... and <out>/head/..., and their sha256s."""
    man: dict = {}
    for which in ("base", "head"):
        for rel, data in side(which).items():
            p = out / which / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            man.setdefault(which, {})[rel] = _sha(data)
    return man


def splice(text: bytes, suggestions: list[dict]) -> bytes:
    """`text` with each suggestion's lines put in place of the lines it replaces, bottom first, and
    every other byte kept -- the final newline, or its absence, included."""
    lines = text.decode("utf-8").splitlines(keepends=True)
    for s in sorted(suggestions, key=lambda s: -s["start_line"]):
        a, b = s["start_line"] - 1, s["line"]
        new = [ln + "\n" for ln in s["lines"]]
        if new and b == len(lines) and not lines[-1].endswith("\n"):
            new[-1] = new[-1][:-1]
        lines[a:b] = new
    return "".join(lines).encode("utf-8")


# ----------------------------------------------------------------------------- the Action, offline

def _git(tree: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=dict(os.environ, **ENV))
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}: {p.stderr.strip()[-200:]}")
    return p.stdout.strip()


def _write(tree: Path, files_: dict[str, bytes]) -> None:
    for rel, data in files_.items():
        p = tree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


def _repo(work: Path) -> dict:
    """Base and head in a local remote that answers fetch-by-sha as GitHub does, with GitHub's test
    merge at refs/pull/1/merge. Only the workflows the change touches are read by the gate."""
    src, remote = work / "src", work / "remote.git"
    subprocess.run(["git", "init", "-q", "-b", "main", str(src)], check=True, capture_output=True)
    _write(src, side("base"))
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "base")
    s = {"src": src, "remote": remote, "base": _git(src, "rev-parse", "HEAD")}
    _git(src, "checkout", "-q", "-b", "live")
    _write(src, side("head"))
    _git(src, "commit", "-q", "-am", "head")
    s["head"] = _git(src, "rev-parse", "HEAD")
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True, capture_output=True)
    _git(remote, "config", "uploadpack.allowAnySHA1InWant", "true")
    s["merge"] = _merge(s, s["head"])
    return s


def _merge(s: dict, head: str) -> str:
    """GitHub's test merge of `head` into the base, pushed as refs/pull/1/{head,merge}."""
    src = s["src"]
    _git(src, "checkout", "-q", "--detach", s["base"])
    _git(src, "merge", "-q", "--no-ff", "-m", "test merge", head)
    m = _git(src, "rev-parse", "HEAD")
    _git(src, "push", "-q", "-f", str(s["remote"]), f"{s['base']}:refs/heads/main", f"{head}:refs/pull/1/head", f"{m}:refs/pull/1/merge")
    return m


def _checkout(work: Path, remote: Path, sha: str, name: str) -> Path:
    """What actions/checkout does on pull_request: the test merge at depth 1, detached."""
    ws = work / name
    subprocess.run(["git", "init", "-q", str(ws)], check=True, capture_output=True)
    _git(ws, "remote", "add", "origin", f"file://{remote}")
    _git(ws, "fetch", "--quiet", "--no-tags", "--depth=1", "origin", f"+{sha}:refs/remotes/checkout")
    _git(ws, "checkout", "-q", "--detach", sha)
    return ws


class Review:
    """GitHub's review-comment API, answered by SWALLOW-11's placement rule: a suggestion whose
    lines sit inside one hunk of the change's diff is accepted (201), any other is refused (422).
    Comments persist across runs, as on GitHub."""

    def __init__(self) -> None:
        self.bodies: list[str] = []
        self.calls: list[dict] = []
        self.tree = self.base = self.head = None

    def __call__(self, method: str, url: str, token: str, payload: dict | None = None):
        if method == "GET":
            return 200, [{"body": b} for b in self.bodies]
        start = payload.get("start_line", payload["line"])
        ok = A.within((start, payload["line"]), A.hunks(self.tree, self.base, self.head, payload["path"]))
        body = payload["body"]
        m = re.search(r"```+suggestion\n(.*?)\n```+", body, re.S)
        self.calls.append({"path": payload["path"], "start_line": start, "line": payload["line"], "side": payload.get("side"),
                           "status": 201 if ok else 422, "key": (re.search(r"styxx-ci-audit:suggest:([0-9a-f]+)", body) or [None, None])[1],
                           "body_sha256": _sha(body.encode("utf-8")), "lines": (m.group(1).split("\n") if m and m.group(1) else [])})
        if ok:
            self.bodies.append(body)
            return 201, {"id": len(self.bodies)}
        return 422, REFUSED


_ANN = re.compile(r"^::(error|warning|notice) (.*?)::(.*)$")


def _unesc(s: str) -> str:
    return s.replace("%0D", "\r").replace("%0A", "\n").replace("%3A", ":").replace("%2C", ",").replace("%25", "%")


def annotations(stdout: str) -> list[dict]:
    """The annotation commands the Action printed, parsed as GitHub parses them."""
    out = []
    for ln in stdout.splitlines():
        m = _ANN.match(ln)
        if not m:
            continue
        props = dict(p.split("=", 1) for p in m.group(2).split(",") if "=" in p)
        out.append({"level": m.group(1), "path": _unesc(props.get("file", "")), "start_line": int(props["line"]) if "line" in props else None,
                    "end_line": int(props.get("endLine", props.get("line", 0))) or None, "title": _unesc(props.get("title", "")),
                    "message": _unesc(m.group(3))})
    return out


def run_action(work: Path, s: dict, head: str, merge: str, review: Review, name: str) -> dict:
    """The Action's entry point, run on a depth-1 checkout of the test merge with the event GitHub
    sends and the review API answered by `review`. Returns what it printed, wrote and posted."""
    import contextlib
    import io
    ws = _checkout(work, s["remote"], merge, name)
    review.tree, review.base, review.head = ws, s["base"], merge
    ev, summary, output = work / f"{name}.event.json", work / f"{name}.summary.md", work / f"{name}.output.txt"
    ev.write_text(json.dumps({"number": 1, "pull_request": {"number": 1, "head": {"sha": head}, "base": {"sha": s["base"]}}}), encoding="utf-8")
    env = {"GITHUB_WORKSPACE": str(ws), "GITHUB_EVENT_NAME": "pull_request", "GITHUB_EVENT_PATH": str(ev), "GITHUB_SHA": merge,
           "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_OUTPUT": str(output), "RUNNER_TEMP": str(work), "STYXX_RECEIPT": str(work / f"{name}.receipt.json"),
           "GITHUB_REPOSITORY": "fathom-lab/styxx", "GITHUB_API_URL": "https://api.github.invalid", "GH_TOKEN": "offline",
           "STYXX_FAIL_ON": "new-hidden", "STYXX_SUGGEST": "true", "STYXX_ANNOTATE": "true"}
    saved, api = dict(os.environ), A.api
    os.environ.update(env)
    A.api = review
    n0 = len(review.calls)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = A.main([])
    finally:
        A.api = api
        os.environ.clear()
        os.environ.update(saved)
    rec = json.loads((work / f"{name}.receipt.json").read_text(encoding="utf-8"))
    return {"rc": rc, "annotations": annotations(buf.getvalue()), "summary": summary.read_text(encoding="utf-8"),
            "suggest_line": next((ln for ln in buf.getvalue().splitlines() if ln.startswith("styxx ci-audit: ")), None),
            "posts": review.calls[n0:], "fires": rec["fires"], "new_hidden": rec["new_hidden"], "annotation_levels": rec.get("annotations"),
            "checks": [{"path": w["path"], "job": x["job"], "step": x["step"], "kind": x["kind"], "verdict": x["verdict"],
                        "repair": (x.get("fix") or {}).get("verified_repair")} for w in rec["workflows"] for x in w.get("new_hidden", [])]}


def plan(work: Path | None = None) -> dict:
    """The prediction: the Action on the live change, again on the same head, and on the head with
    every placed suggestion applied the way the plan says GitHub applies it."""
    own = work is None
    work = Path(tempfile.mkdtemp(prefix="live-click-")) if own else work
    try:
        s = _repo(work)
        review = Review()
        first = run_action(work, s, s["head"], s["merge"], review, "first")
        again = run_action(work, s, s["head"], s["merge"], review, "again")
        placed = [c for c in first["posts"] if c["status"] == 201]
        head_files = side("head")
        expected = {}
        _git(s["src"], "checkout", "-q", "live")
        for path in sorted({c["path"] for c in placed}):
            new = splice(head_files[path], [c for c in placed if c["path"] == path])
            expected[path] = {"sha256": _sha(new), "lines_sha256": _sha("\n".join(new.decode("utf-8").splitlines()).encode("utf-8")),
                              "final_newline": new.endswith(b"\n"), "suggestions": sum(1 for c in placed if c["path"] == path)}
            (s["src"] / path).write_bytes(new)
        _git(s["src"], "commit", "-q", "-am", "the suggestions, applied")
        applied_head = _git(s["src"], "rev-parse", "HEAD")
        applied = run_action(work, s, applied_head, _merge(s, applied_head), review, "applied")
        return {"schema": SCHEMA, "instrument_sha256": instrument_sha256(), "action_sha256": _sha((ROOT / "styxx" / "ciaudit" / "action.py").read_bytes()),
                "action_yml_sha256": _sha((ROOT / "ci-audit" / "action.yml").read_bytes()),
                "differential_living_sha256": _sha((ROOT / "styxx" / "ciaudit" / "differential.py").read_bytes()),
                "fixtures": {n: {"why": w, "base_sha256": _sha(b.encode("utf-8")), "head_sha256": _sha(h.encode("utf-8"))} for n, (w, b, h) in FIXTURES.items()},
                "live_workflow_sha256": _sha(LIVE_WORKFLOW.encode("utf-8")), "base_workflow_sha256": _sha(BASE_WORKFLOW.encode("utf-8")),
                "runs": {"first": first, "again": again, "applied": applied}, "expected_files": expected}
    finally:
        if own:
            shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- what GitHub did

LEVEL = {"failure": "error", "warning": "warning", "notice": "notice"}


def _live_annotations(check_run: dict) -> list[dict]:
    """The Action's own annotations in a check run, in the plan's shape (the runner's own notices dropped)."""
    return [{"level": LEVEL.get(a["annotation_level"], a["annotation_level"]), "path": a["path"], "start_line": a["start_line"],
             "end_line": a["end_line"], "title": a.get("title") or "", "message": a.get("message") or ""}
            for a in check_run.get("annotations", []) if (a.get("title") or "").startswith("styxx ci-audit")]


def _key(a: dict) -> tuple:
    return (a["level"], a["path"], a["start_line"], a["end_line"], a["title"], a["message"])


def receipt(p: dict, obs: dict, clone: Path) -> dict:
    """The plan against what GitHub did: comments, annotations, outcomes and bytes."""
    first, again, applied = p["runs"]["first"], p["runs"]["again"], p["runs"]["applied"]
    h1, h2, base = obs["first"]["head_sha"], obs["applied"]["head_sha"], obs["base_sha"]

    def blob(sha: str, path: str) -> bytes | None:
        q = subprocess.run(["git", "-C", str(clone), "cat-file", "blob", f"{sha}:{path}"], capture_output=True)
        return q.stdout if q.returncode == 0 else None

    head_files, base_files = side("head"), side("base")
    planned = [c for c in first["posts"] if c["status"] == 201]
    comments = obs["comments_after_first"]

    def ckey(c: dict) -> tuple:
        body = c["body"]
        m = re.search(r"```+suggestion\n(.*?)\n```+", body, re.S)
        return (c["path"], c.get("start_line") or c["line"], c["line"], _sha(body.encode("utf-8")), tuple(m.group(1).split("\n")) if m and m.group(1) else ())

    live_comments = sorted(ckey(c) for c in comments)
    plan_comments = sorted((c["path"], c["start_line"], c["line"], c["body_sha256"], tuple(c["lines"])) for c in planned)
    files_ = {}
    for path, e in p["expected_files"].items():
        got = blob(h2, path)
        files_[path] = {"suggestions": e["suggestions"], "present": got is not None,
                        "lines_equal": got is not None and _sha("\n".join(got.decode("utf-8").splitlines()).encode("utf-8")) == e["lines_sha256"],
                        "bytes_equal": got is not None and _sha(got) == e["sha256"], "final_newline_expected": e["final_newline"],
                        "final_newline_got": got is not None and got.endswith(b"\n"), "got_sha256": _sha(got) if got is not None else None}
    untouched = {rel: blob(h2, rel) == data for rel, data in head_files.items() if rel not in p["expected_files"]}
    changed = subprocess.run(["git", "-C", str(clone), "diff", "--name-only", h1, h2], capture_output=True, text=True, encoding="utf-8").stdout.split()

    def same(sha: str, rel: str, want: str) -> bool:
        got = blob(sha, rel)
        return got is not None and _sha(got) == want
    return {
        "schema": SCHEMA, "plan_instrument_sha256": p["instrument_sha256"], "instrument_sha256": instrument_sha256(), "action_sha256": p["action_sha256"],
        "action_yml_sha256": p["action_yml_sha256"], "differential_living_sha256": p["differential_living_sha256"],
        "repo": obs["repo"], "pr": obs["pr"], "base_sha": base, "head_sha": h1, "applied_sha": h2,
        "live_is_the_plan": {"base": {rel: blob(base, rel) == data for rel, data in base_files.items()},
                             "head": {rel: blob(h1, rel) == data for rel, data in head_files.items()},
                             "applied_parent_is_head": obs["applied"].get("parents") == [h1],
                             "applied_by_github": bool(obs["applied"].get("committer_is_github")),
                             "first_run_on_test_merge": bool(obs["first"].get("test_merge")),
                             "gate_code": {"styxx/ciaudit/action.py": same(h1, "styxx/ciaudit/action.py", p["action_sha256"]),
                                           "ci-audit/action.yml": same(h1, "ci-audit/action.yml", p["action_yml_sha256"]),
                                           "styxx/ciaudit/differential.py": same(h1, "styxx/ciaudit/differential.py", p["differential_living_sha256"])}},
        "applied_changed_files": sorted(changed),
        "comments": {"planned": len(plan_comments), "live": len(live_comments), "equal": live_comments == plan_comments,
                     "only_in_plan": [list(k[:3]) for k in plan_comments if k not in live_comments],
                     "only_live": [list(k[:3]) for k in live_comments if k not in plan_comments],
                     "authors": sorted({c.get("author_type") for c in comments}), "commit_ids": sorted({c.get("commit_id") == h1 for c in comments})},
        "refused": {"planned": sorted(c["path"] for c in first["posts"] if c["status"] == 422),
                    "live_first": obs["first"].get("suggest_line"), "live_again": obs["again"].get("suggest_line"),
                    "planned_first": first["suggest_line"], "planned_again": again["suggest_line"]},
        "again": {"comments_before": len(comments), "comments_after": len(obs["comments_after_again"]),
                  "same_ids": sorted(c["id"] for c in comments) == sorted(c["id"] for c in obs["comments_after_again"])},
        "annotations": {run: {"planned": len(p["runs"][run]["annotations"]), "live": len(_live_annotations(obs[run]["check_run"])),
                              "equal": sorted(map(_key, _live_annotations(obs[run]["check_run"]))) == sorted(map(_key, p["runs"][run]["annotations"])),
                              "levels_live": {lv: sum(1 for a in _live_annotations(obs[run]["check_run"]) if a["level"] == lv) for lv in ("error", "warning", "notice")},
                              "levels_planned": {lv: sum(1 for a in p["runs"][run]["annotations"] if a["level"] == lv) for lv in ("error", "warning", "notice")},
                              "only_in_plan": [list(k[:4]) for k in sorted(map(_key, p["runs"][run]["annotations"])) if k not in set(map(_key, _live_annotations(obs[run]["check_run"])))],
                              "only_live": [list(k[:4]) for k in sorted(map(_key, _live_annotations(obs[run]["check_run"]))) if k not in set(map(_key, p["runs"][run]["annotations"]))],
                              "conclusion": obs[run]["check_run"].get("conclusion"), "planned_rc": p["runs"][run]["rc"]}
                        for run in ("first", "again", "applied")},
        "files": files_, "untouched_after_apply": untouched,
        "applied_checks": {"planned": sorted((c["path"], c["step"]) for c in applied["checks"]),
                           "live_annotated": sorted({(a["path"], a["start_line"]) for a in _live_annotations(obs["applied"]["check_run"])}),
                           "planned_annotated": sorted({(a["path"], a["start_line"]) for a in applied["annotations"]})},
    }


# ----------------------------------------------------------------------------- entry point

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("files")
    f.add_argument("--out", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--out", required=True)
    rc = sub.add_parser("receipt")
    rc.add_argument("--plan", required=True)
    rc.add_argument("--observed", required=True)
    rc.add_argument("--clone", required=True)
    rc.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.cmd == "files":
        print(json.dumps(files(Path(a.out)), indent=1))
    elif a.cmd == "plan":
        p = plan()
        Path(a.out).write_text(json.dumps(p, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        r = p["runs"]
        print(json.dumps({k: {"rc": v["rc"], "new_hidden": v["new_hidden"], "levels": v["annotation_levels"],
                              "posts": [(c["path"].rsplit("/", 1)[-1], c["status"]) for c in v["posts"]]} for k, v in r.items()}, indent=1))
    else:
        p = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        obs = json.loads(Path(a.observed).read_text(encoding="utf-8"))
        rec = receipt(p, obs, Path(a.clone))
        rec["seconds"] = round(time.time() - t0, 1)
        Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({k: rec[k] for k in ("comments", "again", "annotations")}, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
