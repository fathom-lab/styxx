"""What CI says about a commit's tests, from GitHub check runs, Actions job steps and statuses.

Evidence is a test-like step of a GitHub Actions job ("Run tests", "pytest"), or, where steps are
not known, a test-like check run or commit status name ("test (3.11)", "ci/circleci: unit"). Passes
and failures are read the same way, so a job named "build" that runs the tests counts through its
steps either way.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

TEST_STEP = re.compile(
    r"\b(?:tests?|testing|pytest|py\.test|tox|nox|jest|vitest|mocha|unit|integration|e2e|spec)\b",
    re.IGNORECASE,
)
NOT_TESTS = re.compile(
    r"\b(?:lint(?:s|ing|er)?|format(?:s|ting|ter)?|fmt|style|type ?check(?:s|ing)?|mypy|"
    r"pyright|eslint|prettier|ruff|flake8|clippy|codeql|security|docs?|deploy|publish|release|"
    r"label|changelog|codecov|coverage upload|preview|vercel|netlify|cla|dco|commit ?lint|title)\b",
    re.IGNORECASE,
)


def is_test_step(name: str) -> bool:
    return bool(TEST_STEP.search(name)) and not NOT_TESTS.search(name)


@dataclass(frozen=True)
class Check:
    name: str
    conclusion: str | None  # "success", "failure", "cancelled", ... or None while running
    app: str | None = None
    steps: tuple[tuple[str, str | None], ...] = ()  # (name, conclusion) of an Actions job


@dataclass
class CiResult:
    head: str
    checks: list[Check] = field(default_factory=list)
    statuses: list[tuple[str, str]] = field(default_factory=list)  # (context, state)

    def tests(self) -> tuple[str, str]:
        """("tests failed" | "tests passed" | "no test result" | "no CI", evidence)."""
        if not self.checks and not self.statuses:
            return "no CI", ""
        evidence: list[tuple[str, str]] = []
        for check in self.checks:
            if check.steps:
                evidence += [
                    (conclusion or "", f"{check.name} › {name}")
                    for name, conclusion in check.steps
                    if is_test_step(name)
                ]
            elif is_test_step(check.name):
                evidence.append((check.conclusion or "", check.name))
        evidence += [
            ("failure" if state in {"failure", "error"} else state, context)
            for context, state in self.statuses
            if is_test_step(context)
        ]
        failed = [where for outcome, where in evidence if outcome == "failure"]
        if failed:
            return "tests failed", failed[0]
        if any(outcome == "success" for outcome, _ in evidence):
            return "tests passed", next(w for o, w in evidence if o == "success")
        return "no test result", ""


# --- GitHub ------------------------------------------------------------------------------------


class GitHubError(RuntimeError):
    pass


def token() -> str | None:
    """A GitHub token from the environment (as in Actions) or from the gh CLI."""
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return out.stdout.strip() or None


def _get(path: str, auth: str | None) -> Any:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "testreceipt"}
    if auth:
        headers["Authorization"] = f"Bearer {auth}"
    request = urllib.request.Request(f"https://api.github.com{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise GitHubError(f"GitHub API {error.code} for {path}") from error
    except urllib.error.URLError as error:
        raise GitHubError(f"GitHub API unreachable: {error.reason}") from error


def _paged(path: str, key: str, auth: str | None, pages: int = 5) -> list[Any]:
    """All items of a listing that comes in pages of 100 (the default page holds 30)."""
    items: list[Any] = []
    for page in range(1, pages + 1):
        batch = _get(f"{path}?per_page=100&page={page}", auth).get(key, [])
        items += batch
        if len(batch) < 100:
            break
    return items


def parse_pr(ref: str) -> tuple[str, int]:
    """ "owner/repo#12" or a pull request URL -> ("owner/repo", 12)."""
    match = re.search(r"github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)", ref) or re.fullmatch(
        r"([\w.-]+/[\w.-]+)#(\d+)", ref.strip()
    )
    if not match:
        raise ValueError(f"not a pull request: {ref!r} (use owner/repo#123 or its URL)")
    return match.group(1), int(match.group(2))


def fetch_pr(repo: str, number: int, auth: str | None = None) -> tuple[str, CiResult]:
    """The pull request's description and what CI says about its head commit."""
    pull = _get(f"/repos/{repo}/pulls/{number}", auth)
    head = pull["head"]["sha"]
    runs = _paged(f"/repos/{repo}/commits/{head}/check-runs", "check_runs", auth)
    checks = []
    for run in runs:
        app = (run.get("app") or {}).get("slug")
        steps: tuple[tuple[str, str | None], ...] = ()
        if app == "github-actions" and run.get("conclusion") in {"success", "failure"}:
            job = _get(f"/repos/{repo}/actions/jobs/{run['id']}", auth)
            steps = tuple((s.get("name") or "", s.get("conclusion")) for s in job.get("steps", []))
        checks.append(Check(run.get("name") or "", run.get("conclusion"), app, steps))
    status = _paged(f"/repos/{repo}/commits/{head}/status", "statuses", auth)
    statuses = [(s.get("context") or "", s.get("state") or "") for s in status]
    return pull.get("body") or "", CiResult(head, checks, statuses)
