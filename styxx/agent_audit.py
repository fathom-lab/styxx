# -*- coding: utf-8 -*-
"""styxx.agent_audit — substrate-grounded auditor for agent session claims.

The next frame of the recursive-discipline arc (papers/PAPER_recursive_discipline_2026_05_27.md §13):
agent session-output claims about the substrate are themselves checkable
against the substrate. This module ships a minimal instrument for that
class of audit — read-only, falsifiable, paper-grade.

Usage:

    from styxx.agent_audit import Claim, AgentClaimAuditor, checkers

    claims = [
        Claim(
            id="C1",
            text="pyproject.toml version equals 7.7.10",
            checker=checkers.package_version_equals,
            args={"path": "pyproject.toml", "version": "7.7.10"},
            expected=True,
        ),
    ]
    results = AgentClaimAuditor(repo_path="/path/to/repo").run(claims)
    for r in results:
        print(r.id, r.verdict, "->", r.evidence[:120])

Each checker returns ``(actual: bool, evidence: str)``. The auditor compares
``actual`` to the claim's ``expected``; verdict is ``"PASS"`` if equal else
``"FAIL"``. Evidence is a short human-readable string with the substrate
witness (commit SHA, file excerpt, page count, etc.).

The instrument has no external services, runs offline, mutates nothing.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable


__all__ = ["Claim", "AuditResult", "AgentClaimAuditor", "checkers"]


@dataclass
class Claim:
    id: str
    text: str
    checker: Callable[..., tuple[bool, str]]
    args: dict[str, Any]
    expected: bool = True


@dataclass
class AuditResult:
    id: str
    text: str
    expected: bool
    actual: bool
    verdict: str  # "PASS" or "FAIL" or "ERROR"
    evidence: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentClaimAuditor:
    """Run a list of Claims against a substrate (typically a git repo).

    The auditor binds a working ``repo_path`` once and passes it as the
    first positional argument to every checker. Individual claims may
    pass additional kwargs via ``Claim.args``.
    """

    def __init__(self, repo_path: str | Path):
        self.repo_path = Path(repo_path).resolve()
        if not self.repo_path.exists():
            raise FileNotFoundError(f"repo_path does not exist: {self.repo_path}")

    def run(self, claims: list[Claim]) -> list[AuditResult]:
        out: list[AuditResult] = []
        for c in claims:
            try:
                actual, evidence = c.checker(self.repo_path, **c.args)
                verdict = "PASS" if actual == c.expected else "FAIL"
                out.append(
                    AuditResult(
                        id=c.id, text=c.text, expected=c.expected,
                        actual=actual, verdict=verdict, evidence=evidence,
                    )
                )
            except Exception as e:  # noqa: BLE001 — surface checker bugs as ERROR
                out.append(
                    AuditResult(
                        id=c.id, text=c.text, expected=c.expected,
                        actual=False, verdict="ERROR",
                        evidence="", error=f"{type(e).__name__}: {e}",
                    )
                )
        return out


class _Checkers:
    """Registered checker functions. Each takes ``repo_path`` first."""

    @staticmethod
    def _run(repo: Path, *cmd: str) -> str:
        r = subprocess.run(
            list(cmd), cwd=str(repo), capture_output=True, text=True, check=False,
        )
        return (r.stdout or "") + ("\n[stderr]\n" + r.stderr if r.stderr else "")

    def git_show_diff_contains(self, repo: Path, *, commit: str, file: str, substring: str) -> tuple[bool, str]:
        diff = self._run(repo, "git", "show", "--format=", f"{commit}", "--", file)
        present = substring in diff
        excerpt = "\n".join(
            l for l in diff.splitlines()
            if substring in l or any(t in l for t in ("---", "+++", "@@"))
        )[:600]
        return present, f"commit {commit[:7]} diff for {file}: {'MATCH' if present else 'no match'}\n{excerpt}"

    def git_branch_contains_commit_chain(self, repo: Path, *, branch: str, commits: list[str]) -> tuple[bool, str]:
        # all commits reachable from branch HEAD AND in monotonic order
        log = self._run(repo, "git", "log", "--format=%H", branch).splitlines()
        positions = []
        for c in commits:
            matches = [i for i, h in enumerate(log) if h.startswith(c)]
            if not matches:
                return False, f"commit {c} not on {branch}"
            positions.append(matches[0])
        ordered = positions == sorted(positions, reverse=True)  # newer-first in log
        evidence = f"{branch} positions (older→newer): " + ", ".join(
            f"{c[:7]}@idx{p}" for c, p in zip(commits, positions)
        )
        return ordered, evidence

    def git_tag_exists(self, repo: Path, *, tag: str) -> tuple[bool, str]:
        out = self._run(repo, "git", "tag", "-l", tag).strip()
        return (out == tag), f"git tag -l {tag} -> {out!r}"

    def file_at_path_contains(self, repo: Path, *, path: str, substring: str) -> tuple[bool, str]:
        p = repo / path
        text = p.read_text(encoding="utf-8", errors="replace")
        present = substring in text
        idx = text.find(substring)
        excerpt = text[max(0, idx - 60): idx + len(substring) + 60] if idx >= 0 else ""
        return present, f"{path}: {'MATCH at offset ' + str(idx) if present else 'no match'}\n{excerpt!r}"

    def python_attr_in_iterable(self, repo: Path, *, module: str, attr: str, iterable: str) -> tuple[bool, str]:
        # import the module from the repo; safer than evaluating arbitrary code
        sys.path.insert(0, str(repo))
        try:
            mod = __import__(module)
            container = getattr(mod, iterable)
            present = attr in container
            return present, f"{module}.{iterable} length={len(container)}; {attr!r} {'present' if present else 'absent'}"
        finally:
            if str(repo) in sys.path:
                sys.path.remove(str(repo))

    def package_version_equals(self, repo: Path, *, path: str, version: str) -> tuple[bool, str]:
        text = (repo / path).read_text(encoding="utf-8", errors="replace")
        m = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if not m:
            return False, f"{path}: no version line found"
        actual = m.group(1)
        return (actual == version), f"{path}: version={actual!r}, expected={version!r}"

    def pdf_page_count_equals(self, repo: Path, *, path: str, n: int) -> tuple[bool, str]:
        from pypdf import PdfReader
        r = PdfReader(str(repo / path))
        actual = len(r.pages)
        return (actual == n), f"{path}: pages={actual}, expected={n}"

    def pdf_contains_section(self, repo: Path, *, path: str, section_title: str) -> tuple[bool, str]:
        from pypdf import PdfReader
        r = PdfReader(str(repo / path))
        for i, page in enumerate(r.pages):
            t = page.extract_text() or ""
            # tolerate hyphenation, whitespace, line breaks
            normalized = re.sub(r"\s+", " ", t)
            target = re.sub(r"\s+", " ", section_title)
            if target in normalized:
                excerpt = normalized[normalized.index(target): normalized.index(target) + 160]
                return True, f"page {i + 1}: {excerpt!r}"
        return False, f"{path}: section title not found across {len(r.pages)} pages"

    def directory_file_count_equals(self, repo: Path, *, glob: str, n: int) -> tuple[bool, str]:
        from glob import glob as _glob
        matches = _glob(str(repo / glob), recursive=True)
        # for `*` patterns that include directories, count only entries matching glob
        actual = len(matches)
        return (actual == n), f"glob {glob!r}: {actual} matches, expected {n}"

    def json_path_equals(self, repo: Path, *, path: str, key_path: str, expected) -> tuple[bool, str]:
        import json as _json
        d = _json.loads((repo / path).read_text(encoding="utf-8", errors="replace"))
        cur = d
        for part in key_path.split("."):
            if part.startswith("[") and part.endswith("]"):
                cur = cur[int(part[1:-1])]
            else:
                cur = cur[part]
        return (cur == expected), f"{path}::{key_path} = {cur!r}, expected {expected!r}"

    def python_attr_equals(self, repo: Path, *, module: str, attr: str, expected) -> tuple[bool, str]:
        sys.path.insert(0, str(repo))
        try:
            mod = __import__(module)
            actual = getattr(mod, attr)
            return (actual == expected), f"{module}.{attr} = {actual!r}, expected {expected!r}"
        finally:
            if str(repo) in sys.path:
                sys.path.remove(str(repo))

    def file_byte_equals(self, repo: Path, *, path_a: str, path_b: str) -> tuple[bool, str]:
        a = (repo / path_a).read_bytes()
        b = (repo / path_b).read_bytes()
        equal = (a == b)
        evidence = f"{path_a} ({len(a)}B) vs {path_b} ({len(b)}B): {'identical' if equal else 'DIFFER'}"
        if not equal:
            # locate first differing byte for evidence
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    ctx_a = a[max(0, i - 30): i + 30]
                    ctx_b = b[max(0, i - 30): i + 30]
                    evidence += f"\nfirst diff at byte {i}: a={ctx_a!r}\n                    b={ctx_b!r}"
                    break
        return equal, evidence


checkers = _Checkers()
