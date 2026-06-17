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

import ntpath
import posixpath
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

# A conservative git ref/pathspec charset. The first character may not be '-'
# (which git would parse as an option), and the body is restricted to the
# characters real refs/tags/SHAs use. Verify-time checker args are attacker-
# controlled (reconstructed from an untrusted attestation artifact), so any ref
# that fails this is refused before it reaches git — closing the argument-
# injection / `--output=` arbitrary-write vector on the verification path.
_SAFE_REF_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./+@^~-]*$")


__all__ = [
    "Claim", "AuditResult", "AgentClaimAuditor", "checkers",
    "extract_claims", "ExtractionReport", "CLAIM_TEMPLATES",
]


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
    """Registered checker functions. Each takes ``repo_path`` first.

    Security boundary: these checkers run on the untrusted-verification path
    (``styxx.attestation.verify_attestation`` reconstructs every ``args`` value
    from an attacker-supplied artifact). File-path args are therefore confined
    to the substrate root via :meth:`_safe_subpath`, and git ref/pathspec args
    are validated via :meth:`_safe_ref` before reaching ``git`` — so a receipt
    can never read a file outside the repo or inject a git option. Checkers that
    must execute substrate code (``python_attr_*``) are additionally refused at
    verify time unless the caller opts into trusting the substrate.
    """

    @staticmethod
    def _safe_subpath(repo: Path, rel: str) -> Path:
        """Resolve ``rel`` under ``repo`` and refuse any escape.

        Rejects absolute paths (POSIX, Windows, or drive-relative) and any
        ``..`` traversal, then resolves symlinks and asserts the result stays
        within ``repo``. Raises ``ValueError`` on any escape — surfaced by the
        auditor as an ERROR verdict (fail-closed), never a silent read.
        """
        if rel is None:
            raise ValueError("path is required")
        raw = str(rel)
        if posixpath.isabs(raw) or ntpath.isabs(raw) or ntpath.splitdrive(raw)[0]:
            raise ValueError(f"absolute paths are not allowed: {raw!r}")
        parts = re.split(r"[\\/]+", raw)
        if ".." in parts:
            raise ValueError(f"path traversal ('..') is not allowed: {raw!r}")
        repo_resolved = repo.resolve()
        resolved = (repo_resolved / raw).resolve()
        if resolved != repo_resolved and not resolved.is_relative_to(repo_resolved):
            raise ValueError(f"path escapes the substrate root: {raw!r}")
        return resolved

    @staticmethod
    def _safe_ref(value: str, *, kind: str = "ref") -> str:
        """Validate a git ref/tag/commit/pathspec arg, refusing option injection."""
        if not isinstance(value, str) or not value:
            raise ValueError(f"{kind} must be a non-empty string")
        if value.startswith("-"):
            raise ValueError(f"{kind} may not begin with '-': {value!r}")
        if not _SAFE_REF_RE.match(value):
            raise ValueError(f"{kind} contains disallowed characters: {value!r}")
        return value

    @staticmethod
    def _safe_module(module: str) -> str:
        """Validate an importable module name is a plain dotted identifier.

        Defense-in-depth for ``python_attr_*`` (which import substrate code):
        blocks path-shaped or option-shaped module args. Note these checkers are
        ALSO refused at verify time (``verify_attestation(trust_substrate=False)``)
        because importing executes top-level code — this guard only narrows the
        blast radius for callers that explicitly opt into trusting the substrate.
        """
        if not isinstance(module, str) or not module:
            raise ValueError("module must be a non-empty string")
        if not all(part.isidentifier() for part in module.split(".")):
            raise ValueError(f"module is not a valid dotted identifier: {module!r}")
        return module

    @staticmethod
    def _safe_glob(repo: Path, pattern: str) -> str:
        """Confine a glob pattern to the substrate root (rejects absolute / ``..``)."""
        if not isinstance(pattern, str) or not pattern:
            raise ValueError("glob is required")
        if posixpath.isabs(pattern) or ntpath.isabs(pattern) or ntpath.splitdrive(pattern)[0]:
            raise ValueError(f"absolute glob is not allowed: {pattern!r}")
        if ".." in re.split(r"[\\/]+", pattern):
            raise ValueError(f"glob traversal ('..') is not allowed: {pattern!r}")
        return str(repo.resolve() / pattern)

    @staticmethod
    def _run(repo: Path, *cmd: str) -> str:
        r = subprocess.run(
            list(cmd), cwd=str(repo), capture_output=True, text=True, check=False,
        )
        return (r.stdout or "") + ("\n[stderr]\n" + r.stderr if r.stderr else "")

    def git_show_diff_contains(self, repo: Path, *, commit: str, file: str, substring: str) -> tuple[bool, str]:
        commit = self._safe_ref(commit, kind="commit")
        diff = self._run(repo, "git", "show", "--format=", commit, "--", file)
        present = substring in diff
        excerpt = "\n".join(
            l for l in diff.splitlines()
            if substring in l or any(t in l for t in ("---", "+++", "@@"))
        )[:600]
        return present, f"commit {commit[:7]} diff for {file}: {'MATCH' if present else 'no match'}\n{excerpt}"

    def git_branch_contains_commit_chain(self, repo: Path, *, branch: str, commits: list[str]) -> tuple[bool, str]:
        # all commits reachable from branch HEAD AND in monotonic order
        branch = self._safe_ref(branch, kind="branch")
        for _c in commits:
            self._safe_ref(_c, kind="commit")
        log = self._run(repo, "git", "log", "--format=%H", branch, "--").splitlines()
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
        tag = self._safe_ref(tag, kind="tag")
        out = self._run(repo, "git", "tag", "-l", tag).strip()
        return (out == tag), f"git tag -l {tag} -> {out!r}"

    def file_at_path_contains(self, repo: Path, *, path: str, substring: str) -> tuple[bool, str]:
        p = self._safe_subpath(repo, path)
        text = p.read_text(encoding="utf-8", errors="replace")
        present = substring in text
        idx = text.find(substring)
        excerpt = text[max(0, idx - 60): idx + len(substring) + 60] if idx >= 0 else ""
        return present, f"{path}: {'MATCH at offset ' + str(idx) if present else 'no match'}\n{excerpt!r}"

    def python_attr_in_iterable(self, repo: Path, *, module: str, attr: str, iterable: str) -> tuple[bool, str]:
        # import the module from the repo; safer than evaluating arbitrary code
        module = self._safe_module(module)
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
        text = self._safe_subpath(repo, path).read_text(encoding="utf-8", errors="replace")
        m = re.search(r'^\s*version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if not m:
            return False, f"{path}: no version line found"
        actual = m.group(1)
        return (actual == version), f"{path}: version={actual!r}, expected={version!r}"

    def pdf_page_count_equals(self, repo: Path, *, path: str, n: int) -> tuple[bool, str]:
        from pypdf import PdfReader
        r = PdfReader(str(self._safe_subpath(repo, path)))
        actual = len(r.pages)
        return (actual == n), f"{path}: pages={actual}, expected={n}"

    def pdf_contains_section(self, repo: Path, *, path: str, section_title: str) -> tuple[bool, str]:
        from pypdf import PdfReader
        r = PdfReader(str(self._safe_subpath(repo, path)))
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
        repo_resolved = repo.resolve()
        matches = [
            m for m in _glob(self._safe_glob(repo, glob), recursive=True)
            if Path(m).resolve().is_relative_to(repo_resolved)
        ]
        # for `*` patterns that include directories, count only entries matching glob
        actual = len(matches)
        return (actual == n), f"glob {glob!r}: {actual} matches, expected {n}"

    def json_path_equals(self, repo: Path, *, path: str, key_path: str, expected) -> tuple[bool, str]:
        import json as _json
        d = _json.loads(self._safe_subpath(repo, path).read_text(encoding="utf-8", errors="replace"))
        cur = d
        for part in key_path.split("."):
            if part.startswith("[") and part.endswith("]"):
                cur = cur[int(part[1:-1])]
            else:
                cur = cur[part]
        return (cur == expected), f"{path}::{key_path} = {cur!r}, expected {expected!r}"

    def python_attr_equals(self, repo: Path, *, module: str, attr: str, expected) -> tuple[bool, str]:
        module = self._safe_module(module)
        sys.path.insert(0, str(repo))
        try:
            mod = __import__(module)
            actual = getattr(mod, attr)
            return (actual == expected), f"{module}.{attr} = {actual!r}, expected {expected!r}"
        finally:
            if str(repo) in sys.path:
                sys.path.remove(str(repo))

    def value_consistent_across_paths(
        self, repo: Path, *, glob: str, pattern: str, expected: str, group: int = 1,
    ) -> tuple[bool, str]:
        """Assert EVERY regex capture of ``pattern`` across ``glob`` equals ``expected``.

        Closes the first-occurrence-only construct ceiling: where the
        single-site checkers verify one location and stop, this scans every
        file matching ``glob``, extracts every occurrence of ``pattern``
        (capture ``group``), and FAILs if any occurrence diverges from
        ``expected`` — surfacing the systematic propagation drift that a
        first-occurrence check silently misses.

        Zero occurrences FAILs loudly: ``all()`` over an empty set is
        vacuously True, which would let a typo'd pattern pass. A claim that a
        value appears consistently is not satisfied when it appears nowhere.
        """
        from glob import glob as _glob

        rx = re.compile(pattern)
        want = str(expected)
        matches = sorted(_glob(self._safe_glob(repo, glob), recursive=True))
        occurrences = 0
        divergent: list[str] = []
        for fp in matches:
            p = Path(fp)
            if not p.is_file():
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            rel = p.relative_to(repo).as_posix() if p.is_relative_to(repo) else fp
            for m in rx.finditer(text):
                occurrences += 1
                got = m.group(group)
                if got != want:
                    line = text.count("\n", 0, m.start()) + 1
                    divergent.append(f"{rel}:{line} -> {got!r}")
        if occurrences == 0:
            return False, (
                f"glob {glob!r} pattern {pattern!r}: ZERO occurrences across "
                f"{len(matches)} path(s) — fails loudly (no vacuous PASS)"
            )
        if divergent:
            shown = "; ".join(divergent[:20])
            more = f" (+{len(divergent) - 20} more)" if len(divergent) > 20 else ""
            return False, (
                f"{occurrences} occurrence(s); {len(divergent)} diverge from "
                f"{want!r}: {shown}{more}"
            )
        return True, f"{occurrences} occurrence(s) across {len(matches)} path(s), all == {want!r}"

    def value_internally_consistent(
        self, repo: Path, *, path: str, pattern: str, group: int = 1,
    ) -> tuple[bool, str]:
        """Assert all captures of ``pattern`` WITHIN one file agree with each other.

        Oracle-free: unlike value_consistent_across_paths (which needs the
        canonical ``expected`` value), this asserts only that a document does
        not contradict ITSELF. It is the automatic form of the L7 off-by-one
        catch — a paper claiming "30 items" in four places and "28" in a fifth
        FAILs with zero configuration. No expected value is supplied.

        Triage semantics, not a gate: legitimate within-document variation
        (e.g. a historical version reference alongside a current one) will also
        FAIL, by design — the instrument flags candidates for review rather
        than adjudicating intent. Zero or one occurrence is trivially
        consistent (nothing contradicts), reported as PASS with the count.
        """
        rx = re.compile(pattern)
        text = self._safe_subpath(repo, path).read_text(encoding="utf-8", errors="replace")
        by_value: dict[str, list[int]] = {}
        for m in rx.finditer(text):
            val = m.group(group)
            line = text.count("\n", 0, m.start()) + 1
            by_value.setdefault(val, []).append(line)
        total = sum(len(v) for v in by_value.values())
        if total <= 1:
            return True, f"{path}: {total} occurrence(s) of {pattern!r} — trivially consistent"
        if len(by_value) == 1:
            val, lines = next(iter(by_value.items()))
            return True, f"{path}: {total} occurrence(s), all == {val!r} (lines {lines[:10]})"
        parts = []
        for val, lines in sorted(by_value.items(), key=lambda kv: -len(kv[1])):
            shown = ",".join(str(n) for n in lines[:8])
            more = f"+{len(lines) - 8}" if len(lines) > 8 else ""
            parts.append(f"{val!r}@[{shown}{more}]")
        return False, (
            f"{path}: {len(by_value)} distinct values across {total} "
            f"occurrence(s): " + "; ".join(parts)
        )

    def file_byte_equals(self, repo: Path, *, path_a: str, path_b: str) -> tuple[bool, str]:
        a = self._safe_subpath(repo, path_a).read_bytes()
        b = self._safe_subpath(repo, path_b).read_bytes()
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


# ---------------------------------------------------------------------------
# Deterministic claim extraction: agent prose -> checkable Claim objects.
#
# This is the bridge from "verifier you must hand-feed" to "paste an agent's
# self-report, get a falsification report". It is DELIBERATELY deterministic
# (regex templates, no LLM): an LLM-based extractor would itself be an
# untrustworthy agent self-report. The closed template set is the honest
# boundary — free-form claims are out of scope and reported as uncovered.
#
# Extraction != verification. extract_claims only PROPOSES checkable claims;
# AgentClaimAuditor.run still mechanically verifies each against substrate.
# Imperfect recall is acceptable; mis-TYPING a claim is not (a wrong checker
# binding yields a meaningless verdict), so templates bind args explicitly.
# ---------------------------------------------------------------------------

@dataclass
class ExtractionReport:
    """Result of extract_claims: the claims found plus honest coverage stats."""
    claims: list[Claim]
    sentences_total: int
    sentences_matched: int

    @property
    def coverage(self) -> float:
        """Fraction of sentences that produced at least one claim (honest recall proxy)."""
        return (self.sentences_matched / self.sentences_total) if self.sentences_total else 0.0


# A PEP440-ish version token: semver core plus optional pre/dev/post suffix, so
# "3.1.0a1" / "1.2.0rc2" survive extraction intact rather than truncating to the
# numeric core (which manufactures a false mismatch against the real version).
_VER = r"\d+\.\d+\.\d+(?:(?:a|b|rc|\.?dev|\.?post)\d*)?"


def _tmpl_version(m: re.Match) -> Claim:
    v = m.group("v") or m.group("v2")  # two alternatives: "version is X" / "styxx==X"
    return Claim(
        id="", text=m.group(0).strip(),
        checker=checkers.package_version_equals,
        args={"path": "pyproject.toml", "version": v}, expected=True,
    )


def _tmpl_version_bump(m: re.Match) -> Claim:
    # A bump line "X -> Y" / "X → Y" / "X to Y" claims the repo is now at the
    # POST-state Y. Capturing the left side (the old version) is the dominant
    # false-positive in retrospective commit-message audits.
    return Claim(
        id="", text=m.group(0).strip(),
        checker=checkers.package_version_equals,
        args={"path": "pyproject.toml", "version": m.group("vb")}, expected=True,
    )


def _tmpl_tag(m: re.Match) -> Claim:
    t = m.group("t")
    return Claim(
        id="", text=m.group(0).strip(),
        checker=checkers.git_tag_exists, args={"tag": t}, expected=True,
    )


def _tmpl_file_contains(m: re.Match) -> Claim:
    return Claim(
        id="", text=m.group(0).strip(),
        checker=checkers.file_at_path_contains,
        args={"path": m.group("path"), "substring": m.group("sub")}, expected=True,
    )


def _tmpl_pdf_pages(m: re.Match) -> Claim:
    return Claim(
        id="", text=m.group(0).strip(),
        checker=checkers.pdf_page_count_equals,
        args={"path": m.group("path"), "n": int(m.group("n"))}, expected=True,
    )


# Each template: (name, compiled regex with named groups, builder). Anchors are
# tight enough that non-checkable prose ("the system is robust") matches nothing.
CLAIM_TEMPLATES: list[tuple[str, "re.Pattern[str]", Callable[[re.Match], Claim]]] = [
    (
        # Bump line FIRST: "X -> Y" / "X to Y" claims the POST-state Y. Listed
        # before version_pin so the migration arrow is consumed as a bump rather
        # than leaking the left-hand (old) version as a state-claim.
        "version_bump",
        re.compile(
            rf"\b{_VER}(?:\s*(?:->|→)\s*|\s+to\s+)[\"']?(?P<vb>{_VER})",
            re.IGNORECASE,
        ),
        _tmpl_version_bump,
    ),
    (
        # State-claim. Trailing guard rejects the left side of a migration
        # ("version 7.7.9 -> ..." must not bind 7.7.9); the bump template above
        # captures the post-state instead.
        "version_pin",
        re.compile(
            rf"(?:\bversion\s+(?:is\s+now\s+|is\s+|=\s*|equals\s+)?[\"']?(?P<v>{_VER})"
            rf"|styxx==(?P<v2>{_VER}))"
            rf"(?!\s*(?:->|→|to\s+\d))",
            re.IGNORECASE,
        ),
        _tmpl_version,
    ),
    (
        "git_tag",
        re.compile(r"\btag\s+[\"']?(?P<t>v?\d+\.\d+\.\d+)[\"']?\s+exists", re.IGNORECASE),
        _tmpl_tag,
    ),
    (
        "file_contains",
        re.compile(
            r"\b(?P<path>[\w./-]+\.[A-Za-z0-9]+)\s+contains\s+[\"'](?P<sub>[^\"']+)[\"']",
        ),
        _tmpl_file_contains,
    ),
    (
        "pdf_pages",
        re.compile(r"\b(?P<path>[\w./-]+\.pdf)\s+(?:has|is)\s+(?P<n>\d+)\s+pages?", re.IGNORECASE),
        _tmpl_pdf_pages,
    ),
]


def extract_claims(text: str, *, id_prefix: str = "X") -> ExtractionReport:
    """Extract deterministic, checkable Claims from agent free-text.

    Splits on sentence/line boundaries, applies each template in CLAIM_TEMPLATES,
    and emits one Claim per match with the checker and args bound explicitly.
    Claims are de-duplicated by (checker name, sorted args). IDs are assigned
    ``{id_prefix}1``, ``{id_prefix}2``, ... in document order.

    Returns an ExtractionReport with the claims plus honest coverage stats.
    Sentences with no checkable assertion produce no claims (by design — the
    closed template set IS the construct ceiling).
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    seen: set[tuple[str, tuple]] = set()
    claims: list[Claim] = []
    matched_sentences = 0
    for s in sentences:
        sentence_hit = False
        for name, rx, builder in CLAIM_TEMPLATES:
            for m in rx.finditer(s):
                claim = builder(m)
                key = (claim.checker.__name__, tuple(sorted(claim.args.items())))
                if key in seen:
                    continue
                seen.add(key)
                claims.append(claim)
                sentence_hit = True
        if sentence_hit:
            matched_sentences += 1
    for i, c in enumerate(claims, start=1):
        c.id = f"{id_prefix}{i}"
    return ExtractionReport(
        claims=claims,
        sentences_total=len(sentences),
        sentences_matched=matched_sentences,
    )
