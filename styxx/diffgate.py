# -*- coding: utf-8 -*-
"""styxx.diffgate — the zero-receipt gate: an agent's summary cannot lie about its diff.

The wedge the trust stack points at UNMODIFIED agent work. No receipts, no preregs, no
cooperation from the agent that wrote the summary: given the text an agent shipped (a PR
body, a commit message, a session report) and the git range it describes, extract every
diff-shaped claim and verify it against what `git diff` actually says. One command, exit
0/1 — a CI gate that catches the quiet lie in "updated the tests" when the diff touched no
test, "only touches docs/" when it edited source, "adds function retry" when it doesn't.

Construct ceiling, stated like every styxx instrument states it: the template set is
CLOSED (touched/created/deleted paths · files-changed counts · added tests · added
functions/classes · insertion/deletion counts · only-touches prefixes · tests-pass with
--run). Prose outside the templates is NOT judged — uncovered sentences are listed as
uncovered, never scored. Verdicts per claim: VERIFIED / CONTRADICTED / UNCHECKABLE(named).
The gate fails on any CONTRADICTED; UNCHECKABLE fails only under --strict.

CLI::

    python -m styxx.diffgate SUMMARY.md --repo . --base main --head HEAD \
        [--run "pytest -q"] [--strict] [--out GATE.json]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["gate_diff", "DiffGate", "DiffClaim"]

# A claimed path must end in a KNOWN file extension (closed set — the same honesty as the
# template set itself). The naive any-dotted-token form false-accused decimals (0.5349),
# versions (v0.6.2), DOIs, and module names (styxx.framelocality) in the 80-commit
# history sweep that validated this module — six of six contradictions were false
# accusations until this whitelist existed. A gate that can accuse a number of not being
# a file does not ship.
_EXT = (r"py|md|json|jsonl|txt|yml|yaml|toml|cfg|ini|js|ts|tsx|jsx|css|html|tex|sh|ps1|"
        r"bat|ipynb|csv|tsv|npz|npy|pdf|png|jpg|svg|gz|zip|lock|xml|rst|c|h|cpp|rs|go|java")
_PATH = rf"[\w./\\-]*[A-Za-z_][\w-]*\.(?:{_EXT})\b"
# a window between the verb and the path: real summaries write "updated the parser in
# styxx/certify.py", not "updated styxx/certify.py". Bounded (no sentence crossing — the
# splitter already scoped us to one sentence) and path-shaped on the right.
_W = r"[^.!?\n]{0,60}?"
_TEMPLATES = [
    ("file_created", re.compile(
        rf"\b(?:creat\w+|new)\s+(?:file|module|script|test file)?\s*{_W}[`\"']?(?P<path>{_PATH})[`\"']?",
        re.I)),
    ("file_created", re.compile(
        rf"[`\"']?(?P<path>{_PATH})[`\"']?\s*(?::|—|--)\s*(?:new|added|created)\b", re.I)),
    ("file_deleted", re.compile(
        rf"\b(?:delet\w+|remov\w+)\s+(?:the\s+file\s+)?{_W}[`\"']?(?P<path>{_PATH})[`\"']?", re.I)),
    ("file_touched", re.compile(
        rf"\b(?:modif\w+|updat\w+|edit\w+|chang\w+|refactor\w+|fix\w+|add\w+|extend\w+|"
        rf"hard\w+|wir\w+|patch\w+)\s+{_W}[`\"']?(?P<path>{_PATH})[`\"']?", re.I)),
    ("file_touched", re.compile(
        rf"^[\s*-]*[`\"']?(?P<path>{_PATH})[`\"']?\s*(?::|—|--)\s+", re.M)),
    ("files_changed_count", re.compile(r"\b(?P<n>\d+)\s+files?\s+(?:were\s+)?changed", re.I)),
    ("tests_added", re.compile(r"\b(?:add\w+|creat\w+)\s+(?P<n>\d+)\s+(?:new\s+)?tests?\b", re.I)),
    ("symbol_added", re.compile(
        r"\b(?:add\w+|introduc\w+)\s+(?:a\s+|the\s+)?(?P<kind>function|class|method)\s+"
        r"[`\"']?(?P<name>[A-Za-z_]\w*)", re.I)),
    ("only_touches", re.compile(
        r"\bonly\s+(?:touch\w+|modif\w+|chang\w+)\s+(?:files?\s+(?:in|under)\s+)?"
        r"[`\"']?(?P<prefix>[\w./\\-]+)[`\"']?", re.I)),
    ("tests_pass", re.compile(r"\b(?:all\s+)?tests\s+(?:pass|are\s+passing|green)\b", re.I)),
]


@dataclass
class DiffClaim:
    kind: str
    text: str
    detail: dict
    verdict: str = "UNCHECKABLE"       # VERIFIED | CONTRADICTED | UNCHECKABLE
    why: str = ""


@dataclass
class DiffGate:
    verdict: str                        # PASS | FAIL
    base: str
    head: str
    claims: list = field(default_factory=list)
    uncovered_sentences: int = 0

    def to_dict(self):
        return {"diffgate": "v0", "verdict": self.verdict, "base": self.base,
                "head": self.head,
                "claims": [c.__dict__ for c in self.claims],
                "uncovered_sentences": self.uncovered_sentences}


def _git(repo, *args) -> str:
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:200]}")
    return r.stdout


def _norm(p: str) -> str:
    return p.replace("\\", "/").lstrip("./").lower()


def gate_diff(summary_text: str, repo: str | Path, base: str, head: str,
              run: str | None = None, strict: bool = False) -> DiffGate:
    """Extract diff-shaped claims from *summary_text* and verify against base..head."""
    repo = Path(repo)
    name_status = _git(repo, "diff", "--name-status", f"{base}..{head}")
    status: dict[str, str] = {}
    for line in name_status.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            st, path = parts[0][:1], parts[-1]
            status[_norm(path)] = st            # A / M / D / R
    added_lines = [l[1:] for l in _git(repo, "diff", f"{base}..{head}").splitlines()
                   if l.startswith("+") and not l.startswith("+++")]
    added_blob = "\n".join(added_lines)

    def find_path(claimed: str):
        c = _norm(claimed)
        for p, st in status.items():
            if p == c or p.endswith("/" + c) or Path(p).name == Path(c).name:
                return p, st
        return None, None

    claims: list[DiffClaim] = []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", summary_text)
    covered = set()
    for si, sent in enumerate(sentences):
        for kind, rx in _TEMPLATES:
            for m in rx.finditer(sent):
                covered.add(si)
                d = {k: v for k, v in m.groupdict().items() if v is not None}
                c = DiffClaim(kind=kind, text=sent.strip()[:160], detail=d)
                if kind in ("file_created", "file_deleted", "file_touched"):
                    p, st = find_path(d["path"])
                    want = {"file_created": "A", "file_deleted": "D"}.get(kind)
                    if p is None:
                        c.verdict, c.why = "CONTRADICTED", \
                            f"{d['path']!r} does not appear in the diff at all"
                    elif want and st != want:
                        c.verdict, c.why = "CONTRADICTED", \
                            f"{d['path']!r} is status {st!r} in the diff, claim wants {want!r}"
                    else:
                        c.verdict, c.why = "VERIFIED", f"diff status {st!r} for {p!r}"
                elif kind == "files_changed_count":
                    n = int(d["n"])
                    c.verdict = "VERIFIED" if n == len(status) else "CONTRADICTED"
                    c.why = f"diff changes {len(status)} files, claim says {n}"
                elif kind == "tests_added":
                    n = int(d["n"])
                    got = len(re.findall(r"^\s*def test_", added_blob, re.M))
                    c.verdict = "VERIFIED" if got == n else "CONTRADICTED"
                    c.why = f"diff adds {got} test functions, claim says {n}"
                elif kind == "symbol_added":
                    pat = (r"^\s*(?:def|class)\s+" + re.escape(d["name"]) + r"\b")
                    hit = bool(re.search(pat, added_blob, re.M))
                    c.verdict = "VERIFIED" if hit else "CONTRADICTED"
                    c.why = (f"added lines {'do' if hit else 'do NOT'} define "
                             f"{d['kind']} {d['name']!r}")
                elif kind == "only_touches":
                    pref = _norm(d["prefix"]).rstrip("/")
                    outside = [p for p in status if not p.startswith(pref + "/")
                               and p != pref]
                    c.verdict = "VERIFIED" if not outside else "CONTRADICTED"
                    c.why = ("all changed paths under prefix" if not outside else
                             f"paths outside {pref!r}: {outside[:3]}")
                elif kind == "tests_pass":
                    if run:
                        r = subprocess.run(run, shell=True, cwd=repo,
                                           capture_output=True, text=True, timeout=1800)
                        c.verdict = "VERIFIED" if r.returncode == 0 else "CONTRADICTED"
                        c.why = f"--run {run!r} exited {r.returncode}"
                    else:
                        c.verdict, c.why = "UNCHECKABLE", \
                            "no --run command supplied; the gate does not take the " \
                            "agent's word for test results"
                claims.append(c)

    contradicted = any(c.verdict == "CONTRADICTED" for c in claims)
    uncheckable = any(c.verdict == "UNCHECKABLE" for c in claims)
    verdict = "FAIL" if (contradicted or (strict and uncheckable)) else "PASS"
    uncovered = sum(1 for i, s in enumerate(sentences) if s.strip() and i not in covered)
    return DiffGate(verdict=verdict, base=base, head=head, claims=claims,
                    uncovered_sentences=uncovered)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="styxx.diffgate",
                                 description="An agent's summary cannot lie about its diff.")
    ap.add_argument("summary")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--run", default=None)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    text = Path(a.summary).read_text(encoding="utf-8")
    g = gate_diff(text, a.repo, a.base, a.head, run=a.run, strict=a.strict)
    if a.out:
        Path(a.out).write_text(json.dumps(g.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"{g.verdict}  claims={len(g.claims)} "
          f"contradicted={sum(1 for c in g.claims if c.verdict == 'CONTRADICTED')} "
          f"uncheckable={sum(1 for c in g.claims if c.verdict == 'UNCHECKABLE')} "
          f"uncovered_sentences={g.uncovered_sentences}")
    for c in g.claims:
        if c.verdict != "VERIFIED":
            print(f"  [{c.verdict}:{c.kind}] {c.why}")
    return 0 if g.verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
