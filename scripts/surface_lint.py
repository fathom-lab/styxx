#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/surface_lint.py: a word lint over the text a stranger meets. It is not a claim check.

What it reads
-------------
The shipped surfaces of this repository (profile ``repo``): the root documents (README.md, action.yml,
pyproject.toml, SECURITY.md and the other root .md files except CHANGELOG.md), the Action's script and the
pre-commit hook definitions, and everything of a prose type under ``styxx/`` (the package the wheel
carries), ``sworn/``, ``integrations/``, ``hooks/``, ``docs/``, ``web/`` and ``packages/``. Records are not
surfaces and are not read: ``papers/``, ``CHANGELOG.md``, ``tests/``, ``conformance/``, ``release/``,
``zenodo/`` and the recorded pull request bodies and harness under ``web/gate/differential/``. Data files
(.json, .jsonl, .csv) are not read under ``repo``. Profile ``site`` reads the canonical site's source tree
(``index.html``, ``styxx.jsx``, ``src/``, ``public/``, JSON included, because the site serves it).
``--files`` reads the named files whole, for outward drafts (letters, release notes, Zenodo text).

What it reports
---------------
withdrawn  sentences the lab has withdrawn. The gate claim that the EXTERNAL-1 preregistration bound the
           lab to withdraw (papers/closed-model-frontier/PREREG_external1_aidev_2026_08_31.md, lines
           93-95), in the wordings the hook titles used; "Every claim machine-verified"; README's "if we ever
           lie about a diff, our own product fails our own build"; the false replacement "today none has";
           and the zero-false-accusation claim the README withdrew. A match with the withdrawal itself
           within WINDOW characters ("is not yet true of this instrument", or a form of "withdraw") is
           counted as allowed, not as a hit, so the sentence that withdraws a claim may quote it.
charter    the words the lab's charter keeps out of its public text. They are stored reversed, as
           scripts/zenodo_deposit_software_v7_49_0.py stores them, so this file does not spell them, and
           every excerpt this script prints masks them.
plan       the word-checkable part of "Must not be said" in papers/PLAN_the_next_level_2026_09_02.md,
           lines 127-131: "Nobody", "0.117", "Sworn makes agents honest", "the contract every agent's
           claims are checked against", and "wire" or "protocol" on a line that also names sworn (the
           plan bans calling sworn either; ``styxx.protocol`` is a module name, so the bare word alone is
           not reported). The ordinal and the self- compound the list also names are charter rules here.

What it cannot do
-----------------
It matches strings. It cannot tell a priority claim from an ordinal number, cannot see a claim phrased in
other words, and cannot check the two items on the plan's list that are not words: a bound-recall number
printed before the leg-2 panel runs, and any use of the sworn documents in this tree as evidence that the
format works. The lab measured that finding claims in prose fails: precision 0.23 (EXTERNAL-1), 0.16 held
out (V14), and 0.4211 on 38 sentences for the best structural detector (the plan above, lines 17-22). So a
clean run says that no listed string was found, and nothing about whether the text is true.

Exit status: 0 when nothing is reported (with ``--baseline``: no withdrawn hit and no count above the
baseline), 1 when something is, 2 on a usage or read error. Standard library only; read-only on every
source (``--rev`` reads git objects with ``git ls-tree`` and ``git cat-file``; ``--write-baseline`` is the
one write, to the path given).
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
WINDOW = 300                    # characters either side of a withdrawn match searched for the withdrawal
MAX_BYTES = 5_000_000           # a larger file is listed as skipped, never read
EXCERPT = 160                   # characters of the line printed around a match
GIT_TIMEOUT_S = 120

PROSE_EXTS = frozenset({".py", ".md", ".rst", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".cff", ".js", ".mjs",
                        ".cjs", ".ts", ".jsx", ".tsx", ".html", ".htm"})

PROFILES: Dict[str, dict] = {
    "repo": {
        "include": ["README.md", "action.yml", "pyproject.toml", "diffgate_action.py", ".pre-commit-hooks.yaml",
                    "SECURITY.md", "CONTRIBUTING.md", "CITATION.cff", "CODE_OF_CONDUCT.md", "POSITIONING.md",
                    "OATH_CONTRACT.md", "BOUNTY.md", "LEADERBOARD.md", "REPLICATIONS.md", "AUTOPILOT.md",
                    "PATENTS.md", "styxx/**", "sworn/**", "integrations/**", "hooks/**", "docs/**", "web/**",
                    "packages/**"],
        "exclude": ["web/gate/differential/**"],
        "exts": PROSE_EXTS,
    },
    "site": {
        "include": ["index.html", "styxx.jsx", "src/**", "public/**"],
        "exclude": [],
        "exts": PROSE_EXTS | {".json"},
    },
}
PRUNE = frozenset({".git", "node_modules", "__pycache__", "dist", "build", ".netlify", ".venv", "venv"})


def _r(s: str) -> str:
    return s[::-1]


# a word is bounded by anything but a letter, a digit or an underscore, so a code identifier that contains a listed
# word (a variable, a file stem) is not read as the word
_B, _E = r"(?<![a-z0-9_])", r"(?![a-z0-9_])"


def _word(reversed_word: str, plural: bool = False) -> str:
    return _B + re.escape(_r(reversed_word)) + ("s?" if plural else "") + _E


def _compound(reversed_words: str, sep: str, plural: bool = False) -> str:
    parts = re.split(r"[ \-]", _r(reversed_words))
    return _B + sep.join(re.escape(p) for p in parts) + ("s?" if plural else "") + _E


class Rule(NamedTuple):
    rid: str
    kind: str                   # withdrawn | charter | plan
    rx: "re.Pattern[str]"
    why: str


def _rule(rid: str, kind: str, pattern: str, why: str) -> Rule:
    return Rule(rid, kind, re.compile(pattern, re.IGNORECASE), why)


RULES: Tuple[Rule, ...] = (
    _rule("withdrawn/gate-claim", "withdrawn",
          r"can(?:not|'t|’t|\s+not)\s+lie\b|cannot\s+post\s+a\s+message\s+that\s+lies"
          r"|whose\s+message\s+lies\s+about\s+its\s+diff",
          "EXTERNAL-1's preregistration bound the lab to withdraw it (PREREG_external1_aidev_2026_08_31.md l.93-95)"),
    _rule("withdrawn/every-claim-verified", "withdrawn", r"every\s+claim\s+(?:is\s+)?machine[\s\-]verified",
          "the sworn census's coverage floor is 0.0156 to 0.2185 (PLAN_the_next_level_2026_09_02.md l.47-48)"),
    _rule("withdrawn/own-build", "withdrawn",
          r"if\s+we\s+ever\s+lie\s+about\s+a\s+diff|our\s+own\s+product\s+fails\s+our\s+own\s+build",
          "a statement about what the gate catches, which no receipt measures"),
    _rule("withdrawn/today-none-has", "withdrawn", r"today\s+none\s+has",
          "precision was measured for kinds that accuse: 0.23, 0.16 and 0.25"),
    _rule("withdrawn/zero-false-accusations", "withdrawn", r"zero[\s\-]+false[\s\-]+accusations?",
          "withdrawn in the README; on the plan's must-not-say list (l.128-129)"),
    _rule("charter/ordinal", "charter", _word("tsrif"), "the ordinal meaning number one"),
    _rule("charter/newness-adjective", "charter", _word("levon"), "a charter word"),
    _rule("charter/revolution-adjective", "charter", _word("yranoitulover"), "a charter word"),
    _rule("charter/ground-adjective", "charter", _word("gnikaerbdnuorg"), "a charter word"),
    _rule("charter/break-noun", "charter", _word("hguorhtkaerb", plural=True), "a charter word"),
    _rule("charter/tamper-compound", "charter", _compound("foorp-repmat", r"[\s\-]?"), "a charter word"),
    _rule("charter/self-compound", "charter", _compound("gniyfirev-fles", r"[\s\-]?"),
          "a charter word, and on the plan's must-not-say list (l.128)"),
    _rule("charter/halluc-compound", "charter", _compound("rotceted noitanicullah", r"[\s\-]+", plural=True),
          "the label the charter forbids for the product"),
    _rule("plan/nobody", "plan", _B + r"nobody" + _E, "plan l.127"),
    _rule("plan/0.117", "plan", r"(?<![0-9.])0\.117(?![0-9])", "plan l.129-130"),
    _rule("plan/sworn-honest", "plan", r"sworn\s+makes\s+agents\s+honest", "plan l.129"),
    _rule("plan/the-contract", "plan",
          r"the\s+contract\s+every\s+agent(?:'|’)?s\s+claims\s+are\s+checked\s+against", "plan l.127-128"),
)
SWORN_LINE_RID = "plan/wire-or-protocol-beside-sworn"
# a dotted module name (styxx.sworn, styxx.protocol) names code, not what sworn is, so it is not read as the word
_SWORN_RX = re.compile(r"(?<![a-z0-9_.])sworn" + _E, re.IGNORECASE)
_WIRE_RX = re.compile(r"(?<![a-z0-9_.])(?:wire|protocol)s?" + _E, re.IGNORECASE)
_WITHDRAWAL_RX = re.compile(r"not\s+yet\s+true\s+of\s+this\s+instrument|" + _B + r"withdr(?:aw|awn|aws|awal|awing|ew)"
                            + _E, re.IGNORECASE)
KINDS = ("withdrawn", "charter", "plan")
_MASK_RULES = tuple(r for r in RULES if r.kind == "charter")


class Hit(NamedTuple):
    path: str
    line: int
    col: int
    rid: str
    kind: str
    excerpt: str


# ---------------------------------------------------------------------------------------------------------
# the lint itself
# ---------------------------------------------------------------------------------------------------------

def _mask(line: str) -> str:
    """The line with every charter word replaced by its rule id, so no excerpt spells one."""
    spans: List[Tuple[int, int, str]] = []
    for rule in _MASK_RULES:
        for m in rule.rx.finditer(line):
            spans.append((m.start(), m.end(), rule.rid))
    spans.sort()
    out, at = [], 0
    for s, e, rid in spans:
        if s < at:
            continue
        out.append(line[at:s])
        out.append("[" + rid + "]")
        at = e
    out.append(line[at:])
    return "".join(out)


def _excerpt(line: str, col0: int) -> str:
    lo = max(0, col0 - EXCERPT // 2)
    piece = line[lo:lo + EXCERPT]
    # a cut can split a charter word in two halves that no longer match; widen the cut to whole tokens
    while lo > 0 and line[lo - 1].isalnum():
        lo -= 1
        piece = line[lo] + piece
    hi = lo + len(piece)
    while hi < len(line) and line[hi].isalnum():
        piece += line[hi]
        hi += 1
    text = _mask(piece).replace("\t", " ").strip()
    return ("..." if lo > 0 else "") + text + ("..." if hi < len(line) else "")


def lint_text(text: str, path: str = "<text>") -> Tuple[List[Hit], List[Hit]]:
    """Return (hits, allowed) for one text. CRLF is read as LF."""
    text = text.replace("\r\n", "\n")
    starts = [0] + [m.end() for m in re.finditer("\n", text)]
    lines = text.split("\n")
    hits: List[Hit] = []
    allowed: List[Hit] = []

    def at(pos: int, rule_id: str, kind: str) -> Hit:
        ln = bisect.bisect_right(starts, pos)
        col0 = pos - starts[ln - 1]
        return Hit(path, ln, col0 + 1, rule_id, kind, _excerpt(lines[ln - 1], col0))

    for rule in RULES:
        for m in rule.rx.finditer(text):
            h = at(m.start(), rule.rid, rule.kind)
            if rule.kind == "withdrawn" and _WITHDRAWAL_RX.search(
                    text, max(0, m.start() - WINDOW), min(len(text), m.end() + WINDOW)):
                allowed.append(h)
            else:
                hits.append(h)
    for i, line in enumerate(lines):
        if _SWORN_RX.search(line):
            for m in _WIRE_RX.finditer(line):
                hits.append(at(starts[i] + m.start(), SWORN_LINE_RID, "plan"))
    key = lambda h: (h.path, h.line, h.col, h.rid)  # noqa: E731
    return sorted(hits, key=key), sorted(allowed, key=key)


# ---------------------------------------------------------------------------------------------------------
# sources: a working tree, a git revision, or named files
# ---------------------------------------------------------------------------------------------------------

def _match(path: str, pattern: str) -> bool:
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return path == pattern


def selected(path: str, profile: dict) -> bool:
    if not any(_match(path, p) for p in profile["include"]):
        return False
    if any(_match(path, p) for p in profile["exclude"]):
        return False
    if any(part in PRUNE for part in path.split("/")[:-1]):
        return False
    name = path.rsplit("/", 1)[-1]
    suffix = os.path.splitext(name)[1].lower()
    if suffix:
        return suffix in profile["exts"]
    return "." not in name.lstrip(".")                     # an extensionless script such as a git hook


def _decode(data: bytes) -> Optional[str]:
    if b"\x00" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def tree_files(root: Path, profile: dict) -> Iterator[Tuple[str, Optional[str]]]:
    """(posix path, text or None when skipped) for each selected file under root."""
    tops = {p.split("/", 1)[0] for p in profile["include"]}
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = sorted(d for d in dirnames if d not in PRUNE and (rel_dir or d in tops))
        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if not selected(rel, profile):
                continue
            p = Path(dirpath) / name
            if p.stat().st_size > MAX_BYTES:
                yield rel, None
                continue
            yield rel, _decode(p.read_bytes())


def _git(root: Path, *args: str, data: Optional[bytes] = None) -> bytes:
    r = subprocess.run(["git", "-C", str(root), *args], input=data, capture_output=True, timeout=GIT_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.decode('utf-8', 'replace').strip()}")
    return r.stdout


def rev_files(root: Path, rev: str, profile: dict) -> Iterator[Tuple[str, Optional[str]]]:
    """The same selection read from git objects at rev; nothing in the working tree is read or written."""
    names = [n for n in _git(root, "ls-tree", "-r", "-z", "--name-only", rev).decode("utf-8").split("\0") if n]
    names = sorted(n for n in names if selected(n, profile))
    if not names:
        return
    out = _git(root, "cat-file", "--batch", data="".join(f"{rev}:{n}\n" for n in names).encode("utf-8"))
    pos = 0
    for n in names:
        nl = out.index(b"\n", pos)
        header = out[pos:nl].decode("utf-8", "replace").split()
        if len(header) < 3 or header[1] != "blob":
            raise RuntimeError(f"git cat-file did not return a blob for {rev}:{n}: {header}")
        size = int(header[2])
        body = out[nl + 1:nl + 1 + size]
        pos = nl + 1 + size + 1
        yield n, (None if size > MAX_BYTES else _decode(body))


def named_files(paths: Sequence[str]) -> Iterator[Tuple[str, Optional[str]]]:
    for p in paths:
        data = Path(p).read_bytes()
        yield Path(p).as_posix(), (None if len(data) > MAX_BYTES else _decode(data))


class Report(NamedTuple):
    files: int
    skipped: List[str]
    hits: List[Hit]
    allowed: List[Hit]


def run(files: Iterator[Tuple[str, Optional[str]]]) -> Report:
    n, skipped, hits, allowed = 0, [], [], []
    for path, text in files:
        if text is None:
            skipped.append(path)
            continue
        n += 1
        h, a = lint_text(text, path)
        hits.extend(h)
        allowed.extend(a)
    return Report(n, skipped, hits, allowed)


def counts(hits: Sequence[Hit]) -> Dict[str, Dict[str, int]]:
    """{path: {rule id: count}} for the rules a baseline may hold (charter and plan; never withdrawn)."""
    out: Dict[str, Dict[str, int]] = {}
    for h in hits:
        if h.kind == "withdrawn":
            continue
        out.setdefault(h.path, {}).setdefault(h.rid, 0)
        out[h.path][h.rid] += 1
    return {p: dict(sorted(v.items())) for p, v in sorted(out.items())}


def over_baseline(hits: Sequence[Hit], baseline: dict) -> List[str]:
    """Every withdrawn hit, and every (path, rule) whose count is above the baseline's."""
    problems = [f"{h.path}:{h.line}:{h.col} {h.rid} (a withdrawn sentence is never baselined)"
                for h in hits if h.kind == "withdrawn"]
    base = baseline.get("counts", {})
    for path, per in counts(hits).items():
        for rid, n in per.items():
            allowed_n = base.get(path, {}).get(rid, 0)
            if n > allowed_n:
                problems.append(f"{path} {rid}: {n} > baseline {allowed_n}")
    return problems


# ---------------------------------------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------------------------------------

def _summary(label: str, profile: str, rep: Report) -> List[str]:
    by_rule: Dict[str, int] = {}
    by_kind = {k: 0 for k in KINDS}
    for h in rep.hits:
        by_rule[h.rid] = by_rule.get(h.rid, 0) + 1
        by_kind[h.kind] += 1
    return [f"surface_lint {label} profile={profile}: {rep.files} files read, {len(rep.hits)} hits, "
            f"{len(rep.allowed)} allowed (withdrawal context), {len(rep.skipped)} skipped",
            "by kind: " + ", ".join(f"{k} {by_kind[k]}" for k in KINDS),
            "by rule: " + (", ".join(f"{r} {n}" for r, n in sorted(by_rule.items())) or "none")]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="surface_lint", description=(
        "A word lint over shipped surfaces: withdrawn sentences, the charter's words and the word-checkable part "
        "of the plan's must-not-say list. It finds words, not claims."))
    ap.add_argument("--root", default=str(ROOT), help="directory to read (default: this repository)")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="repo")
    ap.add_argument("--rev", help="read the selection from this git revision of --root instead of the working tree")
    ap.add_argument("--files", nargs="+", help="lint these files whole, with every rule (outward drafts)")
    ap.add_argument("--baseline", help="exit 0 unless a withdrawn sentence or a count above this baseline is found")
    ap.add_argument("--write-baseline", help="write the charter and plan counts here and exit")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--kinds", default=",".join(KINDS), help="comma-separated kinds to report (default: all)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    profile = PROFILES[a.profile]
    try:
        if a.files:
            label, rep = "files", run(named_files(a.files))
        elif a.rev:
            label, rep = f"{root.as_posix()}@{a.rev}", run(rev_files(root, a.rev, profile))
        else:
            if not root.is_dir():
                raise RuntimeError(f"not a directory: {root}")
            label, rep = root.as_posix(), run(tree_files(root, profile))
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as e:
        print(f"surface_lint: {e}", file=sys.stderr)
        return 2

    kinds = {k.strip() for k in a.kinds.split(",") if k.strip()}
    rep = Report(rep.files, rep.skipped, [h for h in rep.hits if h.kind in kinds],
                 [h for h in rep.allowed if h.kind in kinds])

    if a.write_baseline:
        doc = {"what": "surface_lint.py baseline: charter and plan counts per shipped file, profile repo. A count "
                       "may fall and never rise; a withdrawn sentence is never baselined. Regenerate with "
                       "python scripts/surface_lint.py --write-baseline scripts/surface_lint_baseline.json "
                       "only when a count falls.",
               "profile": a.profile, "counts": counts(rep.hits)}
        Path(a.write_baseline).write_text(json.dumps(doc, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
        print(f"surface_lint: wrote {sum(sum(v.values()) for v in doc['counts'].values())} counted hits over "
              f"{len(doc['counts'])} files to {a.write_baseline}")
        return 0

    problems: Optional[List[str]] = None
    if a.baseline:
        problems = over_baseline(rep.hits, json.loads(Path(a.baseline).read_text(encoding="utf-8")))

    if a.json:
        print(json.dumps({"source": label, "profile": a.profile, "files_read": rep.files, "skipped": rep.skipped,
                          "hits": [h._asdict() for h in rep.hits], "allowed": [h._asdict() for h in rep.allowed],
                          "over_baseline": problems}, indent=1, ensure_ascii=True))
    else:
        out = _summary(label, a.profile, rep)
        out += [f"HIT     {h.path}:{h.line}:{h.col}  {h.rid}  | {h.excerpt}" for h in rep.hits]
        out += [f"ALLOWED {h.path}:{h.line}:{h.col}  {h.rid}  | {h.excerpt}" for h in rep.allowed]
        out += [f"SKIPPED {p} (over {MAX_BYTES} bytes or binary)" for p in rep.skipped]
        if problems is not None:
            out.append(f"over baseline: {len(problems)}")
            out += [f"OVER    {p}" for p in problems]
        sys.stdout.write("\n".join(out) + "\n")

    if problems is not None:
        return 1 if problems else 0
    return 1 if rep.hits else 0


if __name__ == "__main__":
    sys.exit(main())
