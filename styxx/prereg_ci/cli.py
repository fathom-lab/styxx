# -*- coding: utf-8 -*-
"""styxx.prereg_ci.cli: the commands, ``python -m styxx.prereg_ci <verb>`` (DESIGN 3.3).

    lint    PREREG.md [--rev R]                       before data: grammar, totality, vacuity, licences
    seal    PREREG.md [--commit C] [--append]         a seals manifest line
    seal    AMENDMENT.md --amends ID --seen LEVEL [--append]
    stamp   PREREG.md --rfc3161 TSA_URL | --ots [--go]  the one networked command; refuses without --go
    check   --base BASE [--head HEAD]                 the merge gate (DESIGN 3.4, C0 to C9)
    ladder  RECEIPT.json --prereg P.md [--rev R]      the printout alone, for one receipt
    census  [--rev REV]                               every receipt at a revision re-derived

Exit codes (DESIGN 3.3, the audit-claims contract plus one): 0 PASS (or nothing to check), 1 FAIL, 2
an input error, 3 UNCHECKABLE (a shallow clone, a missing commit, git that does not answer, a cover past
its budget, ``openssl`` absent under ``require_time = T3a``). When one run reads a refusal and also an
input error or an unchecked reading, the refusal decides the code (1) and the report lists the rest.
A ``--json`` or ``--summary`` path that does not write is an input error (2, or 1 when the reading
refused something). An exception none of these readings names is no verdict: it prints ``ERROR`` and
the exception, and exits 4 (the Action's ERROR), never 1, the code Python gives an uncaught one, which
would read as FAIL.
``seal`` and ``stamp`` decide nothing, so every refusal of either that is not UNCHECKABLE is an input
error. Step B2 built ``seal``, step B6 ``stamp``; step B7 adds the rest and the report
(``styxx.prereg_ci.report``). The verbs are this module's, run as ``python -m styxx.prereg_ci``, as
``styxx.sworn``, ``styxx.seal`` and ``styxx.charon`` run theirs; ``styxx/cli.py`` registers none of them
(PLAN 2). Inside a checkout that holds styxx itself, ``python -I -m styxx.prereg_ci`` runs the installed
checker: under plain ``-m`` the working directory heads ``sys.path``, so the checkout's own
``styxx/`` would be imported (the Action runs both its python steps with ``-I`` for that reason).

``check --base BASE --head HEAD`` reads the pull request as git's three-dot diff reads it (from the
merge base) and prints, for every receipt the pull request writes or a RESULT it changes names (C1,
PLAN K11): the seal (the manifest line, else the landing rule, C2), the order commit against R, the
commit that added the receipt (C3, ``SEAL_NOT_BEFORE_RECEIPT``), the time ladder (3.5, with the trust
store read at BASE so a branch cannot pin the certificate that verifies its own token), the decision in
effect (C4), the re-score from the sealed bytes (C5: the gates table, or the locked scorer run at its
sealed blobs), the comparison with what the receipt records and every RESULT declares (C6), the
outcome ladder and the single-gate flip table (C7), the licences of the landed row (C8), and what was
not examined (C9). Beside the pairs: sealed decisions moved (C0), receipts rewritten, the manifest's
own rules, and the licences of every receipt already in HEAD's tree (C8: "every later check scans the
shipped surfaces", so a surface a later pull request adds is refused when it is added).

Policies are read from ``prereg-ci.toml`` at the work tree's top as BASE holds it, so a pull request
does not relax the policy that judges it (a change to the file is printed as ``CONFIG_CHANGED``), and a
flag wins over the file. A pull request that changes the file has HEAD's copy read too, and one that
does not read is ``CONFIG_INVALID`` (exit 2) on that pull request, before it lands and locks every
later check; a BASE copy that does not read is exit 2 on every pull request but the one that replaces
it with a copy that reads (or deletes it), which is judged under the flags and the defaults, as a
repository with no config is, and prints ``CONFIG_REPAIRED``. The keys: ``immutability`` (gates |
file), ``amend_after_data`` (report | fail), ``receipt_rewrite`` (fail | note), ``require_time`` (T1 |
T3a), ``preregistered_word`` (note | fail), ``exclude`` (repository-relative path prefixes C1 and
the licence scan leave out), ``surfaces`` and
``surfaces_exclude`` (the globs of the shipped surfaces; default: surface_lint's ``repo`` profile, PLAN
K14). PLAN K8: the file is read with ``tomllib`` where the interpreter has it, else with
``_toml_subset``, a strict reader of what the format uses (bare keys, strings, booleans, arrays of
strings, comments); anything else is ``CONFIG_INVALID``, exit 2. In a glob, ``*`` stays inside one
directory and ``**`` crosses them, and a leading ``**/`` matches at least one directory (so
``**/*.md`` does not select ``README.md``); a path under a build or cache directory (``build/``,
``dist/``, ``node_modules/`` and the like) is never a surface.

``lint PREREG.md`` reads the draft from the file system and the surfaces and the config at ``--rev``
(HEAD by default); ``ladder RECEIPT --prereg P`` reads the receipt at ``--rev`` and the PREREG from the
file system, and prints its reading without a seal, an order or a time; ``census`` re-derives every
record that names a gates digest (R1's rule) and every receipt that names a PREREG (3.2(d)) at
``--rev`` under the landing rule of that revision's chain, and prints a tally. Each of lint, check,
ladder and census writes the JSON report with ``--json PATH``; check appends the markdown job summary to
``--summary PATH`` (the Action passes ``$GITHUB_STEP_SUMMARY``).

``stamp PATH --rfc3161 TSA_URL | --ots [--commit C] [--go]`` is the one networked command: it sends
the sha256 of the blob at the commit (HEAD by default; never the working copy) to an RFC 3161 TSA, or
hands the blob to the OpenTimestamps client, and writes ``PATH.tsr`` or ``PATH.ots`` beside the file.
Without ``--go`` it prints what it would send where and writes nothing, exit 2
(``styxx.prereg_ci.time``).

``seal PATH [--commit C] [--append]`` prints the bytes the seal holds, their sha256 (the digest
``git show <commit>:<path> | sha256sum`` gives), the decision digest, the landing commit L, the add
commit and the order commit, and the manifest line (DESIGN 3.2(b)); the commit is L on HEAD's
parent-one chain unless ``--commit`` names one in HEAD's history. ``seal PATH --amends ID|PATH --seen
LEVEL [--append]`` prints an amendment line: ``--amends`` names a manifest line by id, or a path,
which names that path's newest line that is not withdrawn, or, where the path has none, gets its
default-seal line written before the amendment (PLAN K9); a line that records bytes another line
already records is printed with a note naming it. ``--seen`` on a prereg line states what had been
seen when it was sealed (reproducible-science's freeze records an access level too) and is printed
beside it. A prereg line seals the decision the PREREG landed with on HEAD's parent-one chain:
``--commit`` naming bytes that hold another decision is refused (``MANIFEST_SEAL_NOT_LANDED``,
which check reads too), since a changed decision is an amendment line.
``--append`` appends the line(s) to the manifest's working copy (``prereg-ci.seals.jsonl`` at the
work tree's top), LF, no byte-order mark, after checking that the working copy is HEAD's manifest
with lines appended and that the whole still parses; without it nothing is written. The manifest
is read from blobs everywhere else, so a working copy that a checkout holds with CRLF is read here
with CRLF as LF and its bytes are left as they are.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from styxx.prereg_ci import core as _core
from styxx.prereg_ci import git as _git
from styxx.prereg_ci import report as _report
from styxx.prereg_ci import scorer_run as _scorer_run
from styxx.prereg_ci import time as _time

__all__ = ["main"]

_ADVISORY = ("sealed bytes are read as git stores them; a '-text' line in .gitattributes for a sealed path "
             "keeps a checkout's copy equal to them (not written by this command)")


class _Input(Exception):
    """An argument this command does not take (exit 2)."""


def _repo_path(arg: str, top: str) -> str:
    """*arg*, a path as the caller typed it (relative to the working directory), as a path relative
    to the work tree's top, with ``/``; a path outside the work tree is refused."""
    absolute = os.path.realpath(os.path.join(os.getcwd(), arg))
    root = os.path.realpath(top)
    try:
        rel = os.path.relpath(absolute, root)
    except ValueError:
        raise _Input(f"{arg} is outside the work tree")
    if rel == os.curdir or rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        raise _Input(f"{arg} is outside the work tree")
    return rel.replace(os.sep, "/")


def _say(line: Dict[str, Any], seal: Dict[str, Any]) -> None:
    decision = (f"{seal['form']} {seal['gates_sha256']}" if seal["form"] is not None
                else f"no decision the scanner reads ({seal['code']})")
    if line["amends"]:
        said = f", amends {line['amends']}, {line['decision']}, seen: {line['seen']}"
    else:
        said = "" if line["seen"] is None else f", seen: {line['seen']}"
    print(f"seal {line['path']} (line {line['id']}, {line['kind']}{said})")
    print(f"  bytes {len(seal['bytes'])} at {line['commit']}")
    print(f"  blob sha256 {line['blob_sha256']}")
    print(f"  decision: {decision}")
    landing = seal["landing"] or "none on the parent-one chain of HEAD"
    print(f"  landing commit {landing}; add commit {seal['add'] or 'none'}; order commit {seal['order'] or 'none'}")
    print(_core._manifest_bytes([line]).decode("ascii"), end="")


def _seal(args: argparse.Namespace) -> int:
    with _git._Repo(os.getcwd()) as repo:
        top = repo._where()
        rel = _repo_path(args.path, top)
        man = args.manifest
        problem = _core._path_problem(man)
        if problem:
            raise _Input(f"--manifest {man!r} {problem}")
        head = repo.commit("HEAD")
        wc = os.path.join(top, *man.split("/"))
        existing = b""
        if os.path.exists(wc):
            with open(wc, "rb") as f:
                existing = f.read()
        lf = existing.replace(b"\r\n", b"\n")
        try:
            committed = repo.read(head, man)
        except _git._Refusal as e:
            if e.code != "NOT_A_BLOB":
                raise
            raise _Input(f"{man} at HEAD is not a regular file")
        if committed is not None and not lf.startswith(committed):
            raise _Input(f"the working copy of {man} is not HEAD's {man} with lines appended; restore it "
                         f"(git checkout -- {man}), then seal")
        entries = _git._manifest_parse(lf, man, "the working copy")
        if args.amends is not None and args.seen is None:
            raise _git._Refusal("MANIFEST_SEEN_MISSING", "an amendment line states what had been seen: name --seen, "
                                                         "one of " + ", ".join(repr(s) for s in _core._SEEN))
        amends = args.amends
        if amends is not None and amends not in {e["id"] for e in entries}:
            amends = _repo_path(amends, top)
        new = _git._seal_lines(repo, entries, rel, head, commit=args.commit, amends=amends, seen=args.seen)
        appended = _core._manifest_bytes(new)
        _git._manifest_parse(lf + appended, man, "the working copy with the new lines")
        notes = _git._seal_notes(repo, entries, new, head)
        for line in new:
            _say(line, _git._line_seal(repo, line, head, head))
            for _, note in (n for n in notes if n[0] == line["id"]):
                print(f"  note: {note}")
        if not args.append:
            print(f"  not written: --append appends the line{'s' if len(new) > 1 else ''} to {man}")
        else:
            if existing and not existing.endswith(b"\n"):
                raise _Input(f"the working copy of {man} does not end with a newline")
            with open(wc, "ab") as f:
                f.write(appended)
            with open(wc, "rb") as f:
                back = f.read()
            if back != existing + appended:
                print(f"REFUSED {man} did not read back as written")
                return 2
            print(f"  appended {len(new)} line{'s' if len(new) > 1 else ''} to {man}")
        print(f"  advisory: {_ADVISORY}")
    return 0


def _stamp(args: argparse.Namespace) -> int:
    """``stamp PATH --rfc3161 TSA_URL | --ots [--commit C] [--go]``: the one networked command (DESIGN
    3.3). Without ``--go`` it prints what it would send where and writes nothing (exit 2); the blob at
    the commit is stamped, never the working copy."""
    if bool(args.rfc3161) == bool(args.ots):
        raise _Input("name exactly one of --rfc3161 TSA_URL and --ots")
    with _git._Repo(os.getcwd()) as repo:
        top = repo._where()
        rel = _repo_path(args.path, top)
        try:
            r = _time._stamp(repo, rel, rfc3161=args.rfc3161, ots=args.ots, commit=args.commit, go=args.go)
        except _time._StampError as e:
            raise _Input(str(e))
    print(f"stamp {r['path']} at {r['commit']}: sha256 {r['sha256']} ({r['bytes']} bytes of the blob)")
    print(f"  to {r['to']}; writes {r['writes']}")
    if r["status"] != "WRITTEN":
        print(f"REFUSED {r['detail']}")
        return 2
    print(f"  {r['detail']}")
    if r.get("gen_time"):
        print(f"  time {r['gen_time']} by {r.get('tsa') or 'a TSA that gave no name'} (policy {r.get('policy')}, "
              f"serial {r.get('serial')})")
    return 0


class _Coded(Exception):
    """A refusal with a code of the commands' own (``report._CHECK_KIND``), raised before anything is read."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.kind, self.detail = code, _report._kinds()[code], detail


# ---- the config (PLAN K8) ---------------------------------------------------------------------------------

_CONFIG = "prereg-ci.toml"
_POLICY_VALUES = {"immutability": _git._IMMUTABILITY, "amend_after_data": _core._AMEND_AFTER_DATA,
                  "receipt_rewrite": _git._POLICIES, "require_time": _time._REQUIRE,
                  "preregistered_word": _time._WORD_POLICIES}
_DEFAULTS = {"immutability": "gates", "amend_after_data": "report", "receipt_rewrite": "fail", "require_time": "T1",
             "preregistered_word": "note"}
_LISTS = ("exclude", "surfaces", "surfaces_exclude")
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
_ESCAPES = {"b": "\b", "t": "\t", "n": "\n", "f": "\f", "r": "\r", '"': '"', "\\": "\\"}


def _toml_subset(text: str) -> Dict[str, Any]:
    """The subset of TOML a ``prereg-ci.toml`` uses, read strictly: ``key = value`` lines with a bare
    key, where a value is a basic string (``"..."`` with TOML's escapes), a literal string (``'...'``),
    ``true``, ``false``, or an array of strings that may span lines and hold comments and a trailing
    comma; ``#`` comments; blank lines. A table, a dotted or quoted key, a number, a date, an inline
    table, a multi-line string, a nested array and a key given twice are refused (``ValueError``), as is
    anything after a value but a comment. Where it reads a text, it reads it as ``tomllib.loads`` does
    (``tests/test_prereg_ci_cli.py`` holds the two to each other on every example)."""
    out: Dict[str, Any] = {}
    n = len(text)
    pos = 0
    line = 1

    def fail(why: str) -> ValueError:
        return ValueError(f"line {line}: {why}")

    def blanks(i: int) -> int:
        while i < n and text[i] in " \t":
            i += 1
        return i

    def comment(i: int) -> int:
        if i < n and text[i] == "#":
            while i < n and text[i] not in "\r\n":
                if (ord(text[i]) < 32 and text[i] != "\t") or text[i] == "\x7f":
                    raise fail("a control character in a comment")
                i += 1
        return i

    def newline(i: int) -> int:
        nonlocal line
        if text.startswith("\r\n", i):
            line += 1
            return i + 2
        if i < n and text[i] == "\n":
            line += 1
            return i + 1
        if i < n and text[i] == "\r":
            raise fail("a carriage return that ends no line")
        return i

    def string(i: int) -> Tuple[str, int]:
        quote = text[i]
        i += 1
        buf: List[str] = []
        while True:
            if i >= n or text[i] in "\r\n":
                raise fail("a string that does not close on its line")
            ch = text[i]
            if ch == quote:
                return "".join(buf), i + 1
            if (ord(ch) < 32 and ch != "\t") or ch == "\x7f":
                raise fail("a control character in a string")
            if ch == "\\" and quote == '"':
                esc = text[i + 1:i + 2]
                if esc in _ESCAPES:
                    buf.append(_ESCAPES[esc])
                    i += 2
                    continue
                width = {"u": 4, "U": 8}.get(esc)
                digits = text[i + 2:i + 2 + width] if width else ""
                if not width or len(digits) != width or not re.fullmatch(r"[0-9A-Fa-f]+", digits):
                    raise fail(f"an escape TOML does not name: \\{esc}")
                code = int(digits, 16)
                if code > 0x10FFFF or 0xD800 <= code <= 0xDFFF:
                    raise fail(f"\\{esc}{digits} is not a Unicode scalar value")
                buf.append(chr(code))
                i += 2 + width
                continue
            buf.append(ch)
            i += 1

    def gap(i: int) -> int:
        """Blanks, comments and line ends inside an array."""
        while True:
            j = newline(comment(blanks(i)))
            if j == i:
                return i
            i = j

    def value(i: int) -> Tuple[Any, int]:
        if i < n and text[i] in "\"'":
            if text.startswith('"""', i) or text.startswith("'''", i):
                raise fail("a multi-line string")
            return string(i)
        for word, val in (("true", True), ("false", False)):
            if text.startswith(word, i) and (i + len(word) == n or text[i + len(word)] in " \t\r\n#"):
                return val, i + len(word)
        if i < n and text[i] == "[":
            items: List[str] = []
            i = gap(i + 1)
            while True:
                if i < n and text[i] == "]":
                    return items, i + 1
                if i >= n or text[i] not in "\"'":
                    raise fail("an array holds strings alone")
                if text.startswith('"""', i) or text.startswith("'''", i):
                    raise fail("a multi-line string")
                item, i = string(i)
                items.append(item)
                i = gap(i)
                if i < n and text[i] == ",":
                    i = gap(i + 1)
                elif not (i < n and text[i] == "]"):
                    raise fail("an array's strings are separated by commas and the array closes with ]")
        raise fail("a value is a string, true, false, or an array of strings")

    while True:
        pos = comment(blanks(pos))
        if pos >= n:
            return out
        nxt = newline(pos)
        if nxt != pos:
            pos = nxt
            continue
        m = _BARE_KEY.match(text, pos)
        if m is None:
            raise fail("a line is a bare key = a value, a comment, or blank (no table, no quoted key)")
        key = m.group(0)
        pos = blanks(m.end())
        if pos >= n or text[pos] != "=":
            raise fail(f"{key!r} is not followed by = (a dotted key is not read)")
        val, pos = value(blanks(pos + 1))
        if key in out:
            raise fail(f"{key!r} is given twice")
        out[key] = val
        pos = comment(blanks(pos))
        if pos < n:
            nxt = newline(pos)
            if nxt == pos:
                raise fail(f"after the value of {key!r} only a comment may follow on its line")
            pos = nxt


def _toml(text: str) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:                               # 3.9 and 3.10 (pyproject.toml's floor is 3.9)
        return _toml_subset(text)
    return tomllib.loads(text)


def _config_read(repo: _git._Repo, rev: str) -> Tuple[Dict[str, Any], bool]:
    """``prereg-ci.toml`` at *rev*, read and held to its keys, or ``CONFIG_INVALID``."""
    try:
        data = repo.read(rev, _CONFIG)
    except _git._Refusal as e:
        if e.code != "NOT_A_BLOB":
            raise
        raise _Coded("CONFIG_INVALID", f"{_CONFIG} at {rev[:12]} is not a regular file")
    if data is None:
        return {}, False
    try:
        text = data.decode("utf-8")
        if text.startswith("\ufeff"):
            raise ValueError("it begins with a byte-order mark")
        obj = _toml(text)
    except (UnicodeDecodeError, ValueError) as e:
        raise _Coded("CONFIG_INVALID", f"{_CONFIG} at {rev[:12]} does not read: {e}")
    for key, val in obj.items():
        if key in _POLICY_VALUES:
            if not isinstance(val, str) or val not in _POLICY_VALUES[key]:
                raise _Coded("CONFIG_INVALID", f"{_CONFIG} at {rev[:12]}: {key} is one of "
                                               f"{list(_POLICY_VALUES[key])}, not {val!r}")
        elif key in _LISTS:
            if not isinstance(val, list) or not all(isinstance(v, str) and v for v in val):
                raise _Coded("CONFIG_INVALID", f"{_CONFIG} at {rev[:12]}: {key} is a list of non-empty strings, "
                                               f"not {val!r}")
        else:
            raise _Coded("CONFIG_INVALID", f"{_CONFIG} at {rev[:12]} names {key!r}, which the format does not "
                                           f"name ({', '.join([*_POLICY_VALUES, *_LISTS])})")
    return obj, True


def _policies(repo: _git._Repo, rev: str, args: argparse.Namespace,
              read: Optional[Tuple[Dict[str, Any], bool]] = None) -> Dict[str, Any]:
    """The policies in force: a flag, else ``prereg-ci.toml`` at *rev* (or *read*, the file as the
    caller read it), else the design's default (DESIGN 9: immutability gates, amend_after_data report,
    receipt_rewrite fail, require_time T1, preregistered_word note). ``profile`` is the surface profile
    the licence scan reads."""
    cfg, present = _config_read(repo, rev) if read is None else read
    pol: Dict[str, Any] = {"source": {}}
    for key, default in _DEFAULTS.items():
        flag = getattr(args, key, None)
        if flag is not None:
            pol[key], pol["source"][key] = flag, "flag"
        elif key in cfg:
            pol[key], pol["source"][key] = cfg[key], "config"
        else:
            pol[key], pol["source"][key] = default, "default"
    flag_ex = [x for x in (getattr(args, "exclude", None) or []) if x]
    if flag_ex:
        pol["exclude"], pol["source"]["exclude"] = flag_ex, "flag"
    else:
        pol["exclude"] = list(cfg.get("exclude", []))
        pol["source"]["exclude"] = "config" if "exclude" in cfg else "default"
    if "surfaces" in cfg or "surfaces_exclude" in cfg:
        include = list(cfg.get("surfaces", _core._SURFACE_PROFILE["include"]))
        exclude = list(cfg.get("surfaces_exclude", []))
        if "surfaces" not in cfg:
            exclude = list(_core._SURFACE_PROFILE["exclude"]) + exclude
        pol["profile"] = {"include": include, "exclude": exclude, "exts": _core._SURFACE_EXTS}
        pol["surfaces"] = {"include": include, "exclude": exclude}
    else:
        pol["profile"], pol["surfaces"] = None, None
    pol["config"] = {"path": _CONFIG, "read_at": repo.commit(rev), "present": present, "reads": True}
    return pol


def _public(pol: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in pol.items() if k != "profile"}


# ---- check (DESIGN 3.4) ------------------------------------------------------------------------------------

_PR = re.compile(r"^Merge (?:pull request )?#(\d+)\b")
_SEAL_CODES = ("MANIFEST_SEAL_NOT_LANDED", "MANIFEST_SEAL_PROSE_DIFFERS")


class _Ctx:
    """What one check (or census) reads every pair against: the repository, BASE, HEAD and the merge
    base, the policies, the manifest as HEAD holds it, and the surfaces at HEAD, read once and only
    when a decision declares a licence."""

    def __init__(self, repo: _git._Repo, b: str, h: str, mb: str, pol: Dict[str, Any],
                 entries: Optional[List[Dict[str, Any]]]):
        self.repo, self.b, self.h, self.mb, self.pol, self.entries = repo, b, h, mb, pol, entries
        self.exclude = tuple(pol["exclude"])
        self._surfaces: Optional[List[Tuple[str, Optional[str]]]] = None

    def surfaces(self) -> List[Tuple[str, Optional[str]]]:
        if self._surfaces is None:
            self._surfaces = _git._surfaces(self.repo, self.h, self.pol["profile"])
        return self._surfaces


def _tag(findings: Sequence[Dict[str, Any]], check: str, pair: Optional[str] = None) -> List[Dict[str, Any]]:
    return [{**f, "check": check, "pair": pair} for f in findings]


def _refusal(e: _git._Refusal, path: Optional[str], check: str, pair: Optional[str]) -> Dict[str, Any]:
    return {**e.finding(path), "check": check, "pair": pair}


def _landed_row(rs: Dict[str, Any], decision: Dict[str, Any]) -> Optional[int]:
    if rs.get("code") is not None or rs.get("smoke") or rs.get("verdict") is None:
        return None
    if decision["form"] == "locks":
        outcomes = _core._locks(decision["block"])["outcomes"]
        return outcomes.index(rs["verdict"]) if rs["verdict"] in outcomes else None
    return rs.get("row")


def _rescore(ctx: _Ctx, data: bytes, decision: Dict[str, Any], receipt_path: str, receipt: Any, gov: str,
             pair: Optional[str], findings: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Any]:
    """C5: the gates table over the receipt, or the locked scorer run at the blobs of the commit that
    governs (the seal's, or the amendment in effect). Returns (re-score, the run's record).

    A locked scorer runs as code, so it runs only from blobs BASE's history holds: the governing
    commit is BASE or an ancestor of it. A locks PREREG that arrives in the pull request (C2), or an
    amendment line naming a commit the pull request brings, would run the pull request's own bytes as
    the judge, which DESIGN 3.2(f) rules out ("a pull request cannot change the code that judges
    it"); that receipt reads ``SCORER_NOT_RUN`` (UNCHECKABLE, exit 3) until the PREREG and its locked
    files land (the B8 fresh review, F1). A census reads a revision's own history, so its BASE is the
    revision and every seal in it runs."""
    if decision["form"] != "locks":
        return _core.rescore(data, receipt), None
    try:
        spec = _core._locks(decision["block"])
    except _core._GateSpecError:
        return _core.rescore(data, receipt), None
    if gov != ctx.b and not ctx.repo.is_strict_ancestor(gov, ctx.b):
        findings.append(_report._finding("SCORER_NOT_RUN", (
            f"the locks decision in effect for {receipt_path} is read at {gov[:12]}, which the base {ctx.b[:12]} "
            f"does not hold, so its scorer and locked files are this pull request's own bytes; a locked scorer "
            f"runs as code, and prereg-ci runs one only from blobs the base already holds (DESIGN 3.2(f)). Once "
            f"the PREREG and its locked files have landed, a receipt is judged by the landed scorer"),
            path=receipt_path, check="C5", pair=pair))
        return None, None
    try:
        files = _scorer_run._locked_files(ctx.repo, gov, spec)
        raw = ctx.repo.read(ctx.h, receipt_path)
    except _git._Refusal as e:
        findings.append(_refusal(e, receipt_path, "C5", pair))
        return None, None
    for p in spec["locked"]:
        at_seal, at_head = ctx.repo.blob_id(gov, p), ctx.repo.blob_id(ctx.h, p)
        if at_seal != at_head:
            findings.append(_report._finding("LOCKED_FILE_MOVED", (
                f"{p}: the seal's blob is {at_seal}, HEAD's is {at_head or 'none'}; the run read the seal's blob, "
                f"so a pull request does not change the code that judges it"), path=p, check="C5", pair=pair))
    rs = _scorer_run._rescore_locked(data, receipt_path, raw or b"", files)
    run = rs.get("run")
    return rs, ({k: v for k, v in run.items() if k != "out_bytes"} if run else None)


def _recorded(ctx: _Ctx, rev: str, receipt_path: str, receipt: Any, pair: Optional[str],
              findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """What the receipt records (its verdict and gates) and the digest it was scored under: its own
    ``gates_sha256``, else a ``*.seal.json`` sidecar's (PLAN K2)."""
    rec = _git._fields(receipt, "verdict") if isinstance(receipt, dict) else {"gates_sha256": None}
    source = receipt_path
    try:
        side = _git._recorded(ctx.repo, rev, receipt_path, receipt, ctx.exclude)
    except _git._Refusal as e:
        findings.append(_refusal(e, receipt_path, "C6", pair))
        side = None
    if side is not None and side["source"] != receipt_path:
        rec["gates_sha256"] = side["gates_sha256"]
        for k in ("verdict", "gates"):
            if k not in rec and k in side:
                rec[k] = side[k]
        source = side["source"]
    return {**rec, "source": source}


def _compare(rs: Dict[str, Any], rec: Dict[str, Any], sealed: Sequence[str], declared: Sequence[Dict[str, Any]],
             pair: Optional[str]) -> List[Dict[str, Any]]:
    """C6 in its order: (i) and (ii) through ``core.compare``, then (iii) once per RESULT that declares a
    verdict for the receipt, each naming its RESULT."""
    plain = {k: v for k, v in rec.items() if k != "source"}
    base = _core.compare(rs, plain, sealed=list(sealed))
    out = _tag(base, "C6", pair)
    for d in declared:
        if d.get("verdict") is None:
            continue
        for f in _core.compare(rs, plain, sealed=list(sealed), declared=d["verdict"]):
            if f in base:
                continue
            f = dict(f, detail=f["detail"].replace("the RESULT prints", f"the RESULT {d['result']} prints"),
                     path=d["result"])
            out += _tag([f], "C6", pair)
    return out


def _boundary(form: Optional[str], rung: Optional[str], smoke: bool, network: Optional[str],
              sentences: Sequence[str]) -> List[str]:
    """C9, printed beside every reading at the same prominence (the charter: correctness is not
    coverage)."""
    if form is None:
        return ["the run itself: the PREREG holds no decision the scanner reads, so nothing was re-derived",
                "outcomes, sentences and the order of data and seal: not read"]
    items = [("the run itself: a locks PREREG locks its scorer, not the run that wrote the receipt" if form == "locks"
              else "the run itself: a gates PREREG locks no runner, so the receipt's numbers are read as written")]
    order = f"the order of data and seal: only the time ladder speaks, and it reads {rung or 'no rung'}"
    if sentences:
        order += "; the RESULT's own words on it: " + " / ".join(f'"{s}"' for s in sentences)
    items.append(order)
    items.append("outcomes outside the decision block")
    items.append("sentences outside sworn spans, result fences and licence patterns")
    items.append("discarded runs and receipts never committed")
    if smoke:
        items.append("a smoke receipt is scored by type (its smoke_verdict) and lands on no row, so its licences are "
                     "not read")
    if network and network != "isolated":
        items.append(f"the locked scorer's network: {network}")
    return items


def _results_text(ctx: _Ctx, declared: Sequence[Dict[str, Any]]) -> Tuple[List[str], bool]:
    """The RESULTs' own sentences on the seal's order, and whether one calls itself preregistered."""
    sentences: List[str] = []
    called = False
    for path in dict.fromkeys(d["result"] for d in declared if d.get("result")):
        try:
            data = ctx.repo.read(ctx.h, path) or b""
        except _git._Refusal:
            continue
        text = data.decode("utf-8", "replace")
        sentences += [s for s in _report._seal_sentences(text) if s not in sentences]
        called = called or _report._calls_itself_preregistered(text)
    return sentences[:_report._QUOTED], called


def _pair(ctx: _Ctx, p: Dict[str, Any]) -> Dict[str, Any]:
    """One receipt's reading, C2 to C9 (DESIGN 3.4)."""
    repo, b, h = ctx.repo, ctx.b, ctx.h
    rp, r, obj = p["path"], p["receipt_commit"], p["receipt"]
    out: Dict[str, Any] = {"receipt": rp, "status": p["status"], "receipt_commit": r,
                           "receipt_holders": list(p["receipt_holders"]), "declared": p["declared"], "prereg": None,
                           "seal": None, "effect": None, "earlier": [], "time": None, "blob_history": [],
                           "decision": None, "rescore": None, "run": None, "recorded": None, "ladder": None,
                           "licences": None, "findings": [], "not_examined": [], "reading": None}
    found: List[Dict[str, Any]] = out["findings"]

    def done(form: Optional[str] = None, rung: Optional[str] = None, smoke: bool = False,
             network: Optional[str] = None, sentences: Sequence[str] = ()) -> Dict[str, Any]:
        out["not_examined"] = _boundary(form, rung, smoke, network, sentences)
        out["reading"] = _report._VERDICT[_report._exit(found)]
        return out

    try:
        pp = _git._resolve(obj.get("prereg"), posixpath.dirname(rp), repo._view(h))
    except _git._Refusal as e:
        found.append(_refusal(e, rp, "C1", rp))
        return done()
    out["prereg"] = pp
    try:
        eff = _git._effect(repo, pp, rp, r, b, h, ctx.entries, amend_after_data=ctx.pol["amend_after_data"],
                           exclude=ctx.exclude)
    except _git._Refusal as e:
        found.append(_refusal(e, pp, "C2", rp))
        return done()
    seal, effect = eff["seal"], eff["effect"]
    for f in eff["findings"]:
        found += _tag([f], "C2" if f["code"] in _SEAL_CODES else "C4", rp)
    pr = None
    if seal.get("landing"):
        m = _PR.match(repo.subject(seal["landing"]))
        pr = f"#{m.group(1)}" if m else None
    out["seal"] = {**seal, "pr": pr}
    out["effect"], out["earlier"] = effect, eff["earlier"]
    data = effect["bytes"]
    decision = _core._decision(data)
    out["decision"] = {k: decision[k] for k in ("form", "gates_sha256", "code", "detail")}
    shadow = _core._shadowed(data)
    if shadow is not None:
        found.append(_report._finding("FENCE_SHADOWED", f"{pp}: {shadow}", path=pp, check="C5", pair=rp))
    sentences, called = _results_text(ctx, p["declared"])
    # C3: some minimal holder of the sealed decision is a strict ancestor of R
    order = None
    if seal.get("form") is not None and seal.get("order"):
        holders = list(seal.get("order_holders") or [seal["order"]])
        before = [x for x in holders if repo.is_strict_ancestor(x, r)]
        order = before[-1] if before else seal["order"]
        if not before:
            same = " (the same commit)" if seal["order"] == r else ""
            found.append(_report._finding("SEAL_NOT_BEFORE_RECEIPT", (
                f"the order commit {seal['order'][:12]} of {pp} is not a strict ancestor of {r[:12]}, the commit that "
                f"added {rp}{same}; a seal orders a receipt only from a commit before it"), path=rp, check="C3",
                pair=rp))
        out["blob_history"] = [list(x) for x in repo.blob_history(pp, order, r)]
    # the time ladder (3.5, 3.6), the trust store at BASE
    rung = None
    if seal.get("form") is not None:
        line = next((e for e in ctx.entries or () if e["id"] == seal.get("manifest_id")), None)
        try:
            tr = _time._reading(repo, path=pp, rev=h, seal=seal["bytes"], digests=[seal["gates_sha256"]], line=line,
                                receipt=r, order=order, receipt_path=rp, trust_rev=b)
        except _git._Refusal as e:
            found.append(_refusal(e, pp, "C3", rp))
            tr = None
        if tr is not None:
            if not seal.get("landed") and tr["rung"] == "T3a":
                tr["rung"], tr["name"] = "T1", _time._RUNG_NAME["T1"]
                tr["capped"] = "the PREREG arrives in this pull request (C2), so the rung cannot exceed AUTHOR_ORDER"
            close_after = None
            if decision["form"] == "gates":
                try:
                    spec = _core._gates_spec(decision["block"])
                    close_after = spec.get("close_after") if isinstance(spec, dict) else None
                except _core._GateSpecError:
                    close_after = None
            pol = ctx.pol
            try:
                tf = _time._findings(tr, require_time=pol["require_time"], preregistered=called,
                                     preregistered_word=pol["preregistered_word"], close_after=close_after)
            except ValueError as e:
                found.append(_report._finding("CLOSE_AFTER_UNREAD", f"{pp}: {e}; read without it", path=pp,
                                              check="C3", pair=rp))
                tf = _time._findings(tr, require_time=pol["require_time"], preregistered=called,
                                     preregistered_word=pol["preregistered_word"])
            found += _tag([{**f, "path": pp} for f in tf], "C3", rp)
            out["time"] = {**tr, "line": _time._describe(tr)}
            rung = tr["name"]
    # C5: the re-score from the decision in effect
    gov = next((a["commit"] for a in effect["amendments"] if a["id"] == effect["source"]), seal["commit"])
    rs, run = _rescore(ctx, data, decision, rp, obj, gov, rp, found)
    out["run"] = run
    if rs is None:
        return done(decision["form"], rung, False, None, sentences)
    out["rescore"] = {k: v for k, v in rs.items() if k not in ("run", "output")}
    if rs.get("cover_code"):
        found.append(_report._finding(rs["cover_code"], rs.get("cover_detail") or "", path=pp, check="C7", pair=rp))
    # C6
    rec = _recorded(ctx, h, rp, obj, rp, found)
    out["recorded"] = rec
    found += _compare(rs, rec, effect["sealed"], p["declared"], rp)
    # C7
    out["ladder"] = _core._ladder(rs, data, obj) if decision["form"] == "gates" else None
    # C8: the licences of the landed row
    row = _landed_row(rs, decision)
    if rs.get("code") is None and decision["form"] is not None and not rs.get("smoke"):
        try:
            specs = _core._licence_specs(decision)
        except _core._GateSpecError as e:
            found.append(_report._finding("SCORER_REFUSES", f"{pp}: a licence the format does not read: {e}", path=pp,
                                          check="C8", pair=rp))
            specs = []
        if specs and row is not None:
            lic = _core.licences(data, ctx.surfaces(), landed=rs["verdict"], row=row)
            out["licences"] = lic
            found += _tag(lic["findings"], "C8", rp)
    return done(decision["form"], rung, bool(rs.get("smoke")), (run or {}).get("network"), sentences)


def _tree_reading(ctx: _Ctx, pp: str, x: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The licences of one receipt already in HEAD's tree (C8, every later check), or None where it lands
    on no row: re-scored from the decision in effect for a gates PREREG; for a locks PREREG the receipt's
    recorded verdict is read (its scorer ran when the receipt was added, and is not run again here)."""
    try:
        eff = _git._effect(ctx.repo, pp, x["path"], x["receipt_commit"], ctx.b, ctx.h, ctx.entries,
                           amend_after_data=ctx.pol["amend_after_data"], exclude=ctx.exclude)
    except _git._Refusal as e:
        return {"prereg": pp, "receipt": x["path"], "landed": None, "row": None, "how": f"not read: {e.code}",
                "licences": [], "findings": []}
    data = eff["effect"]["bytes"]
    decision = _core._decision(data)
    if decision["form"] == "gates":
        rs = _core.rescore(data, x["receipt"])
        if rs["code"] is not None or rs["smoke"]:
            return None
        verdict, row, how = rs["verdict"], rs["row"], "re-scored"
    elif decision["form"] == "locks":
        try:
            outcomes = _core._locks(decision["block"])["outcomes"]
        except _core._GateSpecError:
            return None
        verdict = x["receipt"].get("verdict")
        if verdict not in outcomes:
            return None
        row, how = outcomes.index(verdict), "the receipt's recorded verdict (a locks receipt is re-run when it is added)"
    else:
        return None
    try:
        lic = _core.licences(data, ctx.surfaces(), landed=verdict, row=row)
    except ValueError:
        return None
    return {"prereg": pp, "receipt": x["path"], "landed": verdict, "row": row, "how": how,
            "licences": lic["licences"], "findings": _tag([{**f, "prereg": pp} for f in lic["findings"]], "C8",
                                                          x["path"])}


def _tree_licences(ctx: _Ctx, handled: Sequence[str]) -> List[Dict[str, Any]]:
    """C8 over HEAD's tree: every PREREG whose decision declares a licence (a ``*.md`` file, or one a
    receipt or a manifest line at HEAD names under another suffix, ``git._named_preregs``: the B8
    battery, M4), read under every receipt of it at HEAD that this run did not pair already; a PREREG
    with no receipt at all reads its matches as ``LICENCE_PENDING`` (the claim stands on a verdict not
    yet given). A pattern that matches nothing is ``lint``'s refusal and is not repeated here."""
    repo, h = ctx.repo, ctx.h
    out: List[Dict[str, Any]] = []
    md = repo.grep(h, '"withdraws"', ":(icase)*.md")
    named = sorted(p for p in _git._named_preregs(repo, h, ctx.exclude) if not _git._suffix(p, (".md",)))
    for pp in md + [p for p in named if p not in md]:
        if _git._excluded(pp, ctx.exclude):
            continue
        try:
            data = repo.read(h, pp) or b""
        except _git._Refusal:
            continue
        decision = _core._decision(data)
        if decision["form"] is None:
            continue
        try:
            specs = _core._licence_specs(decision)
        except _core._GateSpecError:
            continue
        if not specs:
            continue
        receipts = _git._receipts_of(repo, pp, [h], ctx.exclude)
        for x in receipts:
            if x["path"] in handled:
                continue
            entry = _tree_reading(ctx, pp, x)
            if entry is not None:
                out.append(entry)
        if not receipts:
            lic = _core.licences(data, ctx.surfaces(), landed=None)
            pending = [{**f, "prereg": pp} for f in lic["findings"] if f["code"] == "LICENCE_PENDING"]
            out.append({"prereg": pp, "receipt": None, "landed": None, "row": None, "how": "before data",
                        "licences": lic["licences"], "findings": _tag(pending, "C8", None)})
    return out


def _not_examined(ctx: _Ctx, excluded: Sequence[str], licences_read: bool) -> List[str]:
    items = ["T2 SERVER_ORDER: v0 reads no server, so the forge's merge times are not read",
             "a RESULT this pull request leaves alone or deletes is not read again, and prose verdicts and sworn spans "
             "on other leaves are not read (DESIGN 3.4 C9)",
             "a JSON file the pull request writes that names no PREREG by a top-level prereg key is not paired, though it "
             "records a verdict or a gates digest (DESIGN 3.2(d); census reads R1's records)"]
    if excluded:
        items.append("left out under exclude: " + ", ".join(excluded))
    if licences_read:
        skipped = [p for p, t in ctx.surfaces() if t is None]
        items.append("licences match strings: a wording no registered pattern matches is not refused (DESIGN 7, "
                     "risk 3)" + (f"; {_report._plural(len(skipped), 'surface')} under the globs not read (a symlink, a "
                                  f"binary or an oversize file): {', '.join(skipped[:10])}" if skipped else ""))
    return items


def _run_check(repo: _git._Repo, base: str, head: str, args: argparse.Namespace) -> Dict[str, Any]:
    b, h = repo.commit(base), repo.commit(head)
    repo._full()
    mb = repo.merge_base(b, h)
    config_findings = []
    changed = repo.blob_id(mb, _CONFIG) != repo.blob_id(h, _CONFIG)
    head_reads = True
    if changed:                                     # the copy that governs once this pull request lands
        try:
            _config_read(repo, h)
        except _Coded as e:
            if e.code != "CONFIG_INVALID":
                raise
            head_reads = False
            config_findings.append(_report._finding("CONFIG_INVALID", (
                f"{e.detail}; this pull request writes it, and once it lands every later check that leaves it "
                f"alone reads its policies as an input error"), path=_CONFIG, check="config", pair=None))
    repaired: Optional[_Coded] = None
    try:
        read = _config_read(repo, b)
    except _Coded as e:
        if e.code != "CONFIG_INVALID" or not changed or not head_reads:
            raise
        read, repaired = ({}, False), e
    pol = _policies(repo, b, args, read=read)
    if repaired is not None:
        pol["config"].update(present=True, reads=False)
        config_findings.append(_report._finding("CONFIG_REPAIRED", (
            f"{repaired.detail}; this pull request replaces it with a copy that reads, so it is judged under the "
            f"flags and the defaults, as a repository with no config is, and its copy governs once it lands"),
            path=_CONFIG, check="config", pair=None))
    if changed:
        config_findings.append(_report._finding("CONFIG_CHANGED", (
            f"{_CONFIG} changes in this pull request; the policies are read at base {b[:12]}"
            + (" (whose copy does not read, so the defaults govern)" if repaired is not None else "")
            + ", so the change governs once it lands"), path=_CONFIG, check="config", pair=None))
    decisions = _tag(_git._decisions_moved(repo, b, h, pol["immutability"]), "C0")
    man: Dict[str, Any] = {"path": _git._MANIFEST, "present": False, "lines": 0, "findings": []}
    entries = None
    try:
        entries = _git._manifest_at(repo, h)
    except _git._Refusal as e:
        man["findings"].append(_refusal(e, _git._MANIFEST, "C2", None))
    try:                                            # whether or not HEAD holds one: a deletion is a rewrite (B8, M2)
        man["findings"] += _tag(_git._manifest_append_only(repo, b, h), "C2")
    except _git._Refusal as e:
        man["findings"].append(_refusal(e, _git._MANIFEST, "C2", None))
    if entries is not None:
        man.update(present=True, lines=len(entries))
        for read in (lambda: _git._manifest_blobs(repo, entries, h),
                     lambda: _git._manifest_seen(repo, entries, h, base=b),
                     lambda: _git._manifest_landed(repo, entries, b, h)):
            try:
                man["findings"] += _tag(read(), "C2")
            except _git._Refusal as e:
                man["findings"].append(_refusal(e, _git._MANIFEST, "C2", None))
    exclude = tuple(pol["exclude"])
    rewritten = _tag(_git._receipts_rewritten(repo, b, h, pol["receipt_rewrite"], exclude), "C1")
    pw = _git._pairs_written(repo, b, h, exclude)
    results = {"findings": _tag(pw["findings"], "C6") + _tag(pw["notes"], "C1"), "excluded": pw["excluded"]}
    ctx = _Ctx(repo, b, h, mb, pol, entries)
    pairs = [_pair(ctx, p) for p in pw["pairs"]]
    tree = _tree_licences(ctx, [p["receipt"] for p in pairs])
    findings = (config_findings + decisions + man["findings"] + rewritten + results["findings"]
                + [f for p in pairs for f in p["findings"]] + [f for e in tree for f in e["findings"]])
    code = _report._exit(findings)
    return {"schema": _report._SCHEMA, "command": "check", "tool": _report._tool(), "base": b, "head": h,
            "merge_base": mb, "policies": _public(pol), "config_findings": config_findings, "decisions": decisions,
            "manifest": man, "rewritten": rewritten, "results": results, "pairs": pairs, "tree_licences": tree,
            "findings": findings, "not_examined": _not_examined(ctx, pw["excluded"], ctx._surfaces is not None),
            "verdict": _report._VERDICT[code], "exit": code}


def _emit(report: Dict[str, Any], lines: Sequence[str], args: argparse.Namespace) -> int:
    """Print the reading, then write the JSON report and append the summary; a path that does not
    write is an input error after the reading (exit 2, or 1 when the reading refused something)."""
    for ln in lines:
        print(ln)
    writes = (("json", lambda path: _report._write(path, _report._dumps(report))),
              ("summary", lambda path: _report._write(
                  path, _report._markdown(_report._jsonable(report)).encode("utf-8"), append=True)))
    for flag, write in writes:
        path = getattr(args, flag, None)
        if not path:
            continue
        try:
            write(path)
        except OSError as e:
            print(f"REFUSED --{flag} {path} does not write: {e.strerror or e}")
            return 1 if report["exit"] == 1 else 2
    return report["exit"]


def _check(args: argparse.Namespace) -> int:
    with _git._Repo(os.getcwd()) as repo:
        report = _run_check(repo, args.base, args.head, args)
        return _emit(report, _report._check_lines(report), args)


# ---- lint (DESIGN 3.7) -------------------------------------------------------------------------------------

_LINT_COVER_BUDGET = 1 << 24
_ROW_KEYS = ("when", "verdict", "withdraws")


def _distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _display(path: str, top: str) -> str:
    """A path as a report may print it: repository-relative inside the work tree, else its basename."""
    try:
        return _repo_path(path, top)
    except _Input:
        return os.path.basename(path)


def _lint_gates(decision: Dict[str, Any], add: Any, lines: List[str]) -> bool:
    try:
        spec = _core._gates_spec(decision["block"])
        names, whens = _core._table(spec, smoke=True)
    except _core._GateSpecError as e:
        add("SCORER_REFUSES", str(e))
        return False
    rows = spec["outcomes"]
    verdicts = [r["verdict"] for r in rows]
    for v, n in Counter(verdicts).items():
        if n > 1:
            where = ", ".join(str(i + 1) for i, x in enumerate(verdicts) if x == v)
            add("VERDICTS_REPEATED", f"rows {where} share the verdict {v}; a verdict names one row, so a reader (and "
                                     f"a licence) can tell which row a receipt landed on")
    mentioned = {k for w in whens for k in w}
    for g in names:
        if g not in mentioned:
            add("GATE_VACUOUS", f"gate {g} is named by no outcome row, so no verdict turns on it (a leg that cannot "
                                f"fail must not gate)")
    undeclared = [g for g in names if not (isinstance(spec["gates"][g].get("power_basis"), str)
                                           and spec["gates"][g]["power_basis"].strip())]
    if undeclared:
        add("POWER_BASIS_UNDECLARED", f"gates without a declared power_basis (how the bar was derived): "
                                      f"{', '.join(undeclared)}")
    for i, r in enumerate(rows):
        for k in r:
            if k in _ROW_KEYS:
                continue
            if _distance(k.casefold().strip(), "withdraws") <= 2:
                add("WITHDRAWS_KEY_MISSPELLED", f"row {i + 1} ({r['verdict']}) holds {k!r}, which reads like "
                                                f"'withdraws' and is not it: the scorers pass over it, so the row "
                                                f"would license nothing")
            else:
                add("ROW_KEY_UNREAD", f"row {i + 1} ({r['verdict']}) holds {k!r}, which no scorer reads")
    if "close_after" in spec and (not isinstance(spec["close_after"], str) or _time._epoch(spec["close_after"]) is None):
        add("CLOSE_AFTER_INVALID", f"close_after {spec['close_after']!r} is not YYYY-MM-DDTHH:MM:SSZ")
    try:
        cover = _core._cover(names, whens, budget=_LINT_COVER_BUDGET)
    except _core._CoverBudget as e:
        add("COVER_UNCHECKABLE", str(e))
        return True
    space = 1 << len(names)
    if cover["holes"]:
        add("TABLE_NOT_TOTAL", f"{cover['holes']} of {space} assignments land on no row, one of them "
                               f"{json.dumps(cover['hole'])}")
    for i in cover["unreachable_rows"]:
        add("ROW_UNREACHABLE", f"row {i + 1} ({verdicts[i]}) is reached by no assignment: the rows above it take "
                               f"every assignment it names")
    if not cover["holes"] and not cover["unreachable_rows"]:
        lines.append(f"  total over {space} assignments; {_report._plural(len(rows), 'row')}, each reached")
    return True


def _lint_locks(repo: _git._Repo, c: str, decision: Dict[str, Any], add: Any, lines: List[str]) -> bool:
    try:
        spec = _core._locks(decision["block"])
    except _core._GateSpecError as e:
        add("SCORER_REFUSES", str(e))
        return False
    lines.append(f"  scorer {spec['scorer']}, reads {', '.join(spec['reads']) or 'nothing else'}, "
                 f"{_report._plural(len(spec['outcomes']), 'outcome')}, argv {json.dumps(spec['argv'])}, verdict "
                 f"from {spec['verdict_from']}")
    for p in spec["locked"]:
        try:
            present = repo.read(c, p) is not None
        except _git._Refusal:
            present = False
        if not present:
            add("LOCKED_FILE_ABSENT", f"{p} is locked by the decision and is not a committed file at {c[:12]}; the "
                                      f"sealed blob of each locked path is what runs")
    return True


def _licence_coverage_lines(lic: Dict[str, Any]) -> List[str]:
    out = []
    for e in lic["licences"]:
        n = len(e["surfaces"])
        out.append(f"  licence {e['id']}: {_report._plural(n, 'surface')}" + (f" ({', '.join(e['surfaces'])})" if n else "")
                   + f", pattern {json.dumps(e['pattern'])}, row {e['row'] + 1} ({e['verdict']})")
    return out


def _lint(args: argparse.Namespace) -> int:
    with _git._Repo(os.getcwd()) as repo:
        top = repo._where()
        c = repo.commit(args.rev)
        try:
            with open(args.path, "rb") as fh:
                data = fh.read()
        except OSError as e:
            raise _Input(f"{args.path} does not read: {e}")
        pol = _policies(repo, c, args)
        shown = _display(args.path, top)
        decision = _core._decision(data)
        findings: List[Dict[str, Any]] = []

        def add(code: str, detail: str) -> None:
            findings.append(_report._finding(code, detail, path=shown, check="lint", pair=None))

        lines = [f"prereg-ci lint v0: {shown}, read from the file system (sha256 {hashlib.sha256(data).hexdigest()}); "
                 f"surfaces and config read at {c[:12]}",
                 f"  decision {decision['form'] or 'none'} {decision['gates_sha256'] or ''}".rstrip()]
        read_licences = False
        shadow = _core._shadowed(data)
        if shadow is not None:
            add("FENCE_SHADOWED", shadow)
        if decision["form"] is None:
            add(decision["code"], decision["detail"] or "")
        elif decision["form"] == "gates":
            read_licences = _lint_gates(decision, add, lines)
        else:
            read_licences = _lint_locks(repo, c, decision, add, lines)
        lic = None
        if read_licences:
            surfaces = _git._surfaces(repo, c, pol["profile"])
            lic = _core.licences(data, surfaces, landed=None)
            findings += [{**f, "check": "lint", "pair": None} for f in lic["findings"]]
            lines.append(f"  licences: surfaces read {lic['surfaces']}, not read {len(lic['skipped'])}"
                         + ("" if lic["licences"] else "; the decision declares none"))
            lines += _licence_coverage_lines(lic)
        code = _report._exit(findings)
        report = {"schema": _report._SCHEMA, "command": "lint", "tool": _report._tool(), "rev": c,
                  "prereg": {"path": shown, "sha256": hashlib.sha256(data).hexdigest(), "read_from": "file system"},
                  "policies": _public(pol), "decision": {k: decision[k] for k in ("form", "gates_sha256", "code",
                                                                                   "detail")},
                  "licences": lic, "findings": findings, "verdict": _report._VERDICT[code], "exit": code}
        lines += _report._finding_lines(findings)
        lines.append(f"{report['verdict']}  exit {code}")
        return _emit(report, lines, args)


# ---- ladder ------------------------------------------------------------------------------------------------

def _ladder(args: argparse.Namespace) -> int:
    with _git._Repo(os.getcwd()) as repo:
        top = repo._where()
        c = repo.commit(args.rev)
        rel = _repo_path(args.receipt, top)
        raw = repo.read(c, rel)
        if raw is None:
            raise _git._Refusal("RECEIPT_ABSENT", f"{rel} is not in the tree at {c[:12]}")
        try:
            obj = _git._json(raw)
        except (UnicodeDecodeError, ValueError) as e:
            raise _Input(f"{rel} at {c[:12]} is not JSON: {e}")
        try:
            with open(args.prereg, "rb") as fh:
                prereg = fh.read()
        except OSError as e:
            raise _Input(f"{args.prereg} does not read: {e}")
        pol = _policies(repo, c, args)
        shown = _display(args.prereg, top)
        decision = _core._decision(prereg)
        findings: List[Dict[str, Any]] = []
        run = None
        if decision["form"] == "locks":
            ctx = _Ctx(repo, c, c, c, pol, None)
            rs, run = _rescore(ctx, prereg, decision, rel, obj, c, rel, findings)
        else:
            rs = _core.rescore(prereg, obj)
        lines = [f"prereg-ci ladder v0: receipt {rel} at {c[:12]}",
                 f"  PREREG {shown}, read from the file system (sha256 {hashlib.sha256(prereg).hexdigest()}): a print of "
                 f"a decision, not a seal; no seal, order or time is read",
                 f"  decision {decision['form'] or 'none'} {decision['gates_sha256'] or decision['code'] or ''}".rstrip()]
        lad = lic = cov = rec = None
        if rs is not None:
            rec = {**_git._fields(obj, "verdict"), "source": rel} if isinstance(obj, dict) else {"source": rel}
            findings += _compare(rs, rec, [rs["gates_sha256"]] if rs.get("gates_sha256") else [], (), rel)
            if rs.get("code") is None:
                lines.append(f"  the decision scores this receipt {rs['verdict']}; "
                             + (f"the receipt records {rec['verdict']}" if "verdict" in rec
                                else "the receipt records no verdict"))
            lad = _core._ladder(rs, prereg, obj) if decision["form"] == "gates" else None
            lines += _report._ladder_lines(lad)
            row = _landed_row(rs, decision)
            if row is not None:
                try:
                    specs = _core._licence_specs(decision)
                except _core._GateSpecError as e:
                    findings.append(_report._finding("SCORER_REFUSES", f"a licence the format does not read: {e}",
                                                     check="C8", pair=rel))
                    specs = []
                if specs:
                    surfaces = _git._surfaces(repo, c, pol["profile"])
                    lic = _core.licences(prereg, surfaces, landed=rs["verdict"], row=row)
                    cov = _core._quote_coverage(prereg, surfaces)
                    findings += _tag(lic["findings"], "C8", rel)
                    lines += _report._licence_lines(lic)
                    lines += _report._coverage_lines(cov, lic["surfaces"])
        code = _report._exit(findings)
        report = {"schema": _report._SCHEMA, "command": "ladder", "tool": _report._tool(), "rev": c, "receipt": rel,
                  "prereg": {"path": shown, "sha256": hashlib.sha256(prereg).hexdigest(), "read_from": "file system"},
                  "policies": _public(pol), "rescore": ({k: v for k, v in rs.items() if k not in ("run", "output")}
                                                        if rs else None),
                  "run": run, "recorded": rec, "ladder": lad, "licences": lic, "quote_coverage": cov,
                  "findings": findings, "verdict": _report._VERDICT[code], "exit": code}
        lines += _report._finding_lines(findings)
        lines.append(_report._closing(report))
        return _emit(report, lines, args)


# ---- census ------------------------------------------------------------------------------------------------

def _census_row(ctx: _Ctx, kind: str, rec: Dict[str, Any], label: str) -> Dict[str, Any]:
    repo, c = ctx.repo, ctx.h
    row: Dict[str, Any] = {"label": label, "kind": kind, "receipt": rec.get("receipt_path"),
                           "prereg": rec.get("prereg_path"), "verdict": None, "findings": [], "code": None}
    if "refusal" in rec:
        row["findings"] = _tag([rec["refusal"]], "C1", label)
    else:
        rp = rec["receipt_path"]
        try:
            obj = rec.get("receipt")
            if obj is None:
                obj = _git._json(repo.read(c, rp) or b"")
            pp = rec.get("prereg_path") or _git._resolve(obj.get("prereg"), posixpath.dirname(rp), repo._view(c))
            row["prereg"] = pp
            held = repo._held(rp, c, lambda x: repo._blob_id(x, rp) is not None)
            r = held[-1]
            eff = _git._effect(repo, pp, rp, r, c, c, ctx.entries, amend_after_data=ctx.pol["amend_after_data"],
                               exclude=ctx.exclude)
        except _git._Refusal as e:
            row["findings"] = [_refusal(e, rec.get("record"), "C2", label)]
        except (UnicodeDecodeError, ValueError, AttributeError) as e:
            row["findings"] = [_report._finding("RECEIPT_UNREADABLE", f"{rp}: {e}", check="C1", pair=label)]
        else:
            seal, effect = eff["seal"], eff["effect"]
            found = _tag(eff["findings"], "C4", label)
            decision = _core._decision(effect["bytes"])
            shadow = _core._shadowed(effect["bytes"])
            if shadow is not None:
                found.append(_report._finding("FENCE_SHADOWED", f"{pp}: {shadow}", path=pp, check="C5", pair=label))
            gov = next((a["commit"] for a in effect["amendments"] if a["id"] == effect["source"]), seal["commit"])
            rs, _ = _rescore(ctx, effect["bytes"], decision, rp, obj, gov, label, found)
            if seal.get("form") is not None and seal.get("order"):
                holders = list(seal.get("order_holders") or [seal["order"]])
                if not any(repo.is_strict_ancestor(x, r) for x in holders):
                    found.append(_report._finding("SEAL_NOT_BEFORE_RECEIPT", (
                        f"the order commit {seal['order'][:12]} of {pp} is not a strict ancestor of {r[:12]}, the "
                        f"oldest commit holding {rp}"), path=rp, check="C3", pair=label))
            if rs is not None:
                recorded = rec["recorded"] if kind == "record" else {
                    k: v for k, v in _recorded(ctx, c, rp, obj, label, found).items() if k != "source"}
                found += _compare(rs, recorded, effect["sealed"], (), label)
                row["verdict"] = rs.get("verdict")
            row["findings"] = found
            row["receipt_commit"] = r
    decisive = [f for f in row["findings"] if f["kind"] != "NOTE"]
    row["code"] = decisive[0]["code"] if decisive else "RESCORES"
    return row


def _census(args: argparse.Namespace) -> int:
    with _git._Repo(os.getcwd()) as repo:
        c = repo.commit(args.rev)
        repo._full()
        pol = _policies(repo, c, args)
        head_findings: List[Dict[str, Any]] = []
        entries = None
        try:
            entries = _git._manifest_at(repo, c)
        except _git._Refusal as e:
            head_findings.append(_refusal(e, _git._MANIFEST, "C2", None))
        if entries is not None:
            try:
                head_findings += _tag(_git._manifest_chain(repo, c) + _git._manifest_blobs(repo, entries, c), "C2")
            except _git._Refusal as e:
                head_findings.append(_refusal(e, _git._MANIFEST, "C2", None))
        ctx = _Ctx(repo, c, c, c, pol, entries)
        items: List[Tuple[str, Dict[str, Any], str]] = []
        own = set()
        for rec in _git._pairs(repo, c, ctx.exclude):
            items.append(("record", rec, rec["record"] + (f"#{rec['pointer']}" if rec["pointer"] else "")))
            if rec.get("own"):
                own.add(rec["record"])
        for path in repo.grep(c, '"prereg"', ":(icase)*.json"):
            if path in own or _git._excluded(path, ctx.exclude):
                continue
            try:
                obj = _git._json(repo.read(c, path) or b"")
            except (_git._Refusal, UnicodeDecodeError, ValueError):
                continue
            if _git._receipt_kind(obj) == "receipt":
                items.append(("receipt", {"record": path, "receipt_path": path, "receipt": obj}, path))
        rows = [_census_row(ctx, kind, rec, label) for kind, rec, label in items]
        tree = _tree_licences(ctx, [])
        findings = head_findings + [f for r in rows for f in r["findings"]] + [f for e in tree for f in e["findings"]]
        tally = Counter(r["code"] for r in rows)
        code = _report._exit(findings)
        lines = [f"prereg-ci census v0: at {c[:12]}; {_report._plural(len(items), 'record')} "
                 f"({sum(1 for k, _, _ in items if k == 'record')} naming a gates digest, "
                 f"{sum(1 for k, _, _ in items if k == 'receipt')} receipts naming a PREREG without one)"]
        lines += _report._finding_lines(head_findings)
        for r in rows:
            tail = r["verdict"] if r["code"] == "RESCORES" else next(
                (f["detail"] for f in r["findings"] if f["code"] == r["code"]), "")
            lines.append(f"{r['code']} {r['label']} {tail}")
        for e in tree:
            if e["findings"]:
                lines.append(f"C8 licences of {e['prereg']} " + (f"under receipt {e['receipt']}" if e["receipt"]
                                                                  else "before any receipt has landed") + ":")
                lines += _report._finding_lines(e["findings"])
        lines.append(f"TALLY {json.dumps(dict(sorted(tally.items())))} records {len(items)}")
        report = {"schema": _report._SCHEMA, "command": "census", "tool": _report._tool(), "rev": c,
                  "policies": _public(pol), "rows": rows, "tree_licences": tree, "tally": dict(tally),
                  "findings": findings, "verdict": _report._VERDICT[code], "exit": code}
        lines.append(f"{report['verdict']}  exit {code}")
        return _emit(report, lines, args)


# ---- the parser ----------------------------------------------------------------------------------------------

def _policy_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--immutability", choices=_git._IMMUTABILITY, help="C0: refuse prose moved around a sealed fence "
                                                                       "too (file), or print it (gates, the default)")
    p.add_argument("--amend-after-data", dest="amend_after_data", choices=_core._AMEND_AFTER_DATA,
                   help="C4: print (report, the default) or refuse (fail) an amendment that governs after a receipt")
    p.add_argument("--receipt-rewrite", dest="receipt_rewrite", choices=_git._POLICIES,
                   help="C1: refuse (fail, the default) or print (note) a committed receipt changed in place")
    p.add_argument("--require-time", dest="require_time", choices=_time._REQUIRE,
                   help="the rung a seal must read (T1, the default, or T3a: an RFC 3161 token verified offline)")
    p.add_argument("--preregistered-word", dest="preregistered_word", choices=_time._WORD_POLICIES,
                   help="a RESULT that calls itself preregistered below T3a: note (the default) or fail")
    p.add_argument("--exclude", action="append", metavar="PREFIX",
                   help="a repository-relative path prefix left out of C1 and the licence scan (repeatable)")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m styxx.prereg_ci",
                                 description="prereg-ci: a sealed decision, the verdict it gives, and the sentences "
                                             "that rest on it (DESIGN 3.3)")
    sub = ap.add_subparsers(dest="verb", metavar="VERB")
    sub.required = True
    li = sub.add_parser("lint", help="before data: grammar, totality, vacuity, and each licence's coverage")
    li.add_argument("path", help="the PREREG, read from the file system (the draft)")
    li.add_argument("--rev", default="HEAD", help="where the surfaces and the config are read (default HEAD)")
    li.add_argument("--json", metavar="PATH", help="write the JSON report here")
    li.set_defaults(run=_lint)
    s = sub.add_parser("seal", help="print (and with --append, append) a seals manifest line")
    s.add_argument("path", help="the sealed file, as a path from the working directory")
    s.add_argument("--commit", help="the commit whose blob is sealed (default: the landing commit on HEAD)")
    s.add_argument("--amends", help="the manifest line id, or the path, this amendment amends")
    s.add_argument("--seen", choices=_core._SEEN, help="what had been seen when the amendment was written")
    s.add_argument("--append", action="store_true", help="append the line(s) to the manifest's working copy")
    s.add_argument("--manifest", default=_git._MANIFEST, help="the manifest, relative to the work tree's top")
    s.set_defaults(run=_seal)
    t = sub.add_parser("stamp", help="write an outside time token beside a sealed file: the one networked command, "
                                     "which refuses without --go")
    t.add_argument("path", help="the sealed file, as a path from the working directory")
    t.add_argument("--rfc3161", metavar="TSA_URL", help="send the blob's sha256 to this RFC 3161 TSA; writes PATH.tsr")
    t.add_argument("--ots", action="store_true", help="hand the blob to the OpenTimestamps client on PATH; writes PATH.ots")
    t.add_argument("--commit", default="HEAD", help="the commit whose blob is stamped (default HEAD); never the working copy")
    t.add_argument("--go", action="store_true", help="send it; without --go the command prints what it would send and "
                                                     "writes nothing")
    t.set_defaults(run=_stamp)
    c = sub.add_parser("check", help="the merge gate: every receipt the pull request writes or a RESULT names, "
                                     "re-derived from the sealed bytes")
    c.add_argument("--base", required=True, help="the base commit (the pull request's target)")
    c.add_argument("--head", default="HEAD", help="the head commit (default HEAD)")
    c.add_argument("--json", metavar="PATH", help="write the JSON report here")
    c.add_argument("--summary", metavar="PATH", help="append the markdown job summary here ($GITHUB_STEP_SUMMARY)")
    _policy_flags(c)
    c.set_defaults(run=_check)
    la = sub.add_parser("ladder", help="the printout alone, for one receipt under a PREREG read from the file system")
    la.add_argument("receipt", help="the receipt, as a path from the working directory, read at --rev")
    la.add_argument("--prereg", required=True, help="the PREREG, read from the file system")
    la.add_argument("--rev", default="HEAD", help="where the receipt, the surfaces and the config are read (default HEAD)")
    la.add_argument("--json", metavar="PATH", help="write the JSON report here")
    la.set_defaults(run=_ladder)
    ce = sub.add_parser("census", help="every receipt at a revision re-derived under the landing rule: the post-merge "
                                       "record")
    ce.add_argument("--rev", default="HEAD", help="the revision (default HEAD)")
    ce.add_argument("--json", metavar="PATH", help="write the JSON report here")
    ce.add_argument("--exclude", action="append", metavar="PREFIX",
                    help="a repository-relative path prefix left out (repeatable)")
    ce.add_argument("--amend-after-data", dest="amend_after_data", choices=_core._AMEND_AFTER_DATA)
    ce.set_defaults(run=_census)
    return ap


_CRASH = 4                                            # outside DESIGN 3.3's four codes; the Action reads ERROR


def main(argv: Optional[List[str]] = None) -> int:
    """Run one verb; returns its exit code (argparse exits 2 itself on arguments it does not take)."""
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except _Input as e:
        print(f"REFUSED {e}")
        return 2
    except _Coded as e:
        print(f"{e.kind} {e.code}  {e.detail}")
        return 2 if e.kind == "INPUT" else 3 if e.kind == "UNCHECKABLE" else 1
    except _git._Refusal as e:
        print(f"REFUSED {e.code}: {e.detail}")
        return 3 if e.kind == "UNCHECKABLE" else 2
    except Exception as e:                            # noqa: BLE001 - a crash is no verdict, so never exit 1
        import traceback
        traceback.print_exc()
        print(f"ERROR  {type(e).__name__}: {e} (an internal error: no verdict was read, exit {_CRASH})")
        return _CRASH


if __name__ == "__main__":
    sys.exit(main())
