# -*- coding: utf-8 -*-
"""styxx.prereg_ci.cli: the commands, ``python -m styxx.prereg_ci <verb>`` (DESIGN 3.3).

Step B2 builds ``seal``; step B7 adds ``lint``, ``stamp``, ``check``, ``ladder`` and ``census`` and
the report. Exit codes follow DESIGN 3.3: 0 done, 2 input error, 3 UNCHECKABLE (a shallow clone, a
missing commit, git that does not answer). ``seal`` decides nothing, so every refusal of it that is
not UNCHECKABLE is an input error.

``seal PATH [--commit C] [--append]`` prints the bytes the seal holds, their sha256 (the digest
``git show <commit>:<path> | sha256sum`` gives), the decision digest, the landing commit L, the add
commit and the order commit, and the manifest line (DESIGN 3.2(b)); the commit is L on HEAD's
parent-one chain unless ``--commit`` names one in HEAD's history. ``seal PATH --amends ID|PATH --seen
LEVEL [--append]`` prints an amendment line: ``--amends`` names a manifest line by id, or a path,
which names that path's newest line, or, where the path has none, gets its default-seal line
written before the amendment (PLAN K9). A prereg line seals the decision the PREREG landed with on
HEAD's parent-one chain: ``--commit`` naming bytes that hold another decision is refused
(``MANIFEST_SEAL_NOT_LANDED``, which check reads too), since a changed decision is an amendment line.
``--append`` appends the line(s) to the manifest's working copy (``prereg-ci.seals.jsonl`` at the
work tree's top), LF, no byte-order mark, after checking that the working copy is HEAD's manifest
with lines appended and that the whole still parses; without it nothing is written. The manifest
is read from blobs everywhere else, so a working copy that a checkout holds with CRLF is read here
with CRLF as LF and its bytes are left as they are.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

from styxx.prereg_ci import core as _core
from styxx.prereg_ci import git as _git

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
    decision = (f"gates {seal['gates_sha256']}" if seal["form"] == "gates"
                else f"no decision the scanner reads ({seal['code']})")
    print(f"seal {line['path']} (line {line['id']}, {line['kind']}"
          + (f", amends {line['amends']}, {line['decision']}, seen: {line['seen']}" if line["amends"] else "") + ")")
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
        for line in new:
            _say(line, _git._line_seal(repo, line, head, head))
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


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m styxx.prereg_ci",
                                 description="prereg-ci: a sealed decision, the verdict it gives, and the sentences "
                                             "that rest on it (DESIGN 3.3)")
    sub = ap.add_subparsers(dest="verb", metavar="VERB")
    sub.required = True
    s = sub.add_parser("seal", help="print (and with --append, append) a seals manifest line")
    s.add_argument("path", help="the sealed file, as a path from the working directory")
    s.add_argument("--commit", help="the commit whose blob is sealed (default: the landing commit on HEAD)")
    s.add_argument("--amends", help="the manifest line id, or the path, this amendment amends")
    s.add_argument("--seen", choices=_core._SEEN, help="what had been seen when the amendment was written")
    s.add_argument("--append", action="store_true", help="append the line(s) to the manifest's working copy")
    s.add_argument("--manifest", default=_git._MANIFEST, help="the manifest, relative to the work tree's top")
    s.set_defaults(run=_seal)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    """Run one verb; returns its exit code (argparse exits 2 itself on arguments it does not take)."""
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except _Input as e:
        print(f"REFUSED {e}")
        return 2
    except _git._Refusal as e:
        print(f"REFUSED {e.code}: {e.detail}")
        return 3 if e.kind == "UNCHECKABLE" else 2


if __name__ == "__main__":
    sys.exit(main())
