# -*- coding: utf-8 -*-
"""Generate seals_manifest_from_SEALS_2026_09_13.jsonl: the lab's SEALS record as manifest lines.

  python -B conformance/prereg_ci/gen_seals_manifest.py           write the file where none is committed
  python -B conformance/prereg_ci/gen_seals_manifest.py --check   regenerate in memory; exit 1 if a byte differs

Step B2 of the prereg-ci build; the design's receipt R10 ("8 of 8 rows of
papers/checksum/SEALS_2026_09_13.md:33-40 reproduce under git show <commit>:<path> | sha256sum").
It reads the SEALS file as git stores it at REV (``--rev``), takes the table under "## the digests"
(lines 33-40 there), and for each row prints ``MATCH <commit> <path>`` (R10's own line) once the
sha256 of the blob at that commit and path equals the row's digest, or ``DIFFERS`` and refuses.

How a row becomes a line (DESIGN 3.2(b)), read from the row as written, never guessed:
- ``id`` is the row's ``#``. ``kind``: a number alone is ``prereg``; a number with a letter after
  it is ``amendment`` and amends the row of that number (SEALS' 3b, 6b, 7b, the CORRECTIONs); a
  row whose note says ``**supersedes row N**`` amends row N, and is ``withdrawn`` when its note
  says ``**Withdrawn`` (row 8).
- ``commit`` is the row's abbreviation resolved to the full id; ``blob_sha256`` the sha256 of the
  blob there, equal to the row's.
- ``decision``: ``own`` on a prereg line; on the others ``own`` when the file holds a gates fence
  other than the one in effect for its row's lineage, else ``inherits`` (the three CORRECTIONs
  hold no fence; row 8's blob holds row 7's).
- ``seen`` null (the record states none), ``time`` empty (no token exists, R11).

The lines are rendered by ``styxx.prereg_ci.core._manifest_bytes``, parsed back by
``core._manifest`` (refused if the round trip moves a byte), and checked by
``styxx.prereg_ci.git._manifest_blobs`` at REV. The committed file is never written again (a
receipt is history too): a run that regenerates it byte for byte exits 0, any other prints
DIFFERS and writes nothing. It carries the lab's history as is: paths and commit ids.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conformance.prereg_ci import REV  # noqa: E402
from styxx.prereg_ci import core  # noqa: E402
from styxx.prereg_ci import git as G  # noqa: E402

SEALS = "papers/checksum/SEALS_2026_09_13.md"
OUT = ROOT / "conformance" / "prereg_ci" / "seals_manifest_from_SEALS_2026_09_13.jsonl"
HEADER = "| # | kind | file | frozen at | blob sha256 |"
ROW = re.compile(r"\| (?P<id>[0-9]+[a-z]*) \| `(?P<kind>[^`]+)` \| `(?P<path>[^`]+)` \| `(?P<commit>[0-9a-f]{7,64})` "
                 r"\| `(?P<sha>[0-9a-f]{64})`(?P<note>.*) \|")
SUPERSEDES = re.compile(r"\*\*supersedes row ([0-9]+[a-z]*)\*\*")


class Refused(Exception):
    pass


def _rows(text: str) -> List[Dict[str, Any]]:
    """The table's rows with their 1-based line numbers in the blob."""
    lines = text.split("\n")
    if lines.count(HEADER) != 1:
        raise Refused(f"{SEALS} holds {lines.count(HEADER)} tables headed {HEADER!r}; the generator reads one")
    at = lines.index(HEADER)
    if not lines[at + 1].startswith("|---"):
        raise Refused(f"{SEALS}:{at + 2} is not the table's separator")
    out = []
    for n in range(at + 2, len(lines)):
        if not lines[n].startswith("| "):
            break
        m = ROW.fullmatch(lines[n])
        if m is None:
            raise Refused(f"{SEALS}:{n + 1} is a table row the generator does not read")
        out.append({"line": n + 1, **m.groupdict()})
    if not out:
        raise Refused(f"{SEALS}'s table has no rows")
    return out


def build(repo: G._Repo, rev: str = REV) -> Dict[str, Any]:
    """The rows read at *rev*, the manifest lines and their bytes; Refused on any row that does not
    reproduce or read as the module says."""
    c = repo.commit(rev)
    data = repo.read(c, SEALS)
    if data is None:
        raise Refused(f"{SEALS} is not in the tree at {c[:12]}")
    rows = _rows(data.decode("utf-8"))
    entries: List[Dict[str, Any]] = []
    for row in rows:
        if row["kind"] != "sealed-prereg":
            raise Refused(f"{SEALS}:{row['line']}: kind {row['kind']!r}; the generator reads sealed-prereg rows")
        full = repo.commit(row["commit"])
        blob = repo.read(full, row["path"])
        if blob is None:
            raise Refused(f"DIFFERS {row['commit']} {row['path']}: no file there")
        got = hashlib.sha256(blob).hexdigest()
        row.update(full=full, got=got, match=got == row["sha"])
        if not row["match"]:
            raise Refused(f"DIFFERS {row['commit']} {row['path']}: sha256 {got}, the row records {row['sha']}")
        lettered = re.fullmatch(r"([0-9]+)([a-z]+)", row["id"])
        sup = SUPERSEDES.search(row["note"])
        withdrawn = "**Withdrawn" in row["note"]
        if lettered:
            kind, amends = "amendment", lettered.group(1)
        elif sup:
            kind, amends = ("withdrawn" if withdrawn else "amendment"), sup.group(1)
        elif withdrawn:
            raise Refused(f"{SEALS}:{row['line']}: a withdrawn row that names no row it supersedes")
        else:
            kind, amends = "prereg", None
        if kind == "prereg":
            decision = "own"
        else:
            reading = core._decision(blob)
            if reading["form"] is None and reading["code"] != "NOT_GATED":
                raise Refused(f"{SEALS}:{row['line']}: {row['path']} holds a fence the scanner refuses")
            root = next(e for e in entries if e["id"] == amends)
            while root["kind"] != "prereg":
                root = next(e for e in entries if e["id"] == root["amends"])
            in_effect = core._decision(repo.read(root["commit"], root["path"]) or b"")["gates_sha256"]
            for e in core._lineage(entries, root["id"]):
                if e["kind"] == "amendment" and e["decision"] == "own":
                    in_effect = core._decision(repo.read(e["commit"], e["path"]) or b"")["gates_sha256"]
            decision = "own" if reading["form"] is not None and reading["gates_sha256"] != in_effect else "inherits"
        entries.append({"id": row["id"], "kind": kind, "path": row["path"], "commit": full, "blob_sha256": got,
                        "amends": amends, "decision": decision, "seen": None, "time": []})
        core._manifest(core._manifest_bytes(entries))          # every line so far is a line the parser reads
    raw = core._manifest_bytes(entries)
    if core._manifest(raw) != entries or core._manifest_bytes(core._manifest(raw)) != raw:
        raise Refused("the lines do not round-trip through core._manifest and core._manifest_bytes")
    found = G._manifest_blobs(repo, entries, c)
    if found:
        raise Refused("; ".join(f["detail"] for f in found))
    return {"rev": c, "rows": rows, "entries": entries, "bytes": raw}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="regenerate in memory; exit 1 if a byte differs")
    ap.add_argument("--rev", default=REV, help=f"the commit the SEALS file is read at (default {REV[:12]})")
    args = ap.parse_args(argv)
    try:
        with G._Repo(ROOT) as repo:
            got = build(repo, args.rev)
    except (Refused, G._Refusal) as e:
        print(f"REFUSED {e}")
        return 1
    raw = got["bytes"]
    if args.check or OUT.exists():
        if OUT.exists() and OUT.read_bytes() == raw:
            print(f"SAME {OUT.name}={hashlib.sha256(raw).hexdigest()[:16]} at {got['rev'][:12]}, rows "
                  f"{got['rows'][0]['line']}-{got['rows'][-1]['line']}; nothing written")
        else:
            print(f"DIFFERS {OUT.name}: the run does not regenerate the committed bytes; nothing written. New lines "
                  f"go in new files.")
            return 1
    else:
        OUT.write_bytes(raw)
        if OUT.read_bytes() != raw:
            print(f"REFUSED {OUT.name} did not read back as written")
            return 1
        print(f"WROTE {OUT.name} {len(got['entries'])} lines at {got['rev'][:12]}")
    for row in got["rows"]:
        print(f"MATCH {row['commit']} {row['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
