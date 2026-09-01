# -*- coding: utf-8 -*-
"""styxx.worklog — a record of what an agent actually wrote.

Spec: papers/closed-model-frontier/SPEC_worklog_v01_2026_08_31.md

An agent's account of its own work has had one author. The summary is written by
the model; the diff is produced by the same session; checking one against the
other catches only the disagreements a model left in prose — measured at 22.5%
coverage and 0.23 accusation precision on an external corpus, and switched off.

A worklog has a DIFFERENT author. Not what the agent said it did, and not what
the repository ended up looking like: what the agent's tools actually wrote,
recorded by the harness at the moment of writing.

This module carries NO VERDICT and makes no claim. It records, canonicalises and
re-checks its own integrity. Comparing a worklog against a diff is the
`undeclared` band, and it is deliberately not here: gating on a band whose noise
floor has not been measured is the mistake this lab paid for by disabling a
shipped feature the same day this file was written.

CLI::

    python -m styxx.worklog record LOG.json --path src/x.py --tool edit
    python -m styxx.worklog verify LOG.json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

__all__ = ["Worklog", "WorklogEntry", "load", "verify_worklog", "main"]

SPEC = "styxx-worklog/v0.1"


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _norm(path: str) -> str:
    return (path or "").replace("\\", "/").strip("/")


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class WorklogEntry(dict):
    """One write through an instrumented surface. A record, not an assertion."""

    @classmethod
    def make(cls, seq: int, path: str, tool: str, after: bytes | None,
             before: bytes | None = None) -> "WorklogEntry":
        e = cls({
            "seq": seq,
            "path": _norm(path),
            "tool": tool,
            "at": _now(),
            "after_sha256": None if after is None else _sha256_bytes(after),
            "before_sha256": None if before is None else _sha256_bytes(before),
        })
        return e


class Worklog:
    """Append-only. Digests only — never file contents (spec, out of scope)."""

    def __init__(self, session: str, harness: str, entries: list | None = None):
        self.session = session
        self.harness = harness
        self.entries: list = list(entries or [])

    # ── recording ──────────────────────────────────────────────────────────
    def record(self, path: str, tool: str, after: bytes | None,
               before: bytes | None = None) -> WorklogEntry:
        e = WorklogEntry.make(len(self.entries) + 1, path, tool, after, before)
        self.entries.append(e)
        return e

    def record_file(self, path: str | Path, tool: str,
                    before: bytes | None = None) -> WorklogEntry:
        """Record a write by reading the file as it now stands."""
        p = Path(path)
        after = p.read_bytes() if p.exists() else None
        return self.record(str(path), tool, after, before)

    # ── serialisation ──────────────────────────────────────────────────────
    def core(self) -> dict:
        return {"spec": SPEC, "session": self.session, "harness": self.harness,
                "entries": self.entries}

    def digest(self) -> str:
        from styxx.attestation import jcs
        return _sha256_bytes(jcs(self.core()).encode("utf-8"))

    def to_dict(self) -> dict:
        d = self.core()
        d["digest"] = self.digest()
        return d

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")
        return p


def load(path: str | Path) -> Worklog:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("spec") != SPEC:
        raise SystemExit(f"REFUSED: unknown worklog spec {d.get('spec')!r}")
    return Worklog(d.get("session", ""), d.get("harness", ""), d.get("entries", []))


def verify_worklog(path: str | Path) -> dict:
    """Check exactly what the record alone can support — and nothing more.

    Deliberately NOT checked: the entries against a repository. A worklog
    verified against the tree it produced would be verifying the harness with
    the harness, which is not a check.
    """
    problems: list[str] = []
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"ok": False, "stage": "parse", "problems": [f"unreadable: {e}"]}

    if raw.get("spec") != SPEC:
        return {"ok": False, "stage": "spec",
                "problems": [f"unknown spec {raw.get('spec')!r}"]}

    wl = Worklog(raw.get("session", ""), raw.get("harness", ""), raw.get("entries", []))
    if raw.get("digest") != wl.digest():
        problems.append("digest does not match the entries")

    seqs = [e.get("seq") for e in wl.entries]
    if seqs != list(range(1, len(seqs) + 1)):
        problems.append("sequence numbers are not dense and ordered from 1")
    for i, e in enumerate(wl.entries, 1):
        for k in ("path", "tool", "at"):
            if not e.get(k):
                problems.append(f"entry {i}: missing {k}")
        if e.get("after_sha256") is None and e.get("before_sha256") is None:
            problems.append(f"entry {i}: records neither a before nor an after digest")

    paths = {e.get("path") for e in wl.entries}
    return {
        "ok": not problems,
        "stage": "checked",
        "problems": problems,
        "spec": SPEC,
        "session": wl.session,
        "harness": wl.harness,
        "entries": len(wl.entries),
        "distinct_paths": len(paths),
        "verdict": "UNGATED",          # a worklog never carries one
        "boundary": ("records writes through the instrumented surface only; a write "
                     "the harness did not wrap does not appear here, and this record "
                     "is only as trustworthy as the harness that produced it"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="styxx.worklog",
        description="Record what an agent actually wrote. No claim, no verdict.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="append one write to a worklog")
    r.add_argument("worklog")
    r.add_argument("--path", required=True)
    r.add_argument("--tool", required=True)
    r.add_argument("--session", default="local")
    r.add_argument("--harness", default="unnamed-harness")

    v = sub.add_parser("verify", help="check a worklog's internal consistency")
    v.add_argument("worklog")

    a = ap.parse_args(argv)

    if a.cmd == "record":
        p = Path(a.worklog)
        wl = load(p) if p.exists() else Worklog(a.session, a.harness)
        e = wl.record_file(a.path, a.tool)
        wl.write(p)
        print(f"recorded #{e['seq']} {e['tool']} {e['path']} "
              f"after={str(e['after_sha256'])[:12]}")
        return 0

    rep = verify_worklog(a.worklog)
    print(f"worklog: {rep.get('session')} · harness {rep.get('harness')} · "
          f"{rep.get('entries')} entries over {rep.get('distinct_paths')} paths")
    print(f"verdict: {rep.get('verdict')} — a worklog carries none by construction")
    if rep["ok"]:
        print("INTACT: the digest matches the entries and the sequence is dense.")
        print(f"boundary: {rep['boundary']}")
        return 0
    print("WORKLOG FAILS ITS OWN CHECK:")
    for p_ in rep["problems"]:
        print(f"  - {p_}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
