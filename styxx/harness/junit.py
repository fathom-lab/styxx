# -*- coding: utf-8 -*-
"""A test report through ``styxx.evidence``, minted as a sworn manifest at the rung the caller
declares.

Input: one file of JUnit XML or in-toto test-result bytes, read from disk at the path the caller
gave; ``rung`` (L1 or L2 — required, and the adapter cannot know which is true: it does not see
whether the run happened after the agent's turn or on a machine the agent could not reach);
an optional ``turn`` id; optional byte-objects the caller knows the agent wrote this turn.

Receipts (all ``test_report``, all complete):

  r1  the passed count as ASCII digits (``resolved.passed`` from styxx.evidence)
  r2  the failures count as ASCII digits (``totals.failures``; harness errors stay apart)
  r3  the report bytes, whole
  r4  ``load_evidence``'s returned object as RFC 8785 canonical JSON with one trailing LF, so
      ``r4#/totals/errors``, ``r4#/outcome``, ``r4#/unparsed/0/reason`` are addressable leaves

When the reader could not parse the report (no source, ``unparsed`` non-empty), r1 and r2 are
NOT minted: a zero from a report that did not parse is absence printed as a number, and a span
that names r1 then resolves UNRESOLVED ``manifest_id_missing`` — the verifier saying it could
not see, never a lie made to pass. r3 and r4 are minted regardless.

The path enters r4 as given (``paths_requested``, ``sources[].path``), so a reader handed the
same path re-derives r4 byte for byte; give a path relative to the repository root when a
machine path must not enter a receipt. A report that cannot be read is a usage error at the
command line, not a manifest with an ``unreadable`` reason.

This module imports no clock, environment, network or subprocess module; the manifest's own
timestamps come from ``styxx.sworn.Manifest``, which is where the clock enters a manifest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from styxx.attestation import jcs
from styxx.evidence import load_evidence
from styxx.harness import HARNESS_VERSION, L1_WEAK, LABEL
from styxx.sworn import RUNGS, Manifest

__all__ = ["mint", "harness_string", "main"]


def harness_string(path: str, rung: str, authored_count: int) -> str:
    recorded = ("authored bytes recorded: %d (what the caller handed over; the adapter cannot see "
                "what the agent wrote)" % authored_count if authored_count
                else "authored bytes recorded: none (the caller handed over nothing; the adapter "
                     "cannot see what the agent wrote)")
    weak = ("; " + L1_WEAK) if rung == "L1" else ""
    return ("%s junit over %s; rung %s declared by the caller, not detected%s; %s; %s"
            % (HARNESS_VERSION, path, rung, weak, recorded, LABEL))


def mint(report_path, *, rung: str, turn: str = "", authored: Sequence[bytes] = ()) -> Manifest:
    """Mint a manifest over one report. Raises ValueError for a rung outside the ladder and lets
    OSError propagate for a report that cannot be read; both are usage errors, never verdicts."""
    if rung not in RUNGS:
        raise ValueError("rung must be one of %s, not %r — L3 is reserved and the rung is "
                         "declared, never detected" % (", ".join(RUNGS), rung))
    path = str(report_path)
    report = Path(path).read_bytes()
    ev = load_evidence([path])
    authored = [bytes(b) for b in authored]
    m = Manifest(harness=harness_string(path, rung, len(authored)), turn=turn, rung=rung)
    for blob in authored:
        m.record_authored(blob)
    if ev["sources"]:
        m.add("r1", str(ev["resolved"]["passed"]).encode("ascii"), "test_report", complete=True,
              note="passed count resolved by styxx.evidence from %s (resolved.passed)" % path)
        m.add("r2", str(ev["totals"]["failures"]).encode("ascii"), "test_report", complete=True,
              note="failures count resolved by styxx.evidence from %s (totals.failures; harness "
                   "errors are kept apart in totals.errors)" % path)
    m.add("r3", report, "test_report", complete=True, note="report bytes as read from %s" % path)
    m.add("r4", (jcs(ev) + "\n").encode("utf-8"), "test_report", complete=True,
          note="styxx.evidence.load_evidence over %s, RFC 8785 canonical with one trailing LF, "
               "byte for byte" % path)
    return m


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="styxx.harness junit",
        description="Mint a sworn/manifest/0.2 file from one test report through "
                    "styxx.evidence. An adapter: it derives nothing, signs nothing and exits "
                    "zero whatever the report says; two is a usage error.")
    ap.add_argument("report", help="JUnit XML or in-toto test-result file; the path enters r4 as given")
    ap.add_argument("--rung", required=True, choices=list(RUNGS),
                    help="declared by you, not detected: L1 for a run on the agent's own machine, "
                         "L2 only for a runner that minted after the turn and that the agent "
                         "could not write to")
    ap.add_argument("--turn", default="", help="turn id recorded in the manifest")
    ap.add_argument("--authored", action="append", default=[], metavar="FILE",
                    help="a file you know the agent wrote this turn; its sha256 enters "
                         "authored_sha256 (repeatable)")
    ap.add_argument("--out", required=True, help="manifest path to write (LF-only JSON)")
    a = ap.parse_args(argv)
    try:
        authored = [Path(f).read_bytes() for f in a.authored]
        m = mint(a.report, rung=a.rung, turn=a.turn, authored=authored)
    except (OSError, ValueError) as exc:
        sys.stderr.write("usage: %s\n" % exc)
        return 2
    m.write(a.out)
    withheld = "r1" not in m.receipts
    print("minted %s rung %s receipts=%d (r1,r2 withheld: %s) authored=%d"
          % (a.out, a.rung, len(m.receipts), withheld, len(m.authored_sha256)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
