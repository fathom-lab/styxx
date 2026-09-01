# -*- coding: utf-8 -*-
"""styxx.undeclared — reconcile what an agent's tools wrote against what the diff shows.

Spec lineage: SPEC_worklog_v01_2026_08_31.md, and the pressure-test verdict that
named this comparison after the manifest inversion died to the manifest paradox
(a harness-authored declaration is tautological; a model-authored one is exactly
as unreliable as the prose it replaces).

The comparison here uses TWO artifacts with DIFFERENT authors:

    WORKLOG  — the harness's record of every write it performed
    DIFF     — what the repository actually shows changed

and produces two report-only bands:

    ATTRIBUTED    in the diff and in the worklog: the agent wrote it
    UNATTRIBUTED  in the diff, never written through the instrumented surface —
                  formatter hooks, package managers, code generators, merges

NO VERDICT. Nothing here blocks, fails, or accuses. That restraint is the whole
design: RESULT_v14 measured a shipped accusation at 0.16 precision on prose its
authors had never read, three repair cycles failed to lift it, and this lab
committed in writing to stop gating on bands whose noise floor it has not
measured. The floor for this one is measured — RESULT_collateral_census put
non-substantive files at 10.96% of 1,386,104 changed files — but a floor is not
a precision, and a precision needs a blind panel this band has not had.

The third band, comparing a MODEL-authored declaration against the diff, is
deliberately absent. It requires an artifact that does not exist yet, and
inventing it here would be the manifest paradox again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["reconcile", "main"]

SPEC = "styxx-undeclared/v0.1"


def _norm(p: str) -> str:
    return (p or "").replace("\\", "/").strip("/")


def reconcile(worklog: dict, diff_text: str) -> dict:
    """Two authors, two bands, no verdict.

    The diff side is parsed by the SAME function the gate uses, never a copy of
    it — a second parser would drift, and the day this module was written
    included a correction caused by a harness that disagreed with the parser
    about what a diff said.
    """
    from styxx.diffgate import parse_unified_diff

    status, _added = parse_unified_diff(diff_text)
    in_diff = {_norm(p) for p in status}
    written = {_norm(e.get("path", "")) for e in worklog.get("entries", [])
               if e.get("path")}

    attributed = sorted(in_diff & written)
    unattributed = sorted(in_diff - written)
    # recorded but absent from the diff: written then reverted, or written
    # outside the range the diff covers. Reported, never called a discrepancy.
    recorded_not_in_diff = sorted(written - in_diff)

    n = len(in_diff) or 1
    return {
        "spec": SPEC,
        "verdict": "UNGATED",
        "worklog_spec": worklog.get("spec"),
        "session": worklog.get("session"),
        "harness": worklog.get("harness"),
        "files_in_diff": len(in_diff),
        "files_recorded": len(written),
        "attributed": attributed,
        "unattributed": unattributed,
        "recorded_not_in_diff": recorded_not_in_diff,
        "attributed_share": round(len(attributed) / n, 4),
        "boundary": (
            "UNATTRIBUTED means the file is in the diff and was not written "
            "through the instrumented surface. It does NOT mean concealment: a "
            "formatter, a package manager, a code generator or a merge writes "
            "files no agent claimed and none of that is dishonest. This "
            "comparison carries no verdict and blocks nothing."),
        "not_measured": (
            "the precision of UNATTRIBUTED as a signal has never been measured "
            "by a blind panel; only its noise floor is known"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="styxx.undeclared",
        description="Reconcile a worklog against a diff. Report only, no verdict.")
    ap.add_argument("worklog")
    ap.add_argument("diff", help="a unified diff file")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    wl = json.loads(Path(a.worklog).read_text(encoding="utf-8"))
    diff = Path(a.diff).read_text(encoding="utf-8")
    rep = reconcile(wl, diff)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")

    print(f"session {rep['session']} · harness {rep['harness']}")
    print(f"verdict: {rep['verdict']} — this comparison carries none")
    print(f"  files in diff:   {rep['files_in_diff']}")
    print(f"  ATTRIBUTED:      {len(rep['attributed'])}  "
          f"({rep['attributed_share']:.1%} of the diff)")
    print(f"  UNATTRIBUTED:    {len(rep['unattributed'])}")
    for p in rep["unattributed"][:10]:
        print(f"     {p}")
    if len(rep["unattributed"]) > 10:
        print(f"     ... and {len(rep['unattributed']) - 10} more")
    if rep["recorded_not_in_diff"]:
        print(f"  recorded, not in this diff: {len(rep['recorded_not_in_diff'])}")
    print(f"\nboundary: {rep['boundary']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
