# -*- coding: utf-8 -*-
"""SILENT-PASS — score a detector against real failures that reported success.

    from benchmarks.silent_pass import load_cases, score

    def my_detector(source: str, filename: str) -> set:
        \"\"\"Return the line numbers you flag.\"\"\"
        ...

    print(score(my_detector))

Every case is a real defect with a confirmed fix in this repository's history.
Pre-fix source is fetched from ``fix_commit~1`` rather than copied into the
corpus, so the benchmark cannot drift from the history it cites.

READ THIS BEFORE QUOTING A NUMBER
─────────────────────────────────
This corpus contains ONLY true positives. It therefore measures **recall and
nothing else**. A detector that flags every line in every file scores 1.00 here
and is worthless. Precision has to be estimated separately, by hand, on
unlabelled code — `score()` prints that sentence with every result, and
`papers/CENSUS_absence_2026_08_19.md` shows how we did it on our own detector
(5 of 14, on a random sample).

Reporting recall from this corpus without a precision estimate beside it is the
exact error the corpus exists to document.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

__all__ = ["Case", "ScoreResult", "load_cases", "score", "CAVEAT"]

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent

CAVEAT = (
    "RECALL ONLY. This corpus holds only true positives, so it cannot measure "
    "precision — a detector that flags everything scores 1.00. Estimate "
    "precision separately on unlabelled code and report it beside this number. "
    "Corpus provenance: one codebase (styxx), found by its own authors and "
    "tools, so it over-represents shapes those methods find."
)


@dataclass
class Case:
    id: str
    subtype: str
    module: str
    fix_commit: str
    defect_line: int
    what_failed: str
    what_was_returned: str
    why_it_reads_healthy: str
    consumer: str
    fix: str

    def pre_fix_source(self, repo: Optional[Path] = None) -> Optional[str]:
        """The module as it stood immediately BEFORE the fix landed."""
        repo = Path(repo) if repo else _REPO
        r = subprocess.run(
            ["git", "show", f"{self.fix_commit}~1:{self.module}"],
            capture_output=True, text=True, cwd=str(repo),
        )
        return r.stdout if r.returncode == 0 and r.stdout else None


@dataclass
class ScoreResult:
    recall: Optional[float]
    n_cases: int
    n_caught: int
    n_unavailable: int
    by_subtype: Dict[str, Dict[str, int]] = field(default_factory=dict)
    caught: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)
    unavailable: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"recall": self.recall, "n_cases": self.n_cases,
                "n_caught": self.n_caught, "n_unavailable": self.n_unavailable,
                "by_subtype": self.by_subtype, "caught": list(self.caught),
                "missed": list(self.missed), "unavailable": list(self.unavailable),
                "caveat": CAVEAT}

    def render(self) -> str:
        lines = ["SILENT-PASS — recall against real failures that reported success", ""]
        if self.recall is None:
            lines.append("  NO CASES SCORED — every pre-fix source was unavailable.")
            lines.append("  That is not a score of zero; it is no score. Check that")
            lines.append("  the repository has full history (CI clones shallow).")
            return "\n".join(lines)
        lines.append(f"  recall      {self.n_caught}/{self.n_cases} = {self.recall:.0%}")
        if self.n_unavailable:
            lines.append(f"  unscored    {self.n_unavailable} (pre-fix source unavailable)")
        lines.append("")
        for st in sorted(self.by_subtype):
            b = self.by_subtype[st]
            lines.append(f"    {st:<8} {b['caught']}/{b['total']}")
        if self.missed:
            lines.append("")
            lines.append(f"  missed: {', '.join(self.missed)}")
        lines.append("")
        lines.append(f"  {CAVEAT}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        if self.recall is None:
            return "<ScoreResult NO CASES SCORED — not a zero>"
        return f"<ScoreResult recall={self.n_caught}/{self.n_cases} (recall only)>"


def load_cases(path: Optional[Path] = None) -> List[Case]:
    data = json.loads((Path(path) if path else _HERE / "cases.json")
                      .read_text(encoding="utf-8"))
    return [Case(**c) for c in data["cases"]]


def score(
    detector: Callable[[str, str], Set[int]],
    *,
    tolerance: int = 10,
    repo: Optional[Path] = None,
    cases: Optional[List[Case]] = None,
) -> ScoreResult:
    """Run `detector(source, filename) -> {line numbers}` over the corpus.

    A case counts as caught when the detector flags a line within `tolerance`
    of the recorded defect line — the defect and the value it produces are
    often a few lines apart in a multi-line call.

    A case whose pre-fix source cannot be fetched is reported as UNSCORED, never
    as a miss. Scoring a case you could not load as a failure would be the same
    error this corpus documents.
    """
    cases = cases or load_cases()
    res = ScoreResult(recall=None, n_cases=0, n_caught=0, n_unavailable=0)

    for c in cases:
        src = c.pre_fix_source(repo)
        if not src:
            res.n_unavailable += 1
            res.unavailable.append(c.id)
            continue
        bucket = res.by_subtype.setdefault(c.subtype, {"caught": 0, "total": 0})
        bucket["total"] += 1
        res.n_cases += 1
        try:
            flagged = set(detector(src, c.module) or ())
        except Exception:
            flagged = set()
        if any(abs(int(ln) - c.defect_line) <= tolerance for ln in flagged):
            res.n_caught += 1
            res.caught.append(c.id)
            bucket["caught"] += 1
        else:
            res.missed.append(c.id)

    res.recall = (res.n_caught / res.n_cases) if res.n_cases else None
    return res


def main(argv=None) -> int:
    """Score the detectors that ship with styxx."""
    import argparse
    ap = argparse.ArgumentParser(prog="silent_pass")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    from styxx.absence import scan_source as absence_scan
    from styxx.loops import scan_source as loops_scan

    def absence_detector(src: str, name: str) -> Set[int]:
        try:
            return {f.line for f in absence_scan(src, name)}
        except SyntaxError:
            return set()

    def loops_detector(src: str, name: str) -> Set[int]:
        try:
            v = loops_scan(src, name)
        except SyntaxError:
            return set()
        return {d.line for d in v.derivations} | {t.line for t in v.trust}

    def either(src: str, name: str) -> Set[int]:
        return absence_detector(src, name) | loops_detector(src, name)

    if not a.json:
        for label, det in (("styxx.absence", absence_detector),
                           ("styxx.loops", loops_detector),
                           ("both", either)):
            r = score(det)
            print(f"=== {label} ===")
            print(r.render())
            print()
            if label == "both":
                blind = sorted(st for st, b in r.by_subtype.items() if b["caught"] == 0)
                if blind:
                    print(f"  UNCOVERED BY EITHER TOOL: {', '.join(blind)}")
                    print("  Those are cases whose defect is an ABSENT guard -- code that")
                    print("  was never written. No pass over source can flag that; only a")
                    print("  test or a reviewer can.")
        return 0

    result = score(absence_detector)
    print(json.dumps({"styxx.absence": result.as_dict()}, indent=2)
          if a.json else "styxx.absence\n" + result.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
