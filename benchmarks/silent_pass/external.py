# -*- coding: utf-8 -*-
"""Scoring for SP-EXT — the corpus of silent-pass defects in code we did not write.

Same shape as the internal scorer in ``__init__``, with one difference that
matters: pre-fix source comes from a **clone of the upstream repository**, not
from this one. Ground truth for an SP-EXT case is the upstream maintainers' own
fix commit, so the source has to come from where the fix landed.

    from benchmarks.silent_pass.external import load_external, score_external
    r = score_external(my_detector, clone_root="/tmp/spcorpus")
    print(r.render())

**Score against PRE-FIX source, never post-fix.** A detector run on the fixed code
is being asked whether it can find a defect that is no longer there, and it will
correctly find nothing, and that number means nothing.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

__all__ = ["ExternalCase", "ExternalScore", "load_external", "score_external",
           "EXT_CAVEAT"]

_HERE = Path(__file__).resolve().parent

EXT_CAVEAT = (
    "RECALL ONLY, and on a corpus that is INCOMPLETE. SP-EXT holds only true "
    "positives, so it cannot measure precision — a detector that flags every "
    "line scores 1.00. Estimate precision separately on unlabelled code and "
    "report it beside this number. Recall of the HARVEST is unknown: cases come "
    "from a frozen commit-message regex intersected with a frozen diff shape, so "
    "fixes described any other way are invisible to it. SP-EXT is a lower bound "
    "on incidence and must never be quoted as a rate."
)


@dataclass
class ExternalCase:
    id: str
    repo: str                     # owner/name, e.g. "Giskard-AI/giskard"
    fix_commit: str
    module: str                   # path within the upstream repository
    subtype: str
    defect_line: Optional[int] = None
    symbol: str = ""
    url: str = ""
    what_failed: str = ""
    what_was_returned: str = ""
    why_it_reads_healthy: str = ""
    consumer: str = ""
    fix: str = ""
    prefix_code: str = ""
    extra: Dict = field(default_factory=dict)

    @property
    def clone_name(self) -> str:
        return self.repo.split("/")[-1]

    def pre_fix_source(self, clone_root) -> Optional[str]:
        """The upstream module as it stood immediately BEFORE the fix landed."""
        repo_dir = Path(clone_root) / self.clone_name
        if not (repo_dir / ".git").exists():
            return None
        r = subprocess.run(
            ["git", "-C", str(repo_dir), "show", f"{self.fix_commit}~1:{self.module}"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r.stdout if r.returncode == 0 and r.stdout else None


@dataclass
class ExternalScore:
    recall: Optional[float] = None
    n_cases: int = 0
    n_caught: int = 0
    n_unavailable: int = 0
    caught: List[str] = field(default_factory=list)
    missed: List[str] = field(default_factory=list)
    unavailable: List[str] = field(default_factory=list)
    by_repo: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def render(self) -> str:
        if self.recall is None:
            return ("SCORED NOTHING — NOT A CLEAN RESULT: no case's upstream "
                    f"source could be fetched ({self.n_unavailable} unavailable). "
                    "Clone the repositories first.")
        lines = [f"SP-EXT recall {self.recall:.0%}  "
                 f"({self.n_caught}/{self.n_cases} caught, "
                 f"{self.n_unavailable} unavailable and NOT counted as misses)"]
        for repo, b in sorted(self.by_repo.items()):
            lines.append(f"    {repo:28} {b['caught']}/{b['total']}")
        if self.unavailable:
            lines.append(f"    UNSCORED: {', '.join(self.unavailable)}")
        lines.append("  " + EXT_CAVEAT)
        return "\n".join(lines)


def load_external(path: Optional[Path] = None) -> List[ExternalCase]:
    data = json.loads((Path(path) if path else _HERE / "external.json")
                      .read_text(encoding="utf-8"))
    known = set(ExternalCase.__dataclass_fields__) - {"extra"}
    out = []
    for c in data["cases"]:
        kw = {k: v for k, v in c.items() if k in known}
        kw["extra"] = {k: v for k, v in c.items() if k not in known}
        out.append(ExternalCase(**kw))
    return out


def score_external(
    detector: Callable[[str, str], Set[int]],
    *,
    clone_root,
    tolerance: int = 10,
    cases: Optional[List[ExternalCase]] = None,
) -> ExternalScore:
    """Run ``detector(source, filename) -> {line numbers}`` over SP-EXT.

    A case whose upstream source cannot be fetched is **UNSCORED, never a miss**.
    Counting a case you could not load as a failure would be the same error this
    corpus documents, and a corpus that does it has no standing to complain.

    A case with no recorded ``defect_line`` is likewise UNSCORED — there is
    nothing to localise against, and scoring it either way would be inventing a
    result.
    """
    cases = cases or load_external()
    res = ExternalScore()

    for c in cases:
        src = c.pre_fix_source(clone_root)
        if not src or c.defect_line is None:
            res.n_unavailable += 1
            res.unavailable.append(c.id)
            continue
        bucket = res.by_repo.setdefault(c.repo, {"caught": 0, "total": 0})
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
