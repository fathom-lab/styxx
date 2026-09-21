# -*- coding: utf-8 -*-
"""styxx.ciaudit — when one CI step's tools fail, what does the workflow do?

    styxx ci-audit .                       # this checkout's .github/workflows
    styxx ci-audit OWNER/REPO              # any public repository: workflow files only, no runner, no token
    styxx ci-audit . --format json         # the receipt
    styxx ci-audit . --counted             # the counted reading of a dropped check (see engine)

A green CI light means every job exited 0. It does not mean every check ran, or that every
check could have failed. This command asks the question the light does not answer: for each
`run:` step that reaches a tool, if that step's tools failed and everything else stayed healthy,
would the workflow go red? Would a check be silently skipped? Would a check run, fail, and be
hidden? It answers by simulating the workflow -- outputs, env, `if:`, `needs:`, `fromJSON`
matrices -- in a sandbox where every tool is a stub, with no runner, no token, and no code.

Verdicts per fault, in precedence: RED (loud), FAIL_OPEN (a check that would have run is silently
not run), SWALLOWED (a check ran and its failure was hidden), ABSORBED, NO_CHECK; BASELINE_RED /
BASELINE_SKIPPED when the step cannot be read under the model.

The engine is the instrument of `papers/harness/RESULT_swallow2_which_way_it_falls_2026_09_21.md`
(30,642 faults through 2,178 workflows of the 100 repositories agents send the most pull requests
to: 92% RED; a check's own failure hidden in 23 repositories; a check silently dropped in 2). It
reproduces this repository's own #137 -- the discover step's tools fail, both gated steps are
skipped, the job is green -- and finds the repair RED.

What it does not say: a step that is loud is not thereby correct; a check that is an action
(`uses:`) is never executed and is invisible to it; a job is not a status; the check rule is a
stated heuristic. Read the receipt, which keeps every step's name and first line.

Needs PyYAML (`pip install 'styxx[ciaudit]'`); nothing here imports it, or anything heavy, until
`audit()` is called.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

CIAUDIT_VERSION = "styxx.ci-audit/v1"
VERDICTS = ("RED", "FAIL_OPEN", "SWALLOWED", "ABSORBED", "NO_CHECK")
UNINTERPRETABLE = ("BASELINE_RED", "BASELINE_SKIPPED")

__all__ = ["audit", "render", "CIAUDIT_VERSION", "VERDICTS", "UNINTERPRETABLE", "main"]


def _require_yaml() -> None:
    try:
        import yaml  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only on a bare install
        raise ImportError("styxx ci-audit needs PyYAML: pip install 'styxx[ciaudit]'") from e


def audit(target: str, *, counted: bool = False, work: Optional[str] = None, deadline_seconds: Optional[float] = None) -> dict:
    """Audit one repository. `target` is a checkout path, or `owner/repo` for a blob-less sparse
    clone of its `.github/workflows` (git and network needed). Returns the receipt: every fault
    with its verdict, every dropped check with its mechanism, and a summary."""
    import shutil
    import subprocess
    import tempfile
    import time

    _require_yaml()
    from . import engine

    t0 = time.time()
    path = Path(target)
    cloned = None
    repo = None
    if path.exists() and path.is_dir():
        tree = path.resolve()
    elif "/" in target and not target.startswith((".", "/")) and target.count("/") == 1:
        repo = target
        cloned = Path(work or tempfile.mkdtemp(prefix="ciaudit_"))
        cloned.mkdir(parents=True, exist_ok=True)
        dest, err = engine.sparse_clone(repo, cloned)
        if dest is None:
            raise RuntimeError(f"could not clone {repo}: {err}")
        tree = dest
    else:
        raise FileNotFoundError(f"{target}: not a directory, and not an owner/repo")

    rec = engine.analyse_tree(tree, repo, deadline=(t0 + deadline_seconds) if deadline_seconds else None)
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    rec.update(
        schema=CIAUDIT_VERSION,
        instrument="styxx/ciaudit/engine.py",
        instrument_sha256=engine.instrument_sha256(),
        target=target,
        head=head.stdout.strip() if head.returncode == 0 else None,
        counted=bool(counted),
        seconds=round(time.time() - t0, 1),
    )
    rec["summary"] = summarize(rec, counted=counted)
    if cloned is not None and work is None:
        shutil.rmtree(cloned, ignore_errors=True)
    return rec


def summarize(rec: dict, *, counted: bool = False) -> dict:
    """Counts for one repository, under the preregistered reading or the counted one."""
    key = "verdict_counted" if counted else "verdict"
    faults = rec.get("faults", [])
    by: dict = {}
    for f in faults:
        v = f.get(key) or f["verdict"]
        by[v] = by.get(v, 0) + 1
    interp = [f for f in faults if (f.get(key) or f["verdict"]) in VERDICTS]
    live = [f for f in interp if f.get("self_live")]
    ws = rec.get("workflow_summaries", {})
    return {
        "workflows": rec.get("workflows", 0),
        "fault_sites": len(faults),
        "interpretable": len(interp),
        "by_verdict": by,
        "red_rate": round(by.get("RED", 0) / len(interp), 3) if interp else None,
        "checks": sum(s.get("verification_steps", 0) for s in ws.values()),
        "checks_reached_in_healthy_world": sum(s.get("verification_reached_in_plus", 0) for s in ws.values()),
        "live_checks_red_under_own_fault": sum(1 for f in live if (f.get(key) or f["verdict"]) == "RED"),
        "live_checks": len(live),
        "hidden": sum(1 for f in faults if (f.get(key) or f["verdict"]) == "SWALLOWED"),
        "dropped": sum(1 for f in faults if (f.get(key) or f["verdict"]) == "FAIL_OPEN"),
        "artifact_failures_in_healthy_world": {fl: sum(s.get("artifact_failures_in_plus", {}).get(fl, 0) for s in ws.values()) for fl in ("x", "empty")},
        "reading": "counted" if counted else "preregistered",
    }


def render(rec: dict, *, counted: bool = False, width: int = 96) -> str:
    """The card: what a reader needs in one screen, then every finding, one line each."""
    from .report import card
    return card(rec, counted=counted, width=width)


def main(argv=None) -> int:
    """`python -m styxx.ciaudit TARGET [--format card|json] [--counted] [--out FILE]`."""
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(prog="styxx ci-audit", description=__doc__.split("\n\n")[1])
    ap.add_argument("target", help="a checkout path, or owner/repo (public; workflow files only)")
    ap.add_argument("--format", choices=["card", "json"], default="card")
    ap.add_argument("--counted", action="store_true", help="report the counted reading of a dropped check (reached fewer times than in the healthy world)")
    ap.add_argument("--out", default=None, help="also write the receipt (JSON) here")
    ap.add_argument("--work", default=None, help="where to clone owner/repo (default: a temporary directory, removed afterwards)")
    ap.add_argument("--deadline", type=float, default=None, help="seconds to spend at most; a capped audit says so")
    a = ap.parse_args(argv)
    try:
        rec = audit(a.target, counted=a.counted, work=a.work, deadline_seconds=a.deadline)
    except (FileNotFoundError, RuntimeError, ImportError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.out:
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    if a.format == "json":
        print(json.dumps(rec, indent=1, default=str))
    else:
        print(render(rec, counted=a.counted))
    s = rec["summary"]
    return 1 if (s["hidden"] or s["dropped"]) else 0
