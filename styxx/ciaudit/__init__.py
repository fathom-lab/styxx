# -*- coding: utf-8 -*-
"""styxx.ciaudit — when one CI step's tools fail, what does the workflow do?

    styxx ci-audit .                       # this checkout's .github/workflows
    styxx ci-audit OWNER/REPO              # any public repository: workflow files only, no runner, no token
    styxx ci-audit . --format json         # the receipt
    styxx ci-audit . --counted             # the counted reading of a dropped check (see engine)
    styxx ci-audit . --no-actions          # SWALLOW-2's reading: checks are `run:` steps only
    styxx ci-audit . --repair              # for every finding, a verified repair of the workflow text, or why there is none
    styxx ci-audit . --history             # for every finding, the commit it has been hidden since: born hidden, or acquired, and by what
    styxx ci-audit . --base origin/main    # the pull request's gate: what this HEAD hides that its base did not (exit 1 if anything)
    styxx ci-audit . --pr 42               # the same gate on pull request #42 of origin, from GitHub's own test-merge ref, no checkout
    styxx ci-audit OWNER/REPO --pr 42      # ... on any public repository

A green CI light means every job exited 0. It does not mean every check ran, or that every
check could have failed. This command asks the question the light does not answer: for each
`run:` step that reaches a tool, if that step's tools failed and everything else stayed healthy,
would the workflow go red? Would a check be silently skipped? Would a check run, fail, and be
hidden? It answers by simulating the workflow -- outputs, env, `if:`, `needs:`, `fromJSON`
matrices -- with its tools stubbed, no runner and no token. The steps' shell itself runs, on this
machine, in a temporary directory with an empty environment: what the stubs do not cover -- `rm`,
`mkdir`, a redirect -- is real, a path a script names outside that directory is this machine's,
and a value the empty flavour empties can make a relative path absolute (`rm -rf "$D"/build`
with `D` empty). Run it where that is safe -- a CI runner, a container -- and not on a checkout
you do not trust (papers/harness/RESULT_swallow14_the_empty_list_2026_09_22.md, §0).

Verdicts per fault, in precedence: RED (loud), FAIL_OPEN (a check that would have run is silently
not run), SWALLOWED (a check ran and its failure was hidden), ABSORBED, NO_CHECK; BASELINE_RED /
BASELINE_SKIPPED when the step cannot be read under the model.

The engine is the instrument of `papers/harness/RESULT_swallow2_which_way_it_falls_2026_09_21.md`
(30,642 faults through 2,178 workflows of the 100 repositories agents send the most pull requests
to: 92% RED; a check's own failure hidden in 23 repositories; a check silently dropped in 2). It
reproduces this repository's own #137 -- the discover step's tools fail, both gated steps are
skipped, the job is green -- and finds the repair RED.

A check that is an action (`uses: pre-commit/action`, `lycheeverse/lychee-action`, CodeQL's
`analyze`) is never executed, but it is counted: the declared list in `actions.py` (SWALLOW-3's
catalogue, every entry with its reason) says which actions are checks, and a fault that closes the
gate in front of one is FAIL_OPEN. `verdict_runs_only` on every fault keeps SWALLOW-2's reading,
without the catalogue.

`--repair` asks the next question of every finding: is there a small change to the workflow that
makes the same fault loud and leaves a healthy run exactly as it was? Two stated repairs are tried
at the fault site (`repair.py`: remove the step's `continue-on-error`; make its shell strict), each
verified on both halves against the same model, and the card prints the diff -- or which half
failed, and what the original line was protecting (SWALLOW-4). For a finding those leave
unverified, two edits to the script's logic are tried next (`repair_structural.py`: a guard whose
failing tool is not its green path; a query without its `|| echo` default), verified the same way
(SWALLOW-5). For what those leave, a third stage (`repair_frontier.py`, SWALLOW-13 and -14): a
`$(...)` whose status its line throws away hoisted onto a line of its own that keeps it, a list read
from a process substitution whose command's failure is waited for (`wait $!`) instead of read as an
empty list, a background job whose early death nobody waits for checked once, `|| exit 0` removed,
a fallback hidden by a continued line, each alone or with the `continue-on-error` removed --
verified the same way. A finding none of the
three stages repairs carries two readings of its script, when they match: its failure is routed (a
flag written to GITHUB_ENV or GITHUB_OUTPUT that a later step reads), or the script says it is not
fatal (a `::warning`, or stated words). A reading is context for the reviewer, not a verdict.

`--history` asks since when. Every revision of the workflow on the checkout's own mainline
(first-parent from HEAD) is read with the same engine and the finding's step is followed as a
lineage -- the same job and step name -- back to the commit that created it. The card says whether
the step was born hidden (the `|| true` was written with it) or acquired its hiding later, in which
commit, by which mechanism (`continue-on-error`, `|| true`, `set +e`, a `|| echo` default, gating,
a rewrite, or a change elsewhere), and whether that commit's message says why in one of a stated
list of words (`history.py`, SWALLOW-6). For `owner/repo` the clone is deepened to the GitHub
Actions YAML era (2019-08) first, blob-less.

`--base <rev>` is the gate for a pull request. Only the workflows that changed between the
merge-base of <rev> and HEAD are read, every step is matched across the two revisions, and the
card says what HEAD hides that the base did not -- a check born hidden, or an existing one made
hidden, each with the verified repair -- what it made loud or removed, and what was hidden on both
sides (not this change's doing). With `--base` the exit status is 1 only when the change brings a
new hidden check (`differential.py`, SWALLOW-7).

What it does not say: a step that is loud is not thereby correct; an action check cannot be seen
to fail (no fault is injected into one); a local action, a reusable workflow and `github-script`
cannot be read; a job is not a status; the check rule and the catalogue are stated heuristics.
Read the receipt, which keeps every step's name and first line.

Stated plainly: the catalogue ships under an INVALID receipt
(`papers/harness/RESULT_swallow3_the_checks_that_are_actions_2026_09_21.md`). Its own gates
held -- every verdict it moves is one of the declared transitions, RED and SWALLOWED never move --
but the cycle's reproduction gate found that the engine underneath is not bit-reproducible on the
population: 8 records of 30,642 differ between two runs at the same commits, because of the real
`date`, a tie in the order stubs are created, and a background subshell racing the log that counts
what a step reached. That is true of this command with or without the catalogue, and it is the
next cycle's first item. A verdict here can move by those mechanisms; a repository's card can
differ between two runs on a step that calls `date`, backgrounds a command, or ties two stubs.

Needs PyYAML (`pip install 'styxx[ciaudit]'`); nothing here imports it, or anything heavy, until
`audit()` is called.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

CIAUDIT_VERSION = "styxx.ci-audit/v2"
VERDICTS = ("RED", "FAIL_OPEN", "SWALLOWED", "ABSORBED", "NO_CHECK")
UNINTERPRETABLE = ("BASELINE_RED", "BASELINE_SKIPPED")

__all__ = ["audit", "render", "CIAUDIT_VERSION", "VERDICTS", "UNINTERPRETABLE", "main"]


def _require_yaml() -> None:
    try:
        import yaml  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only on a bare install
        raise ImportError("styxx ci-audit needs PyYAML: pip install 'styxx[ciaudit]'") from e


def audit(target: str, *, counted: bool = False, actions: bool = True, repair: bool = False, history: bool = False,
          base: Optional[str] = None, pr: Optional[int] = None, remote: str = "origin", work: Optional[str] = None,
          deadline_seconds: Optional[float] = None, confined: Optional[bool] = None) -> dict:
    """Audit one repository. `target` is a checkout path, or `owner/repo` for a blob-less sparse
    clone of its `.github/workflows` (git and network needed). Returns the receipt: every fault
    with its verdict, every dropped check with its mechanism, and a summary. `actions=False` is
    SWALLOW-2's reading (checks are `run:` steps only); the default counts the catalogue's actions.

    `confined`: the part that simulates -- every step's shell -- runs in a child process confined by
    Landlock (`confine.run`: writes only in its own scratch directory, no TCP, no signal outside the
    audit). True requires it (`confine.Unconfinable` where the kernel has none), False runs it
    unconfined, None (the default) confines where the kernel can. The clone, `--history`'s deepening
    and `--pr`'s fetch come first, unconfined: they need the network and write only the clone's
    `.git`. The receipt's `confinement` says which it was."""
    import shutil
    import subprocess
    import tempfile
    import time

    _require_yaml()
    from . import confine
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
    deadline = (t0 + deadline_seconds) if deadline_seconds else None
    history_clone = None
    if history and cloned is not None:                  # the network, before the confinement
        from .history import deepen
        history_clone = deepen(tree)
    if pr is not None and not base:
        from .differential import pr_fetch
        pr_fetch(tree, int(pr), remote=remote)

    def analyse() -> dict:
        rec = engine.analyse_tree(tree, repo, deadline=deadline, actions=actions)
        if repair:
            from .repair import REPAIRS, repair_faults
            from .repair_frontier import REPAIRS as FRONTIER
            from .repair_structural import REPAIRS as STRUCTURAL
            rec["repairs"] = repair_faults(tree, rec["faults"])
            rec["repair_catalogue"] = list(REPAIRS) + list(STRUCTURAL) + list(FRONTIER)
        if history:
            from .history import since
            if history_clone is not None:
                rec["history_clone"] = history_clone
            rec["history"] = since(tree, rec["faults"], deadline=deadline)
        if base:
            from .differential import tree_differential
            try:
                rec["differential"] = tree_differential(tree, base)
            except RuntimeError as e:
                rec["differential"] = {"base_ref": base, "error": str(e)[:200], "fires": False, "workflows": [], "new_hidden": 0, "removed_hidden": 0, "still_hidden": 0}
        elif pr is not None:
            from .differential import pr_differential
            try:
                rec["differential"] = pr_differential(tree, int(pr), remote=remote, fetch=False)
            except RuntimeError as e:
                rec["differential"] = {"base_ref": f"pull request #{pr}", "error": str(e)[:200], "fires": False, "workflows": [], "new_hidden": 0, "removed_hidden": 0, "still_hidden": 0}
        head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        rec["head"] = head.stdout.strip() if head.returncode == 0 else None
        return rec

    try:
        if confined or (confined is None and confine.abi()):
            rec, info = confine.run(analyse)
        else:
            rec = analyse()
            info = {"schema": confine.SCHEMA, "confined": False,
                    "why": "unconfined by request" if confined is False else "this kernel has no Landlock"}
    finally:
        if cloned is not None and work is None:
            shutil.rmtree(cloned, ignore_errors=True)
    rec.update(
        schema=CIAUDIT_VERSION,
        instrument="styxx/ciaudit/engine.py",
        instrument_sha256=engine.instrument_sha256(),
        target=target,
        counted=bool(counted),
        actions=bool(actions),
        confinement=info,
        seconds=round(time.time() - t0, 1),
    )
    rec["summary"] = summarize(rec, counted=counted)
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
        "action_checks": sum(s.get("action_checks", 0) for s in ws.values()),
        "action_checks_reached_in_healthy_world": sum(s.get("action_checks_reached_in_plus", 0) for s in ws.values()),
        "action_checks_unverified": sum(s.get("action_checks_unverified", 0) for s in ws.values()),
        "dropped_action_checks": sum(len(f.get("dropped_actions", [])) for f in faults if (f.get(key) or f["verdict"]) == "FAIL_OPEN"),
        "moved_by_the_catalogue": sum(1 for f in faults if f.get("verdict_runs_only") and f["verdict_runs_only"] != f["verdict"]),
        "unreadable_steps": sum(n for s in ws.values() for c, n in s.get("step_classes", {}).items() if c in ("local", "docker", "workflow", "not:unreadable")),
        "reading": ("counted" if counted else "preregistered") + ("" if rec.get("actions", True) else ", run: steps only"),
        "repairs": ({"targets": len(rec["repairs"]), "verified": sum(1 for t in rec["repairs"] if t.get("verified_repair")),
                     "by_repair": _count(t["verified_repair"] for t in rec["repairs"] if t.get("verified_repair")),
                     "rejected_changes_healthy_run": sum(1 for t in rec["repairs"] if not t.get("verified_repair")
                                                         and any(c.get("applies") and c.get("unchanged") is False for c in t["candidates"])),
                     "not_loud": sum(1 for t in rec["repairs"] if not t.get("verified_repair") and any(c.get("applies") for c in t["candidates"])
                                     and all(c.get("unchanged") is not False for c in t["candidates"])),
                     "no_candidate_applies": sum(1 for t in rec["repairs"] if all(not c.get("applies") for c in t["candidates"])),
                     "routed": sum(1 for t in rec["repairs"] if (t.get("readings") or {}).get("routed")),
                     "declared": sum(1 for t in rec["repairs"] if (t.get("readings") or {}).get("declared"))}
                    if "repairs" in rec else None),
    }


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


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
    ap.add_argument("--no-actions", action="store_true", help="SWALLOW-2's reading: a check is a run: step only; the catalogue of checking actions is not applied")
    ap.add_argument("--repair", action="store_true", help="for every finding, try the stated repairs of the workflow text, in three stages, and verify each: loud under the same fault, healthy run unchanged")
    ap.add_argument("--history", action="store_true", help="for every finding, the commit it has been hidden since -- born hidden, or acquired later and by what -- from the checkout's git history")
    ap.add_argument("--base", default=None, help="the pull request's gate: read only the workflows changed since the merge-base with this revision, and exit 1 only if HEAD hides a check the base did not")
    ap.add_argument("--pr", type=int, default=None, help="the same gate on pull request N of the remote, from GitHub's refs/pull/N/merge (its test merge against the base branch) -- no checkout of the branch needed")
    ap.add_argument("--remote", default="origin", help="with --pr: the remote to fetch the pull request from (default origin)")
    ap.add_argument("--out", default=None, help="also write the receipt (JSON) here")
    ap.add_argument("--work", default=None, help="where to clone owner/repo (default: a temporary directory, removed afterwards)")
    ap.add_argument("--deadline", type=float, default=None, help="seconds to spend at most; a capped audit says so")
    ap.add_argument("--unconfined", action="store_true",
                    help="run the steps' shell without confining it -- on this machine, as this user; only where that is safe "
                         "(a CI runner, a container). Also STYXX_CIAUDIT_UNCONFINED=1")
    a = ap.parse_args(argv)
    from . import confine
    unconfined = a.unconfined or confine.unconfined_allowed()
    if not unconfined and not confine.abi():
        print("error: this machine cannot confine the audit (it needs Linux's Landlock, 5.13+). ci-audit runs each workflow step's "
              "shell with its tools stubbed; what the stubs do not cover -- rm, mkdir, a redirect -- would act on this machine, and an "
              "empty value can make a scoped path the root. Run it on a CI runner or in a container, or pass --unconfined to run it "
              "here anyway.", file=sys.stderr)
        return 2
    try:
        rec = audit(a.target, counted=a.counted, actions=not a.no_actions, repair=a.repair, history=a.history, base=a.base, pr=a.pr, remote=a.remote,
                    work=a.work, deadline_seconds=a.deadline, confined=not unconfined)
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
    if rec.get("differential") is not None:
        d = rec["differential"]
        return 2 if d.get("error") else (1 if d.get("fires") else 0)
    return 1 if (s["hidden"] or s["dropped"]) else 0
