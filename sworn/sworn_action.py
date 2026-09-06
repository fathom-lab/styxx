# -*- coding: utf-8 -*-
"""GitHub Action entry for sworn output — report-only, injection-safe, no network.

Built to ``papers/sworn/SPEC_sworn_action_v01_2026_09_05.md``, frozen before this file existed.
Leg 3, item 4 of the plan: *report-only until the measurement prices FAILED; dogfooded only after
the operator merges the workflow*.

THE RUNG, STATED. This action declares the rung the workflow declares (``rung``,
``after-turn-on-base``, ``base-pinned-workflow``), hands the declarations to the adapters
unchanged, and lowers to L1 with a printed reason wherever the GitHub adapter would refuse L2 —
on a ``pull_request`` from a fork without ``base-pinned-workflow``, and whenever
``after-turn-on-base`` is not declared. It never raises a rung and detects nothing. On a pull
request from a fork the minting job is the claimant's: the ``pull_request`` event runs the
workflow file, the action ref, the command and the code from the pull request's head with a
read-only token, so the manifest declares L1 (weak), never L2, and the summary says why.

WHAT IT DOES, IN ORDER. Reads the event from ``GITHUB_EVENT_PATH`` (the body and every path go
from the event file and from git plumbing into Python; the only string a shell sees is
``command``, the workflow's own configuration, which arrives through an environment variable).
Runs ``command`` after the turn with ``SWORN_JUNIT`` exported. Mints, after the command
returned, the JUnit adapter's manifest over the report the command wrote and the GitHub
adapter's manifest over the event bytes and a diff taken with git from the checkout, both at
the minted rung; composes one manifest from them — r1 to r4 the JUnit adapter's receipts id for
id, r5 to r9 the GitHub adapter's r1 to r5 shifted by four, an absence never renumbering — and
records every blob the turn added or modified into ``authored_sha256``. Verifies the pull
request body as submitted and every changed ``.md`` carrying ``<sworn``, bytes read at the head
commit through ``styxx.sworn.GitTree`` and never from the working tree, against the composed
manifest; writes a verdict receipt per document; writes the job summary with the rung and the
harness string on every row; sets outputs; exits zero.

EXIT. Zero on every verdict, on every refusal it reports, on DID NOT RUN, on a command that
failed or timed out. Two only for a usage error before anything ran (no event path, an empty
command). No input can fail the job, and the file carries no error or warning annotation
command, by test (``tests/test_sworn_action.py`` greps for them).

It imports the adapters and the verifier; nothing under ``styxx/`` imports it.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from styxx.harness import github as github_adapter
from styxx.harness import junit as junit_adapter
from styxx.sworn import (RUNGS, GitTree, Manifest, _CANDIDATE, _headline,
                         _write_json_lf, issue_receipt, verify)

__all__ = ["ACTION_VERSION", "LAYOUT", "FORK_RULE", "REPORT_ONLY", "decide_rung", "compose", "main"]

ACTION_VERSION = "sworn-action/0.1"
REPORT_ONLY = "report-only until the measurement prices FAILED"
EVENTS = ("pull_request", "push")
NULL_SHA = "0" * 40
_HEX40 = re.compile(r"[0-9a-f]{40}")

# The composed layout, fixed per action version and never renumbered (SPEC, "The manifest the
# runner mints"). An absence leaves every other id where it is.
LAYOUT = (
    ("r1", "the passed count as ASCII digits", "JUnit adapter r1, id for id; withheld with r2 when the reader parsed no source"),
    ("r2", "the failures count as ASCII digits (harness errors kept apart)", "JUnit adapter r2, id for id; withheld with r1"),
    ("r3", "the report bytes, whole", "JUnit adapter r3, id for id"),
    ("r4", "load_evidence's object as RFC 8785 canonical JSON, one trailing LF (r4#/totals/errors, r4#/outcome are leaves)", "JUnit adapter r4, id for id"),
    ("r5", "the diff between base and head as git printed it", "GitHub adapter r1; omitted when the diff could not be taken"),
    ("r6", "the base sha as ASCII", "GitHub adapter r2"),
    ("r7", "the head sha as ASCII", "GitHub adapter r3"),
    ("r8", "the event name as ASCII", "GitHub adapter r4"),
    ("r9", "the event payload bytes, whole (r9#/pull_request/number is a leaf)", "GitHub adapter r5"),
)
JUNIT_IDS = ("r1", "r2", "r3", "r4")
GITHUB_SHIFT = {"r1": "r5", "r2": "r6", "r3": "r7", "r4": "r8", "r5": "r9"}

# The README's sentence, verbatim (SPEC, "The rung, declared by the workflow, and the rule for
# forks"). Printed into the summary, run.json and the composed manifest's harness string on every
# fork pull request.
FORK_RULE = (
    "On a pull request from a fork, the minting job is the claimant's. The pull_request event "
    "runs the workflow file, the action ref, the command and the code from the pull request's "
    "head — every one of them under the pull request author's control — with a read-only token. "
    "The runner did mint after the turn, but \"a runner the agent could not write to\" is not true "
    "of a job the agent configured, so the manifest declares rung L1 (weak), never L2, and the job "
    "summary says why. L2 is declared only when the head repository is the base repository, or "
    "when the workflow declares base-pinned-workflow because it is one. Do not switch the trigger "
    "to pull_request_target with a checkout of the head to change this: that hands the base "
    "repository's token to the claimant's code.")

NOT_AFTER_TURN = ("L2 was declared without after-turn-on-base: the declaration L2 rests on — this "
                  "job ran after the agent's turn ended, on a runner and from a workflow file the "
                  "agent could not write to — was not made; minted at L1")


# ----------------------------------------------------------------------------------------------
# small helpers: git plumbing (argument lists, never a shell), LF writes, safe names
# ----------------------------------------------------------------------------------------------

def _git(ws: str, *args: str, stdin: Optional[bytes] = None) -> Tuple[int, bytes]:
    try:
        r = subprocess.run(["git", "-C", ws, *args], input=stdin, capture_output=True, check=False)
    except (OSError, ValueError):
        return 127, b""
    return r.returncode, r.stdout


def _commit_present(ws: str, sha: str) -> bool:
    if not sha or not _HEX40.fullmatch(sha) or sha == NULL_SHA:
        return False
    rc, out = _git(ws, "cat-file", "-t", sha)
    return rc == 0 and out.strip() == b"commit"


def _changed_paths(ws: str, base: str, head: str, event_name: str) -> Optional[List[str]]:
    """Paths added or modified by the turn, or None when git could not compare."""
    rng = ("%s..%s" if event_name == "push" else "%s...%s") % (base, head)
    rc, out = _git(ws, "-c", "diff.renames=false", "diff", "--name-only", "--diff-filter=AM", "-z", rng)
    if rc != 0:
        return None
    return [p.decode("utf-8", errors="surrogateescape") for p in out.split(b"\0") if p]


def _diff_bytes(ws: str, base: str, head: str, event_name: str) -> Optional[bytes]:
    rng = ("%s..%s" if event_name == "push" else "%s...%s") % (base, head)
    rc, out = _git(ws, "-c", "core.quotepath=false", "-c", "core.abbrev=7", "-c", "diff.noprefix=false",
                   "-c", "diff.mnemonicPrefix=false", "-c", "diff.algorithm=myers", "-c", "diff.renames=false",
                   "diff", "--no-color", "--no-ext-diff", rng)
    return out if rc == 0 else None


def _blobs_at(ws: str, head: str, paths: List[str]) -> Tuple[List[bytes], List[str]]:
    """Bytes of every blob among ``paths`` at ``head`` through one ``cat-file --batch``; the paths
    that were not blobs (a submodule, a symlink, a name git could not spell on one line)."""
    usable = [p for p in paths if "\n" not in p]
    skipped = [p for p in paths if "\n" in p]
    if not usable:
        return [], skipped
    stdin = "".join("%s:%s\n" % (head, p) for p in usable).encode("utf-8", errors="surrogateescape")
    rc, out = _git(ws, "cat-file", "--batch", stdin=stdin)
    if rc != 0:
        return [], skipped + usable
    blobs: List[bytes] = []
    pos = 0
    for p in usable:
        nl = out.find(b"\n", pos)
        if nl < 0:
            skipped.append(p)
            continue
        header = out[pos:nl].split()
        pos = nl + 1
        if len(header) < 3 or header[1] != b"blob" or not header[2].isdigit():
            skipped.append(p)            # "<spec> missing", a tree, a commit
            continue
        size = int(header[2])
        blobs.append(out[pos:pos + size])
        pos += size + 1                  # the LF cat-file prints after each object
    return blobs, skipped


def _safe_name(name: str, taken: Dict[str, int]) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "document"
    n = taken.get(stem, 0)
    taken[stem] = n + 1
    return stem if n == 0 else "%s.%d" % (stem, n)


def _cell(text: str) -> str:
    return str(text).replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _write_text_lf(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def _under(path: Path, ws: str) -> str:
    """A path relative to the workspace when it lies inside it (so run.json carries no machine
    path for the common layout), else as it is."""
    try:
        return Path(path).resolve().relative_to(Path(ws).resolve()).as_posix()
    except ValueError:
        return str(path)


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _say(text: str) -> None:
    """stdout, survivable on a console that cannot spell the headline's characters: a verdict
    line must never turn into a crash and a nonzero exit."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    sys.stdout.write(text.encode(enc, errors="replace").decode(enc, errors="replace") + "\n")


# ----------------------------------------------------------------------------------------------
# the rung, the command, the composition
# ----------------------------------------------------------------------------------------------

def decide_rung(declared: str, after_turn_on_base: bool, base_pinned_workflow: bool,
                event_name: str, fork: Optional[bool]) -> Tuple[str, Optional[str]]:
    """(minted rung, reason it was lowered or None). Mirrors the GitHub adapter's refusals; the
    adapter is still called with the declarations unchanged and is the authority."""
    if declared not in RUNGS:
        return "L1", ("declared rung %r is not in the ladder %s; minted at L1"
                      % (declared, "/".join(RUNGS)))
    if declared == "L1":
        return "L1", None
    if not after_turn_on_base:
        return "L1", NOT_AFTER_TURN
    if event_name == "pull_request" and fork is not False and not base_pinned_workflow:
        why = FORK_RULE
        if fork is None:
            why = ("the head repository is absent from the payload (a deleted fork) and is treated "
                   "as a fork. " + why)
        return "L1", why
    return "L2", None


def run_command(command: str, ws: str, junit_path: Path, timeout_s: float) -> Tuple[int, bytes, bool]:
    """(exit status, stdout+stderr, timed out). The command is the workflow's configuration; it is
    the one string a shell sees. Its failure is recorded and never becomes the action's exit."""
    env = dict(os.environ, SWORN_JUNIT=str(junit_path))
    try:
        r = subprocess.run(command, shell=True, cwd=ws, env=env, capture_output=True,
                           timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or b"") + (exc.stderr or b""), True
    except OSError as exc:
        return 127, str(exc).encode("utf-8", errors="replace"), False
    return r.returncode, r.stdout + r.stderr, False


def compose(harness: str, turn: str, rung: str, junit_m: Optional[Manifest],
            github_m: Manifest, authored: List[bytes]) -> Manifest:
    """One manifest from two: every entry is the adapter's entry with only its id changed."""
    m = Manifest(harness=harness, turn=turn, rung=rung)
    for blob in authored:
        m.record_authored(blob)
    if junit_m is not None:
        for rid in JUNIT_IDS:
            if rid in junit_m.receipts:
                m.receipts[rid] = dict(junit_m.receipts[rid])
    for rid, new in GITHUB_SHIFT.items():
        if rid in github_m.receipts:
            entry = dict(github_m.receipts[rid])
            entry["id"] = new
            m.receipts[new] = entry
    return m


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------

def _summary_head(run: dict) -> List[str]:
    lines = ["## styxx sworn — %s" % REPORT_ONLY, "",
             "`%s` · exit 0 on every verdict; the command's exit status is a record, never the "
             "job's." % ACTION_VERSION, ""]
    return lines


def _did_not_run(run: dict, why: str, out_dir: Path, env: Dict[str, str]) -> int:
    run["did_not_run"] = why
    lines = _summary_head(run) + ["**DID NOT RUN.** %s" % why, "",
                                  "_Nothing was minted and nothing was verified; this is not a verdict._"]
    _finish(run, lines, out_dir, env, {})
    _say("::notice title=styxx sworn::DID NOT RUN — %s" % _cell(why))
    return 0


def _finish(run: dict, lines: List[str], out_dir: Path, env: Dict[str, str], outputs: Dict[str, str]) -> None:
    text = "\n".join(lines) + "\n"
    _write_text_lf(out_dir / "summary.md", text)
    step_summary = env.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    _write_json_lf(out_dir / "run.json", run)
    outputs = dict(outputs)
    outputs.setdefault("out-dir", str(out_dir))
    gh_out = env.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8", newline="\n") as fh:
            for key, value in outputs.items():
                delim = "sworn_%s" % secrets.token_hex(8)
                fh.write("%s<<%s\n%s\n%s\n" % (key, delim, value, delim))


def main(argv: Optional[List[str]] = None) -> int:
    env = os.environ
    event_path = env.get("GITHUB_EVENT_PATH", "")
    command = env.get("SWORN_COMMAND", "")
    if not event_path or not command.strip():
        sys.stderr.write("usage: GITHUB_EVENT_PATH and a non-empty SWORN_COMMAND are required; "
                         "nothing ran\n")
        return 2

    ws = os.path.abspath(env.get("GITHUB_WORKSPACE") or os.getcwd())
    out_dir = Path(os.path.abspath(env.get("SWORN_OUT_DIR") or os.path.join(ws, ".sworn-action")))
    out_dir.mkdir(parents=True, exist_ok=True)
    event_name = env.get("GITHUB_EVENT_NAME", "")
    declared = (env.get("SWORN_RUNG") or "L2").strip()
    after_turn = _truthy(env.get("SWORN_AFTER_TURN_ON_BASE", "true"))
    base_pinned = _truthy(env.get("SWORN_BASE_PINNED_WORKFLOW", "false"))
    try:
        timeout_s = float(env.get("SWORN_TIMEOUT_MINUTES") or "30") * 60.0
    except ValueError:
        timeout_s = 30.0 * 60.0
    action_repo = env.get("GITHUB_ACTION_REPOSITORY", "")
    action_ref = env.get("GITHUB_ACTION_REF", "")
    harness_name = "sworn/sworn_action.py" + (" (%s@%s)" % (action_repo, action_ref)
                                              if action_repo or action_ref else "")

    run: dict = {
        "what": "a sworn action run: report-only, exit 0 on every verdict",
        "action": ACTION_VERSION, "harness": harness_name, "event": event_name,
        "out_dir": _under(out_dir, ws), "command": command,
        "declared": {"rung": declared, "after_turn_on_base": after_turn,
                     "base_pinned_workflow": base_pinned},
    }

    # --- the event: read as bytes, parsed in Python; its text never reaches a shell ----------
    try:
        event_bytes = Path(event_path).read_bytes()
        event = json.loads(event_bytes.decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        return _did_not_run(run, "the event payload could not be read as JSON (%s)." % exc, out_dir, env)
    if event_name == "pull_request_target":
        return _did_not_run(run, "event pull_request_target: this action executes the head's "
                            "command, and under that event it would do so with the base "
                            "repository's token; use the pull_request trigger, where the fork rule "
                            "prints L1 instead.", out_dir, env)
    if event_name not in EVENTS:
        return _did_not_run(run, "event %r is not minted (pull_request and push only)." % event_name,
                            out_dir, env)
    try:
        base, head = github_adapter.shas(event, event_name)
        fork = github_adapter.fork_status(event, event_name)
    except ValueError as exc:
        return _did_not_run(run, "the payload cannot be read by the GitHub adapter: %s." % exc, out_dir, env)
    if not _HEX40.fullmatch(head):
        return _did_not_run(run, "the head sha %r is not forty lowercase hex characters." % head, out_dir, env)
    repository = ((event.get("repository") or {}).get("full_name")
                  if isinstance(event.get("repository"), dict) else None) or "repository"
    number = (event.get("pull_request") or {}).get("number") if event_name == "pull_request" else None
    turn = ("%s#%s@%s" % (repository, number, head)) if event_name == "pull_request" else "%s@%s" % (repository, head)
    run.update({"repository": repository, "number": number, "base": base, "head": head,
                "fork": fork, "turn": turn})
    if not _commit_present(ws, head):
        return _did_not_run(run, "the head commit %s is not in the checkout at %s; check out "
                            "`ref: ${{ github.event.pull_request.head.sha }}` with `fetch-depth: 0` "
                            "(GITHUB_SHA on a pull_request event is a merge commit the author could "
                            "not have cited)." % (head, ws), out_dir, env)

    # --- the rung: declared by the workflow, lowered with a reason, never raised --------------
    rung, rung_reason = decide_rung(declared, after_turn, base_pinned, event_name, fork)
    run["rung"] = rung
    run["rung_reason"] = rung_reason

    # --- the command, after the turn -----------------------------------------------------------
    junit_given = env.get("SWORN_JUNIT", "").strip()
    junit_as_given = junit_given or str(out_dir / "junit.xml")
    junit_abs = Path(junit_as_given) if os.path.isabs(junit_as_given) else Path(ws) / junit_as_given
    junit_abs.parent.mkdir(parents=True, exist_ok=True)
    if junit_abs.exists():
        junit_abs.unlink()               # a report from before the turn is not this run's
    exit_code, output, timed_out = run_command(command, ws, junit_abs, timeout_s)
    (out_dir / "command.log").write_bytes(output)
    run.update({"exit_code": exit_code, "timed_out": timed_out, "junit": junit_as_given,
                "junit_present": junit_abs.exists()})

    # --- discovery: what the turn changed, and the diff, from the checkout -------------------
    notes: List[str] = []
    changed: Optional[List[str]] = None
    diff: Optional[bytes] = None
    authored: List[bytes] = []
    if _commit_present(ws, base):
        changed = _changed_paths(ws, base, head, event_name)
        diff = _diff_bytes(ws, base, head, event_name)
    if changed is None:
        notes.append("changed-file discovery unavailable: the base commit %s is not in the checkout "
                     "(a shallow clone, or a push whose before is the null sha); set `fetch-depth: 0`. "
                     "Only the pull request body was verified, authored_sha256 is empty and r5 is "
                     "absent." % base)
        run["discovery"] = "unavailable"
    else:
        authored, unreadable = _blobs_at(ws, head, changed)
        run["discovery"] = "ok"
        if unreadable:
            notes.append("%d changed path(s) were not blobs at the head commit and did not enter "
                         "authored_sha256: %s" % (len(unreadable), ", ".join(sorted(unreadable)[:20])))
    if diff is None and changed is not None:
        notes.append("the diff between base and head could not be taken; r5 is absent.")

    # --- the adapters, called as library code, after the command returned --------------------
    junit_m: Optional[Manifest] = None
    junit_note: Optional[str] = None
    saved_cwd = os.getcwd()
    try:
        os.chdir(ws)                     # the report path enters r4 as given; relative means workspace
        if junit_abs.exists():
            try:
                junit_m = junit_adapter.mint(junit_as_given, rung=rung, turn=turn, authored=authored)
            except (OSError, ValueError) as exc:
                junit_note = "the JUnit adapter could not mint over %s: %s" % (junit_as_given, exc)
        else:
            junit_note = ("the command wrote no report at $SWORN_JUNIT (%s); r1 to r4 are absent and "
                          "a span that cites them reads UNRESOLVED manifest_id_missing" % junit_as_given)
    finally:
        os.chdir(saved_cwd)
    if junit_note:
        notes.append(junit_note)
    try:
        github_m = github_adapter.mint(event, event_bytes, event_name, diff=diff,
                                       diff_complete=diff is not None, rung=rung,
                                       ran_after_turn_on_base=after_turn,
                                       base_pinned_workflow=base_pinned, turn=turn)
    except ValueError as exc:
        # the adapter is the authority on L2; decide_rung mirrors it, and this branch is the net
        rung, rung_reason = "L1", "the GitHub adapter refused L2: %s; minted at L1" % exc
        run["rung"], run["rung_reason"] = rung, rung_reason
        github_m = github_adapter.mint(event, event_bytes, event_name, diff=diff,
                                       diff_complete=diff is not None, rung="L1",
                                       ran_after_turn_on_base=after_turn,
                                       base_pinned_workflow=base_pinned, turn=turn)
        if junit_m is not None:
            junit_m = junit_adapter.mint(junit_as_given, rung="L1", turn=turn, authored=authored)

    fork_word = {True: "true", False: "false", None: "unknown"}[fork]
    harness = ("%s %s; turn %s; event %s; fork: %s; rung %s declared by the workflow "
               "(rung=%s, after-turn-on-base=%s, base-pinned-workflow=%s), not detected%s; "
               "composed: r1-r4 are the JUnit adapter's r1-r4 id for id [%s]; r5-r9 are the GitHub "
               "adapter's r1-r5 shifted by four [%s]; authored_sha256 = every blob the turn added "
               "or modified at head (%d); %s"
               % (ACTION_VERSION, harness_name, turn, event_name, fork_word, rung, declared,
                  str(after_turn).lower(), str(base_pinned).lower(),
                  ("; minted at L1 because: " + rung_reason) if rung_reason else "",
                  junit_m.harness if junit_m is not None else "no report: " + (junit_note or ""),
                  github_m.harness, len(authored), REPORT_ONLY))
    composed = compose(harness, turn, rung, junit_m, github_m, authored)
    manifest_path = composed.write(out_dir / "sworn.manifest.json")
    if junit_m is not None:
        junit_m.write(out_dir / "junit.manifest.json")
    github_m.write(out_dir / "github.manifest.json")
    manifest = Manifest.load(manifest_path)      # verified against the file's bytes, as the CLI would
    run["manifest"] = {"path": manifest_path.name, "digest": manifest.digest(),
                       "junit": junit_m.digest() if junit_m is not None else None,
                       "github": github_m.digest(), "authored": len(composed.authored_sha256),
                       "receipts": sorted(composed.receipts, key=lambda r: int(r[1:]))}

    # --- the documents: the body as submitted, every changed .md carrying <sworn, at head -----
    docs: List[Tuple[str, bytes, str]] = []
    skipped: List[dict] = []
    if event_name == "pull_request":
        body = (event.get("pull_request") or {}).get("body")
        # C1 (SPEC_action_finds_what_the_lexer_finds_v01_2026_09_06): ask the LEXER what a
        # tag-shaped candidate is, rather than spelling it a second time. `"<sworn" in body` was
        # wrong in both directions — case-sensitive, so `<SWORN …>` was reported as "carries no
        # <sworn tag" for a document the verifier calls SWORN-FAILED; and a substring test, so
        # `<swornish>` matched where the lexer's negative lookahead does not. A second spelling of
        # "what a tag looks like" is the drift the U+0085 path-segment defect was.
        if isinstance(body, str) and _CANDIDATE.search(body.encode("utf-8")):
            docs.append(("pull_request_body.md", body.encode("utf-8"), "body"))
        elif isinstance(body, str) and body.strip():
            skipped.append({"path": "pull_request_body.md", "why": "the body carries no <sworn tag"})
    tree = GitTree(ws, head)
    if changed is not None:
        for path in changed:
            if path.endswith(".sworn.json"):
                skipped.append({"path": path, "why": "a sidecar is never verified here: its embedded "
                                "manifest is the one it was sworn against, and the verifier refuses "
                                "a supplied manifest that disagrees; the repository's own tests "
                                "re-derive committed sidecars at their commits"})
                continue
            if not path.endswith(".md"):
                continue
            data, why = tree.blob(path)
            if data is None:
                skipped.append({"path": path, "why": "could not be read at the head commit: %s" % why})
            elif _CANDIDATE.search(data) is None:          # C1: the lexer decides, not a copy
                skipped.append({"path": path, "why": "carries no <sworn tag"})
            else:
                docs.append((path, data, "changed"))

    rows: List[dict] = []
    verdicts: Dict[str, str] = {}
    taken: Dict[str, int] = {}
    docs_dir, receipts_dir = out_dir / "documents", out_dir / "receipts"
    docs_dir.mkdir(exist_ok=True)
    receipts_dir.mkdir(exist_ok=True)
    for name, raw, source in docs:
        safe = _safe_name(name, taken)
        (docs_dir / safe).write_bytes(raw)
        row = {"name": name, "source": source, "document": "documents/" + safe}
        try:
            core = verify(raw, name=name, manifest=manifest, tree=tree, commit=head)
        except SystemExit as exc:
            row.update({"verdict": "REFUSED", "message": str(exc)})
            verdicts[name] = "REFUSED"
            rows.append(row)
            continue
        rec = issue_receipt(core)
        rpath = _write_json_lf(receipts_dir / (safe + ".sworn-receipt.json"), rec)
        row.update({"verdict": core["document_verdict"], "counts": core["counts"],
                    "headline": _headline(core), "rungs": core.get("rungs", {}),
                    "receipt": "receipts/" + rpath.name, "receipt_digest": rec["digest"],
                    "non_held": [{"verdict": s["verdict"], "reason": s.get("reason"),
                                  "receipt": s["receipt"], "at": s["at"]}
                                 for s in core["spans"] if s["verdict"] != "HELD"],
                    "certifies": core.get("certifies")})
        verdicts[name] = core["document_verdict"]
        rows.append(row)
    run["documents"] = rows
    run["skipped"] = skipped
    run["notes"] = notes

    # --- the summary: the rung and the harness string on every row ---------------------------
    lines = _summary_head(run)
    lines += ["**Turn.** `%s` event on `%s`%s · head `%s` · base `%s` · fork: %s"
              % (event_name, repository, (" #%s" % number) if number is not None else "", head, base, fork_word), ""]
    if rung_reason:
        lines += ["**Rung L1** — %s" % rung_reason, ""]
    else:
        lines += ["**Manifest minted at rung %s after the turn** by `%s`." % (rung, harness_name), ""]
    lines += ["**Command** `%s` exited %d%s; report at `$SWORN_JUNIT` = `%s`: %s."
              % (_cell(command), exit_code, " (timed out)" if timed_out else "", _cell(junit_as_given),
                 "written" if junit_abs.exists() else "not written"), ""]
    if rows:
        lines += ["| document | verdict | held | failed | unresolved | malformed | rung | harness |",
                  "|---|---|---|---|---|---|---|---|"]
        for row in rows:
            if row["verdict"] == "REFUSED":
                lines.append("| `%s` | REFUSED — %s | | | | | %s | %s |"
                             % (_cell(row["name"]), _cell(row["message"]), rung, _cell(harness)))
                continue
            c = row["counts"]
            lines.append("| `%s` | %s | %d | %d | %d | %d | %s | %s |"
                         % (_cell(row["name"]), row["verdict"], c["HELD"], c["FAILED"],
                            c["UNRESOLVED"], c["MALFORMED"], rung, _cell(harness)))
        lines.append("")
        for row in rows:
            if row["verdict"] == "REFUSED":
                continue
            lines.append("`%s`: %s" % (_cell(row["name"]), _cell(row["headline"])))
            for s in row["non_held"]:
                lines.append("- %s %s `%s` @%s" % (s["verdict"], s["reason"] or "", _cell(s["receipt"]), s["at"]))
            lines.append("")
    else:
        lines += ["_No sworn document to verify: the body and the changed markdown carry no `<sworn` "
                  "tag. The manifest was minted regardless._", ""]
    lines += ["**Receipt layout** (fixed per action version; an absence never renumbers):", "",
              "| id | bytes | from | minted |", "|---|---|---|---|"]
    for rid, what, origin in LAYOUT:
        lines.append("| %s | %s | %s | %s |" % (rid, what, origin, "yes" if rid in composed.receipts else "no"))
    lines.append("")
    for s in skipped:
        lines.append("- not verified: `%s` — %s" % (_cell(s["path"]), _cell(s["why"])))
    for n in notes:
        lines.append("- %s" % _cell(n))
    if skipped or notes:
        lines.append("")
    lines += ["Reproduce: `python -m styxx.sworn verify <out-dir>/documents/<name> --repo . --commit %s "
              "--manifest <out-dir>/sworn.manifest.json`; the two adapter manifests are beside it." % head, ""]
    certifies = next((r.get("certifies") for r in rows if r.get("certifies")), None)
    if certifies:
        lines += ["_%s_" % _cell(certifies), ""]
    lines += ["_%s. Exit 0 on every verdict. On a pull request from a fork the minting job is the "
              "claimant's and the manifest declares L1._" % REPORT_ONLY]
    outputs = {"manifest": str(manifest_path), "receipts-dir": str(receipts_dir),
               "out-dir": str(out_dir), "rung": rung,
               "verdicts": json.dumps(verdicts, sort_keys=True, separators=(",", ":"))}
    _finish(run, lines, out_dir, env, outputs)
    for row in rows:
        _say("::notice title=styxx sworn (%s)::%s: %s"
             % (rung, _cell(row["name"]), _cell(row.get("headline") or "REFUSED — " + row.get("message", ""))))
    _say("styxx sworn: rung %s, %d document(s) verified, %d skipped, manifest %s"
         % (rung, len(rows), len(skipped), manifest.digest()[:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
