# -*- coding: utf-8 -*-
"""A GitHub event payload and a diff the caller fetched, minted as a sworn manifest at the rung
the caller declares.

Input: the parsed ``GITHUB_EVENT_PATH`` JSON and its raw bytes; the event name (``pull_request``,
``pull_request_target``, ``push`` — anything else is refused); the diff bytes or ``None``;
``diff_complete`` (r1 is complete only when the caller asserts it, so ``absent`` over a diff the
caller streamed and truncated is MALFORMED rather than a hollow oath); ``rung``;
``ran_after_turn_on_base``; ``base_pinned_workflow``.

Receipts:

  r1  the diff bytes (omitted when None)          http_fetch     complete as the caller asserted
  r2  the base sha as ASCII                        harness_note   complete
  r3  the head sha as ASCII                        harness_note   complete
  r4  the event name as ASCII                      harness_note   complete
  r5  the event payload bytes, whole               harness_note   complete
      (so ``r5#/pull_request/number`` and ``r5#/pull_request/head/repo/fork`` are leaves)

The fork caveat (``FORK_CAVEAT``) is printed into every manifest's harness string, fork or not,
with ``fork: true|false|unknown`` read off the payload. L2 needs the caller's declaration
``ran_after_turn_on_base``; on a fork ``pull_request`` (or one whose head repository is absent)
L2 further needs ``base_pinned_workflow``, which the event bytes cannot show. A refusal is a
ValueError at mint time — a usage error, never a verdict — and the caller may mint at L1.

No network is opened here and no environment variable is read below ``main``; the command line
reads ``GITHUB_EVENT_PATH`` and ``GITHUB_EVENT_NAME`` as argument defaults and nowhere else.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

from styxx.harness import HARNESS_VERSION, L1_WEAK, LABEL
from styxx.sworn import RUNGS, Manifest

__all__ = ["EVENT_NAMES", "FORK_CAVEAT", "fork_status", "shas", "mint", "main"]

EVENT_NAMES = ("pull_request", "pull_request_target", "push")

# The design's paragraph, verbatim in substance, printed on every manifest this adapter mints.
FORK_CAVEAT = (
    "fork caveat: on a pull_request event from a fork, the job that mints the manifest runs the "
    "workflow file as it exists in the pull request's head, on a runner the claimant's changes "
    "configured; the manifest is then minted by a party the claimant controls, and 'a party other "
    "than the claimant' — the sentence L2 rests on — does not hold. It holds only for a workflow "
    "pinned to the base branch (pull_request_target, which runs the base's workflow file with the "
    "base's secrets and must not check out and execute head content) or for a manifest attested "
    "outside the job")


def fork_status(event: dict, event_name: str) -> Optional[bool]:
    """True/False from the payload; None when the head repository is absent (a deleted fork)."""
    if event_name == "push":
        return False
    pr = event.get("pull_request") if isinstance(event, dict) else None
    if not isinstance(pr, dict):
        raise ValueError("a %s event needs a pull_request object in the payload" % event_name)
    head = pr.get("head") if isinstance(pr.get("head"), dict) else {}
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    head_repo = head.get("repo")
    base_repo = base.get("repo") if isinstance(base.get("repo"), dict) else {}
    if not isinstance(head_repo, dict):
        return None
    if head_repo.get("fork") is True:
        return True
    head_name, base_name = head_repo.get("full_name"), base_repo.get("full_name")
    if not isinstance(head_name, str) or not isinstance(base_name, str):
        return None
    return head_name != base_name


def shas(event: dict, event_name: str) -> Tuple[str, str]:
    """(base, head): pull_request.base.sha / .head.sha, or push.before / push.after."""
    try:
        if event_name == "push":
            base, head = event["before"], event["after"]
        else:
            pr = event["pull_request"]
            base, head = pr["base"]["sha"], pr["head"]["sha"]
    except (KeyError, TypeError) as exc:
        raise ValueError("the %s payload carries no base/head sha (%s)" % (event_name, exc))
    if not isinstance(base, str) or not isinstance(head, str) or not base or not head:
        raise ValueError("base/head sha must be non-empty strings")
    return base, head


def mint(event: dict, event_bytes: bytes, event_name: str, *, diff: Optional[bytes],
         diff_complete: bool, rung: str, ran_after_turn_on_base: bool,
         base_pinned_workflow: bool, turn: str = "") -> Manifest:
    if event_name not in EVENT_NAMES:
        raise ValueError("event name must be one of %s, not %r" % (", ".join(EVENT_NAMES), event_name))
    if rung not in RUNGS:
        raise ValueError("rung must be one of %s, not %r — L3 is reserved and the rung is "
                         "declared, never detected" % (", ".join(RUNGS), rung))
    if rung == "L2" and not ran_after_turn_on_base:
        raise ValueError("L2 needs the caller's declaration ran_after_turn_on_base; the adapter "
                         "cannot detect it — mint at L1 instead")
    fork = fork_status(event, event_name)
    if rung == "L2" and event_name == "pull_request" and fork is not False and not base_pinned_workflow:
        raise ValueError("L2 on a pull_request from a fork (or with the head repository absent) "
                         "needs the caller's declaration base_pinned_workflow — the event bytes "
                         "cannot show it; mint at L1 instead")
    base, head = shas(event, event_name)
    fork_word = {True: "true", False: "false",
                 None: "unknown (head repository absent from the payload; treated as a fork)"}[fork]
    weak = ("; " + L1_WEAK) if rung == "L1" else ""
    harness = ("%s github %s; fork: %s; rung %s declared by the caller, not detected%s; %s; %s"
               % (HARNESS_VERSION, event_name, fork_word, rung, weak, FORK_CAVEAT, LABEL))
    m = Manifest(harness=harness, turn=turn, rung=rung)
    if diff is not None:
        m.add("r1", bytes(diff), "http_fetch", complete=bool(diff_complete),
              note="diff supplied by the caller; completeness asserted by the caller, not observed")
    base_where = "push.before" if event_name == "push" else "pull_request.base.sha"
    head_where = "push.after" if event_name == "push" else "pull_request.head.sha"
    m.add("r2", base.encode("ascii", errors="replace"), "harness_note", complete=True,
          note="base sha from %s event (%s)" % (event_name, base_where))
    m.add("r3", head.encode("ascii", errors="replace"), "harness_note", complete=True,
          note="head sha from %s event (%s)" % (event_name, head_where))
    m.add("r4", event_name.encode("ascii"), "harness_note", complete=True,
          note="GITHUB_EVENT_NAME as given")
    m.add("r5", bytes(event_bytes), "harness_note", complete=True,
          note="GITHUB_EVENT_PATH bytes as given")
    return m


def main(argv: Optional[Sequence[str]] = None) -> int:
    import os                                        # the one ambient read: argparse defaults
    ap = argparse.ArgumentParser(
        prog="styxx.harness github",
        description="Mint a sworn/manifest/0.2 file from a GitHub event payload and a diff you "
                    "fetched. An adapter: no network, no signature, no verdict; exit two is a "
                    "usage error (an event or rung this adapter refuses).")
    ap.add_argument("--event", default=os.environ.get("GITHUB_EVENT_PATH"),
                    help="event payload JSON (default: $GITHUB_EVENT_PATH)")
    ap.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME"),
                    help="pull_request | pull_request_target | push (default: $GITHUB_EVENT_NAME)")
    ap.add_argument("--diff", default=None, help="diff bytes you fetched; omitted -> no r1")
    ap.add_argument("--diff-complete", action="store_true",
                    help="assert the diff is whole; without it r1 is complete:false")
    ap.add_argument("--rung", required=True, choices=list(RUNGS))
    ap.add_argument("--after-turn-on-base", action="store_true", dest="after_turn_on_base",
                    help="declare that the run happened after the agent's turn on a branch or "
                         "machine the agent could not write to (required for L2)")
    ap.add_argument("--base-pinned-workflow", action="store_true", dest="base_pinned_workflow",
                    help="declare that the workflow file came from the base branch (required for "
                         "L2 on a fork pull_request)")
    ap.add_argument("--turn", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if not a.event or not a.event_name:
        sys.stderr.write("usage: --event and --event-name are required when GITHUB_EVENT_PATH / "
                         "GITHUB_EVENT_NAME are not set\n")
        return 2
    try:
        event_bytes = Path(a.event).read_bytes()
        event = json.loads(event_bytes.decode("utf-8"))
        diff = Path(a.diff).read_bytes() if a.diff else None
        m = mint(event, event_bytes, a.event_name, diff=diff, diff_complete=a.diff_complete,
                 rung=a.rung, ran_after_turn_on_base=a.after_turn_on_base,
                 base_pinned_workflow=a.base_pinned_workflow, turn=a.turn)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        sys.stderr.write("usage: %s\n" % exc)
        return 2
    m.write(a.out)
    print("minted %s rung %s receipts=%d event=%s" % (a.out, a.rung, len(m.receipts), a.event_name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
