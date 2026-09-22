# -*- coding: utf-8 -*-
"""styxx.ciaudit.differential -- the check that arrives hidden, caught at the pull request.

The living copy of SWALLOW-7's instrument (`benchmarks/harness_mutation/differential.py`, frozen;
the test suite holds this file to it). `styxx ci-audit . --base <rev>` reads only the workflows
that changed between the merge-base of <rev> and HEAD, with the engine, matches every step across
the two revisions -- the same job and step name, else id, else first line; a renamed step by its
script -- and says:

  new hidden          hidden at HEAD, and at BASE loud, not a check, or not there at all: the gate
                      FIRES (exit 1), and for each the repair the engine verifies on HEAD's text
  hidden after unread hidden at HEAD, uninterpretable at BASE: reported, not a firing
  removed hidden      hidden at BASE, and at HEAD loud (a repair), not a check, or gone
  still hidden        hidden at both: reported, not a firing -- the pull request did not bring it
"""
from __future__ import annotations

import time
from pathlib import Path

from . import history as H              # the living copy of SWALLOW-6's instrument; names read as in benchmarks/harness_mutation/differential.py
from . import repair
from . import repair_structural as rs

SCHEMA = "styxx.ci-audit.differential/v1"
FIRES = ("loud", "other")           # what a check was at BASE for its hiding at HEAD to be a firing


# ----------------------------------------------------------------------------- git

def changed_workflows(clone: Path, base: str, head: str) -> list[dict]:
    """The workflow paths that differ between two commits, renames followed; hand-written only."""
    out = H._git(clone, ["diff", "--name-status", "-M", base, head, "--", ".github/workflows"], check=False)
    changed = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status = parts[0][0]
        if status == "R" and len(parts) >= 3:
            rec = {"status": "R", "path": parts[2], "from": parts[1]}
        else:
            rec = {"status": status, "path": parts[-1], "from": parts[-1]}
        if H._is_workflow(rec["path"]) and not rec["path"].endswith(".lock.yml"):
            changed.append(rec)
    return changed


def _text(clone: Path, sha: str, path: str) -> str | None:
    return H.texts_at(clone, [(sha, path)]).get((sha, path))


# ----------------------------------------------------------------------------- the gate

def _sides(fl: dict | None, snap: dict) -> dict:
    """Every step of one revision keyed by (job, key): its state, verdict, fault record and snapshot."""
    out = {}
    for (jid, i), s in snap.items():
        k = (jid, s["key"])
        if k in out:                                   # two steps share a key: the first is followed
            continue
        f = (fl or {}).get((jid, i))
        out[k] = {"state": H.state_of(f["verdict"]) if f else "other", "verdict": f["verdict"] if f else None, "fault": f,
                  "snap": s, "index": i, "run_sha": s["run_sha"]}
    return out


def audit_pair(base_text: str | None, head_text: str | None, wf_name: str, reader: H.Reader, fix: bool = True) -> dict:
    """The gate on one workflow: what changed between its BASE text and its HEAD text."""
    out = {"workflow": wf_name, "new_hidden": [], "hidden_after_unread": [], "removed_hidden": [], "still_hidden": [],
           "unread_after_hidden": [], "base_unparseable": None, "head_unparseable": None, "reads": 0}
    b = h = {}
    if base_text is not None:
        fl, snap, err = reader.read(base_text, wf_name)
        out["base_unparseable"] = err
        b = _sides(fl, snap) if err is None else {}
    if head_text is not None:
        fl, snap, err = reader.read(head_text, wf_name)
        out["head_unparseable"] = err
        h = _sides(fl, snap) if err is None else {}
    out["reads"] = reader.reads
    # renames: a key only at BASE and a key only at HEAD, same job, same script
    only_b = {k for k in b if k not in h}
    only_h = {k for k in h if k not in b}
    renamed = {}
    for kb in sorted(only_b):
        if b[kb]["run_sha"] is None:
            continue
        match = next((kh for kh in sorted(only_h) if kh[0] == kb[0] and h[kh]["run_sha"] == b[kb]["run_sha"] and kh not in renamed.values()), None)
        if match is not None:
            renamed[kb] = match
    pairs = [(k, k) for k in b if k in h] + sorted(renamed.items())
    born = [k for k in h if k in only_h and k not in renamed.values()]          # document order
    gone = [k for k in b if k in only_b and k not in renamed]

    def where(side, k):
        s = side[k]
        st = s["snap"]["step"]
        return {"job": k[0], "step": k[1], "index": s["index"], "name": st.get("name") if isinstance(st.get("name"), str) else None,
                "run_head": (st["run"].strip().splitlines() or [""])[0][:160] if isinstance(st.get("run"), str) else "",
                "continue_on_error": bool(s["fault"] and s["fault"].get("continue_on_error"))}

    for kb, kh in pairs:
        sb, sh = b[kb]["state"], h[kh]["state"]
        if sh == "hidden" and sb in FIRES:
            rec = where(h, kh)
            rec.update(kind="acquired" if sb == "loud" else "became a check, hidden", verdict=h[kh]["verdict"], verdict_before=b[kb]["verdict"],
                       mechanism=H.mechanism({"step": b[kb]["snap"]["step"], "job": b[kb]["snap"]["job"]}, {"step": h[kh]["snap"]["step"], "job": h[kh]["snap"]["job"]}),
                       renamed_from=kb[1] if kb != kh else None)
            out["new_hidden"].append(rec)
        elif sh == "hidden" and sb == "unread":
            rec = where(h, kh)
            rec.update(kind="hidden after unread", verdict=h[kh]["verdict"], verdict_before=b[kb]["verdict"])
            out["hidden_after_unread"].append(rec)
        elif sb == "hidden" and sh in FIRES:
            rec = where(h, kh)
            rec.update(kind="repaired" if sh == "loud" else "no longer a check", verdict=h[kh]["verdict"], verdict_before=b[kb]["verdict"],
                       mechanism=H.mechanism({"step": b[kb]["snap"]["step"], "job": b[kb]["snap"]["job"]}, {"step": h[kh]["snap"]["step"], "job": h[kh]["snap"]["job"]}))
            out["removed_hidden"].append(rec)
        elif sb == "hidden" and sh == "hidden":
            rec = where(h, kh)
            rec.update(verdict=h[kh]["verdict"], changed=b[kb]["run_sha"] != h[kh]["run_sha"])
            out["still_hidden"].append(rec)
        elif sb == "hidden" and sh == "unread":
            rec = where(h, kh)
            rec.update(kind="unread after hidden", verdict=h[kh]["verdict"], verdict_before=b[kb]["verdict"])
            out["unread_after_hidden"].append(rec)
    for k in born:
        if h[k]["state"] == "hidden":
            rec = where(h, k)
            rec.update(kind="born hidden", verdict=h[k]["verdict"], verdict_before=None, mechanism=None, renamed_from=None)
            out["new_hidden"].append(rec)
    for k in gone:
        if b[k]["state"] == "hidden":
            rec = where(b, k)
            rec.update(kind="removed", verdict=None, verdict_before=b[k]["verdict"], mechanism=None)
            out["removed_hidden"].append(rec)
    if fix and head_text is not None:
        for rec in out["new_hidden"]:
            rec["fix"] = fix_for(head_text, wf_name, rec["job"], rec["index"], reader.runner)
    out["fires"] = bool(out["new_hidden"])
    return out


def fix_for(text: str, wf_name: str, jid: str, i: int, runner) -> dict:
    """SWALLOW-4's repairs, then SWALLOW-5's, on HEAD's text: the first verified one with its diff."""
    r = repair.try_repairs(text, wf_name, jid, i, runner)
    stage = "swallow-4"
    if r.get("verified_repair") is None:
        r2 = rs.try_structural(text, wf_name, jid, i, runner)
        if r2.get("verified_repair") is not None:
            r, stage = r2, "swallow-5"
    v = r.get("verified_repair")
    c = next((c for c in r.get("candidates", []) if c["repair"] == v), None) if v else None
    return {"verified_repair": v, "stage": stage if v else None, "lines_changed": c["lines_changed"] if c else None,
            "diff": c["diff"] if c else None,
            "why_not": None if v else "; ".join(f"{c['repair']}: {c.get('why', '')}" for c in r.get("candidates", []) if c.get("why"))[:200]}


def audit_commit(clone: Path, base: str, head: str, readers: dict | None = None, fix: bool = True) -> dict:
    """The gate at one commit: every changed hand-written workflow between BASE and HEAD."""
    t0 = time.time()
    readers = readers if readers is not None else {}
    out = {"base": base, "head": head, "workflows": [], "fires": False, "new_hidden": 0, "removed_hidden": 0, "still_hidden": 0,
           "hidden_after_unread": 0}
    for ch in changed_workflows(clone, base, head):
        path, before = ch["path"], ch["from"]
        wf_name = path.rsplit("/", 1)[-1]
        reader = readers.setdefault(path, H.Reader())
        bt = None if ch["status"] == "A" else _text(clone, base, before)
        ht = None if ch["status"] == "D" else _text(clone, head, path)
        w = audit_pair(bt, ht, wf_name, reader, fix=fix)
        w.update(path=path, status=ch["status"])
        out["workflows"].append(w)
        for k in ("new_hidden", "removed_hidden", "still_hidden", "hidden_after_unread"):
            out[k] += len(w[k])
    out["fires"] = out["new_hidden"] > 0
    out["seconds"] = round(time.time() - t0, 2)
    return out


def tree_differential(tree: Path, base: str, fix: bool = True) -> dict:
    """This checkout's HEAD against `base` (any revision git resolves)."""
    head = H._git(tree, ["rev-parse", "HEAD"]).strip()
    base_sha = H._git(tree, ["rev-parse", "--verify", f"{base}^{{commit}}"]).strip()
    merge_base = H._git(tree, ["merge-base", base_sha, head], check=False).strip() or base_sha
    out = audit_commit(tree, merge_base, head, readers={}, fix=fix)
    out.update(base_ref=base, merge_base=merge_base)
    return out


def pr_differential(tree: Path, number: int, remote: str = "origin", fix: bool = True) -> dict:
    """The gate on pull request `number` of `remote`, without checking it out. GitHub keeps two refs
    per pull request: `refs/pull/N/head`, the branch as pushed, and -- while the pull request is
    open and mergeable -- `refs/pull/N/merge`, its test merge into the base branch. With the merge
    ref, BASE is the merge's first parent (the base branch as GitHub would merge into) and HEAD the
    merge itself: exactly what merging would bring, no history needed. Without it (closed, or
    conflicting), BASE is the merge-base of the head with the remote's default branch, which needs
    history, and the record says which reading it is."""
    pfx = f"refs/ciaudit/pull/{number}"
    # a clone that is already shallow (owner/repo's sparse clone) stays so: the merge and its two parents are enough;
    # a real checkout is never made shallow by this command
    depth = ["--depth=2"] if H._git(tree, ["rev-parse", "--is-shallow-repository"], check=False).strip() == "true" else []
    H._git(tree, ["fetch", "--quiet", "--no-tags", "--no-write-fetch-head", *depth, remote, f"+refs/pull/{number}/merge:{pfx}/merge"], check=False, timeout=600)
    got_merge = H._git(tree, ["rev-parse", "--verify", "--quiet", f"{pfx}/merge^{{commit}}"], check=False).strip() != ""
    H._git(tree, ["fetch", "--quiet", "--no-tags", "--no-write-fetch-head", *depth, remote, f"+refs/pull/{number}/head:{pfx}/head"], check=False, timeout=600)
    head = H._git(tree, ["rev-parse", "--verify", "--quiet", f"{pfx}/head^{{commit}}"], check=False).strip()
    if not head:
        raise RuntimeError(f"{remote} has no refs/pull/{number}/head")
    if got_merge:
        merge = H._git(tree, ["rev-parse", f"{pfx}/merge"]).strip()
        base = H._git(tree, ["rev-parse", "--verify", "--quiet", f"{merge}^1"], check=False).strip()
        if not base:
            raise RuntimeError(f"refs/pull/{number}/merge has no first parent in this clone")
        out = audit_commit(tree, base, merge, readers={}, fix=fix)
        out.update(base_ref=f"pull request #{number}", merge_base=base, pr=number, pr_head=head, pr_merge=merge,
                   reading="GitHub's test merge (refs/pull/N/merge) against its first parent, the base branch")
        return out
    default = H._git(tree, ["symbolic-ref", "--quiet", "--short", f"refs/remotes/{remote}/HEAD"], check=False).strip() or f"{remote}/HEAD"
    base_tip = H._git(tree, ["rev-parse", "--verify", "--quiet", f"{default}^{{commit}}"], check=False).strip()
    if not base_tip:
        raise RuntimeError(f"refs/pull/{number}/merge is not there (closed, or conflicting) and {remote}'s default branch is unknown here")
    merge_base = H._git(tree, ["merge-base", base_tip, head], check=False).strip()
    if not merge_base:
        raise RuntimeError(f"refs/pull/{number}/merge is not there (closed, or conflicting) and the head shares no history with {default} in this clone")
    out = audit_commit(tree, merge_base, head, readers={}, fix=fix)
    out.update(base_ref=f"pull request #{number}", merge_base=merge_base, pr=number, pr_head=head, pr_merge=None,
               reading=f"the head against its merge-base with {default} (no refs/pull/N/merge: the pull request is closed, or conflicts)")
    return out


def instrument_sha256() -> str:
    return H._sha(Path(__file__))
