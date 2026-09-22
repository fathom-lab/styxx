# -*- coding: utf-8 -*-
"""The GitHub Action for `styxx ci-audit`: the gate on the change that triggered the workflow, the
check it hides marked on its own line, and the verified repair one click away.

    - uses: actions/checkout@v4
    - uses: fathom-lab/styxx/ci-audit@main

It reads the event payload from GITHUB_EVENT_PATH (no event text ever reaches a shell), works out
the comparison the event implies, fetches only the commits that comparison needs, and runs the
gate (`differential.audit_commit`: only the workflows that changed, every step matched across the
two revisions, each read by simulating its tools failing in a sandbox of stubs). Then it reports
in five places:

  annotations   an error on the exact line of every check the change hides -- the
                `continue-on-error:` key, or the `run:` block -- so the Files-changed view shows it
  job summary   the verdicts, and every verified repair as a diff
  outputs       fires, new-hidden, workflows-changed, receipt
  receipt       the gate's full record, as JSON
  suggestions   (opt-in, `suggest: true`) a review comment on the line whose suggestion applies the
                verified repair with one click; posted once, found again by its marker on re-runs

The comparison:

  pull_request    the default checkout is GitHub's test merge (refs/pull/N/merge): BASE is its
                  first parent -- the base branch as GitHub would merge into -- and HEAD the merge
                  itself, exactly what merging would bring. Any other checkout: the pull request's
                  merge ref is fetched and read the same way (`pr_differential`).
  merge_group     BASE the queue's base_sha, HEAD the group's commit.
  push            BASE the push's `before`, HEAD its `after`; a new branch has nothing to compare.
  pull_request_target  refused: that context carries the base repository's write token and
                  secrets, and the gate reads the pull request's workflow text.
  anything else   reported, not gated.

Exit status: 1 when the change brings a check that hides its own failure (`fail-on: new-hidden`,
the default); 0 when it does not, or `fail-on: never`; 2 when the gate could not run -- a gate that
could not run has not passed, and the job says so. Before the gate reads anything, the process
drops every token from its environment; the one it may need for suggestions is held in memory.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from . import differential as D
from . import repair as R
from . import repair_structural as RS

SCHEMA = "styxx.ci-audit-action/v1"
MARK = "styxx-ci-audit"
ZERO = "0" * 40
TOKEN_ENV = ("GH_TOKEN", "GITHUB_TOKEN", "INPUT_GITHUB-TOKEN", "INPUT_GITHUB_TOKEN", "ACTIONS_RUNTIME_TOKEN",
             "ACTIONS_ID_TOKEN_REQUEST_TOKEN", "ACTIONS_ID_TOKEN_REQUEST_URL")
READINGS = {
    "test-merge": "GitHub's test merge, against its first parent (the base branch)",
    "merge-group": "the merge queue's group, against its base",
    "push": "the push: its old tip against its new tip",
}


# ----------------------------------------------------------------------------- git

def git(tree: Path, *args: str, timeout: int = 600) -> str:
    p = subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    return p.stdout if p.returncode == 0 else ""


def have(tree: Path, sha: str) -> bool:
    return subprocess.run(["git", "-C", str(tree), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def parents(tree: Path, sha: str) -> list[str]:
    """The parents the commit object names -- read from the object, so a shallow checkout (where the
    commit is grafted as parentless) still says what they are."""
    return [ln.split()[1] for ln in git(tree, "cat-file", "-p", sha).split("\n\n", 1)[0].splitlines() if ln.startswith("parent ")]


def ensure(tree: Path, sha: str, remote: str = "origin") -> None:
    """The commit in the clone: fetched from the remote by name when missing (depth 1 in a shallow clone)."""
    if have(tree, sha):
        return
    shallow = git(tree, "rev-parse", "--is-shallow-repository").strip() == "true"
    p = subprocess.run(["git", "-C", str(tree), "fetch", "--quiet", "--no-tags", "--no-write-fetch-head", *(["--depth=1"] if shallow else []), remote, sha],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    if not have(tree, sha):
        raise RuntimeError(f"could not fetch {sha[:12]} from {remote}: {(p.stderr or '').strip()[-200:] or 'not found'}")


# ----------------------------------------------------------------------------- what the event asks

def resolve(event_name: str, event: dict, tree: Path, github_sha: Optional[str] = None) -> dict:
    """The comparison the event implies: {base, head, reading, ...}, {pr_differential: N}, or
    {skip: why}. Raises RuntimeError when the event asks for a comparison that cannot be made."""
    head = git(tree, "rev-parse", "HEAD").strip()
    if event_name == "pull_request":
        pr = event.get("pull_request") or {}
        n = pr.get("number") or event.get("number")
        ps = parents(tree, head) if head else []
        if head and len(ps) == 2 and (not github_sha or head == github_sha):
            ensure(tree, ps[0])
            return {"base": ps[0], "head": head, "reading": "test-merge", "pr": n, "pr_head": (pr.get("head") or {}).get("sha"),
                    "base_ref": f"pull request #{n}" if n else "pull request"}
        if n is None:
            raise RuntimeError("a pull_request event without a pull request number")
        return {"pr_differential": int(n), "pr": n, "pr_head": (pr.get("head") or {}).get("sha")}
    if event_name == "pull_request_target":
        raise RuntimeError("refused on pull_request_target: that context carries the base repository's write token and secrets, "
                           "and the gate reads the pull request's workflow text; run it on pull_request")
    if event_name == "merge_group":
        base = (event.get("merge_group") or {}).get("base_sha")
        if not base or not head:
            raise RuntimeError("a merge_group event without a base_sha")
        ensure(tree, base)
        return {"base": base, "head": head, "reading": "merge-group", "base_ref": "the merge queue's base"}
    if event_name == "push":
        if event.get("deleted"):
            return {"skip": "a deleted branch: nothing to read"}
        before, after = event.get("before") or "", event.get("after") or head
        if not before or before == ZERO:
            return {"skip": "a new branch: its first push has nothing to be compared against"}
        ensure(tree, before)
        ensure(tree, after)
        return {"base": before, "head": after, "reading": "push", "base_ref": "the push's old tip"}
    return {"skip": f"the {event_name or 'unknown'} event is not gated (pull_request, merge_group, push)"}


def readable(tree: Path, base: str, head: str) -> None:
    """Both revisions in the clone and git able to compare them -- else the gate would read an empty
    change and pass it. `git diff --quiet` exits 0 (no change) or 1 (a change); anything else is an error."""
    for sha in (base, head):
        if not have(tree, sha):
            raise RuntimeError(f"{sha[:12]} is not in the clone")
    p = subprocess.run(["git", "-C", str(tree), "diff", "--quiet", base, head, "--", ".github/workflows"], capture_output=True, text=True, timeout=600)
    if p.returncode not in (0, 1):
        raise RuntimeError(f"git cannot compare {base[:12]} and {head[:12]}: {(p.stderr or '').strip()[-200:]}")


def run_gate(tree: Path, res: dict) -> dict:
    if "pr_differential" in res:
        rec = D.pr_differential(tree, int(res["pr_differential"]))
        res.update(base=rec["merge_base"], head=rec["head"], reading="pr-ref", reading_text=rec.get("reading"))
        readable(tree, res["base"], res["head"])
        return rec
    readable(tree, res["base"], res["head"])
    rec = D.audit_commit(tree, res["base"], res["head"], readers={}, fix=True)
    rec.update(base_ref=res.get("base_ref"), merge_base=res["base"], reading=READINGS.get(res.get("reading"), res.get("reading")))
    if res.get("pr") is not None:
        rec.update(pr=res["pr"], pr_head=res.get("pr_head") or res["head"], pr_merge=res["head"] if res.get("reading") == "test-merge" else None)
    return rec


# ----------------------------------------------------------------------------- where a check is, in the text

def positions(text: str, jid: str, i: int) -> Optional[dict]:
    """1-based inclusive line ranges of a step, its `run:` value, its `continue-on-error:` and its
    job's, from the parser's marks. None when the step cannot be located."""
    import yaml
    try:
        root = yaml.compose(text)
    except Exception:  # noqa: BLE001
        return None
    _, jobs = R._get(root, "jobs")
    _, job = R._get(jobs, jid) if jobs is not None else (None, None)
    _, steps = R._get(job, "steps") if job is not None else (None, None)
    if steps is None or not hasattr(steps, "value") or i >= len(steps.value):
        return None
    st = steps.value[i]
    lines = text.splitlines()

    def span(start0: int, end_mark) -> tuple[int, int]:
        stop = end_mark.line if end_mark.column == 0 else end_mark.line + 1
        stop = max(stop, start0 + 1)
        while stop > start0 + 1 and stop - 1 < len(lines) and not lines[stop - 1].strip():
            stop -= 1                                        # trailing blank lines are not the step's
        return start0 + 1, stop

    loc = R.locate(text, jid, i) or {}
    out = {"step": span(st.start_mark.line, st.end_mark), "run": None, "step_coe": None, "job_coe": None}
    _, rv = R._get(st, "run")
    if loc.get("run") and rv is not None:
        out["run"] = span(loc["run"]["line"], rv.end_mark)
    for k in ("step_coe", "job_coe"):
        if loc.get(k):
            out[k] = (loc[k]["line"] + 1, loc[k]["line"] + 1)
    return out


def target(pos: dict, rec: dict) -> tuple[int, int, str]:
    """The lines an annotation for this hidden check points at, and what they are: the
    `continue-on-error:` that hides it when one does, else the `run:` block, else the step."""
    if rec.get("continue_on_error"):
        if pos.get("step_coe"):
            return (*pos["step_coe"], "continue-on-error")
        if pos.get("job_coe"):
            return (*pos["job_coe"], "job continue-on-error")
    if pos.get("run"):
        return (*pos["run"], "run")
    return (pos["step"][0], pos["step"][0], "step")


# ----------------------------------------------------------------------------- the repair, as a one-click suggestion

def repaired_text(text: str, wf_name: str, rec: dict) -> Optional[str]:
    """The verified repair's text, rebuilt by the same function that built it, and held to the
    receipt's diff: None unless it is the text the verification read."""
    fx = rec.get("fix") or {}
    name = fx.get("verified_repair")
    if not name:
        return None
    if fx.get("stage") == "swallow-5":
        new, _ = RS.apply_structural(text, rec["job"], rec["index"], name)
    else:
        new, _ = R.apply_repair(text, rec["job"], rec["index"], name)
    if new is None:
        return None
    if fx.get("diff") and R.unified_diff(text, new, wf_name) != fx["diff"]:
        return None
    return new


def suggestion(text: str, new: str) -> Optional[dict]:
    """The smallest run of lines of `text` whose replacement gives `new`: {start_line, line (both
    1-based, inclusive), lines (the replacement), replaces (the lines replaced)}. A pure insertion
    is anchored on the line above it (below it, at the top of the file): a suggestion replaces lines."""
    a, b = text.splitlines(), new.splitlines()
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return None
    i1, i2, j1, j2 = ops[0][1], ops[-1][2], ops[0][3], ops[-1][4]
    if i1 == i2:
        if i1 > 0:
            i1, j1 = i1 - 1, j1 - 1
        else:
            i2, j2 = i2 + 1, j2 + 1
    return {"start_line": i1 + 1, "line": i2, "lines": b[j1:j2], "replaces": a[i1:i2]}


def apply_suggestion(text: str, s: dict) -> str:
    a = text.splitlines()
    return "\n".join(a[: s["start_line"] - 1] + s["lines"] + a[s["line"]:]) + "\n"


def hunks(tree: Path, base: str, head: str, path: str, before: Optional[str] = None, context: int = 3) -> list[tuple[int, int]]:
    """The head-side line ranges (1-based, inclusive) of the change's diff for one file, with the
    context a pull request's diff shows: the lines a review comment can be placed on."""
    out = []
    args = ["diff", f"-U{context}", "-M", base, head, "--"] + ([before] if before and before != path else []) + [path]
    for ln in git(tree, *args).splitlines():
        if ln.startswith("@@"):
            try:
                plus = ln.split(" ")[2]                      # +c,d
                c, _, d = plus[1:].partition(",")
                c, d = int(c), int(d) if d else 1
            except (IndexError, ValueError):
                continue
            if d > 0:
                out.append((c, c + d - 1))
    return out


def within(span: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
    return any(a <= span[0] and span[1] <= b for a, b in ranges)


# ----------------------------------------------------------------------------- reporting

def esc_data(s: str) -> str:
    return str(s).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def esc_prop(s: str) -> str:
    return esc_data(s).replace(":", "%3A").replace(",", "%2C")


def annotation(level: str, message: str, title: str, path: Optional[str] = None, line: Optional[int] = None, end: Optional[int] = None) -> str:
    props = []
    if path:
        props.append(f"file={esc_prop(path)}")
    if line:
        props.append(f"line={int(line)}")
        if end and end != line:
            props.append(f"endLine={int(end)}")
    props.append(f"title={esc_prop(title)}")
    return f"::{level} {','.join(props)}::{esc_data(message)}"


def code(s: str) -> str:
    """An inline code span that no backtick, pipe or newline in `s` can break out of."""
    s = str(s).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    tick = "`" * (max((len(r) for r in re.findall(r"`+", s)), default=0) + 1)
    pad = " " if s.startswith("`") or s.endswith("`") else ""
    return f"{tick}{pad}{s}{pad}{tick}"


def fence(body: str, info: str = "") -> str:
    n = max([len(m) for m in re.findall(r"`{3,}", body)] + [2]) + 1
    return f"{'`' * n}{info}\n{body.rstrip()}\n{'`' * n}"


def _what(x: dict) -> str:
    how = x["kind"] + (": " + ", ".join(x["mechanism"]) if x.get("mechanism") else "")
    if x.get("continue_on_error") and "continue-on-error" not in how:
        how += " (continue-on-error)"
    return how


def summary_md(rec: dict, res: dict, located: dict) -> str:
    n, r, still, unread = rec.get("new_hidden", 0), rec.get("removed_hidden", 0), rec.get("still_hidden", 0), rec.get("hidden_after_unread", 0)
    wfs = rec.get("workflows", [])
    head = "### styxx ci-audit — " + (f"{n} check{'s' if n != 1 else ''} newly hidden: the gate fires" if n else
                                        ("nothing newly hidden" if wfs else "no workflow changed"))
    reading = READINGS.get(res.get("reading"), res.get("reading_text") or res.get("reading") or "")
    heads = f"head `{(res.get('head') or '')[:8]}`"
    if res.get("pr_head") and res.get("pr_head") != res.get("head"):
        heads = f"test merge `{(res.get('head') or '')[:8]}` of the pull request's head `{res['pr_head'][:8]}`"
    out = [head, "", f"{reading} · base `{(res.get('base') or '')[:8]}` → {heads} · {len(wfs)} workflow{'s' if len(wfs) != 1 else ''} changed", ""]
    if n:
        out += ["| the check | what hides it | verified repair |", "|---|---|---|"]
        for w in wfs:
            for x in w.get("new_hidden", []):
                where = f"{code(w['workflow'])} › {code(x['job'])} › {code(x.get('name') or 'step ' + str(x['index']))}"
                ln = located.get((w["path"], x["job"], x["index"]))
                if ln:
                    where += f" (line {ln[0]})"
                fx = x.get("fix") or {}
                rep = (f"{code(fx['verified_repair'])}, {fx['lines_changed']} line{'s' if fx['lines_changed'] != 1 else ''}" if fx.get("verified_repair")
                       else "none verified")
                out.append(f"| {where} | {x['verdict']} — {code(_what(x))} | {rep} |")
        diffs = [(w, x) for w in wfs for x in w.get("new_hidden", []) if (x.get("fix") or {}).get("diff")]
        if diffs:
            out += ["", "<details open><summary>the repairs — each verified: loud under the same fault, the healthy run unchanged</summary>", ""]
            for w, x in diffs:
                out += [fence(x["fix"]["diff"], "diff"), ""]
            out += ["</details>"]
    extra = []
    if r:
        extra.append(f"{r} hidden check{'s' if r != 1 else ''} made loud or removed")
    if still:
        extra.append(f"{still} hidden on both sides (already there: not this change's doing)")
    if unread:
        extra.append(f"{unread} hidden here that the base could not be read for")
    if extra:
        out += ["", " · ".join(extra)]
    out += ["", "<sub>The gate reads only the workflows this change touched, simulating each step's tools failing in a sandbox of stubs: "
                "no runner, no token, no code run. RED is loud, not correct. "
                "<a href=\"https://github.com/fathom-lab/styxx\">styxx ci-audit</a></sub>", ""]
    return "\n".join(out)


def _write(path: Optional[str], text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)


def outputs(**kv) -> None:
    lines = "".join(f"{k}={v}\n" for k, v in kv.items())
    _write(os.environ.get("GITHUB_OUTPUT"), lines)


# ----------------------------------------------------------------------------- suggestions (opt-in)

def api(method: str, url: str, token: str, payload: Optional[dict] = None) -> tuple[int, object]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Accept": "application/vnd.github+json", "User-Agent": "styxx-ci-audit-action", "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28", **({"Content-Type": "application/json"} if data else {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(body) if body.strip() else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:300]
    except (urllib.error.URLError, OSError, ValueError) as e:
        return 0, str(e)[:300]


def suggestion_body(x: dict, w: dict, s: dict, key: str) -> str:
    fx = x["fix"]
    return "\n".join([
        f"**styxx ci-audit** — this step hides its own failure ({x['verdict']}: {_what(x)}): "
        f"if {code(x.get('run_head') or 'its tools')} fails, the job stays green.",
        "",
        fence("\n".join(s["lines"]), "suggestion"),
        "",
        f"The repair is verified: under the same fault the workflow goes red, and the healthy run is unchanged "
        f"({code(fx['verified_repair'])}, {fx['lines_changed']} line{'s' if fx['lines_changed'] != 1 else ''}).",
        f"<!-- {MARK}:suggest:{key} -->",
    ])


def suggest(tree: Path, rec: dict, res: dict, token: str, repo: str, api_url: str) -> list[dict]:
    """One review suggestion per newly hidden check that has a verified repair, on the pull
    request's own head, never twice. Every outcome is recorded; nothing here can fail the gate."""
    outcomes: list[dict] = []
    pr, pr_head = res.get("pr"), res.get("pr_head")
    if not (token and repo and pr and pr_head):
        return [{"posted": False, "why": "not a pull request with a token, or no head sha in the event"}]
    existing: Optional[set] = None
    for w in rec.get("workflows", []):
        for x in w.get("new_hidden", []):
            o = {"path": w["path"], "job": x["job"], "step": x["step"], "posted": False}
            outcomes.append(o)
            text = D._text(tree, rec["head"], w["path"])
            new = repaired_text(text, w["workflow"], x) if text is not None else None
            if new is None:
                o["why"] = "no verified repair" if not (x.get("fix") or {}).get("verified_repair") else "the repair could not be rebuilt on this text"
                continue
            s = suggestion(text, new)
            if s is None or apply_suggestion(text, s).splitlines() != new.splitlines():
                o["why"] = "the suggestion would not reproduce the verified repair"
                continue
            if pr_head != rec["head"]:
                try:
                    ensure(tree, pr_head)
                except RuntimeError as e:
                    o["why"] = f"the pull request's head could not be read: {str(e)[:120]}"
                    continue
                if D._text(tree, pr_head, w["path"]) != text:
                    o["why"] = "the base branch changed this file since the branch was cut: the lines would not match the pull request's"
                    continue
            key = hashlib.sha256("|".join([w["path"], x["job"], x["step"], x["fix"]["verified_repair"], "\n".join(s["lines"])]).encode()).hexdigest()[:16]
            o.update(key=key, start_line=s["start_line"], line=s["line"], replaces=len(s["replaces"]))
            if existing is None:
                existing = set()
                for page in range(1, 11):
                    st, body = api("GET", f"{api_url}/repos/{repo}/pulls/{pr}/comments?per_page=100&page={page}", token)
                    if st != 200 or not isinstance(body, list):
                        break
                    existing.update(c.get("body") or "" for c in body)
                    if len(body) < 100:
                        break
            if any(f"{MARK}:suggest:{key}" in b for b in existing):
                o["why"] = "already suggested on this pull request"
                continue
            payload = {"body": suggestion_body(x, w, s, key), "commit_id": pr_head, "path": w["path"], "line": s["line"], "side": "RIGHT"}
            if s["start_line"] < s["line"]:
                payload.update(start_line=s["start_line"], start_side="RIGHT")
            st, body = api("POST", f"{api_url}/repos/{repo}/pulls/{pr}/comments", token, payload)
            o["status"] = st
            if st == 201:
                o["posted"] = True
                existing.add(payload["body"])
            else:
                o["why"] = {403: "the token cannot write to pull requests (a fork's pull request, or permissions: pull-requests: write is missing)",
                            422: "the lines are outside the pull request's diff"}.get(st, f"HTTP {st}: {str(body)[:120]}")
    return outcomes


# ----------------------------------------------------------------------------- the entry point

def main(argv: Optional[list[str]] = None) -> int:
    env = os.environ
    fail_on = (env.get("STYXX_FAIL_ON") or "new-hidden").strip().lower()
    want_suggest = (env.get("STYXX_SUGGEST") or "false").strip().lower() == "true"
    want_annotate = (env.get("STYXX_ANNOTATE") or "true").strip().lower() != "false"
    token = env.get("GH_TOKEN") or ""
    for k in TOKEN_ENV:                                   # nothing below the gate reads a token; the gate reads untrusted text
        os.environ.pop(k, None)
    tree = Path(env.get("GITHUB_WORKSPACE") or ".").resolve()
    event_name = env.get("GITHUB_EVENT_NAME", "")
    receipt_path = Path(env.get("STYXX_RECEIPT") or Path(env.get("RUNNER_TEMP") or tempfile.gettempdir()) / "styxx-ci-audit.json")
    t0 = time.time()
    level_err = "warning" if fail_on == "never" else "error"
    try:
        if fail_on not in ("new-hidden", "never"):
            raise RuntimeError(f"fail-on must be new-hidden or never, not {fail_on!r}")
        event = json.loads(Path(env["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8")) if env.get("GITHUB_EVENT_PATH") else {}
        res = resolve(event_name, event, tree, env.get("GITHUB_SHA"))
        if res.get("skip"):
            print(f"styxx ci-audit: {res['skip']}; not gated.")
            _write(env.get("GITHUB_STEP_SUMMARY"), f"### styxx ci-audit — not gated\n\n{res['skip']}.\n")
            outputs(fires="false", **{"new-hidden": 0, "workflows-changed": 0, "measured": "false", "receipt": ""})
            return 0
        rec = run_gate(tree, res)
    except Exception as e:  # noqa: BLE001 -- any failure to read is reported as one, never as a pass
        why = f"{type(e).__name__}: {str(e)[:300]}"
        print(annotation(level_err, f"{why}. The gate did not run; this is not a pass.", "styxx ci-audit — the gate did not run"))
        _write(env.get("GITHUB_STEP_SUMMARY"), f"### styxx ci-audit — UNMEASURED\n\n**The gate did not run.** {code(why)}\n\nThis is not a pass.\n")
        outputs(fires="false", **{"new-hidden": 0, "workflows-changed": 0, "measured": "false", "receipt": ""})
        return 0 if fail_on == "never" else 2

    located: dict = {}
    notes = []
    for w in rec.get("workflows", []):
        text = D._text(tree, rec["head"], w["path"]) if w.get("new_hidden") else None
        for x in w.get("new_hidden", []):
            pos = positions(text, x["job"], x["index"]) if text is not None else None
            if pos is None:                                   # the file, not the line: still an annotation on it
                notes.append(annotation(level_err, f"{x['verdict']}: {x.get('name') or x['step']} ({x['job']}) hides its own failure — {_what(x)}.",
                                        "styxx ci-audit — a check that hides its own failure", w["path"]))
                continue
            a, b, what = target(pos, x)
            located[(w["path"], x["job"], x["index"])] = (a, b, what)
            fx = x.get("fix") or {}
            fix_line = (f" Verified repair: {fx['verified_repair']} ({fx['lines_changed']} line{'s' if fx['lines_changed'] != 1 else ''}; the diff is in the job summary)."
                        if fx.get("verified_repair") else " No verified repair.")
            msg = (f"{x['verdict']}: the step '{x.get('name') or x['step']}' in job '{x['job']}' hides its own failure — {_what(x)}. "
                   f"If `{x.get('run_head') or 'its tools'}` fails, the job stays green.{fix_line}")
            notes.append(annotation(level_err, msg, "styxx ci-audit — a check that hides its own failure", w["path"], a, b))

    sugg = suggest(tree, rec, res, token, env.get("GITHUB_REPOSITORY", ""), env.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")) \
        if want_suggest and rec.get("new_hidden") and res.get("pr") is not None else []
    rec_out = {"schema": SCHEMA, "event": event_name, "base": res.get("base"), "head": res.get("head"), "reading": res.get("reading"),
               "pr": res.get("pr"), "fires": rec["fires"], "new_hidden": rec["new_hidden"], "removed_hidden": rec["removed_hidden"],
               "still_hidden": rec["still_hidden"], "hidden_after_unread": rec.get("hidden_after_unread", 0), "workflows": rec["workflows"],
               "lines": {f"{k[0]}::{k[1]}::{k[2]}": v for k, v in located.items()}, "suggestions": sugg, "seconds": round(time.time() - t0, 2),
               "fail_on": fail_on}
    receipt = str(receipt_path)
    try:
        receipt_path.write_text(json.dumps(rec_out, indent=1, default=str) + "\n", encoding="utf-8")
    except OSError:
        receipt = ""

    from .report import _differential_section
    stop = secrets.token_hex(16)
    print(f"::stop-commands::{stop}")                  # the card quotes the change's own workflow text: no line of it may act as a command
    print("\n".join(_differential_section(rec, 96)))
    print(f"::{stop}::")
    if want_annotate:
        for ln in notes:
            print(ln)
    posted = [o for o in sugg if o.get("posted")]
    if sugg:
        print(f"styxx ci-audit: {len(posted)} suggestion{'s' if len(posted) != 1 else ''} posted"
              + "".join(f"; {o['path']} {o['step']}: {o['why']}" for o in sugg if not o.get("posted") and o.get("why")))
    _write(env.get("GITHUB_STEP_SUMMARY"), summary_md(rec, res, located)
           + (f"\n{len(posted)} one-click suggestion{'s' if len(posted) != 1 else ''} posted on the pull request.\n" if posted else ""))
    outputs(fires="true" if rec["fires"] else "false", **{"new-hidden": rec["new_hidden"], "workflows-changed": len(rec.get("workflows", [])),
                                                            "measured": "true", "receipt": receipt})
    if rec["fires"] and fail_on == "new-hidden":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
