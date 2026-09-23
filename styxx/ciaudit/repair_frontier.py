# -*- coding: utf-8 -*-
"""`styxx ci-audit --repair`, third stage (SWALLOW-13, SWALLOW-14): for the hidden checks SWALLOW-4's
and SWALLOW-5's repairs leave unverified, more edits, verified on the same two halves -- loud under
the same fault, the healthy run unchanged. Tried in this order; the first verified is the repair:

  hoist-local           (SWALLOW-14) a `$(...)` whose status the line throws away, put on a line of
                        its own that keeps it -- `__subN="$(...)" || exit $?`, pipefail inside the
                        substitution when it is a pipeline -- and nothing else in the script
                        changed: no `set -e` for every other line, no `|| true` elsewhere removed
  wait-list             (SWALLOW-14) a list read from a process substitution -- `done < <(cmd)`,
                        `mapfile -t a < <(cmd)` -- whose command fails into an empty list the loop
                        reads as nothing to check: `wait $! || exit $?` after the statement (bash
                        4.4+: `$!` is the substitution; before, `$!` is empty and the line does
                        nothing), pipefail inside it when it is a pipeline
  hoist-substitution    a `$(...)` whose status the line throws away -- inside an `echo`/`printf`
                        argument, a `for ... in` list, an `export`/`local` assignment, a one-test
                        `if [ ... ]` -- moved onto its own line as `__subN="$(...)"`, and the shell
                        made strict, so the command's failure stops the step (a grep's, a diff's,
                        a `jq -e`'s status 1 -- an answer, not a failure -- kept)
  background-liveness   a job started with `&` whose early death nobody waits for: after it,
                        `__bgN=$!`, a short sleep, and `wait` on it if it has already exited -- a
                        job that died failing fails the step; one still running, or that exited
                        cleanly, does not
  no-exit-zero          `|| exit 0` -- the step told to succeed when its command fails -- removed,
                        and the shell made strict
  no-default-joined     SWALLOW-5's no-default over the script's backslash-continued lines joined
                        into one, where the `|| echo …` fallback starts a continuation line of its
                        own command
  no-coe+<edit>         continue-on-error removed together with a script edit, where neither
                        alone is loud: <edit> is SWALLOW-5's no-default or guard-status, or one of
                        the four above

And two readings for what no edit should be asked to repair:

  routed    the step's status is kept, not lost: a flag set on the failure path is written to
            GITHUB_ENV or GITHUB_OUTPUT under a name a later step reads (a ratchet's baseline, an
            agent's trigger). The job stays green; the failure went somewhere.
  declared  the script says a failure is not fatal: it emits a `::warning`, or it says so in stated
            words (non-blocking, best effort, allowed to fail, don't fail, soft fail, ⚠ ...).

The readings are rules on the text, not judgements of intent; each says what it matched. The gate
still fires on the check: a reading is what the reviewer reads next to it.
"""
from __future__ import annotations

import re

from . import engine as faults
from . import repair
from . import repair_structural as rs
from .repair import _first_change, _healthy_shape, analyse, diff_size, locate, strict_shell, unified_diff

SCHEMA = "styxx.ci-audit-repair-frontier/v1"
EDITS = ("hoist-local", "wait-list", "hoist-substitution", "background-liveness", "no-exit-zero", "no-default-joined")
REPAIRS = EDITS + tuple(f"no-coe+{e}" for e in ("no-default", "guard-status") + EDITS)
# the stage as SWALLOW-13 ran it, for replaying its receipt and its tests
S13_EDITS = ("hoist-substitution", "background-liveness", "no-exit-zero", "no-default-joined")
S13_REPAIRS = S13_EDITS + tuple(f"no-coe+{e}" for e in ("no-default", "guard-status") + S13_EDITS)


# ----------------------------------------------------------------------------- reading a line of shell

def _close(s: str, k: int) -> int | None:
    """Index just past the `)` closing a `$(` whose body starts at k; quotes and nested
    substitutions read. None when the line ends first."""
    depth, n = 1, len(s)
    while k < n:
        ch = s[k]
        if ch == "\\":
            k += 2
            continue
        if ch == "'":
            j = s.find("'", k + 1)
            if j < 0:
                return None
            k = j + 1
            continue
        if ch == '"':
            k += 1
            while k < n and s[k] != '"':
                if s[k] == "\\":
                    k += 2
                    continue
                if s.startswith("$(", k):
                    j = _close(s, k + 2)
                    if j is None:
                        return None
                    k = j
                    continue
                k += 1
            if k >= n:
                return None
            k += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return k + 1
        k += 1
    return None


def scan(line: str) -> tuple[list[tuple[int, int]], str] | None:
    """The outermost `$( … )` substitutions of one line as [start, end) spans, and the line with
    every quoted string and substitution blanked out (so its operators can be read). `$((`
    arithmetic is not a substitution. None when the line's quoting does not close."""
    subs, mask, i, n = [], list(line), 0, len(line)
    dq = False
    while i < n:
        ch = line[i]
        if ch == "\\":
            for j in range(i, min(i + 2, n)):
                mask[j] = "_"
            i += 2
            continue
        if not dq and ch == "'":
            j = line.find("'", i + 1)
            if j < 0:
                return None
            for q in range(i, j + 1):
                mask[q] = "_"
            i = j + 1
            continue
        if ch == '"':
            dq = not dq
            mask[i] = "_"
            i += 1
            continue
        if not dq and ch == "#" and (i == 0 or line[i - 1] in " \t;"):
            for q in range(i, n):
                mask[q] = "_"
            break
        if line.startswith("$(", i) and not line.startswith("$((", i):
            j = _close(line, i + 2)
            if j is None:
                return None
            subs.append((i, j))
            for q in range(i, j):
                mask[q] = "_"
            i = j
            continue
        if dq:
            mask[i] = "_"
        i += 1
    if dq:
        return None
    return subs, "".join(mask)


_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
_ECHO = re.compile(r"^\s*(echo|printf)\s")
_FOR = re.compile(r"^\s*for\s+[A-Za-z_][A-Za-z0-9_]*\s+in\s")
_DECL = re.compile(r"^\s*(export|local|readonly|declare(\s+-[A-Za-z]+)*)\s+[A-Za-z_][A-Za-z0-9_]*=")
_IF_TEST = re.compile(r"^\s*if\s+\[\[?\s.*\]\]?\s*;\s*then\s*$")
_OPS = re.compile(r"&&|\|\||;|\||&|<\(|>\(")


def _logical(lines: list[str]) -> list[tuple[int, int]]:
    """Physical line ranges [a, b] of logical lines (backslash continuations joined), heredoc
    bodies excluded."""
    out, i, n = [], 0, len(lines)
    while i < n:
        a = i
        while i < n and lines[i].rstrip().endswith("\\") and not lines[i].rstrip().endswith("\\\\"):
            i += 1
        out.append((a, min(i, n - 1)))
        m = _HEREDOC.search(lines[a]) if not lines[a].lstrip().startswith("#") else None
        i += 1
        if m:
            end = m.group(2)
            while i < n and lines[i].strip() != end:
                i += 1
            i += 1
    return out


def _hoistable(first: str, masked: str) -> bool:
    """Whether a logical line throws away the status of its substitutions and hoisting them keeps
    its meaning: the line is one simple command -- no `&&`, `||`, `;`, pipe, `&` or process
    substitution outside quotes -- of a kind whose status is not the substitution's."""
    body = masked.strip()
    if _FOR.match(first):
        head = re.split(r";\s*do\b|\bdo\b", masked, maxsplit=1)[0]
        return not _OPS.search(head.replace(";", ""))
    if _IF_TEST.match(first):
        inner = re.sub(r";\s*then\s*$", "", body)
        return not _OPS.search(inner)
    if _ECHO.match(first) or _DECL.match(first):
        return not _OPS.search(body)
    return False


# commands whose status 1 is an answer, not a failure: no match (grep and its kin, pgrep), a
# difference (diff, cmp, `git diff --exit-code`/`--quiet`), not found (which, `command -v`), false
# (`jq -e`). A hoisted substitution of one keeps its 1 -- `|| [ $? -eq 1 ]` -- so only a status above 1
# stops the step, as guard-status reads a guard's. The model's stubs never answer 1; this is the
# real run's "no". (Added after SWALLOW-13's run; see its RESULT.)
_STATUS_ONE = re.compile(r"(?:^|[|;&(`!])\s*(?:e?grep|fgrep|zgrep|rg|ag|pgrep|diff|cmp|which|command\s+-v|"
                         r"git\s+diff\b[^|;&]*--(?:exit-code|quiet)|jq\b[^|;&]*\s(?:-[a-zA-Z]*e[a-zA-Z]*|--exit-status))(?=\s|$|\))")


_PIPE = re.compile(r"(?<!\|)\|(?!\|)")


def _hoisted(sub: str, v: str, ind: str, local: bool) -> str:
    """The line a hoisted substitution becomes. Global (SWALLOW-13): `__subN="$(…)"`, the script made
    strict around it. Local (SWALLOW-14): the substitution keeps its own status and nothing else
    changes -- `|| exit $?`, pipefail inside it when it is a pipeline -- so no other line of the
    script is made strict, and no `|| true` elsewhere is taken away."""
    body = sub[2:-1]
    answer = bool(_STATUS_ONE.search(body))
    if not local:
        return f'{ind}{v}="{sub}"' + (" || [ $? -eq 1 ]" if answer else "")
    sc = scan(body)
    if _PIPE.search(sc[1] if sc else body):
        sub = "$(set -o pipefail; " + body + ")"
    return f'{ind}{v}="{sub}"' + (' || { __rc=$?; [ "$__rc" -eq 1 ] || exit "$__rc"; }' if answer else " || exit $?")


def _hoist(run: str, local: bool) -> str:
    lines = run.splitlines()
    taken = set(re.findall(r"__sub(\d+)", run))
    k = 0

    def name() -> str:
        nonlocal k
        k += 1
        while str(k) in taken:
            k += 1
        return f"__sub{k}"

    inserts: dict[int, list[str]] = {}
    for a, b in _logical(lines):
        if a != b:
            continue                                   # a continued line: left alone
        line = lines[a]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        sc = scan(line)
        if sc is None:
            continue
        subs, masked = sc
        if not subs or not _hoistable(line, masked):
            continue
        if _FOR.match(line):
            cut = re.search(r";\s*do\b|\bdo\b", masked)
            subs = [s for s in subs if cut is None or s[1] <= cut.start()]
            if not subs:
                continue
        ind = line[: len(line) - len(line.lstrip())]
        new, pre, last = [], [], 0
        for s0, s1 in subs:
            v = name()
            pre.append(_hoisted(line[s0:s1], v, ind, local))
            new.append(line[last:s0] + "${" + v + "}")
            last = s1
        new.append(line[last:])
        lines[a] = "".join(new)
        inserts[a] = pre
    if not inserts:
        return run
    out = []
    for i, line in enumerate(lines):
        out.extend(inserts.get(i, []))
        out.append(line)
    text = "\n".join(out) + ("\n" if run.endswith("\n") else "")
    return text if local else strict_shell(text)


def hoist_substitution(run: str) -> str:
    """The hoist-substitution repair (SWALLOW-13: the whole script made strict); the script unchanged
    when no substitution can be hoisted."""
    return _hoist(run, local=False)


def hoist_local(run: str) -> str:
    """The hoist-local repair (SWALLOW-14): the same hoist, the substitution's status kept by its own
    line and nothing else in the script changed; the script unchanged when none can be hoisted."""
    return _hoist(run, local=True)


# a list read from a process substitution: `done < <(cmd)`, `mapfile -t a < <(cmd)`. Bash drops the
# status of `<(…)` -- `set -e` and pipefail never see it -- so a failed `cmd` is an empty list and the
# loop runs zero times. From bash 4.4, `$!` is the process substitution and `wait $!` returns its status.
_PROCSUB = re.compile(r"<\s*<\(")
# `$!` is empty before bash 4.4 when no job ran: the line then waits for nothing, as the step did
WAIT_LINE = {False: '[ -z "$!" ] || wait $! || exit $?  # stop if the command that made the list failed',
             True: '[ -z "$!" ] || wait $! || { __rc=$?; [ "$__rc" -eq 1 ] || exit "$__rc"; }  # stop if the command that made the list failed'}
_READS = re.compile(r"^\s*(?:mapfile|readarray|read)\b|\bdone\s*$")


def _masked_lines(lines: list[str]) -> list[str | None]:
    """Each line with quotes, substitutions and comments blanked (None when its quoting does not
    close on the line), for reading its operators and keywords."""
    out = []
    for ln in lines:
        sc = scan(ln)
        out.append(sc[1] if sc else None)
    return out


def _mask_text(text: str) -> str:
    """A script fragment, lines and all, with its quoted strings and comments blanked."""
    out, i, n, q = [], 0, len(text), None
    while i < n:
        ch = text[i]
        if q is None:
            if ch == "\\":
                out.append("_" + ("\n" if text[i + 1:i + 2] == "\n" else "_" * len(text[i + 1:i + 2])))   # a continued line keeps its newline
                i += 2
                continue
            if ch in "'\"":
                q = ch
                out.append("_")
                i += 1
                continue
            if ch == "#" and (i == 0 or text[i - 1] in " \t\n;"):
                j = text.find("\n", i)
                j = n if j < 0 else j
                out.append("_" * (j - i))
                i = j
                continue
            out.append(ch)
            i += 1
            continue
        if ch == "\\" and q == '"':
            out.append("_" + ("\n" if text[i + 1:i + 2] == "\n" else "_" * len(text[i + 1:i + 2])))
            i += 2
            continue
        if ch == q:
            q = None
        out.append("\n" if ch == "\n" else "_")
        i += 1
    return "".join(out)


def _loop_start(masked: list[str | None], end: int, prefix: str) -> int | None:
    """The line where the loop whose `done` ends `prefix` (line `end`, up to the substitution) begins;
    None when a line on the way cannot be read."""
    depth = 0
    for i in range(end, -1, -1):
        m = prefix if i == end else masked[i]
        if m is None:
            return None
        depth += len(re.findall(r"\bdone\b", m)) - len(re.findall(r"\bdo\b", m))
        if depth <= 0:
            return i
    return None


def wait_list(run: str) -> str:
    """The wait-list repair (SWALLOW-14): after a statement that reads its list from a process
    substitution, `wait $! || exit $?` -- the list's command stops the step when it fails, instead of
    leaving an empty list the loop reads as nothing to check; pipefail inside the substitution when
    it is a pipeline, for that command only; a command whose status 1 is an answer keeps its 1. The
    script unchanged when there is none, when a background job or another substitution inside the
    loop would take `$!`, or when the substitution is followed by more than a comment."""
    lines = run.splitlines()
    in_script = {i for a, b in _logical(lines) for i in range(a, b + 1)}     # heredoc bodies excluded
    text = "\n".join(lines)
    masked = _mask_text("\n".join(ln if i in in_script else " " * len(ln) for i, ln in enumerate(lines))).split("\n")
    if any(m.rstrip().endswith("&") and not m.rstrip().endswith("&&") for m in masked):
        return run                                                            # a background job would take `$!`
    starts, pos = [], 0
    for ln in lines:
        starts.append(pos)
        pos += len(ln) + 1

    def line_of(idx: int) -> int:
        return max(j for j in range(len(lines)) if starts[j] <= idx)

    edits = []                                                   # (index just past `<(`, closing line, piped, answer, first line)
    for mt in _PROCSUB.finditer(text):
        li = line_of(mt.start())
        col = mt.start() - starts[li]
        if li not in in_script or masked[li][col] != "<":
            continue                                             # a heredoc body, or inside quotes or a comment
        prefix = masked[li][:col]
        if not _READS.search(prefix):
            continue
        close = _close(text, mt.end())
        if close is None:
            continue
        cl = line_of(close - 1)
        if not re.match(r"^\s*(?:#.*)?$", text[close:starts[cl] + len(lines[cl])]):
            continue                                             # more than a comment after it
        if re.search(r"\bdone\s*$", prefix):
            st = _loop_start(masked, li, prefix)
            if st is None or "<(" in "\n".join(masked[st:li]) + prefix:
                continue                                         # another substitution inside the loop takes `$!` last
        body = _mask_text(text[mt.end():close - 1])
        edits.append((mt.end(), cl, bool(_PIPE.search(body)), bool(_STATUS_ONE.search(body)), li))
    if not edits:
        return run
    out_lines = list(lines)
    for k, cl, piped, _, _ in sorted(edits, key=lambda e: -e[0]):
        if piped:
            li = line_of(k - 1)
            col = k - starts[li]
            out_lines[li] = out_lines[li][:col] + ("set -o pipefail;" if col == len(lines[li]) else "set -o pipefail; ") + out_lines[li][col:]
    adds = {cl: lines[li][: len(lines[li]) - len(lines[li].lstrip())] + WAIT_LINE[answer] for _, cl, _, answer, li in edits}   # at the statement's indent
    final = []
    for i, ln in enumerate(out_lines):
        final.append(ln)
        if i in adds:
            final.append(adds[i])
    return "\n".join(final) + ("\n" if run.endswith("\n") else "")


_WORD = re.compile(r"^\s*[A-Za-z0-9_./~$\"'-]")


def background_liveness(run: str) -> str:
    """The background-liveness repair; the script unchanged when no command is started with `&`."""
    lines = run.splitlines()
    taken = set(re.findall(r"__bg(\d+)", run))
    k, out, changed = 0, [], False
    spans = _logical(lines)
    ends = {b: a for a, b in spans}
    for i, line in enumerate(lines):
        out.append(line)
        if i not in ends:
            continue
        a = ends[i]
        first = lines[a]
        sc = scan(" ".join(l.rstrip().rstrip("\\") for l in lines[a:i + 1]))
        if sc is None:
            continue
        masked = sc[1].rstrip()
        if not masked.endswith("&") or masked.endswith("&&") or not _WORD.match(first):
            continue
        if re.match(r"^\s*(wait|sleep|kill|echo|printf|\(|\{|done|fi)\b", first):
            continue
        k += 1
        while str(k) in taken:
            k += 1
        ind = first[: len(first) - len(first.lstrip())]
        v = f"__bg{k}"
        out += [f"{ind}{v}=$!", f"{ind}sleep 5", f'{ind}if ! kill -0 "${v}" 2>/dev/null; then wait "${v}" || exit $?; fi']
        changed = True
    if not changed:
        return run
    return "\n".join(out) + ("\n" if run.endswith("\n") else "")


_EXIT_ZERO = re.compile(r"\s*\|\|\s*exit\s+0\b")


def no_exit_zero(run: str) -> str:
    """The no-exit-zero repair; the script unchanged when no `|| exit 0` is found outside quotes."""
    out, changed = [], False
    for line in run.splitlines():
        sc = scan(line)
        if sc is not None:
            masked = sc[1]
            for m in reversed(list(_EXIT_ZERO.finditer(masked))):
                line = line[: m.start()] + line[m.end():]
                changed = True
        out.append(line)
    if not changed:
        return run
    return strict_shell("\n".join(out) + ("\n" if run.endswith("\n") else ""))


def no_default_joined(run: str) -> str:
    """SWALLOW-5's no-default with continued lines joined; the script unchanged when joining finds
    no fallback the unjoined script hides."""
    lines = run.splitlines()
    joined = []
    for a, b in _logical(lines):
        if a == b:
            joined.append(lines[a])
            continue
        parts = [lines[a].rstrip()[:-1].rstrip()] + [l.strip().rstrip("\\").rstrip() for l in lines[a + 1:b + 1]]
        joined.append(" ".join(p for p in parts if p))
    if len(joined) == len(lines):
        return run
    j = "\n".join(joined) + ("\n" if run.endswith("\n") else "")
    if rs.no_default(run) != run:
        return run                                      # the unjoined script already shows its fallback: SWALLOW-5's to repair
    nd = rs.no_default(j)
    if nd == j:
        return run
    return strict_shell(nd)


def transform(run: str, name: str) -> tuple[str | None, str | None]:
    """(repaired script, reason it does not apply) for a script-level edit of this stage or the last."""
    if name == "no-exit-zero":
        new = no_exit_zero(run)
        return (new, None) if new != run else (None, "no `|| exit 0`")
    if name == "no-default-joined":
        new = no_default_joined(run)
        return (new, None) if new != run else (None, "no `|| echo …` fallback hidden by a continued line")
    if name == "hoist-substitution":
        new = hoist_substitution(run)
        return (new, None) if new != run else (None, "no substitution whose status the line throws away")
    if name == "hoist-local":
        new = hoist_local(run)
        return (new, None) if new != run else (None, "no substitution whose status the line throws away")
    if name == "wait-list":
        new = wait_list(run)
        return (new, None) if new != run else (None, "no list read from a process substitution, or `$!` not its")
    if name == "background-liveness":
        new = background_liveness(run)
        return (new, None) if new != run else (None, "no command started with `&`")
    if name in ("no-default", "guard-status"):
        return rs.transform(run, name)
    return None, "unknown edit"


def apply_frontier(text: str, jid: str, i: int, name: str) -> tuple[str | None, str | None]:
    """(repaired workflow text, reason it does not apply)."""
    if name.startswith("no-coe+"):
        pos = locate(text, jid, i)
        if pos is None:
            return None, "step not located in the workflow text"
        if not (pos["step_coe"] or pos["job_coe"]):
            return None, "no continue-on-error on the step or its job"
        text2, why = repair.apply_repair(text, jid, i, "no-continue-on-error")
        if text2 is None:
            return None, why
        text, name = text2, name[len("no-coe+"):]
    pos = locate(text, jid, i)
    if pos is None or pos["run"] is None:
        return None, "step not located in the workflow text, or no run:"
    new_run, why = transform(pos["run"]["value"], name)
    if new_run is None:
        return None, why
    out = repair._replace_run(text.splitlines(), pos["run"], new_run)
    if out is None:
        return None, "run: value could not be rewritten in place"
    return "\n".join(out) + "\n", None


# ----------------------------------------------------------------------------- verification (SWALLOW-4's, over these candidates)

def try_frontier(text: str, wf_name: str, jid: str, i: int, runner: faults.Runner) -> dict:
    import yaml
    doc = yaml.safe_load(text)
    base_faults, base_plus = analyse(doc, wf_name, runner)
    base = base_faults.get((jid, i))
    rec = {"job": jid, "index": i, "baseline": None, "candidates": [], "verified_repair": None}
    if base is None:
        rec["baseline"] = {"verdict": None, "note": "not a fault site on this machine"}
        return rec
    rec["baseline"] = {"verdict": base["verdict"], "by_flavour": base["by_flavour"], "name": base["name"], "continue_on_error": base["continue_on_error"]}
    interpretable = [f for f, v in base["by_flavour"].items() if v in faults._PRECEDENCE]
    base_shape = {f: _healthy_shape(base_plus[f]) for f in faults.FLAVOURS}
    for name in REPAIRS:
        cand = {"repair": name}
        try:
            new_text, why = apply_frontier(text, jid, i, name)
        except Exception as e:  # noqa: BLE001 -- an edit that cannot read a script does not apply to it; the audit goes on
            new_text, why = None, f"the edit could not read this script ({type(e).__name__})"
        if new_text is None:
            cand.update(applies=False, why=why)
            rec["candidates"].append(cand)
            continue
        try:
            new_doc = yaml.safe_load(new_text)
        except Exception as e:  # noqa: BLE001
            cand.update(applies=False, why=f"repaired text does not parse: {str(e)[:80]}")
            rec["candidates"].append(cand)
            continue
        new_faults, new_plus = analyse(new_doc, wf_name, runner)
        after = new_faults.get((jid, i))
        cand.update(applies=True, diff=unified_diff(text, new_text, wf_name), lines_changed=diff_size(text, new_text))
        if after is None:
            cand.update(loud=False, unchanged=None, verified=False, why="the repaired step is no longer a fault site (reaches no tool alone)")
            rec["candidates"].append(cand)
            continue
        cand["verdict_after"] = after["verdict"]
        cand["by_flavour_after"] = after["by_flavour"]
        unchanged_by = {f: _healthy_shape(new_plus[f]) == base_shape[f] for f in faults.FLAVOURS}
        loud_by = {f: after["by_flavour"].get(f) == "RED" for f in interpretable}
        unchanged = all(unchanged_by.values())
        loud = bool(loud_by) and all(loud_by.values())
        cand.update(loud=loud, loud_by_flavour=loud_by, unchanged=unchanged, unchanged_by_flavour=unchanged_by, verified=loud and unchanged)
        if not unchanged:
            bad = [f for f, ok in unchanged_by.items() if not ok]
            cand["why"] = "changes the healthy run in flavour " + ", ".join(bad) + ": what the repaired line was protecting"
            cand["healthy_change"] = _first_change(base_plus, new_plus, bad[0])
        elif not loud:
            cand["why"] = "not loud: " + ", ".join(f"{f}={after['by_flavour'].get(f)}" for f in interpretable)
        rec["candidates"].append(cand)
        if cand["verified"] and rec["verified_repair"] is None:
            rec["verified_repair"] = name
    return rec


# ----------------------------------------------------------------------------- the two readings

_FLAG_SET = re.compile(r"(?:^|[\s;(|&])([A-Za-z_][A-Za-z0-9_]*)=(\$\?|[01]|true|false)(?=\s|;|$|\))")
_EXPORT = re.compile(r"""echo\s+["']?([A-Za-z_][A-Za-z0-9_-]*)=["']?\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?["']?[^\n]*>>\s*["']?\$\{?(GITHUB_ENV|GITHUB_OUTPUT)\}?""")
_DECLARED = re.compile(r"::warning\b|non[- ]?blocking|best[- _]?effort|allow(?:ed)?[- _]?(?:to[- _])?fail|don'?t fail|do not fail|"
                       r"(?:will|should|must) not fail|soft[- _]?fail|ignor(?:e|ing) (?:the )?(?:failure|error)s?|⚠", re.I)


def _steps(doc: dict, jid: str) -> list:
    job = (doc.get("jobs") or {}).get(jid) or {}
    return [s for s in (job.get("steps") or []) if isinstance(s, dict)]


def routed(doc: dict, jid: str, i: int) -> dict | None:
    """The step's status routed to a later step: a flag -- a variable the script sets to two of
    0/1/true/false, or to `$?` -- written to GITHUB_ENV or GITHUB_OUTPUT, under a name a later
    step of the workflow reads. None when there is none. What it returns names the flag, the name
    it travels under, and each reader: whether the reader can fail the job on it (`exit` in its
    script) or only conditions itself on it (`if:`)."""
    steps = _steps(doc, jid)
    if i >= len(steps) or not isinstance(steps[i].get("run"), str):
        return None
    run = steps[i]["run"]
    values: dict = {}
    for m in _FLAG_SET.finditer(run):
        values.setdefault(m.group(1), set()).add(m.group(2))
    flags = {v for v, vals in values.items() if "$?" in vals or len(vals & {"0", "1"}) == 2 or len(vals & {"true", "false"}) == 2}
    for v, vals in values.items():                       # a counter: set to 0, then counted up
        if "0" in vals and re.search(r"\b" + re.escape(v) + r"=\$\(\(\s*\$?" + re.escape(v) + r"\s*\+|\(\(\s*" + re.escape(v) + r"\s*(\+\+|\+=)", run):
            flags.add(v)
    exported = [(m.group(1), m.group(3)) for m in _EXPORT.finditer(run) if m.group(2) in flags]
    if not exported:
        return None
    sid = steps[i].get("id")
    readers = []
    later = [(jid, k, s) for k, s in enumerate(steps) if k > i]
    for ojid in (doc.get("jobs") or {}):
        if ojid != jid:
            later += [(ojid, k, s) for k, s in enumerate(_steps(doc, ojid))]
    for name, where in exported:
        pats = [re.compile(r"\benv\." + re.escape(name) + r"\b"), re.compile(r"\$\{?" + re.escape(name) + r"\b")]
        if where == "GITHUB_OUTPUT" and sid:
            pats = [re.compile(r"steps\." + re.escape(str(sid)) + r"\.outputs\." + re.escape(name) + r"\b")] + pats
        for ojid, k, s in later:
            cond, script = str(s.get("if", "")), s.get("run") if isinstance(s.get("run"), str) else ""
            hit_if = any(p.search(cond) for p in pats)
            hit_run = any(p.search(script) for p in pats)
            if hit_if or hit_run:
                readers.append({"name": name, "via": where, "job": ojid, "step": k, "step_name": str(s.get("name") or s.get("id") or f"step {k}"),
                                "can_fail": bool(hit_run and re.search(r"\bexit\s+[1-9]", script)), "conditions": hit_if})
    if not readers:
        return None
    return {"flags": sorted(flags), "exported": sorted({n for n, _ in exported}), "readers": readers,
            "a_reader_can_fail": any(r["can_fail"] for r in readers)}


def declared(doc: dict, jid: str, i: int) -> str | None:
    """The words by which the step's script declares a failure non-fatal, or None."""
    steps = _steps(doc, jid)
    if i >= len(steps) or not isinstance(steps[i].get("run"), str):
        return None
    m = _DECLARED.search(steps[i]["run"])
    return m.group(0) if m else None


def readings(text: str, jid: str, i: int) -> dict:
    """Both readings of one step, on the workflow's text: {"routed": ..., "declared": ...}."""
    import yaml
    try:
        doc = yaml.safe_load(text) or {}
    except Exception:  # noqa: BLE001
        return {"routed": None, "declared": None}
    if not isinstance(doc, dict):
        return {"routed": None, "declared": None}
    return {"routed": routed(doc, jid, i), "declared": declared(doc, jid, i)}


def say(r: dict) -> str | None:
    """One line for a card, an annotation or a summary: what the readings found, or None."""
    parts = []
    ro = r.get("routed")
    if ro:
        rd = ro["readers"][0]
        more = len(ro["readers"]) - 1
        parts.append(f"its failure is routed, not lost: {', '.join(ro['exported'])} via {rd['via']}, read by '{rd['step_name']}' (job {rd['job']})"
                     + (f" and {more} more" if more else "") + (" -- a reader can fail the job" if ro["a_reader_can_fail"] else " -- no reader fails the job on it"))
    if r.get("declared"):
        parts.append(f"the script says '{r['declared']}'")
    return "; ".join(parts) or None
