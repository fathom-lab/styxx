"""SWALLOW-2 -- single-fault injection: when one step's tools fail, what does the workflow do?

    python -m benchmarks.harness_mutation.faults --tree /path/to/checkout [--out receipt.json]
    python -m benchmarks.harness_mutation.faults --repos repos.json --out receipt.json [--work DIR]

SWALLOW-1 executed every `run:` step alone, with every external command failing, and found that
in 67 of 87 repositories a step cannot fail. It could not say which way such a step falls: a
query that fails and is answered with "nothing changed" may skip the check (fail-open) or run
everything (fail-closed), and both exit 0. This module answers that by *simulating the workflow*
rather than the step, in two worlds:

    W+      every external command succeeds: exits 0 and prints one line, `x`
    W-A     the same, except that in step A every external command fails (exit 127, no output)

Steps are executed in job order with the contexts GitHub Actions gives them: what a step writes
to $GITHUB_OUTPUT and $GITHUB_ENV is read back, `${{ steps.<id>.outputs.<k> }}`,
`${{ env.<k> }}` and `${{ needs.<job>.outputs.<k> }}` are substituted from it, every step's `if:`
and every job's `if:` and `needs:` are evaluated, and a job whose matrix comes from an empty
`fromJSON(...)` runs zero times. A value the simulation cannot know (an action's outputs,
`github.*`, `inputs.*`, `matrix.*`, `secrets.*`) is *unknown*: it substitutes as `x` in a script,
and a condition that depends on it is treated as true, so the simulation skips a step only when
the condition is false on what it actually knows.

For every bash `run:` step A that reaches at least one external command alone, the workflow is
run in W-A and compared with W+, and the fault gets one verdict:

    RED         a job goes red in W-A: the failure of A's tools is loud
    FAIL_OPEN   nothing goes red, and a verification step (test / lint / typecheck) that reached
                a tool in W+ is skipped, or runs without reaching a tool, in W-A: the failure
                silently removed a check
    SWALLOWED   nothing goes red, A is itself a verification step, and it reached its tool in
                W-A: the check ran and its failure was hidden (`|| true`, `continue-on-error`)
    ABSORBED    nothing goes red and every verification step reaches its tools exactly as in W+:
                A's failure changed nothing about the checks
    NO_CHECK    no verification step in A's job or downstream of it reaches a tool in W+: there
                was no check to protect
    BASELINE_RED  A's job is already red in W+ before A runs (the script fails on its own
                logic under `x`): the fault is uninterpretable and is reported, not counted

Precedence is the order above. A verification step is one SWALLOW-1's `categorize` calls test,
lint or typecheck; the receipt keeps every step's text and the reader may disagree.

What this does not say. Injected faults are faults of *tools*: every external command of one
step fails at once. An action (`uses:`) never fails here and its outputs are unknown. The OK
world's `x` is a model of success, not a runner: a script that `exit 1`s on `x` is BASELINE_RED
and says nothing. Windows and non-bash steps are not executed. RED means the pipeline is loud
when A's tools fail; it does not mean A is correct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .census import _kill_group, _shell_for, categorize, sparse_clone

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------- what is a check
#
# SWALLOW-1's category was a coarse tag on name + first line, and its own RESULT said it mis-filed
# installs and cleanups under "test". Here "is this step a check" decides a verdict, so the rule
# is tighter and stated: a step is a VERIFICATION step when its first command (comments, `set`,
# `export`, `cd`, `echo`, assignments skipped) invokes a known test / lint / typecheck runner or a
# script whose name says test / lint / check / verify / validate, or when its name says so and
# does not also say install / build / setup / report / generate / ... . A fixer (`--fix`,
# `:fix`, `--write`) is not a check. The receipt keeps every name and script head; the rule is a
# heuristic and the RESULT reads what it selects.
_RUNNER = (r"(?:pytest|py\.test|unittest|jest|vitest|mocha|karma|cypress|playwright\s+test|tox|nox|ctest|bats|phpunit|rspec|"
           r"minitest|eslint|ruff|flake8|pylint|prettier|black|isort|mypy|pyright|tsc|clippy|rustfmt|gofmt|golangci-lint|shellcheck|"
           r"yamllint|markdownlint|stylelint|rubocop|swiftlint|ktlint|checkstyle|vale\s+(?!sync)|codespell|biome|oxlint|knip|actionlint|"
           r"hadolint|luacheck|xcodebuild\s+test|swift\s+test|zig\s+build\s+test|cargo\s+(?:test|clippy|fmt|nextest)|go\s+(?:test|vet)|"
           r"\S*dotnet(?:\.sh)?\s+(?:test|format)|mvn\s+(?:test|verify)|(?:\./)?gradlew?\s+(?:test|check|lint)|pre-commit\s+run|"
           r"deno\s+(?:test|lint|check)|bun\s+test|node\s+--test|jac\s+test|sfw\s+bun\s+run\s+test)")
_SCRIPT = (r"(?:(?:npm|pnpm|yarn|bun|deno)\s+(?:(?:--filter|-F|-w|--workspace)\s*\S+\s+)*(?:run\s+)?"
           r"(?:\S*(?:test|lint|typecheck|type-check|check|verify|validate)\S*)|"
           r"make\s+\S*(?:test|lint|check|verify|validate)\S*|"
           r"(?:uv|poetry|pipenv|hatch|pdm|rye)\s+run\s+" + _RUNNER + r"|"
           r"(?:python3?|node|bash|sh|ruby|pwsh|bun|npx|tsx|deno\s+run)\s+(?:-m\s+)?(?:-u\s+)?\S*(?:tests?|lint|check|verify|validat)\S*|"
           r"(?:\./)?\S*/(?:tests?|ci|scripts?)/\S*(?:tests?|lint|check|verify|validat)\S*|"
           r"\./\S*(?:tests?|lint|check|verify|validat)\S*\.(?:sh|py|js|ts|rb))")
TOOL_RX = re.compile(r"^(?:\S*/)?(?:python3?\s+-m\s+)?(?:" + _RUNNER + r"|" + _SCRIPT + r")\b", re.I)
FIXER_RX = re.compile(r"(?::fix\b|--fix\b|-fix\b|--write\b|:write\b)", re.I)
NAME_RX = re.compile(r"\b(tests?|testing|lint(?:ing|ers?)?|type-?check(?:ing|s)?|mypy|pyright|eslint|ruff|clippy|rustfmt|gofmt|prettier|"
                     r"verify|verification|validate|validation|smoke[- ]?tests?|e2e|end-to-end|check\s+(?:formatting|format|types|lint|style|licen[cs]es?)|"
                     r"format(?:ting)?\s+check|static\s+analysis|sanity|regression)\b", re.I)
EXCL_RX = re.compile(r"\b(install|installing|setup|set\s*up|cache|caching|download|upload|publish|deploy|release|build|building|compile|compiling|"
                     r"docker\s+(?:build|push|login)|image|artifact|report|summary|comment|notify|notification|clean|cleanup|tear\s*down|checkout|"
                     r"badge|merge|dispatch|prepare|generate|convert|disable|enable|create|configure|start|stop|launch|wait|version|dependencies|deps|"
                     r"prerequisites|bootstrap|fixture|seed|migrat\w*|snapshot|collect|coverage|gather|extract|resolve|detect|discover|find|locate|list|"
                     r"count|filter|record|save|store|write|print|show|display|dump|log|patch|sync|update|fix|apply|restore|rebuild|push|pull|fetch|clone|login|auth)\b", re.I)
_SKIP_LINE = re.compile(r"^(set\s|export\s|cd\s|echo\s|printf\s|mkdir\s|source\s|\.\s|[A-Za-z_][A-Za-z0-9_]*=)")


_SEGMENT = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")


def _logical_lines(run: str) -> list[str]:
    """Physical lines joined at backslash continuations."""
    out, buf = [], ""
    for line in run.splitlines():
        if line.rstrip().endswith("\\"):
            buf += line.rstrip()[:-1] + " "
            continue
        out.append(buf + line)
        buf = ""
    if buf:
        out.append(buf)
    return out


def commands(run: str) -> list[str]:
    """Every command segment of a script: lines minus comments, split on && || ; |, in order."""
    out = []
    for line in _logical_lines(run):
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        for seg in _SEGMENT.split(t):
            seg = seg.strip()
            if seg and not _SKIP_LINE.match(seg):
                out.append(seg)
    return out


def first_command(run: str) -> str:
    c = commands(run)
    return c[0] if c else ""


def verification(name, run: str) -> str | None:
    """'tool' | 'name' | None -- why a step counts as a check, or that it does not."""
    for cmd in commands(run):
        if TOOL_RX.search(cmd):
            return None if FIXER_RX.search(cmd) else "tool"
    n = str(name or "")
    if NAME_RX.search(n) and not EXCL_RX.search(n):
        return "name"
    return None

STEP_TIMEOUT_OK = 10        # the OK world cannot fail out of a loop, so a spin is a spin
STEP_TIMEOUT_FAIL = 30      # SWALLOW-1's
REPO_SECONDS_CAP = 900      # per repository; a capped repository is recorded as such

# Three worlds. "fail" is SWALLOW-1's: every external command is not found. The two healthy
# worlds differ in one thing, what a successful command prints -- one line `x`, or nothing --
# because a healthy run answers some queries with something (`git diff --name-only` in a
# discover step) and others with nothing (`git status --porcelain` in a clean-tree check), and
# the simulation cannot know which a script wants. Both are tried. A healthy command also drains
# its standard input, as a real `tee` or `jq` would, so a pipeline into it is not a SIGPIPE.
# In the healthy worlds the plain filesystem utilities are real: a healthy `mkdir -p` makes the
# directory a later redirection writes into, a healthy `tee -a $GITHUB_OUTPUT` records the output
# a later `if:` reads. They are not what a fault model is about. Everything else -- git, gh, npm,
# pytest, docker, jq, grep, and every other tool -- is the world's stub. `sleep` returns at once.
REAL_IN_HEALTHY = ("mkdir", "touch", "rm", "rmdir", "tee", "dirname", "basename", "realpath", "readlink", "chmod",
                   "mktemp", "pwd", "date", "seq", "env", "printenv", "true", "false")
_HEALTHY_CD = ('_mkd() { /bin/mkdir -p "$1" 2>/dev/null || /usr/bin/mkdir -p "$1" 2>/dev/null; }\n'
               'cd() { builtin cd "$@" 2>/dev/null || { _mkd "${@: -1}" && builtin cd "$@"; }; }\n'
               'pushd() { builtin pushd "$@" >/dev/null 2>&1 || { _mkd "${@: -1}" && builtin pushd "$@" >/dev/null; }; }\n')
# grep is the one tool whose "found nothing" is an exit status, not an empty line: in the empty
# flavour grep and its kin exit 1 with no output (no match), so `! grep pattern .` passes and
# `if cmd | grep -q x; then` takes its else branch; in the x flavour they match. Likewise
# `git diff --quiet` / `--exit-code` answer with a status: in the x flavour ("something") they
# exit 1, there are differences; in the empty flavour ("nothing") they exit 0.
_GREP_EMPTY = 'case "$1" in grep|egrep|fgrep|rg|ag|ack) return 1;; esac; '
_GIT_DIFF_X = 'if [ "$1" = git ]; then case " $* " in *" diff "*" --quiet"*|*" diff "*" --exit-code"*|*" diff-index "*" --quiet"*) return 1;; esac; fi; '
_DRAIN = '[ -t 0 ] || while IFS= read -r _; do :; done; '
PROLOGUE = {
    "fail":  'command_not_found_handle() { printf "%s\\n" "$1" >> "$STUB_LOG"; return 127; }\n',
    "x":     'command_not_found_handle() { printf "%s\\n" "$1" >> "$STUB_LOG"; ' + _DRAIN + _GIT_DIFF_X + 'printf "x\\n"; return 0; }\n',
    "empty": 'command_not_found_handle() { printf "%s\\n" "$1" >> "$STUB_LOG"; ' + _DRAIN + _GREP_EMPTY + 'return 0; }\n',
}
PROLOGUE["x"] += _HEALTHY_CD
PROLOGUE["empty"] += _HEALTHY_CD
# "fault" is the fail world of a step inside a simulated workflow: its tools fail, but the
# directories the healthy world would have are still there -- `cd sub && tool` must reach the
# tool and fail on it, not fail on the cd. "fail" alone is SWALLOW-1's world, kept for the
# per-step alone verdict so that G-S2-3 compares like with like.
PROLOGUE["fault"] = PROLOGUE["fail"] + _HEALTHY_CD
STUB_BODY = {
    "fail":  'exit 127\n',
    "fault": 'exit 127\n',
    "x":     _DRAIN + 'printf "x\\n"\nexit 0\n',
    "empty": _DRAIN + 'exit 0\n',
}


# ----------------------------------------------------------------------------- expressions

class _Unknown:
    """A context value the simulation cannot know. Substitutes as `x`; makes a condition true."""
    def __repr__(self) -> str:
        return "UNKNOWN"


UNKNOWN = _Unknown()


class _Answer:
    """A value the healthy world produced from a tool's output: the model's `x`. It is an answer
    of unknown content -- non-empty, so `!= ''` is true and `== ''` is false -- but it is not any
    particular string, so `== 'true'` is UNKNOWN rather than false."""
    def __repr__(self) -> str:
        return "ANSWER"


ANSWER = _Answer()


def _known(v):
    """Turn the model's placeholder into ANSWER; leave every other value as it is."""
    if isinstance(v, str) and v.strip() and all(line.strip() == "x" for line in v.strip().splitlines()):
        return ANSWER
    return v
_TOKEN = re.compile(r"""\s*(?:(?P<str>'(?:[^']|'')*')|(?P<num>-?\d+(?:\.\d+)?)|(?P<op>&&|\|\||==|!=|<=|>=|[!<>()\[\],.*])|(?P<id>[A-Za-z_][A-Za-z0-9_\-]*))""")


def _tokens(expr: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    while pos < len(expr):
        m = _TOKEN.match(expr, pos)
        if not m or m.end() == pos:
            if expr[pos:].strip() == "":
                break
            raise ValueError(f"cannot tokenize expression at {expr[pos:pos+20]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind:
            out.append((kind, m.group(kind)))
    return out


def truthy(v):
    if v is UNKNOWN:
        return UNKNOWN
    if v is ANSWER:
        return True
    if v is None or v is False:
        return False
    if isinstance(v, str):
        return v != ""
    if isinstance(v, (int, float)):
        return v != 0
    return True


def _num(v):
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return None
    return None


def _eq(a, b):
    if a is UNKNOWN or b is UNKNOWN:
        return UNKNOWN
    if a is ANSWER or b is ANSWER:
        other = b if a is ANSWER else a
        if other is ANSWER:
            return UNKNOWN
        return False if other in ("", None) else UNKNOWN
    na, nb = _num(a), _num(b)
    if na is not None and nb is not None and not (isinstance(a, str) and isinstance(b, str)):
        return na == nb
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    if a is None or b is None:
        return (a in (None, "")) and (b in (None, ""))
    return a == b


class Evaluator:
    """A recursive-descent evaluator for the GitHub Actions expression subset workflows use."""

    def __init__(self, ctx: "Context"):
        self.ctx = ctx

    def eval(self, expr: str):
        try:
            self.t = _tokens(expr)
        except ValueError:
            return UNKNOWN
        self.i = 0
        try:
            v = self._or()
            return v if self.i >= len(self.t) else UNKNOWN
        except (IndexError, ValueError, KeyError):
            return UNKNOWN

    def _peek(self, *ops):
        if self.i < len(self.t):
            k, v = self.t[self.i]
            if k == "op" and v in ops:
                return v
        return None

    def _take(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def _or(self):
        v = self._and()
        while self._peek("||"):
            self._take()
            r = self._and()
            tv = truthy(v)
            v = v if tv is True else (r if tv is False else UNKNOWN if truthy(r) is not True else r)
        return v

    def _and(self):
        v = self._not()
        while self._peek("&&"):
            self._take()
            r = self._not()
            tv = truthy(v)
            v = r if tv is True else (v if tv is False else UNKNOWN if truthy(r) is not False else r)
        return v

    def _not(self):
        if self._peek("!"):
            self._take()
            v = truthy(self._not())
            return UNKNOWN if v is UNKNOWN else (not v)
        return self._cmp()

    def _cmp(self):
        v = self._prim()
        op = self._peek("==", "!=", "<", ">", "<=", ">=")
        if not op:
            return v
        self._take()
        r = self._prim()
        if op == "==":
            return _eq(v, r)
        if op == "!=":
            e = _eq(v, r)
            return UNKNOWN if e is UNKNOWN else (not e)
        if v is UNKNOWN or r is UNKNOWN or v is ANSWER or r is ANSWER:
            return UNKNOWN
        a, b = _num(v), _num(r)
        if a is None or b is None:
            a, b = str(v).lower(), str(r).lower()
        return {"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]

    def _prim(self):
        k, v = self._take()
        if k == "str":
            return v[1:-1].replace("''", "'")
        if k == "num":
            return _num(v)
        if k == "op" and v == "(":
            inner = self._or()
            if not self._peek(")"):
                raise ValueError("expected )")
            self._take()
            return inner
        if k == "id":
            if self._peek("("):
                self._take()
                args = []
                if not self._peek(")"):
                    args.append(self._or())
                    while self._peek(","):
                        self._take()
                        args.append(self._or())
                if not self._peek(")"):
                    raise ValueError("expected )")
                self._take()
                return self._call(v.lower(), args)
            if v.lower() in ("true", "false", "null"):
                return {"true": True, "false": False, "null": None}[v.lower()]
            path = [v]
            while self._peek(".", "["):
                if self._peek("."):
                    self._take()
                    kk, vv = self._take()
                    if kk == "op" and vv == "*":
                        path.append("*")
                    else:
                        path.append(vv)
                else:
                    self._take()
                    idx = self._or()
                    if not self._peek("]"):
                        raise ValueError("expected ]")
                    self._take()
                    path.append("x" if idx is UNKNOWN else str(idx))
            return self.ctx.resolve(path)
        raise ValueError("unexpected token")

    def _call(self, name: str, args: list):
        if name == "success":
            return self.ctx.success()
        if name == "failure":
            s = self.ctx.success()
            return UNKNOWN if s is UNKNOWN else (not s)
        if name == "always":
            return True
        if name == "cancelled":
            return False
        if any(a is UNKNOWN or a is ANSWER for a in args):
            return UNKNOWN
        if name == "contains" and len(args) == 2:
            a, b = args
            if isinstance(a, list):
                return any(_eq(x, b) is True for x in a)
            return str(b).lower() in str(a).lower()
        if name == "startswith" and len(args) == 2:
            return str(args[0]).lower().startswith(str(args[1]).lower())
        if name == "endswith" and len(args) == 2:
            return str(args[0]).lower().endswith(str(args[1]).lower())
        if name == "format" and args:
            s = str(args[0])
            for n, a in enumerate(args[1:]):
                s = s.replace("{" + str(n) + "}", str(a))
            return s
        if name == "join":
            sep = str(args[1]) if len(args) > 1 else ","
            return sep.join(str(x) for x in args[0]) if isinstance(args[0], list) else str(args[0])
        if name == "tojson" and args:
            return json.dumps(args[0])
        if name == "fromjson" and args:
            try:
                return json.loads(str(args[0]))
            except (ValueError, TypeError):
                return "" if str(args[0]).strip() == "" else UNKNOWN
        return UNKNOWN


class Context:
    """What a step can see: this job's steps so far, the needed jobs, the env, and nothing else known."""

    def __init__(self, steps: dict, needs: dict, env: dict, job_failed):
        self.steps, self.needs, self.env, self.job_failed = steps, needs, env, job_failed

    def success(self):
        return UNKNOWN if self.job_failed is UNKNOWN else (not self.job_failed)

    def resolve(self, path: list[str]):
        head = path[0].lower()
        if len(path) >= 3 and path[1] == "*" and head in ("steps", "needs"):
            # the object filter: needs.*.result, steps.*.outcome -- a list over what is known
            src = self.steps if head == "steps" else self.needs
            what = path[2].lower()
            if not src:
                return UNKNOWN
            vals = []
            for v in src.values():
                if head == "needs" and what == "result":
                    vals.append(v["result"])
                elif head == "steps" and what in ("outcome", "conclusion"):
                    vals.append(v[what])
                else:
                    return UNKNOWN
            return vals
        if head == "steps" and len(path) >= 3:
            st = self.steps.get(path[1])
            if st is None:
                return UNKNOWN
            what = path[2].lower()
            if what == "outputs":
                if st["outputs"] is UNKNOWN:
                    return UNKNOWN
                return _known(st["outputs"].get(path[3], "")) if len(path) > 3 else UNKNOWN
            if what in ("outcome", "conclusion"):
                return st[what]
            return UNKNOWN
        if head == "needs" and len(path) >= 3:
            j = self.needs.get(path[1])
            if j is None:
                return UNKNOWN
            what = path[2].lower()
            if what == "result":
                return j["result"]
            if what == "outputs":
                if j["outputs"] is UNKNOWN:
                    return UNKNOWN
                return _known(j["outputs"].get(path[3], "")) if len(path) > 3 else UNKNOWN
            return UNKNOWN
        if head == "env" and len(path) >= 2:
            v = self.env.get(path[1])
            return UNKNOWN if v is None else _known(v)
        if head == "job" and len(path) >= 2 and path[1].lower() == "status":
            s = self.success()
            return UNKNOWN if s is UNKNOWN else ("success" if s else "failure")
        return UNKNOWN


_EXPR = re.compile(r"\$\{\{(.*?)\}\}", re.S)


def substitute(text: str, ctx: Context) -> str:
    def one(m):
        v = Evaluator(ctx).eval(m.group(1).strip())
        if v is UNKNOWN or v is ANSWER:
            return "x"
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (list, dict)):
            return json.dumps(v)
        return str(v)
    return _EXPR.sub(one, text)


def condition(expr, ctx: Context):
    """True, False, or UNKNOWN for an `if:` value. Missing = success(). `${{ }}` wrappers allowed."""
    if expr is None:
        return ctx.success()
    if isinstance(expr, bool):
        return expr
    s = str(expr).strip()
    m = _EXPR.fullmatch(s)
    if m:
        s = m.group(1).strip()
    return truthy(Evaluator(ctx).eval(s))


# ----------------------------------------------------------------------------- execution

def _parse_kv_file(path: Path) -> dict:
    """$GITHUB_OUTPUT / $GITHUB_ENV: `k=v` lines and `k<<DELIM ... DELIM` blocks."""
    out: dict = {}
    if not path.exists():
        return out
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_\-.]*)<<(.+)$", line)
        if m:
            k, delim = m.group(1), m.group(2).strip()
            buf = []
            i += 1
            while i < len(lines) and lines[i].strip() != delim:
                buf.append(lines[i])
                i += 1
            out[k] = "\n".join(buf)
            i += 1
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            if re.match(r"^[A-Za-z_][A-Za-z0-9_\-.]*$", k):
                out[k] = v
        i += 1
    return out


_REL_CMD = re.compile(r"^(?:\.{1,2}/|[A-Za-z0-9_.-]+/)[A-Za-z0-9_./-]+$")
_TMP_PREFIX = re.compile(r"^\$\{?(RUNNER_TEMP|GITHUB_WORKSPACE|HOME)\}?/([A-Za-z0-9_./-]+)$")


_CMD_POSITION = re.compile(r"(?:^|\$\(|`|[A-Za-z_][A-Za-z0-9_]*=|&&|\|\||;|\|)\s*")


def _command_heads(text: str) -> list[str]:
    """The first word of every command position in a script: segment starts, `$(`, backticks and
    the right-hand side of an assignment, with a leading interpreter (bash, python, node ...)
    stripped."""
    heads = []
    for line in _logical_lines(text):
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        for seg in _CMD_POSITION.split(t):
            seg = seg.strip()
            for pre in ("bash ", "sh ", "python ", "python3 ", "node ", "bun ", "ruby ", "exec ", "sudo ", "time "):
                if seg.startswith(pre):
                    seg = seg[len(pre):].lstrip()
            if seg:
                heads.append(seg.split()[0].strip("\"'`)};"))
    return heads


def _relative_commands(text: str) -> set[str]:
    """Paths in command position that the sandbox can hold a stub for: relative ones
    (./ci/test.sh, scripts/lint.py) and ones under $RUNNER_TEMP, $GITHUB_WORKSPACE or $HOME,
    returned relative to the sandbox root; never other $-expansions or absolute paths."""
    out = set()
    for head in _command_heads(text):
        if ".." in head:
            continue
        if _REL_CMD.match(head):
            out.add("work/" + head)
        else:
            m = _TMP_PREFIX.match(head)
            if m:
                out.add(("work/" if m.group(1) == "GITHUB_WORKSPACE" else "") + m.group(2))
    return out


class Runner:
    """Executes one step's shell in one world, memoised on (world, text, env)."""

    def __init__(self):
        self.cache: dict = {}
        self.executions = 0

    def run(self, text: str, world: str, env: dict) -> dict:
        key = (world, text, tuple(sorted(env.items())))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        self.executions += 1
        res = self._execute(text, world, env)
        self.cache[key] = res
        return res

    @staticmethod
    def _execute(text: str, world: str, env: dict) -> dict:
        # mkdtemp + rmtree(ignore_errors) rather than TemporaryDirectory(ignore_cleanup_errors=...),
        # which Python 3.9 -- still in this repository's test matrix -- does not have. A step may
        # leave a backgrounded builtin loop writing into the directory after bash returns; the
        # group is killed, but the race is real and the instrument must not stop for it.
        td = tempfile.mkdtemp(prefix="faults_")
        try:
            return Runner._execute_in(Path(td), text, world, env)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    @staticmethod
    def _execute_in(t: Path, text: str, world: str, env: dict) -> dict:
        (t / "bin").mkdir()
        (t / "work").mkdir()
        if world not in ("fail", "fault"):
            for name in REAL_IN_HEALTHY:
                real = shutil.which(name)
                if real:
                    (t / "bin" / name).symlink_to(real)
            (t / "bin" / "sleep").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
            (t / "bin" / "sleep").chmod(0o755)
        log = t / "reached.log"
        script = t / "step.sh"
        script.write_text(PROLOGUE[world] + text, encoding="utf-8")
        # bash never calls command_not_found_handle for a command with a slash in it, so a
        # script the repository ships (./ci/test.sh, scripts/lint.py) would be "No such file"
        # in both worlds. Each relative path a segment starts with gets a stub that behaves as
        # the world says: logs itself, then exits 127 (fail) or prints x and exits 0 (ok).
        for rel in sorted(_relative_commands(text), key=lambda r: -r.count("/")):   # deepest first: a/b/c.sh before a/b
            stub = t / rel
            try:
                if stub.exists():
                    continue
                stub.parent.mkdir(parents=True, exist_ok=True)
                stub.write_text('#!/bin/bash\nprintf "%s\\n" "$0" >> "$STUB_LOG"\n' + STUB_BODY[world], encoding="utf-8")
                stub.chmod(0o755)
            except OSError:
                pass
        full_env = {"PATH": str(t / "bin"), "STUB_LOG": str(log), "HOME": str(t), "LANG": "C.UTF-8",
                    "GITHUB_OUTPUT": str(t / "output"), "GITHUB_ENV": str(t / "env"), "GITHUB_PATH": str(t / "path"),
                    "GITHUB_STEP_SUMMARY": str(t / "summary"), "GITHUB_WORKSPACE": str(t / "work"), "RUNNER_TEMP": str(t),
                    "GITHUB_ACTIONS": "true", "CI": "true"}
        for k, v in env.items():
            if k not in full_env and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k):
                full_env[k] = v
        proc = subprocess.Popen(["/bin/bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
                                cwd=str(t / "work"), env=full_env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        timeout = STEP_TIMEOUT_FAIL if world in ("fail", "fault") else STEP_TIMEOUT_OK
        try:
            _, err = proc.communicate(timeout=timeout)
            rc = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            rc, err, timed_out = -1, "", True
        _kill_group(proc)
        reached = log.read_text(encoding="utf-8", errors="replace").split() if log.exists() else []
        return {"exit": rc, "timeout": timed_out, "reached": reached, "n_reached": len(reached),
                "outputs": _parse_kv_file(t / "output"), "env": _parse_kv_file(t / "env"),
                "tail": ((err or "").strip().splitlines() or [""])[-1][:160]}


# ----------------------------------------------------------------------------- simulation

def _needs(job: dict) -> list[str]:
    n = job.get("needs")
    if n is None:
        return []
    return [n] if isinstance(n, str) else [str(x) for x in n]


def _job_order(jobs: dict) -> list[str]:
    order, seen = [], set()

    def visit(j, stack=()):
        if j in seen or j not in jobs or j in stack:
            return
        for d in _needs(jobs[j] if isinstance(jobs[j], dict) else {}):
            visit(d, stack + (j,))
        seen.add(j)
        order.append(j)
    for j in jobs:
        visit(j)
    return order


def _env_map(mapping, ctx: Context) -> dict:
    out = {}
    if isinstance(mapping, dict):
        for k, v in mapping.items():
            out[str(k)] = substitute("" if v is None else str(v), ctx)
    return out


def _matrix_empty(strategy, ctx: Context):
    """True when the matrix comes from a fromJSON(...) that is empty in this world."""
    if not isinstance(strategy, dict) or "matrix" not in strategy:
        return False
    m = strategy["matrix"]
    texts = []
    if isinstance(m, str):
        texts.append(m)
    elif isinstance(m, dict):
        for v in m.values():
            if isinstance(v, str):
                texts.append(v)
    for t in texts:
        mm = _EXPR.fullmatch(t.strip())
        if not mm:
            continue
        v = Evaluator(ctx).eval(mm.group(1).strip())
        if v is UNKNOWN or v is ANSWER:
            continue
        if v in ("", None, [], {}) or (isinstance(v, dict) and all(x in ([], "", None) for x in v.values())):
            return True
    return False


def simulate(doc: dict, runner: Runner, fault: tuple[str, int] | None, flavour: str = "x",
             artifacts: set | None = None) -> dict:
    """Run one workflow document in W+ (fault=None) or W-A (fault=(job id, step index)), the
    healthy commands printing `x` (flavour "x") or nothing (flavour "empty").

    The healthy world is the world in which every step passes. A healthy-flavour step that
    exits non-zero anyway has failed on its own logic under the model (`x` is not the version
    string it expected), which is an artifact of the model, not a finding: in W+ every such
    failure is recorded on the step as `artifact` and the job carries on. In W-A the same
    artifacts -- the set collected from W+ -- are ignored again, so that a failure W-A has and
    W+ does not is caused by the fault, and a failure both have is not."""
    jobs = {k: v for k, v in (doc.get("jobs") or {}).items() if isinstance(v, dict)}
    wf_env_raw = doc.get("env") if isinstance(doc.get("env"), dict) else {}
    results: dict = {}
    for jid in _job_order(jobs):
        job = jobs[jid]
        needs_ctx = {d: results[d] for d in _needs(job) if d in results}
        needs_ok = all(results[d]["result"] == "success" for d in _needs(job) if d in results)
        base_ctx = Context({}, needs_ctx, {}, job_failed=(not needs_ok))
        env = _env_map(wf_env_raw, base_ctx)
        env.update(_env_map(job.get("env"), Context({}, needs_ctx, env, job_failed=(not needs_ok))))
        ctx0 = Context({}, needs_ctx, env, job_failed=(not needs_ok))
        cond = condition(job.get("if"), ctx0)
        rec = {"result": "success", "outputs": {}, "steps": [], "skipped": None}
        if cond is False:
            rec.update(result="skipped", skipped="if" if job.get("if") is not None else "needs")
            results[jid] = rec
            continue
        if _matrix_empty(job.get("strategy"), ctx0):
            rec.update(result="skipped", skipped="empty-matrix")
            results[jid] = rec
            continue
        steps_ctx: dict = {}
        failed = False
        job_coe = job.get("continue-on-error") is True
        for i, st in enumerate(job.get("steps") or []):
            if not isinstance(st, dict):
                continue
            sid = str(st.get("id") or f"__{i}")
            ctx = Context(steps_ctx, needs_ctx, env, job_failed=failed)
            srec = {"index": i, "id": sid, "name": st.get("name"), "ran": False, "why": None,
                    "exit": None, "reached": [], "n_reached": 0, "timeout": False}
            c = condition(st.get("if"), ctx)
            if c is False:
                srec["why"] = "if"
                steps_ctx[sid] = {"outputs": {}, "outcome": "skipped", "conclusion": "skipped"}
                rec["steps"].append(srec)
                continue
            if "run" not in st or not isinstance(st.get("run"), str):
                srec["why"] = "uses"
                steps_ctx[sid] = {"outputs": UNKNOWN, "outcome": "success", "conclusion": "success"}
                rec["steps"].append(srec)
                continue
            shell = _shell_for(st, job, doc)
            if shell not in ("bash", "sh"):
                srec["why"] = "not-bash"
                steps_ctx[sid] = {"outputs": UNKNOWN, "outcome": "success", "conclusion": "success"}
                rec["steps"].append(srec)
                continue
            step_env = dict(env)
            step_env.update(_env_map(st.get("env"), ctx))
            text = substitute(st["run"], ctx)
            world = "fault" if fault == (jid, i) else flavour
            r = runner.run(text, world, step_env)
            ok = r["exit"] == 0 or r["timeout"]      # a step the simulation could not finish is not a failure of the script
            if not ok and world not in ("fail", "fault") and c is UNKNOWN and r["n_reached"] == 0:
                # a step whose condition the simulation cannot know, that reaches no tool and
                # fails: a guard written to fire on an abnormal condition (`if: ... ; run: exit 1`).
                # The healthy world is the normal condition, so the guard does not fire.
                srec["why"] = "if-unknown-guard"
                steps_ctx[sid] = {"outputs": {}, "outcome": "skipped", "conclusion": "skipped"}
                rec["steps"].append(srec)
                continue
            coe = st.get("continue-on-error") is True or job_coe
            srec.update(ran=True, exit=r["exit"], reached=r["reached"], n_reached=r["n_reached"],
                        timeout=r["timeout"], world=world)
            if not ok and world not in ("fail", "fault") and (artifacts is None or (jid, i) in artifacts):
                srec["artifact"] = True          # failed on its own logic under the model; the healthy world carries on
                ok = True
            env.update(r["env"])
            steps_ctx[sid] = {"outputs": dict(r["outputs"]), "outcome": "success" if ok else "failure",
                              "conclusion": "success" if (ok or coe) else "failure"}
            if not ok and not coe:
                failed = True
                srec["failed_job"] = True
            rec["steps"].append(srec)
        if failed:
            rec["result"] = "failure"
        outs = job.get("outputs")
        if isinstance(outs, dict):
            ctx = Context(steps_ctx, needs_ctx, env, job_failed=failed)
            for k, v in outs.items():
                val = Evaluator(ctx).eval(_EXPR.fullmatch(str(v).strip()).group(1).strip()) if _EXPR.fullmatch(str(v).strip()) else substitute(str(v), ctx)
                rec["outputs"][str(k)] = UNKNOWN if val is UNKNOWN else ("x" if val is ANSWER else "" if val is None else val if isinstance(val, str) else json.dumps(val))
        if any(v is UNKNOWN for v in rec["outputs"].values()):
            known = {k: v for k, v in rec["outputs"].items() if v is not UNKNOWN}
            rec["outputs"] = _PartialOutputs(known)
        rec["coe"] = job_coe
        results[jid] = rec
    return results


class _PartialOutputs(dict):
    """Job outputs where some are unknown: a missing key is UNKNOWN, not ''."""
    def get(self, k, default=None):
        return dict.get(self, k, UNKNOWN)


def _downstream(jobs: dict, jid: str) -> set[str]:
    out, frontier = set(), [jid]
    while frontier:
        j = frontier.pop()
        for k, v in jobs.items():
            if isinstance(v, dict) and j in _needs(v) and k not in out:
                out.add(k)
                frontier.append(k)
    return out


def _verification_steps(doc: dict) -> dict[tuple[str, int], str]:
    out = {}
    for jid, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        for i, st in enumerate(job.get("steps") or []):
            if isinstance(st, dict) and isinstance(st.get("run"), str):
                why = verification(st.get("name"), st["run"])
                if why:
                    out[(jid, i)] = why
    return out


def check_commands(run: str) -> set[str]:
    """The command words of the runner segments of a check (`python` for `python -m pytest`,
    `./ci/test.sh`); empty for a step that is a check by name only."""
    out = set()
    for seg in commands(run):
        if TOOL_RX.search(seg) and not FIXER_RX.search(seg):
            head = seg.split()[0].strip("\"'")
            if head:
                out.add(head)
    return out


def _reached(sim: dict, jid: str, i: int, names: set[str] | None = None) -> int | None:
    """How many times step (jid, i) reached its check tool -- one of `names` when the check has
    named runners, any external command otherwise; 0 when it did not run; None when the step is
    not executed at all (an action, or not bash). A check that reaches its runner fewer times
    than in the healthy world has done less: a loop over a failed query runs its body less often."""
    job = sim.get(jid)
    if not job or job["result"] == "skipped":
        return 0
    for s in job["steps"]:
        if s["index"] == i:
            if s["why"] in ("uses", "not-bash"):
                return None
            if not s["ran"]:
                return 0
            if names:
                return sum(1 for r in s["reached"] if r in names)
            return s["n_reached"]
    return 0


def _red(sim: dict) -> str | None:
    for jid, job in sim.items():
        if job["result"] == "failure" and not job.get("coe"):
            return jid
    return None


def _fail_index(job: dict) -> int | None:
    for s in job["steps"]:
        if s.get("failed_job"):
            return s["index"]
    return None


def _red_caused(plus: dict, minus: dict) -> str | None:
    """The first job that is red in W-A because of the fault: red in W-A and not red in W+, or
    red in W-A at an earlier step than in W+."""
    for jid, job in minus.items():
        if job["result"] != "failure" or job.get("coe"):
            continue
        pj = plus.get(jid)
        if not pj or pj["result"] != "failure":
            return jid
        fi, pi = _fail_index(job), _fail_index(pj)
        if fi is not None and (pi is None or fi < pi):
            return jid
    return None


def _artifacts(sim: dict) -> set:
    return {(jid, s["index"]) for jid, job in sim.items() for s in job["steps"] if s.get("artifact")}


def _fault_site_is_artifact(sim: dict, jid: str, i: int) -> bool:
    """Step i fails on its own logic in this healthy world: a fault there cannot be read."""
    job = sim.get(jid)
    if not job:
        return False
    return any(s["index"] == i and s.get("artifact") for s in job["steps"])


def _fault_verdict(doc: dict, jobs: dict, verify: dict, plus: dict, runner: Runner, jid: str, i: int, flavour: str) -> dict | None:
    """One fault in one healthy flavour; None when the flavour cannot interpret it."""
    st = jobs[jid]["steps"][i]
    if _fault_site_is_artifact(plus, jid, i):
        return {"uninterpretable": "BASELINE_RED"}
    plus_step = next((s for s in plus[jid]["steps"] if s["index"] == i), None) if plus.get(jid) else None
    if plus_step is None or not plus_step["ran"]:
        return {"uninterpretable": "BASELINE_SKIPPED", "why": plus_step["why"] if plus_step else "job-skipped"}
    scope = {jid} | _downstream(jobs, jid)
    checks = [(k, c) for k, c in verify.items() if k[0] in scope]
    minus = simulate(doc, runner, (jid, i), flavour, artifacts=_artifacts(plus))
    red = _red_caused(plus, minus)
    names = {k: check_commands(jobs[k[0]]["steps"][k[1]]["run"]) for k, _ in checks}
    count_plus = {k: (_reached(plus, *k, names[k]) or 0) for k, _ in checks}
    live = [k for k, _ in checks if count_plus[k] > 0]
    # Two readings of "the check did less". The preregistered one: the check reached its runner
    # in W+ and not at all in W-A. The counted one, added after the freeze and reported beside
    # it, never in its place: the check reached its runner fewer times than in W+ -- a loop over
    # a failed query runs its body less often, and a step whose query and runner are the same
    # script (`./tests/db/compose.sh config` then `./tests/db/compose.sh run ... pytest`) reaches
    # that script for the query and never for the run.
    dropped, dropped_counted = [], []
    for k in live:
        cm = _reached(minus, *k, names[k]) or 0
        if cm < count_plus[k]:
            mjob = minus.get(k[0])
            if not mjob or mjob["result"] == "skipped":
                mech = "job-" + str((mjob or {}).get("skipped") or "skipped")
            else:
                ms = next((s for s in mjob["steps"] if s["index"] == k[1]), None)
                mech = ("unreached" if cm == 0 else "fewer") if (ms and ms["ran"]) else ("step-if" if (ms and ms["why"] == "if") else "not-run")
            d = {"job": k[0], "index": k[1], "name": jobs[k[0]]["steps"][k[1]].get("name"),
                 "check_by": verify[k], "mechanism": mech, "cross_step": k != (jid, i), "runs_plus": count_plus[k], "runs_minus": cm}
            dropped_counted.append(d)
            if cm == 0:
                dropped.append(d)
    self_check = (jid, i) in verify and bool(_reached(minus, jid, i, check_commands(st["run"])))

    def _verdict(drops):
        if red:
            return "RED"
        if drops:
            return "FAIL_OPEN"
        if self_check:
            return "SWALLOWED"
        if live:
            return "ABSORBED"
        return "NO_CHECK"
    verdict = _verdict(dropped)
    ms = next((s for s in minus[jid]["steps"] if s["index"] == i), None) if minus.get(jid) else None
    return {"verdict": verdict, "red_job": red, "dropped": dropped, "checks_in_scope": len(checks), "checks_live": len(live),
            "verdict_counted": _verdict(dropped_counted), "dropped_counted": dropped_counted,
            "self_live": (jid, i) in live,
            "exit_minus": ms["exit"] if ms else None, "timeout": bool(ms and ms["timeout"]) or bool(plus_step["timeout"])}


_PRECEDENCE = ("RED", "FAIL_OPEN", "SWALLOWED", "ABSORBED", "NO_CHECK")
FLAVOURS = ("x", "empty")


def analyse_workflow(doc: dict, wf_name: str, runner: Runner, alone: dict) -> tuple[list[dict], dict]:
    """Every fault verdict for one workflow. `alone` maps (job, index) -> SWALLOW-1 verdict.

    Each fault is judged in both healthy flavours; the verdict is the strongest by precedence
    among the flavours that can interpret it (RED > FAIL_OPEN > SWALLOWED > ABSORBED > NO_CHECK),
    and the per-flavour verdicts are kept beside it. A fault no flavour can interpret is
    BASELINE_RED or BASELINE_SKIPPED."""
    jobs = {k: v for k, v in (doc.get("jobs") or {}).items() if isinstance(v, dict)}
    plus = {f: simulate(doc, runner, None, f) for f in FLAVOURS}
    verify = _verification_steps(doc)
    faults = []
    for jid, job in jobs.items():
        for i, st in enumerate(job.get("steps") or []):
            if not isinstance(st, dict) or not isinstance(st.get("run"), str):
                continue
            if _shell_for(st, job, doc) not in ("bash", "sh"):
                continue
            av = alone.get((jid, i))
            if av not in ("PROPAGATES", "SWALLOWS"):
                continue                       # TOOLLESS has no tool to fail; SYNTAX / TIMEOUT are not executable
            rec = {"workflow": wf_name, "generated": wf_name.endswith(".lock.yml"), "job": jid, "index": i,
                   "id": str(st.get("id") or f"__{i}"), "name": st.get("name"),
                   "category": categorize(st.get("name"), st["run"]), "check": verification(st.get("name"), st["run"]), "alone": av,
                   "continue_on_error": st.get("continue-on-error") is True or job.get("continue-on-error") is True,
                   "run_sha256": hashlib.sha256(st["run"].encode()).hexdigest()[:16],
                   "run_head": st["run"].strip().splitlines()[0][:160] if st["run"].strip() else ""}
            per = {f: _fault_verdict(doc, jobs, verify, plus[f], runner, jid, i, f) for f in FLAVOURS}
            interpretable = {f: v for f, v in per.items() if v and "verdict" in v}
            rec["by_flavour"] = {f: (v["verdict"] if "verdict" in v else v["uninterpretable"]) for f, v in per.items()}
            if not interpretable:
                kinds = {v["uninterpretable"] for v in per.values()}
                rec.update(verdict="BASELINE_SKIPPED" if kinds == {"BASELINE_SKIPPED"} else "BASELINE_RED", dropped=[], red_job=None)
                faults.append(rec)
                continue
            best_f = min(interpretable, key=lambda f: _PRECEDENCE.index(interpretable[f]["verdict"]))
            best = dict(interpretable[best_f])
            best_c = min(interpretable, key=lambda f: _PRECEDENCE.index(interpretable[f]["verdict_counted"]))
            best["verdict_counted"] = interpretable[best_c]["verdict_counted"]
            best["dropped_counted"] = interpretable[best_c]["dropped_counted"]
            rec.update(flavour=best_f, **best)
            faults.append(rec)
    def reached_any(k):
        names = check_commands(jobs[k[0]]["steps"][k[1]]["run"])
        return any(_reached(plus[f], *k, names) for f in FLAVOURS)
    summary = {"jobs": len(jobs), "artifact_failures_in_plus": {f: len(_artifacts(plus[f])) for f in FLAVOURS},
               "steps_run_in_plus": {f: sum(1 for j in plus[f].values() for s in j["steps"] if s["ran"]) for f in FLAVOURS},
               "verification_steps": len(verify),
               "verification_reached_in_plus": sum(1 for k in verify if reached_any(k)),
               "verification_not_executed": sum(1 for k in verify if _reached(plus["x"], *k) is None)}
    return faults, summary


def alone_verdict(run: str, runner: Runner) -> str:
    """SWALLOW-1's per-step verdict, taken by this executor (x for every ${{ }}, stubs for the
    repository's own scripts): PROPAGATES / SWALLOWS / TOOLLESS / SYNTAX / TIMEOUT."""
    r = runner.run(re.sub(r"\$\{\{.*?\}\}", "x", run), "fail", {})
    if r["timeout"]:
        return "TIMEOUT"
    if r["exit"] == 0:
        return "SWALLOWS" if r["n_reached"] else "TOOLLESS"
    if r["exit"] == 2 and "syntax error" in r["tail"]:
        return "SYNTAX"
    return "PROPAGATES"


def analyse_tree(tree: Path, repo: str | None = None, deadline: float | None = None) -> dict:
    import yaml
    runner = Runner()
    wdir = tree / ".github" / "workflows"
    out = {"repo": repo, "workflows": 0, "faults": [], "workflow_summaries": {}, "unparseable": [], "capped": False}
    if not wdir.exists():
        out["no_workflows_dir"] = True
        return out
    for wf in sorted(list(wdir.glob("*.yml")) + list(wdir.glob("*.yaml"))):
        if deadline and time.time() > deadline:
            out["capped"] = True
            break
        try:
            doc = yaml.safe_load(wf.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception as e:  # noqa: BLE001
            out["unparseable"].append({"workflow": wf.name, "error": str(e)[:120]})
            continue
        if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
            continue
        out["workflows"] += 1
        alone: dict = {}
        for jid, job in doc["jobs"].items():
            if not isinstance(job, dict):
                continue
            for i, st in enumerate(job.get("steps") or []):
                if isinstance(st, dict) and isinstance(st.get("run"), str) and _shell_for(st, job, doc) in ("bash", "sh"):
                    alone[(jid, i)] = alone_verdict(st["run"], runner)
        try:
            faults, summary = analyse_workflow(doc, wf.name, runner, alone)
        except RecursionError:
            out["unparseable"].append({"workflow": wf.name, "error": "recursion"})
            continue
        out["faults"].extend(faults)
        out["workflow_summaries"][wf.name] = summary
    out["executions"] = runner.executions
    return out


def _summary(results: list[dict]) -> dict:
    faults = [f for r in results for f in r.get("faults", [])]
    hand = [f for f in faults if not f["generated"]]
    by = {}
    for f in faults:
        by[f["verdict"]] = by.get(f["verdict"], 0) + 1
    by_hand = {}
    for f in hand:
        by_hand[f["verdict"]] = by_hand.get(f["verdict"], 0) + 1
    fo = [f for f in faults if f["verdict"] == "FAIL_OPEN"]
    foc = [f for f in faults if f.get("verdict_counted") == "FAIL_OPEN"]
    return {
        "by_verdict_counted": _count(f.get("verdict_counted") for f in faults if f.get("verdict_counted")),
        "by_verdict_counted_hand_written": _count(f.get("verdict_counted") for f in hand if f.get("verdict_counted")),
        "repos_with_fail_open_counted_hand_written": sum(1 for r in results if any(f.get("verdict_counted") == "FAIL_OPEN" and not f["generated"] for f in r.get("faults", []))),
        "fail_open_counted_by_mechanism": _count(d["mechanism"] for f in foc for d in f["dropped_counted"]),
        "repos": len(results), "repos_clone_failed": sum(1 for r in results if r.get("clone_error")),
        "repos_with_a_fault_site": sum(1 for r in results if r.get("faults")),
        "faults": len(faults), "by_verdict": by, "hand_written_faults": len(hand), "by_verdict_hand_written": by_hand,
        "repos_with_fail_open": sum(1 for r in results if any(f["verdict"] == "FAIL_OPEN" for f in r.get("faults", []))),
        "repos_with_fail_open_hand_written": sum(1 for r in results if any(f["verdict"] == "FAIL_OPEN" and not f["generated"] for f in r.get("faults", []))),
        "fail_open_cross_step": sum(1 for f in fo if any(d["cross_step"] for d in f["dropped"])),
        "fail_open_in_step": sum(1 for f in fo if all(not d["cross_step"] for d in f["dropped"])),
        "fail_open_by_mechanism": _count(d["mechanism"] for f in fo for d in f["dropped"]),
        "verification_steps": sum(s["verification_steps"] for r in results for s in r.get("workflow_summaries", {}).values()),
        "verification_reached_in_plus": sum(s["verification_reached_in_plus"] for r in results for s in r.get("workflow_summaries", {}).values()),
        "verification_not_executed": sum(s["verification_not_executed"] for r in results for s in r.get("workflow_summaries", {}).values()),
        "artifact_failures_in_plus": {f: sum(s["artifact_failures_in_plus"][f] for r in results for s in r.get("workflow_summaries", {}).values()) for f in FLAVOURS},
        "steps_run_in_plus": {f: sum(s["steps_run_in_plus"][f] for r in results for s in r.get("workflow_summaries", {}).values()) for f in FLAVOURS},
        "faults_by_flavour_used": _count(f.get("flavour") for r in results for f in r.get("faults", []) if f.get("flavour")),
        "workflows": sum(r.get("workflows", 0) for r in results),
        "repos_capped": sum(1 for r in results if r.get("capped")),
        "executions": sum(r.get("executions", 0) for r in results),
    }


def _count(it) -> dict:
    out: dict = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", help="JSON list of {repo: owner/name, ...}")
    ap.add_argument("--tree", help="one checkout on disk")
    ap.add_argument("--out", default="faults_receipt.json")
    ap.add_argument("--work", default=None)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.tree:
        rec = analyse_tree(Path(a.tree).resolve())
        Path(a.out).write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
        print(json.dumps(_summary([rec]), indent=1))
        return 0
    repos = json.loads(Path(a.repos).read_text(encoding="utf-8"))
    if a.limit:
        repos = repos[: a.limit]
    work = Path(a.work or tempfile.mkdtemp(prefix="faults_"))
    work.mkdir(parents=True, exist_ok=True)
    partial = Path(a.out).with_suffix(".partial.jsonl")
    results, done = [], set()
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                results.append(rec)
                done.add(rec["repo"])
    for k, r in enumerate(repos):
        name = r["repo"] if isinstance(r, dict) else r
        if name in done:
            continue
        dest, err = sparse_clone(name, work)
        if dest is None:
            rec = {"repo": name, "clone_error": err, "faults": [], "workflows": 0}
        else:
            t1 = time.time()
            rec = analyse_tree(dest, name, deadline=t1 + REPO_SECONDS_CAP)
            rec["head"] = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            rec["seconds"] = round(time.time() - t1, 1)
            if isinstance(r, dict):
                rec["population"] = {k2: v for k2, v in r.items() if k2 != "repo"}
            shutil.rmtree(dest, ignore_errors=True)
        results.append(rec)
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        by = _count(f["verdict"] for f in rec.get("faults", []))
        sys.stderr.write(f"{k+1}/{len(repos)} {name}: {rec.get('workflows', 0)} workflows, {len(rec.get('faults', []))} faults {by} {rec.get('seconds', '')}s\n")
    receipt = {"schema": "styxx.harness-faults/v1", "instrument": "benchmarks/harness_mutation/faults.py",
               "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "census_sha256": hashlib.sha256((Path(__file__).parent / "census.py").read_bytes()).hexdigest(),
               "population_file": str(a.repos), "population_sha256": hashlib.sha256(Path(a.repos).read_bytes()).hexdigest(),
               "population_size": len(repos),
               "method": "W+: every external command exits 0 and prints x (flavour x) or nothing (flavour empty); W-A: the same "
                         "with every external command of step A failing (127). Steps executed in job order with outputs, env, "
                         "if:, needs: and fromJSON matrices simulated; unknown context values are x and make a condition true; "
                         "a fault's verdict is the strongest across the flavours that can interpret it.",
               "repos": results, "seconds": round(time.time() - t0, 1)}
    order = {(r["repo"] if isinstance(r, dict) else r): i for i, r in enumerate(repos)}
    receipt["repos"] = sorted(results, key=lambda x: order.get(x["repo"], 1 << 30))
    receipt["summary"] = _summary(receipt["repos"])
    Path(a.out).write_text(json.dumps(receipt, indent=1, default=str) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps(receipt["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
