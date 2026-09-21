"""SWALLOW-1 — the census: how many CI steps in the wild cannot fail.

    python -m benchmarks.harness_mutation.census --repos repos.json --out receipt.json [--work DIR]
    python -m benchmarks.harness_mutation.census --tree /path/to/checkout      # one repository, already on disk

MUTE-1 found that 37 of this repository's 38 `run:` steps could be wrapped `( ... ) || true` and
no test would notice; #137 was a real instance -- a `git diff ... || true` that turned a failed
comparison into "nothing to verify" and went green for 138 runs. MUTE-2's behavioural guard
catches that shape by *executing* every step with an empty PATH: every external command fails,
so a step that propagates failure goes red and a step that swallows it goes green.

That method needs nothing but the workflow files, so it can be run on any repository without
cloning its code. This module does that for a list of repositories: a sparse, blob-less clone of
`.github/workflows` only, then every `run:` step executed the way the guard executes it, and
classified:

    PROPAGATES   the step exited non-zero -- the failure of what it called was not hidden
    SWALLOWS     the step exited 0 although at least one external command was reached and failed
    TOOLLESS     the step exited 0 and reached no external command (under `x` for every `${{ }}`)
    SYNTAX       bash could not parse the step after substitution (exit 2, "syntax error")
    TIMEOUT      the step did not finish (a loop with no external command to fail on)
    NOT_BASH     the step's shell is not bash/sh (pwsh, powershell, cmd, python, ...), or the job
                 runs on Windows with the default shell; not executed

Two YAML-level swallows are recorded beside the shell verdict, because they hide failure without
touching the script: `continue-on-error: true` on the step or the job, and `if: always()` /
`if: failure()`-style conditions are NOT swallows and are not counted. Each step is also tagged
by what it appears to be for (test, lint, typecheck, build, install, publish, other) from its name
and its first command; the tag is a heuristic and the receipt keeps the text so a reader can
disagree.

What this does not say. A swallowed step is not a defect: a "post a comment, best effort" step is
right to swallow. The census counts the shape; the RESULT reads the steps whose *purpose* is to
verify something, because those are the ones where a swallow is a check reporting green while
measuring nothing -- the SILENT-PASS shape, in CI, at scale.
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

ROOT = Path(__file__).resolve().parents[2]

PROLOGUE = 'command_not_found_handle() { printf "%s\\n" "$1" >> "$STUB_LOG"; return 127; }\n'
STEP_TIMEOUT = 30

CATEGORY_RULES = [
    ("test", re.compile(r"\b(pytest|unittest|jest|vitest|mocha|karma|cypress|playwright|go test|cargo test|dotnet test|mvn (test|verify)|gradle(w)? test|phpunit|rspec|minitest|tox|nox|ctest|bats|coverage|nyc)\b|\btests?\b", re.I)),
    ("lint", re.compile(r"\b(lint|ruff|flake8|pylint|eslint|prettier|black|isort|clippy|rustfmt|gofmt|golangci|shellcheck|yamllint|markdownlint|stylelint|rubocop|swiftlint|ktlint|checkstyle|spotless|fmt)\b", re.I)),
    ("typecheck", re.compile(r"\b(tsc|mypy|pyright|typecheck|type-check|flow check)\b", re.I)),
    ("build", re.compile(r"\b(build|compile|bundle|webpack|vite build|cargo build|go build|make\b|cmake|msbuild|xcodebuild|gradle(w)? (build|assemble)|docker build)\b", re.I)),
    ("install", re.compile(r"\b(install|setup|pip |npm ci|npm i\b|yarn|pnpm i|poetry|uv sync|apt-get|brew|bundle install|composer|nuget|restore)\b", re.I)),
    ("publish", re.compile(r"\b(publish|release|deploy|upload|push|twine|npm publish|cargo publish|gh release|docker push|pages)\b", re.I)),
]


def categorize(name: str | None, run: str) -> str:
    text = f"{name or ''}\n{run.strip().splitlines()[0] if run.strip() else ''}"
    for cat, rx in CATEGORY_RULES:
        if rx.search(text):
            return cat
    return "other"


def execute(run_text: str) -> tuple[str, int, list[str], str]:
    """(verdict, exit status, commands reached, last stderr line) for one step's shell."""
    # ignore_cleanup_errors: a step may leave a backgrounded builtin loop writing into the
    # directory after bash returns; the step is run in its own session and the whole group is
    # killed afterwards, but the race is real and the census must not stop for it
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        t = Path(td)
        (t / "bin").mkdir()
        (t / "work").mkdir()
        log = t / "reached.log"
        script = t / "step.sh"
        script.write_text(PROLOGUE + re.sub(r"\$\{\{.*?\}\}", "x", run_text), encoding="utf-8")
        env = {"PATH": str(t / "bin"), "STUB_LOG": str(log), "HOME": str(t), "LANG": "C.UTF-8",
               "GITHUB_OUTPUT": str(t / "output"), "GITHUB_ENV": str(t / "env"), "GITHUB_PATH": str(t / "path"),
               "GITHUB_STEP_SUMMARY": str(t / "summary"), "GITHUB_WORKSPACE": str(t / "work"), "RUNNER_TEMP": str(t)}
        proc = subprocess.Popen(["/bin/bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
                                cwd=str(t / "work"), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, start_new_session=True)
        try:
            out, err = proc.communicate(timeout=STEP_TIMEOUT)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            reached = log.read_text(encoding="utf-8", errors="replace").split() if log.exists() else []
            return "TIMEOUT", -1, reached, ""
        _kill_group(proc)
        r = subprocess.CompletedProcess(proc.args, rc, out, err)
        reached = log.read_text(encoding="utf-8", errors="replace").split() if log.exists() else []
        tail = (r.stderr.strip().splitlines() or [""])[-1][:200]
        if r.returncode == 0:
            return ("TOOLLESS" if not reached else "SWALLOWS"), 0, reached, tail
        if r.returncode == 2 and "syntax error" in r.stderr:
            return "SYNTAX", 2, reached, tail
        return "PROPAGATES", r.returncode, reached, tail


def _kill_group(proc: subprocess.Popen) -> None:
    """Whatever the step left running in its session dies with it."""
    import signal
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _shell_for(step: dict, job: dict, doc: dict) -> str:
    sh = step.get("shell") or (job.get("defaults") or {}).get("run", {}).get("shell") \
        or (doc.get("defaults") or {}).get("run", {}).get("shell")
    if sh:
        return str(sh).split()[0].lower()
    runs_on = json.dumps(job.get("runs-on", "")).lower()
    return "pwsh" if "windows" in runs_on else "bash"


def census_tree(tree: Path, repo: str | None = None) -> dict:
    """Every `run:` step under <tree>/.github/workflows, executed and classified."""
    import yaml
    wdir = tree / ".github" / "workflows"
    out = {"repo": repo, "workflows": 0, "steps": [], "unparseable": []}
    if not wdir.exists():
        out["no_workflows_dir"] = True
        return out
    for wf in sorted(list(wdir.glob("*.yml")) + list(wdir.glob("*.yaml"))):
        try:
            doc = yaml.safe_load(wf.read_text(encoding="utf-8", errors="replace")) or {}
        except Exception as e:  # noqa: BLE001 - the census records, it does not stop
            out["unparseable"].append({"workflow": wf.name, "error": str(e)[:120]})
            continue
        if not isinstance(doc, dict):
            out["unparseable"].append({"workflow": wf.name, "error": "not a mapping"})
            continue
        out["workflows"] += 1
        for jid, job in (doc.get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            job_coe = bool(job.get("continue-on-error") is True)
            for i, st in enumerate(job.get("steps") or []):
                if not isinstance(st, dict) or "run" not in st or not isinstance(st.get("run"), str):
                    continue
                shell = _shell_for(st, job, doc)
                rec = {"workflow": wf.name, "generated": wf.name.endswith(".lock.yml"),
                       "job": jid, "index": i, "name": st.get("name"),
                       "category": categorize(st.get("name"), st["run"]), "shell": shell,
                       "if": st.get("if"), "continue_on_error": bool(st.get("continue-on-error") is True) or job_coe,
                       "run_sha256": hashlib.sha256(st["run"].encode()).hexdigest()[:16],
                       "run_head": st["run"].strip().splitlines()[0][:160] if st["run"].strip() else "",
                       "has_or_true": bool(re.search(r"\|\|\s*(true|:|echo\b|exit 0)", st["run"])),
                       # #137's exact shape: a git query whose failure is turned into an empty answer
                       "git_query_or_true": bool(re.search(r"git\s+(diff|log|status|ls-files|rev-parse|describe|fetch)[^\n]*\|\|\s*(true|:|echo\b)", st["run"]))}
                if shell not in ("bash", "sh"):
                    rec.update(verdict="NOT_BASH", exit=None, reached=[], tail="")
                else:
                    v, rc, reached, tail = execute(st["run"])
                    rec.update(verdict=v, exit=rc, reached=reached[:8], tail=tail)
                out["steps"].append(rec)
    return out


def sparse_clone(repo: str, work: Path) -> tuple[Path | None, str | None]:
    """Blob-less sparse clone of .github/workflows only. Returns (path, error)."""
    dest = work / repo.replace("/", "__")
    if dest.exists():
        shutil.rmtree(dest)
    url = f"https://github.com/{repo}.git"
    try:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", url, str(dest)],
                       check=True, capture_output=True, text=True, timeout=180)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", ".github/workflows"],
                       check=True, capture_output=True, text=True, timeout=180)
    except subprocess.CalledProcessError as e:
        return None, (e.stderr or "").strip().splitlines()[-1][:200] if (e.stderr or "").strip() else f"exit {e.returncode}"
    except subprocess.TimeoutExpired:
        return None, "timeout"
    return dest, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", help="JSON list of {repo: owner/name, ...}")
    ap.add_argument("--tree", help="census one checkout on disk instead")
    ap.add_argument("--out", default="census_receipt.json")
    ap.add_argument("--work", default=None)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.tree:
        rec = census_tree(Path(a.tree).resolve())
        Path(a.out).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(_summary([rec]))
        return 0
    repos = json.loads(Path(a.repos).read_text(encoding="utf-8"))
    if a.limit:
        repos = repos[: a.limit]
    work = Path(a.work or tempfile.mkdtemp(prefix="census_"))
    work.mkdir(parents=True, exist_ok=True)
    partial = Path(a.out).with_suffix(".partial.jsonl")     # one line per repository, so a crash costs one repository
    results = []
    done = set()
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
            rec = {"repo": name, "clone_error": err, "steps": [], "workflows": 0}
            results.append(rec)
            with partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            sys.stderr.write(f"\r{k+1}/{len(repos)} {name}: clone failed ({err})\n")
            continue
        rec = census_tree(dest, name)
        head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        rec["head"] = head
        if isinstance(r, dict):
            rec["population"] = {k2: v for k2, v in r.items() if k2 != "repo"}
        results.append(rec)
        with partial.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        shutil.rmtree(dest, ignore_errors=True)
        sys.stderr.write(f"\r{k+1}/{len(repos)} {name}: {rec['workflows']} workflows, {len(rec['steps'])} steps   ")
    sys.stderr.write("\n")
    receipt = {"schema": "styxx.harness-census/v1", "instrument": "benchmarks/harness_mutation/census.py",
               "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "population_file": str(a.repos), "population_sha256": hashlib.sha256(Path(a.repos).read_bytes()).hexdigest(),
               "population_size": len(repos),
               "method": "every run: step executed with bash -eo pipefail, empty PATH, ${{ }} -> x, "
                         "command_not_found_handle logging every external command; 30 s timeout",
               "repos": results, "seconds": round(time.time() - t0, 1), "summary": _summary(results)}
    order = {(r["repo"] if isinstance(r, dict) else r): i for i, r in enumerate(repos)}
    receipt["repos"] = sorted(results, key=lambda x: order.get(x["repo"], 1 << 30))
    Path(a.out).write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps(receipt["summary"], indent=1))
    return 0


def _summary(results: list[dict]) -> dict:
    steps = [s for r in results for s in r.get("steps", [])]
    by_v: dict = {}
    for s in steps:
        by_v[s["verdict"]] = by_v.get(s["verdict"], 0) + 1
    executed = [s for s in steps if s["verdict"] in ("PROPAGATES", "SWALLOWS")]
    ver = [s for s in executed if s["category"] in ("test", "lint", "typecheck")]
    def cannot_fail(s):
        return s["verdict"] == "SWALLOWS" or (s["continue_on_error"] and s["verdict"] in ("PROPAGATES", "SWALLOWS"))
    per_repo = []
    for r in results:
        ex = [s for s in r.get("steps", []) if s["verdict"] in ("PROPAGATES", "SWALLOWS")]
        if ex:
            per_repo.append(sum(1 for s in ex if s["verdict"] == "SWALLOWS") / len(ex))
    per_repo.sort()
    return {
        "repos_with_an_executed_step": len(per_repo),
        "median_per_repo_shell_swallow_rate": per_repo[len(per_repo) // 2] if per_repo else None,
        "repos_with_a_step_that_cannot_fail": sum(1 for r in results if any(cannot_fail(s) for s in r.get("steps", []))),
        "repos_with_a_verification_step_that_cannot_fail": sum(1 for r in results if any(cannot_fail(s) and s["category"] in ("test", "lint", "typecheck") for s in r.get("steps", []))),
        "repos_with_generated_agentic_workflows": sum(1 for r in results if any(s.get("generated") for s in r.get("steps", []))),
        "generated_steps": sum(1 for s in steps if s.get("generated")),
        "generated_steps_continue_on_error": sum(1 for s in steps if s.get("generated") and s["continue_on_error"]),
        "steps_syntax_or_timeout": sum(1 for s in steps if s["verdict"] in ("SYNTAX", "TIMEOUT")),
        "steps_git_query_or_true": sum(1 for s in steps if s.get("git_query_or_true")),
        "repos": len(results),
        "repos_with_workflows": sum(1 for r in results if r.get("workflows", 0) > 0),
        "repos_clone_failed": sum(1 for r in results if r.get("clone_error")),
        "steps": len(steps), "by_verdict": by_v,
        "swallow_rate_among_executed": round(sum(1 for s in executed if s["verdict"] == "SWALLOWS") / len(executed), 4) if executed else None,
        "verification_steps_executed": len(ver),
        "verification_swallows": sum(1 for s in ver if s["verdict"] == "SWALLOWS"),
        "verification_continue_on_error": sum(1 for s in steps if s["category"] in ("test", "lint", "typecheck") and s["continue_on_error"]),
        "repos_with_a_swallow": sum(1 for r in results if any(s["verdict"] == "SWALLOWS" for s in r.get("steps", []))),
        "repos_with_a_verification_swallow": sum(1 for r in results if any(s["verdict"] == "SWALLOWS" and s["category"] in ("test", "lint", "typecheck") for s in r.get("steps", []))),
    }


if __name__ == "__main__":
    sys.exit(main())
