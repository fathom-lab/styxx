"""`gauntlet-pr.yml` said no submission lands without CI verification. It verified nothing.

The workflow's job is to re-run `styxx gauntlet` on an outside researcher's method and refuse the
PR if the numbers they reported are not the numbers the gauntlet produces. Its first step found
which submissions the PR touched:

    changed_dirs=$(git diff --name-only origin/${{ github.base_ref }}...HEAD -- 'submissions/' \\
      | awk -F/ 'NF >= 3 {print $1"/"$2}' | sort -u || true)

`origin/<base>` is not in the checkout. `actions/checkout@v4` at its default `fetch-depth: 1`
fetches exactly one refspec on a `pull_request` event -- `+<sha>:refs/remotes/pull/<n>/merge`
(`src/ref-helper.ts`, `getRefSpec`; the `+refs/heads/*:refs/remotes/origin/*` spec lives in
`getRefSpecForAllHistory`, which only runs at `fetch-depth: 0`). So the diff exited 128 with
"unknown revision", the `|| true` turned that into an empty string, every later step was skipped
by `if: steps.discover.outputs.dirs != ''`, and the job printed *"no submissions/ files changed in
this PR -- nothing to verify"* and went green.

Two different facts -- *this PR changes no submission* and *I could not work out what this PR
changes* -- arrived at CI as the same green check. That is SP-1 (`benchmarks/silent_pass/`): an
absent measurement surfacing as a passing check. `benchmarks/silent_pass/CORPUS.md` collects these
from other people's repositories. This one is ours, on the public submission path.

What is tested here is behaviour, not wording. The first test runs the shipped step's own shell
with a base commit that cannot be fetched and demands a non-zero exit; the second runs the
historical line, in the same harness, and shows it exits 0 with an empty result -- so the first
test is known to discriminate rather than merely to pass.

`.github/workflows/leaderboard-submission.yml` handles the other submission protocol
(`submissions/GAUNTLET.md` documents both) and never had this defect: it fetches its base ref and
exits 1 when it finds nothing. It is pinned here too, so the good half cannot quietly drift into
the shape the bad half had.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
GAUNTLET_PR = WORKFLOWS / "gauntlet-pr.yml"
LEADERBOARD = WORKFLOWS / "leaderboard-submission.yml"

# A commit that exists in no repository on earth, standing in for "the base is not fetchable".
UNFETCHABLE = "0123456789abcdef0123456789abcdef01234567"

# What GitHub substitutes before bash ever sees the script. Anything not named here becomes an
# obvious placeholder rather than silently expanding to nothing.
CONTEXT = {
    "github.event.pull_request.base.sha": UNFETCHABLE,
    "github.base_ref": "main",
    "steps.discover.outputs.dirs": "",
}


def _expand(script: str) -> str:
    def sub(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        return CONTEXT.get(key, "__UNSET_%s__" % re.sub(r"\W+", "_", key))
    return re.sub(r"\$\{\{([^}]*)\}\}", sub, script)


def _steps(path: Path) -> list[dict]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [s for job in doc["jobs"].values() for s in job["steps"]]


def _step(path: Path, name: str) -> dict:
    for s in _steps(path):
        if s.get("name") == name:
            return s
    raise AssertionError("%s has no step named %r; it has %r"
                         % (path.name, name, [s.get("name") for s in _steps(path)]))


def _run_shell(script: str) -> subprocess.CompletedProcess:
    """Run a workflow step's shell the way the runner would: bash, in a git repo, with an
    `origin` that does not have the base commit."""
    with tempfile.TemporaryDirectory() as td:
        work = Path(td) / "work"
        bare = Path(td) / "origin.git"
        subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
        work.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
               "GITHUB_OUTPUT": str(Path(td) / "gh_output")}
        for cmd in (["git", "init", "-q", "-b", "main"],
                    ["git", "remote", "add", "origin", str(bare)]):
            subprocess.run(cmd, cwd=work, check=True, env=env)
        (work / "submissions").mkdir()
        (work / "submissions" / "x.txt").write_text("x", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=work, check=True, env=env)
        subprocess.run(["git", "commit", "-qm", "c"], cwd=work, check=True, env=env)
        sh = Path(td) / "step.sh"
        sh.write_text(script, encoding="utf-8")
        proc = subprocess.run(["bash", str(sh)], cwd=work, env=env,
                              capture_output=True, text=True, timeout=120)
        out = Path(env["GITHUB_OUTPUT"])
        proc.gh_output = out.read_text(encoding="utf-8") if out.exists() else ""  # type: ignore[attr-defined]
        return proc


# ---- the defect, run rather than described ----------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="the step is bash; the runner is ubuntu")
def test_a_comparison_that_cannot_be_made_is_not_reported_as_nothing_to_verify():
    """The shipped step, on a base commit nothing can fetch, must fail."""
    proc = _run_shell(_expand(_step(GAUNTLET_PR, "Discover changed submissions")["run"]))
    assert proc.returncode != 0, (
        "the discovery step succeeded when it could not reach the base commit. Everything after "
        "it is guarded by `if: steps.discover.outputs.dirs != ''`, so a success here means the "
        "gauntlet is never re-run and the PR is reported green unverified.\n"
        "stdout:\n%s\nstderr:\n%s" % (proc.stdout[-2000:], proc.stderr[-2000:]))
    assert "dirs<<EOF" not in proc.gh_output, (  # type: ignore[attr-defined]
        "it failed but still wrote an output the later steps read: %r" % proc.gh_output)  # type: ignore[attr-defined]


@pytest.mark.skipif(sys.platform == "win32", reason="the step is bash; the runner is ubuntu")
def test_the_line_this_replaced_passes_the_test_above_so_the_test_discriminates():
    """The historical shape, in the same harness. It must exit 0 with an empty answer.

    Without this, the test above could be passing because the harness is broken rather than
    because the step is fixed. This is the control, and it is the defect verbatim.
    """
    historical = _expand(
        "set -e\n"
        "changed_dirs=$(git diff --name-only origin/${{ github.base_ref }}...HEAD "
        "-- 'submissions/' | awk -F/ 'NF >= 3 {print $1\"/\"$2}' | sort -u || true)\n"
        'echo "changed submissions:"\n'
        'echo "$changed_dirs"\n'
        'echo "dirs<<EOF" >> "$GITHUB_OUTPUT"\n'
        'echo "$changed_dirs" >> "$GITHUB_OUTPUT"\n'
        'echo "EOF" >> "$GITHUB_OUTPUT"\n'
    )
    proc = _run_shell(historical)
    assert proc.returncode == 0, "the control no longer reproduces the defect: %s" % proc.stderr
    assert "fatal" in proc.stderr.lower(), (
        "git was expected to refuse the unknown revision; it said: %r" % proc.stderr)
    body = proc.gh_output.split("dirs<<EOF", 1)[1]  # type: ignore[attr-defined]
    assert body.strip().strip("EOF").strip() == "", (
        "the control was expected to hand later steps an empty list: %r" % proc.gh_output)  # type: ignore[attr-defined]


# ---- the shape of the repair ------------------------------------------------------------------

def _code_lines(run: str) -> list[str]:
    """Executable lines only. The step deliberately quotes the old, broken line in a comment, so
    a check that reads the whole blob finds `git diff` before `git fetch` and is measuring the
    comment."""
    return [ln for ln in run.splitlines() if not ln.strip().startswith("#")]


def test_the_discover_step_fetches_the_base_before_diffing_it():
    code = _code_lines(_step(GAUNTLET_PR, "Discover changed submissions")["run"])
    fetch = next((i for i, ln in enumerate(code) if "git fetch" in ln), None)
    diff = next((i for i, ln in enumerate(code) if "git diff" in ln), None)
    assert fetch is not None, "nothing fetches the base commit, so the diff has nothing to compare to"
    assert diff is not None, "the step no longer diffs anything"
    assert fetch < diff, "the fetch must come before the diff"


def test_no_git_command_in_the_discover_step_swallows_its_status():
    for line in _code_lines(_step(GAUNTLET_PR, "Discover changed submissions")["run"]):
        assert not re.search(r"git .*\|\|\s*(true|:)", line.strip()), (
            "`|| true` on a git command is how this workflow reported a failed comparison as an "
            "empty one: %r" % line)


def test_every_workflow_run_block_is_valid_shell():
    """A syntax error in a workflow that fires twice a year surfaces twice a year."""
    broken = []
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
        for jname, job in (doc.get("jobs") or {}).items():
            for st in job.get("steps") or []:
                if "run" not in st:
                    continue
                with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False,
                                                 encoding="utf-8") as fh:
                    fh.write(st["run"])
                    tmp = fh.name
                try:
                    r = subprocess.run(["bash", "-n", tmp], capture_output=True, text=True)
                    if r.returncode:
                        broken.append("%s / %s / %s: %s"
                                      % (wf.name, jname, st.get("name"), r.stderr.strip()[:200]))
                finally:
                    os.unlink(tmp)
    assert not broken, "\n".join(broken)


# ---- the filter, and the protocol it is allowed to drop ---------------------------------------

def _filter(paths: list[str]) -> list[str]:
    """The step's own awk, run rather than paraphrased."""
    run = _step(GAUNTLET_PR, "Discover changed submissions")["run"]
    m = re.search(r"(awk -F/ [^|\n]*)", run)
    assert m, "the directory filter is no longer a readable awk one-liner"
    proc = subprocess.run(["bash", "-c", "%s | sort -u" % m.group(1)],
                          input="\n".join(paths) + "\n", capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return [x for x in proc.stdout.splitlines() if x]


def test_the_filter_keeps_a_gauntlet_submission():
    assert "submissions/baseline_002_classifier" in _filter([
        "submissions/baseline_002_classifier/method.py",
        "submissions/baseline_002_classifier/submission.json",
    ])


def test_the_filter_drops_a_flat_file_because_the_other_workflow_owns_it():
    """`submissions/<name>.py` is a Cognometry Detector Interface v0 submission. Dropping it here
    is correct -- leaderboard-submission.yml runs it -- and is recorded so that a future reader
    does not 'fix' the filter and double-run every v0 entry."""
    assert _filter(["submissions/styxx.py", "submissions/README.md"]) == []
    trigger = yaml.safe_load(LEADERBOARD.read_text(encoding="utf-8"))
    paths = trigger[True]["pull_request"]["paths"] if True in trigger else trigger["on"]["pull_request"]["paths"]
    assert "submissions/*.py" in paths


def test_the_other_protocols_workflow_still_fails_loudly_when_it_finds_nothing():
    """The half that got it right, pinned so it cannot drift into the half that did not."""
    run = _step(LEADERBOARD, "Identify the submission file changed in this PR")["run"]
    assert "git fetch origin" in run, "it stopped fetching its base ref"
    assert "exit 1" in run, "it stopped failing when no submission file is found"
