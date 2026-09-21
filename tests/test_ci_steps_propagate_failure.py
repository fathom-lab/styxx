"""Every `run:` step under .github/workflows must be able to go red.

MUTE-1 wrapped each of the harness's 38 `run:` steps in `( ... ) || true` -- the step still
exists, still has its name, still appears in the log, and can no longer fail -- and 37 of the 38
wrappers went unnoticed by this suite. The one that was noticed was caught by a test that RAN the
step's shell; the textual guard beside it, which looked for `|| true` on a git line, saw nothing,
because the wrapper puts `|| true` on a line of its own. A guard that reads the text of a check is
fooled by anything that leaves the text in place. So this one does not read: it runs.

The method. Each step's `run:` block is executed the way Actions executes it (`bash -eo
pipefail`) in an empty directory, with an EMPTY PATH: every external command -- python, pip,
git, npm, cat, basename, styxx, anything -- is caught by bash's `command_not_found_handle`,
logged, and fails with 127. Actions expressions `${{ ... }}` are replaced by `x`. Under those
conditions a step that propagates the failure of the tools it calls exits non-zero; a step that
swallows it exits zero. The test asserts non-zero for every step that reached an external
command at all.

What is exempt, and why it is stated rather than hidden. A step that exits zero WITHOUT having
reached any external command has nothing to propagate in this context (it printed, or it took
a branch that `x` made empty). Such a step is reported as skipped with its name, not passed:
the guard does not pretend to have checked it. Wrapping one of those in `|| true` is invisible
to this test, and papers/harness/PREREG_mute2 lists which ones those are, by name, in advance.

What this does NOT say: that any step does the right thing when its tools succeed. It says the
step cannot hide their failure.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

PROLOGUE = 'command_not_found_handle() { printf "%s\\n" "$1" >> "$STUB_LOG"; return 127; }\n'


def run_steps() -> list[tuple[str, str]]:
    """(id, run text) for every `run:` step, id = workflow::job::step-name (index only on a clash)."""
    out = []
    for wf in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
        for jid, job in (doc.get("jobs") or {}).items():
            seen: dict[str, int] = {}
            for i, st in enumerate(job.get("steps") or []):
                if "run" not in st:
                    continue
                name = st.get("name") or f"step {i}"
                seen[name] = seen.get(name, 0) + 1
                label = name if seen[name] == 1 else f"{name} #{seen[name]}"
                out.append((f"{wf.name}::{jid}::{label}", st["run"]))
    return out


def execute(run_text: str) -> tuple[int, list[str], str]:
    """Run one step's shell with an empty PATH. Returns (exit status, commands reached, stderr tail)."""
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        (t / "bin").mkdir()                      # empty: nothing on PATH resolves
        (t / "work").mkdir()
        log = t / "reached.log"
        script = t / "step.sh"
        script.write_text(PROLOGUE + re.sub(r"\$\{\{.*?\}\}", "x", run_text), encoding="utf-8")
        env = {
            "PATH": str(t / "bin"), "STUB_LOG": str(log), "HOME": str(t),
            "GITHUB_OUTPUT": str(t / "output"), "GITHUB_ENV": str(t / "env"),
            "GITHUB_STEP_SUMMARY": str(t / "summary"), "GITHUB_WORKSPACE": str(t / "work"),
            "RUNNER_TEMP": str(t), "LANG": "C.UTF-8",
        }
        r = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-eo", "pipefail", str(script)],
                           cwd=str(t / "work"), env=env, capture_output=True, text=True, timeout=120)
        reached = log.read_text(encoding="utf-8").split() if log.exists() else []
        tail = (r.stderr.strip().splitlines() or [""])[-1][:200]
        return r.returncode, reached, tail


STEPS = run_steps()


@pytest.mark.skipif(not os.path.exists("/bin/bash"), reason="needs bash, as Actions does")
@pytest.mark.parametrize("step_id,run_text", STEPS, ids=[s[0] for s in STEPS])
def test_the_step_cannot_hide_the_failure_of_what_it_calls(step_id, run_text):
    rc, reached, tail = execute(run_text)
    if rc == 0 and not reached:
        pytest.skip(f"{step_id}: reached no external command under an empty PATH; nothing to propagate here")
    assert rc != 0, (
        f"{step_id}: every external command fails in this environment ({', '.join(reached[:5])} "
        f"reached), yet the step exited 0 -- it swallows the failure of what it calls. "
        f"Last stderr line: {tail!r}")


def test_there_are_steps_to_hold():
    assert len(STEPS) >= 30, f"only {len(STEPS)} run steps found under .github/workflows"


def test_the_propagation_guard_rejects_a_swallowed_step():
    """Control (MUTE-3). The guard above, called directly on a step that swallows the failure of
    the tool it calls, must refuse it. If its assert is ever hollowed to `assert True`, this is
    the test that notices; `test_the_method_sees_a_swallow...` below checks the method, not the
    guard, and would not."""
    with pytest.raises(AssertionError):
        test_the_step_cannot_hide_the_failure_of_what_it_calls("fixture.yml::job::swallowed", "( python -m pytest tests -q\n) || true\n")


def test_the_method_sees_a_swallow_and_passes_a_propagating_step():
    """The guard has to be able to see the thing it claims to see, in both directions."""
    rc, reached, _ = execute("python -m pytest tests -q\n")
    assert rc != 0 and reached == ["python"]
    rc, reached, _ = execute("( python -m pytest tests -q\n) || true\n")
    assert rc == 0 and reached == ["python"], "a swallowed step must exit 0 with the tool reached"
    rc, reached, _ = execute('echo "nothing external here"\n')
    assert rc == 0 and reached == [], "a toolless step is the exempt case, and must look like one"
