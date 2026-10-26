# -*- coding: utf-8 -*-
"""prereg-ci B7: the composite Action, ``prereg-ci/action.yml`` (outside ``.github/``, DESIGN 3.1, 9).

DESIGN 7, risks 9 and 10: full history is needed, so the checkout is ``fetch-depth: 0`` with the
partial-clone filter ``blob:none`` (git fetches only the blobs ``check`` reads, from the repository's
own remote, while it runs); the Action is pinned by commit sha in the caller's ``uses:`` line and needs
``contents: read`` only. Every input reaches the shell through the environment, as the sworn action's
do (``sworn/action.yml``, ``tests/test_sworn_action.py``), so no input text is ever spliced into a
script. The check step's own script is run here under bash on a ``git init`` repository: its exit code
is the command's, ``report-only`` makes it 0 with the verdict still written, and the job summary is
appended to. No workflow is added and nothing reaches a network.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "prereg-ci" / "action.yml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.test_prereg_ci_cli import demo, result_md  # noqa: E402
from tests.test_prereg_ci_package import HOUSE_RX, PRIORITY_RX  # noqa: E402


@pytest.fixture(scope="module")
def action():
    return yaml.safe_load(ACTION.read_text(encoding="utf-8"))


def _steps(action):
    return action["runs"]["steps"]


def _check_step(action):
    (step,) = [s for s in _steps(action) if s.get("id") == "check"]
    return step


def test_the_action_lives_outside_dot_github_and_is_composite(action):
    assert ACTION.is_file() and ".github" not in ACTION.parts
    assert action["runs"]["using"] == "composite"
    raw = ACTION.read_bytes()
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")


def test_the_checkout_is_full_history_with_a_blob_none_partial_clone(action):
    (co,) = [s for s in _steps(action) if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert co["with"]["fetch-depth"] == 0 and co["with"]["filter"] == "blob:none"
    assert co["with"]["ref"] == "${{ inputs.head }}"           # the tree the check reads arrives in one batch
    assert "inputs.checkout" in co["if"]
    desc = action["description"]
    for needle in ("fetch-depth: 0", "blob:none", "contents: read", "commit sha", "report-only"):
        assert needle in desc, needle


def test_every_input_reaches_the_shell_through_the_environment(action):
    for step in _steps(action):
        script = step.get("run")
        if script is None:
            continue
        assert "${{ inputs." not in script and "${{ github.event" not in script, script
    env = _check_step(action)["env"]
    for name, value in env.items():
        assert value.startswith("${{ inputs.") or value.startswith("${{ github."), (name, value)
    assert set(action["inputs"]) >= {"base", "head", "checkout", "report-only", "immutability", "amend-after-data",
                                     "receipt-rewrite", "require-time", "preregistered-word", "exclude", "out-dir",
                                     "python-version", "styxx-source"}
    assert set(action["outputs"]) == {"verdict", "exit-code", "report"}


def test_the_check_step_runs_the_merge_gate(action):
    script = _check_step(action)["run"]
    assert "python -m styxx.prereg_ci" in script and " check " in script.replace('"${args[@]}"', " check ")
    assert "--summary" in script and "--json" in script and "GITHUB_OUTPUT" in script


def test_the_word_law_over_the_action():
    text = ACTION.read_text(encoding="utf-8")
    assert not HOUSE_RX.search(text) and not PRIORITY_RX.search(text)


def _bash():
    found = shutil.which("bash")
    if found and "system32" in found.lower():                 # WSL's launcher, not a POSIX shell on this box
        git = shutil.which("git")
        found = None
        if git:
            for cand in (Path(git).resolve().parents[1] / "bin" / "bash.exe",
                         Path(git).resolve().parents[2] / "bin" / "bash.exe"):
                if cand.is_file():
                    found = str(cand)
                    break
    return found


def run_step(action, repo: Path, tmp: Path, **inputs):
    """Run the check step's script with the environment the composite sets for these inputs."""
    bash = _bash()
    if bash is None:
        pytest.skip("no bash on this box to run the step's script")
    step = _check_step(action)
    defaults = {k: (v.get("default", "") if isinstance(v, dict) else "") for k, v in action["inputs"].items()}
    values = {**defaults, **inputs}
    env = dict(os.environ)
    for name, expr in step["env"].items():
        m = re.fullmatch(r"\$\{\{ inputs\.([a-z-]+) \}\}", expr)
        if m:
            env[name] = str(values[m.group(1)])
    env.update(GITHUB_OUTPUT=str(tmp / "out.txt"), GITHUB_STEP_SUMMARY=str(tmp / "summary.md"),
               PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1",
               PREREG_CI_TEST_PYTHON=Path(sys.executable).as_posix())
    script = 'python() { "$PREREG_CI_TEST_PYTHON" -B "$@"; }\n' + step["run"]
    (tmp / "step.sh").write_text(script, encoding="utf-8", newline="\n")
    r = subprocess.run([bash, str(tmp / "step.sh")], cwd=str(repo), env=env, capture_output=True)
    outputs = {}
    if (tmp / "out.txt").exists():
        for line in (tmp / "out.txt").read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            outputs[k] = v
    return r.returncode, r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace"), outputs


def test_the_step_passes_the_commands_exit_code_and_writes_its_outputs(action, tmp_path):
    g, rec = demo(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    code, text, outputs = run_step(action, g.root, out, base=g("rev-parse", "main"), head=g("rev-parse", "result"),
                                   **{"out-dir": str(out / "report")})
    assert code == 0, text
    assert outputs["verdict"] == "PASS" and outputs["exit-code"] == "0"
    assert json.loads(Path(outputs["report"]).read_text(encoding="ascii"))["verdict"] == "PASS"
    assert (out / "summary.md").read_text(encoding="utf-8").startswith("## prereg-ci check: PASS")


def test_a_refusal_fails_the_step_and_report_only_writes_it_and_exits_zero(action, tmp_path):
    g, rec = demo(tmp_path)
    g("switch", "-q", "-c", "typed", "result")
    g.write("RESULT_digits.md", result_md("HOLD__calibration"))
    g.commit("type the verdict")
    for report_only, want in (("false", 1), ("true", 0)):
        out = tmp_path / f"o_{report_only}"
        out.mkdir()
        code, text, outputs = run_step(action, g.root, out, base=g("rev-parse", "result"),
                                       head=g("rev-parse", "typed"), **{"out-dir": str(out / "report"),
                                                                        "report-only": report_only})
        assert code == want, text
        assert outputs["verdict"] == "FAIL" and outputs["exit-code"] == "1"
        assert "FAIL VERDICT_DIFFERS" in text


def test_policy_inputs_become_flags(action, tmp_path):
    g, rec = demo(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    code, text, outputs = run_step(action, g.root, out, base=g("rev-parse", "main"), head=g("rev-parse", "result"),
                                   **{"out-dir": str(out / "r"), "require-time": "T3a", "exclude": "conformance/\ndocs/"})
    assert code == 1 and "FAIL TIME_BELOW_POLICY" in text and "require_time T3a (flag)" in text, text
    rep = json.loads(Path(outputs["report"]).read_text(encoding="ascii"))
    assert rep["policies"]["exclude"] == ["conformance/", "docs/"]
