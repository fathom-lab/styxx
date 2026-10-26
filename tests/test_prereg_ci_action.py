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

DESIGN 3.2(f), "a pull request cannot change the code that judges it": both python steps run with
``-I``, so the working directory (the checked-out pull request), ``PYTHONPATH`` and the user site stay
off ``sys.path`` (the B7 review's F1: under plain ``-m``, a ``styxx/prereg_ci/__main__.py`` in the pull
request's tree was the judge). The step is therefore run with an interpreter that has the package
under test installed, as a runner has after the install step: a venv made with ``--without-pip``
(no download) whose ``.pth`` names this checkout. No test runs pip.
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


def _committed_bytes(path: Path) -> bytes:
    """*path* as the index holds it (what a commit stores), else the working copy: a checkout under
    ``core.autocrlf=true`` writes CRLF into the working copy of an LF blob (the B7 review's F6)."""
    rel = path.relative_to(ROOT).as_posix()
    try:
        r = subprocess.run(["git", "-C", str(ROOT), "show", f":{rel}"], capture_output=True)
    except OSError:
        return path.read_bytes()
    return r.stdout if r.returncode == 0 else path.read_bytes()


def test_the_action_lives_outside_dot_github_and_is_composite(action):
    assert ACTION.is_file() and ".github" not in ACTION.parts
    assert action["runs"]["using"] == "composite"
    raw = _committed_bytes(ACTION)
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    assert not ACTION.read_bytes().startswith(b"\xef\xbb\xbf")


def test_the_checkout_is_full_history_with_a_blob_none_partial_clone(action):
    (co,) = [s for s in _steps(action) if str(s.get("uses", "")).startswith("actions/checkout@")]
    assert co["with"]["fetch-depth"] == 0 and co["with"]["filter"] == "blob:none"
    assert co["with"]["ref"] == "${{ inputs.head }}"           # the tree the check reads arrives in one batch
    assert "inputs.checkout" in co["if"]
    desc = action["description"]
    for needle in ("fetch-depth: 0", "blob:none", "contents: read", "commit sha", "report-only"):
        assert needle in desc, needle


def test_every_python_the_action_runs_is_isolated_from_the_checkout(action):
    """Each ``python`` in a step's script is ``python -I``: under plain ``-m`` the working directory, which
    is the pull request's tree, is ``sys.path[0]``, and ``python -m pip`` or ``python -m styxx.prereg_ci``
    would import what the pull request wrote there."""
    seen = 0
    for step in _steps(action):
        for line in (step.get("run") or "").splitlines():
            code = line.split("#", 1)[0]
            for m in re.finditer(r"(?<![\w./-])python(?:3(?:\.\d+)?)?(?![\w.-])", code):
                seen += 1
                assert code[m.end():].startswith(" -I "), line
    assert seen == 2
    desc = action["description"]
    assert "-I" in desc and "fathom-lab/styxx/prereg-ci@<sha>" in desc and "never uses: ./prereg-ci" in desc
    assert "./prereg-ci" in action["inputs"]["styxx-source"]["description"]


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
    assert "python -I -X utf8 -m styxx.prereg_ci" in script and " check " in script.replace('"${args[@]}"', " check ")
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


def _isolated_python(where: Path) -> Path:
    """An interpreter with this checkout installed and nothing else of the test's on its path: a venv
    made with ``--without-pip`` (nothing is downloaded) that sees the interpreter's site-packages,
    with a ``.pth`` naming this checkout ahead of them. Under ``-I`` it imports styxx from here, as a
    runner imports the copy the install step put in site-packages, whatever the working directory
    holds."""
    import site
    import venv

    venv.EnvBuilder(system_site_packages=True, with_pip=False, symlinks=os.name != "nt").create(str(where))
    exe = where / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    r = subprocess.run([str(exe), "-I", "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                       capture_output=True, check=True)
    purelib = Path(r.stdout.decode("utf-8").strip())
    lines = [str(ROOT)]
    if sys.prefix != sys.base_prefix:                    # a venv's own site-packages are not the base's
        lines += [p for p in site.getsitepackages() if os.path.isdir(p)]
    (purelib / "_prereg_ci_under_test.pth").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return exe


def run_step(action, repo: Path, tmp: Path, **inputs):
    """Run the check step's script with the environment the composite sets for these inputs, under an
    interpreter that has this checkout installed (``_isolated_python``) and no ``PYTHONPATH``."""
    bash = _bash()
    if bash is None:
        pytest.skip("no bash on this box to run the step's script")
    step = _check_step(action)
    python = _isolated_python(tmp / "py")
    defaults = {k: (v.get("default", "") if isinstance(v, dict) else "") for k, v in action["inputs"].items()}
    values = {**defaults, **inputs}
    env = dict(os.environ)
    for name, expr in step["env"].items():
        m = re.fullmatch(r"\$\{\{ inputs\.([a-z-]+) \}\}", expr)
        if m:
            env[name] = str(values[m.group(1)])
    env.pop("PYTHONPATH", None)
    env.update(GITHUB_OUTPUT=str(tmp / "out.txt"), GITHUB_STEP_SUMMARY=str(tmp / "summary.md"),
               PYTHONDONTWRITEBYTECODE="1", PREREG_CI_TEST_PYTHON=python.as_posix())
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


def test_the_step_is_report_only_by_default_and_still_writes_the_refusal(action, tmp_path):
    """B9: the README calls the Action report-only by default. With no report-only input a FAIL exits 0, and
    the verdict, the exit code and the report still say FAIL."""
    assert action["inputs"]["report-only"]["default"] == "true"
    assert "report-only by default" in " ".join(action["description"].split())
    g, rec = demo(tmp_path)
    g("switch", "-q", "-c", "typed", "result")
    g.write("RESULT_digits.md", result_md("HOLD__calibration"))
    g.commit("type the verdict")
    out = tmp_path / "o"
    out.mkdir()
    code, text, outputs = run_step(action, g.root, out, base=g("rev-parse", "result"), head=g("rev-parse", "typed"),
                                   **{"out-dir": str(out / "report")})
    assert code == 0, text
    assert outputs["verdict"] == "FAIL" and outputs["exit-code"] == "1" and "FAIL VERDICT_DIFFERS" in text
    assert "report-only, so this step exits 0" in text
    assert json.loads(Path(outputs["report"]).read_text(encoding="ascii"))["verdict"] == "FAIL"


def test_policy_inputs_become_flags(action, tmp_path):
    g, rec = demo(tmp_path)
    out = tmp_path / "o"
    out.mkdir()
    code, text, outputs = run_step(action, g.root, out, base=g("rev-parse", "main"), head=g("rev-parse", "result"),
                                   **{"out-dir": str(out / "r"), "require-time": "T3a", "exclude": "conformance/\ndocs/",
                                      "report-only": "false"})
    assert code == 1 and "FAIL TIME_BELOW_POLICY" in text and "require_time T3a (flag)" in text, text
    rep = json.loads(Path(outputs["report"]).read_text(encoding="ascii"))
    assert rep["policies"]["exclude"] == ["conformance/", "docs/"]


SHADOW = {"styxx/__init__.py": "", "styxx/prereg_ci/__init__.py": "",
          "styxx/prereg_ci/__main__.py": "import sys\nprint('PASS  a judge the pull request wrote')\nsys.exit(0)\n"}


def test_a_pull_request_that_brings_its_own_checker_is_judged_by_the_installed_one(action, tmp_path):
    """The B7 review's F1: a typed-verdict pull request that also adds ``styxx/prereg_ci/__main__.py`` to its
    tree still reads exit 1, FAIL VERDICT_DIFFERS, and the shadow never runs. The same step with ``-I``
    taken out reads the shadow's PASS, so this test sees the flag it guards."""
    g, rec = demo(tmp_path)
    g("switch", "-q", "-c", "typed", "result")
    g.write("RESULT_digits.md", result_md("HOLD__calibration"))
    for path, text in SHADOW.items():
        g.write(path, text)
    g.commit("type the verdict and bring a judge")
    base, head = g("rev-parse", "result"), g("rev-parse", "typed")
    out = tmp_path / "o"
    out.mkdir()
    code, text, outputs = run_step(action, g.root, out, base=base, head=head,
                                   **{"out-dir": str(out / "r"), "report-only": "false"})
    assert code == 1 and outputs["verdict"] == "FAIL", text
    assert "FAIL VERDICT_DIFFERS" in text and "a judge the pull request wrote" not in text
    plain = json.loads(json.dumps(action))
    step = _check_step(plain)
    assert step["run"].count("python -I -X utf8 -m styxx.prereg_ci") == 1
    step["run"] = step["run"].replace("python -I -X utf8 -m styxx.prereg_ci", "python -m styxx.prereg_ci")
    out2 = tmp_path / "o2"
    out2.mkdir()
    code, text, outputs = run_step(plain, g.root, out2, base=base, head=head,
                                   **{"out-dir": str(out2 / "r"), "report-only": "false"})
    assert code == 0 and outputs["verdict"] == "PASS" and "a judge the pull request wrote" in text, text
