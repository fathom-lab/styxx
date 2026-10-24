# -*- coding: utf-8 -*-
"""integrations/gitlab/diffgate.gitlab-ci.yml — the job's script lines, run the way GitLab runs them.

The job file is parsed as YAML, its `script` lines are executed in a temporary repository with
CI_MERGE_REQUEST_DESCRIPTION, CI_MERGE_REQUEST_DIFF_BASE_SHA and CI_COMMIT_SHA set the way a
merge request pipeline sets them (the `pip install` line is skipped: the release under test is
the one on sys.path). The JSON artifact is written either way.

The job reports by default, as the GitHub Action does since it changed its default (CHANGELOG,
"the GitHub Action reports by default"): a lying description is reported, its lies named in the
log, and the job passes. Only STYXX_SOFT_FAIL set to "false" blocks, read the way the Action's
``_soft_fail`` reads soft-fail: compared case-insensitively with surrounding whitespace stripped,
unset as "true", every other value reporting with a warning naming it. An error that leaves no
report fails the job in either mode.

GitLab stores only the leading 2700 characters of a longer description in the variable and sets
CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED to "true" (#209). The job then gates the leading part up to its
last whole sentence, names the cut, prints CUT where the gate prints PASS, and records the cut in
diffgate.json; the tests hand it description[:2700] with the flag, as GitLab does.
"""
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "integrations" / "gitlab" / "diffgate.gitlab-ci.yml"
yaml = pytest.importorskip("yaml")

LYING = "Refactored src/retry.py. Added 3 tests. Only touches files under src/."
HONEST = "Modified src/retry.py, adds function retry_once, added 1 test. 2 files changed."
# One claim of each verdict: file_touched VERIFIED, tests_added CONTRADICTED, tests_pass UNCHECKABLE.
MIXED = "Refactored src/retry.py. Added 3 tests. All tests pass."
PASSES = 'so the job passes; set STYXX_SOFT_FAIL to "false" to fail it.'
REPORTED = "WARNING: styxx diffgate found failures but STYXX_SOFT_FAIL is on, as it is by default, " + PASSES
REPORTED_ODD = ("WARNING: styxx diffgate found failures but STYXX_SOFT_FAIL has a value this job does not "
                "recognise (named above), so it reports as it does by default, " + PASSES)
BLOCKED = 'styxx diffgate: FAIL, and STYXX_SOFT_FAIL is "false", so the job fails.'
NOT_RECOGNISED = "which is neither \"true\" nor \"false\""
NO_REPORT = "without writing a FAIL report: that is an error, not a verdict"
# The mode line the job prints on every run, PASS included, so that a variable that never reached
# the job (a Protected one on an unprotected branch, say) shows as "not set" in the log.
MODE = "styxx diffgate mode: STYXX_SOFT_FAIL "
REPORTS = ", so this job reports and the gate's verdicts do not fail it."
MODE_UNSET = MODE + "is not set" + REPORTS + " A variable marked Protected reaches a merge request pipeline"
MODE_TRUE = MODE + 'reads as "true"' + REPORTS
MODE_ODD = MODE + "has a value this job does not recognise" + REPORTS
MODE_FALSE = MODE + 'is "false", so a FAIL verdict fails this job.'


def _changelog_entry(changelog: str, title: str) -> str:
    """The CHANGELOG entry whose heading starts with `title`: `## [Unreleased] — title` while it waits
    for a release, `### title` once a release cut folds it under a version heading (7.49.0's cut set
    the pattern). It ends at the next heading of level 2 or 3."""
    m = re.search(r"^#{2,3} (?:\[Unreleased\] — )?" + re.escape(title), changelog, re.M)
    assert m, f"CHANGELOG.md has no entry headed {title!r} at level 2 or 3"
    rest = changelog[m.end():]
    nxt = re.search(r"\n#{2,3} ", rest)
    return rest[: nxt.start()] if nxt else rest


def _mode(soft_fail):
    """The mode line the job should print for this value of STYXX_SOFT_FAIL (None: unset)."""
    if soft_fail is None:
        return MODE_UNSET
    v = soft_fail.strip().lower()
    return MODE_FALSE if v == "false" else MODE_TRUE if v == "true" else MODE_ODD


def _one_mode_line(stdout, soft_fail):
    assert stdout.count(MODE) == 1, stdout
    assert _mode(soft_fail) in stdout, stdout


def _bash():
    """The bash that runs the job's script, as a full path, or None when there is none.

    A bare "bash" is the wrong name to hand subprocess on Windows (#185): CreateProcess searches
    the system directory before PATH, so it starts WSL's launcher (System32\\bash.exe) even where
    shutil.which("bash") found Git's bash, and the launcher can exit 1 without running a line.
    So the script runs under the path found here, and a launcher under the Windows directory or
    the WindowsApps alias never counts as bash. On Windows, Git's own bash is looked for beside
    git.exe when the one on PATH is a launcher.
    """
    found = shutil.which("bash")
    if os.name != "nt":
        return found

    def usable(p):
        if not p:
            return False
        low = str(Path(p).resolve()).lower()
        windir = str(Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve()).lower()
        return not (low.startswith(windir + os.sep) or os.sep + "windowsapps" + os.sep in low)

    if usable(found):
        return found
    git = shutil.which("git")
    if git:
        for parent in Path(git).resolve().parents:
            for rel in (("bin", "bash.exe"), ("usr", "bin", "bash.exe")):
                cand = parent.joinpath(*rel)
                if cand.is_file() and usable(cand):
                    return str(cand)
    return None


BASH = _bash()
needs_bash = pytest.mark.skipif(BASH is None, reason="the job's script needs bash (WSL's launcher does not count)")


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", check=True).stdout.strip()


def _job():
    return yaml.safe_load(JOB.read_text(encoding="utf-8"))["diffgate"]


@pytest.fixture
def mr_repo(tmp_path):
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "clone", "-q", str(origin), str(work))
    _git(work, "config", "user.email", "t@t")
    _git(work, "config", "user.name", "t")
    (work / "src").mkdir()
    (work / "src" / "retry.py").write_text("def retry(n):\n    return n\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "base")
    _git(work, "push", "-q", "origin", "HEAD:main")
    base = _git(work, "rev-parse", "HEAD")
    (work / "src" / "retry.py").write_text(
        "def retry(n):\n    return n\n\ndef retry_once(n):\n    return retry(1)\n", encoding="utf-8")
    (work / "tests").mkdir()
    (work / "tests" / "test_retry.py").write_text("def test_retry_once():\n    assert True\n", encoding="utf-8")
    _git(work, "add", "-A")
    _git(work, "commit", "-qm", "the change")
    head = _git(work, "rev-parse", "HEAD")
    return work, base, head


def _run_job(work, base, head, description, soft_fail=None, strict=False, gate=None, truncated=None):
    """Run the job's script lines under bash. ``soft_fail`` None leaves STYXX_SOFT_FAIL unset;
    ``strict`` adds --strict to the styxx.diffgate line, the edit the README describes; ``gate``,
    a script path and its arguments, stands in for the whole styxx.diffgate command (a CLI that
    exits without the report its exit promises); ``truncated`` is the value GitLab gives
    CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED (None leaves it unset)."""
    job = _job()
    assert job["rules"][0]["if"] == '$CI_PIPELINE_SOURCE == "merge_request_event"'
    env = dict(os.environ, CI_MERGE_REQUEST_DESCRIPTION=description,
               CI_MERGE_REQUEST_DIFF_BASE_SHA=base, CI_COMMIT_SHA=head,
               CI_PIPELINE_SOURCE="merge_request_event", PYTHONPATH=str(ROOT))
    env.pop("STYXX_SOFT_FAIL", None)
    if soft_fail is not None:
        env["STYXX_SOFT_FAIL"] = soft_fail
    env.pop("CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED", None)
    if truncated is not None:
        env["CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED"] = truncated
    lines = [l for l in job["script"] if not l.startswith("pip install")]
    script = "\n".join(lines)
    if strict:
        assert script.count(" --out diffgate.json") == 1
        script = script.replace(" --out diffgate.json", " --strict --out diffgate.json")
    if gate is not None:
        stub = "python " + " ".join(shlex.quote(str(a)) for a in gate)
        script, n = re.subn(r"(?m)^python -m styxx\.diffgate [^\n]* --out diffgate\.json"
                            r"(?=(?: > diffgate\.log)? \|\| styxx_gate=\$\?$)", lambda _: stub, script)
        assert n == 1, script
    # bash reads the backslashes of a Windows path as escapes, so the interpreter is passed with
    # forward slashes and quoted (#185); on POSIX this is the path as it was. The job names it
    # `python` three times, at the start of a line: the cut of a description GitLab truncated (#209),
    # the gate, and the reading of STYXX_SOFT_FAIL.
    python = shlex.quote(Path(sys.executable).as_posix())
    script, n = re.subn(r"(?m)^python(?= )", lambda _: python, script)
    assert n == 3, script
    return subprocess.run([BASH, "-c", "set -e\n" + script + "\n"], cwd=work, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")


def _report(work):
    return json.loads((work / "diffgate.json").read_text(encoding="utf-8"))


@needs_bash
def test_by_default_a_lying_description_is_reported_with_the_lies_named_and_the_job_passes(mr_repo):
    work, base, head = mr_repo
    r = _run_job(work, base, head, LYING)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "FAIL" in r.stdout and "[CONTRADICTED:tests_added]" in r.stdout
    assert "[CONTRADICTED:only_touches]" in r.stdout and "tests/test_retry.py" in r.stdout
    assert REPORTED in r.stdout and REPORTED_ODD not in r.stdout
    assert NOT_RECOGNISED not in r.stdout      # unset is the default, not a value
    _one_mode_line(r.stdout, None)
    assert _report(work)["verdict"] == "FAIL"


@needs_bash
def test_with_soft_fail_false_a_lying_description_fails_the_job_with_the_lies_named(mr_repo):
    work, base, head = mr_repo
    r = _run_job(work, base, head, LYING, soft_fail="false")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "FAIL" in r.stdout and "[CONTRADICTED:tests_added]" in r.stdout
    assert "[CONTRADICTED:only_touches]" in r.stdout and "tests/test_retry.py" in r.stdout
    assert BLOCKED in r.stdout and PASSES not in r.stdout
    _one_mode_line(r.stdout, "false")
    assert _report(work)["verdict"] == "FAIL"


@needs_bash
@pytest.mark.parametrize("soft_fail", [None, "true", "false"])
def test_an_honest_description_passes_the_job_and_the_log_names_the_mode(mr_repo, soft_fail):
    """A PASS run prints the mode it ran in too: a STYXX_SOFT_FAIL "false" that never reached the
    job (a variable marked Protected, on a merge request whose branches are not both protected)
    shows as "not set" on every run, not only on a run that FAILs."""
    work, base, head = mr_repo
    r = _run_job(work, base, head, HONEST, soft_fail=soft_fail)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.startswith("PASS")
    assert "WARNING" not in r.stdout
    _one_mode_line(r.stdout, soft_fail)
    assert ("marked Protected" in r.stdout) is (soft_fail is None)
    assert _report(work)["verdict"] == "PASS"


@needs_bash
@pytest.mark.parametrize("soft_fail", [None, "false"])
def test_the_log_names_each_contradicted_or_uncheckable_claim_and_the_artifact_holds_every_verdict(
        mr_repo, soft_fail):
    """The job has no job summary: the CLI prints the gate's verdict, its claim counts and each
    claim that is not VERIFIED, and only diffgate.json carries every claim's verdict. The Action's
    job summary lists every claim; this job's log does not, so its docs must not say it does."""
    work, base, head = mr_repo
    r = _run_job(work, base, head, MIXED, soft_fail=soft_fail)
    assert r.returncode == (1 if soft_fail == "false" else 0), r.stdout + r.stderr
    assert re.search(r"(?m)^FAIL +claims=3 contradicted=1 uncheckable=1\b", r.stdout), r.stdout
    assert "[CONTRADICTED:tests_added]" in r.stdout and "[UNCHECKABLE:tests_pass]" in r.stdout
    assert "[VERIFIED" not in r.stdout and "file_touched" not in r.stdout
    _one_mode_line(r.stdout, soft_fail)
    verdicts = {c["kind"]: c["verdict"] for c in _report(work)["claims"]}
    assert verdicts == {"file_touched": "VERIFIED", "tests_added": "CONTRADICTED",
                        "tests_pass": "UNCHECKABLE"}


def _action_soft_fail():
    spec = importlib.util.spec_from_file_location("_diffgate_action_for_gitlab", ROOT / "diffgate_action.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._soft_fail


# The seven values tests/test_diffgate_action.py reads through the Action, and three more.
@needs_bash
@pytest.mark.parametrize("value, blocks, warned", [
    ("", False, True),
    (" true", False, False),
    ("TRUE", False, False),
    ("yes", False, True),
    ("0", False, True),
    ("False ", True, False),
    ("false", True, False),
    ("\tFALSE\n", True, False),
    ("no", False, True),
    ("fals", False, True),
])
def test_how_the_soft_fail_value_is_read(mr_repo, value, blocks, warned):
    """Only "false" blocks, compared case-insensitively with surrounding whitespace stripped;
    "true" reports; every other value, the empty string included, reports and is named in a
    warning. The Action's reader gives the same answer for each value."""
    assert _action_soft_fail()(value) == (not blocks, value if warned else None)
    work, base, head = mr_repo
    r = _run_job(work, base, head, LYING, soft_fail=value)
    assert r.returncode == (1 if blocks else 0), r.stdout + r.stderr
    assert "[CONTRADICTED:tests_added]" in r.stdout
    assert (NOT_RECOGNISED in r.stdout) is warned
    if warned:
        assert f"WARNING: STYXX_SOFT_FAIL is {value!r}, which is neither" in r.stdout
    assert (BLOCKED in r.stdout) is blocks
    assert (PASSES in r.stdout) is (not blocks)
    assert (REPORTED in r.stdout) is (not blocks and not warned)
    assert (REPORTED_ODD in r.stdout) is (not blocks and warned)
    _one_mode_line(r.stdout, value)
    assert (MODE_FALSE in r.stdout) is blocks and (MODE_ODD in r.stdout) is warned
    assert _report(work)["verdict"] == "FAIL"


@needs_bash
@pytest.mark.parametrize("soft_fail, rc", [(None, 0), ("false", 1)])
def test_strict_makes_an_uncheckable_claim_a_failure_that_blocks_only_with_soft_fail_false(mr_repo, soft_fail, rc):
    work, base, head = mr_repo
    r = _run_job(work, base, head, "Added 1 test. All tests pass.", soft_fail=soft_fail, strict=True)
    assert r.returncode == rc, r.stdout + r.stderr
    assert r.stdout.startswith("FAIL") and "contradicted=0" in r.stdout
    assert "[UNCHECKABLE:tests_pass]" in r.stdout
    assert (REPORTED in r.stdout) is (rc == 0) and (BLOCKED in r.stdout) is (rc == 1)
    _one_mode_line(r.stdout, soft_fail)
    assert _report(work)["verdict"] == "FAIL"


@needs_bash
@pytest.mark.parametrize("soft_fail", [None, "true", "false"])
def test_a_gate_that_errors_fails_the_job_in_either_mode_and_a_stale_report_is_not_read(mr_repo, soft_fail):
    """A head the repository does not have makes the CLI raise, exit 1 and write no report. That
    is an error, not a FAIL verdict, so reporting does not swallow it; and a diffgate.json left
    from an earlier run, saying FAIL, is removed before the gate runs rather than read as this
    run's verdict."""
    work, base, _head = mr_repo
    (work / "diffgate.json").write_text(json.dumps({"diffgate": "v0", "verdict": "FAIL"}), encoding="utf-8")
    r = _run_job(work, base, "0" * 40, LYING, soft_fail=soft_fail)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "styxx diffgate exited 1 " + NO_REPORT in r.stdout
    assert PASSES not in r.stdout and BLOCKED not in r.stdout
    _one_mode_line(r.stdout, soft_fail)
    assert not (work / "diffgate.json").exists()


STUB_GATE = '''import json, sys
verdict, code = sys.argv[1], int(sys.argv[2])
if verdict != "none":
    with open("diffgate.json", "w", encoding="utf-8") as f:
        json.dump({"diffgate": "v0", "verdict": verdict}, f)
print("stub gate: verdict", verdict, "exit", code)
sys.exit(code)
'''


@needs_bash
@pytest.mark.parametrize("soft_fail", [None, "false"])
@pytest.mark.parametrize("verdict, code, rc, error", [
    ("PASS", 0, 0, None),                    # the CLI's PASS: exit 0 with a PASS report
    ("none", 0, 1, "exited 0 without writing a PASS report"),
    ("FAIL", 0, 1, "exited 0 without writing a PASS report"),
    ("PASS", 1, 1, "exited 1 without writing a FAIL report"),
    ("none", 2, 2, "exited 2 without writing a matching report"),
    ("FAIL", 2, 2, "exited 2 without writing a matching report"),
])
def test_a_gate_whose_report_does_not_match_its_exit_fails_the_job_in_either_mode(
        mr_repo, soft_fail, verdict, code, rc, error):
    """Only an exit 0 with a PASS report passes as a PASS, and only an exit 1 with a FAIL report is
    read as a FAIL verdict. A CLI that exits 0 and writes nothing (a shadowed or altered one; the
    released CLI writes --out before it returns) is an error, not a PASS, and so is any other
    pairing, in either mode. A PASS report an earlier run left behind is removed before the gate
    runs, so it cannot stand in for this run's."""
    work, base, head = mr_repo
    stub = work.parent / "stub_gate.py"
    stub.write_text(STUB_GATE, encoding="utf-8")
    (work / "diffgate.json").write_text(json.dumps({"diffgate": "v0", "verdict": "PASS"}), encoding="utf-8")
    r = _run_job(work, base, head, LYING, soft_fail=soft_fail,
                 gate=[stub.as_posix(), verdict, code])
    assert r.returncode == rc, r.stdout + r.stderr
    assert f"stub gate: verdict {verdict} exit {code}" in r.stdout
    _one_mode_line(r.stdout, soft_fail)
    if error is None:
        assert "that is an error, not a verdict" not in r.stdout
    else:
        assert f"styxx diffgate {error}: that is an error, not a verdict" in r.stdout
    assert PASSES not in r.stdout and BLOCKED not in r.stdout
    assert (work / "diffgate.json").exists() is (verdict != "none")


def test_the_job_never_expands_the_description_through_the_shell():
    """The one shell expansion of the description is the printf that writes it to a file. The flag that
    says GitLab cut it (#209) is read by Python from the environment, never through the shell."""
    job = _job()
    writes = [l for l in job["script"] if re.search(r"\$\{?CI_MERGE_REQUEST_DESCRIPTION(?![A-Z_])", l)]
    assert writes == ["printf '%s' \"$CI_MERGE_REQUEST_DESCRIPTION\" > mr_description.md"]
    assert not any(re.search(r"\$\{?CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED", l) for l in job["script"])
    assert 'os.environ.get("CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED")' in _cut_step()
    assert job["artifacts"]["when"] == "always" and "diffgate.json" in job["artifacts"]["paths"]


def test_the_job_leaves_soft_fail_unset_so_a_pipeline_or_project_variable_reaches_it():
    """In GitLab a variable in the job's own `variables:` outranks one in the pipeline's top-level
    `variables:`, so a default written into the job would silently override a pipeline that sets
    STYXX_SOFT_FAIL: "false". The default lives in the script (unset reads as "true") instead.
    The job's commented-out line is the copy-paste opt-in."""
    job = _job()
    assert "STYXX_SOFT_FAIL" not in (job.get("variables") or {})
    text = JOB.read_text(encoding="utf-8")
    assert re.search(r'(?m)^    # STYXX_SOFT_FAIL: "false"', text)


def test_the_docs_say_the_job_reports_by_default():
    readme = (ROOT / "integrations" / "gitlab" / "README.md").read_text(encoding="utf-8")
    header = JOB.read_text(encoding="utf-8").split("\ndiffgate:", 1)[0]
    for text in (readme, header):
        flat = " ".join(text.replace("#", " ").split())
        assert "reports by default" in flat
        assert 'Only STYXX_SOFT_FAIL set to "false" blocks' in flat
        assert "The job fails only on a CONTRADICTED claim" not in flat


def test_the_docs_warn_that_a_protected_variable_does_not_reach_a_merge_request_pipeline():
    """GitLab hands a protected variable to a merge request pipeline only when its source and
    target branches are both protected branches of the same project and the project allows it
    (docs.gitlab.com, merge request pipelines, "Control access to protected variables and
    runners"); otherwise the job reads it as unset and reports. Someone who set STYXX_SOFT_FAIL
    in settings to keep blocking would lose blocking without a word, so the README, the job's
    header and the CHANGELOG entry each say so, and say what to do instead."""
    readme = (ROOT / "integrations" / "gitlab" / "README.md").read_text(encoding="utf-8")
    header = JOB.read_text(encoding="utf-8").split("\ndiffgate:", 1)[0]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = _changelog_entry(changelog, "the GitLab job reports by default")
    for text in (readme, header, entry):
        flat = " ".join(text.replace("#", " ").replace("`", "").split())
        assert "marked Protected" in flat
        assert "reaches a merge request pipeline only when its source and target branches are both protected" in flat
        assert "never from a fork" in flat
        assert "leave Protect cleared or set the variable in .gitlab-ci.yml" in flat


def test_the_docs_say_what_goes_to_the_log_and_what_to_the_artifact():
    """The README, the job's header and the CHANGELOG entry once said every verdict goes to the job
    log, a line copied from the Action, whose job summary lists every claim. This job's log holds
    the gate's verdict, its claim counts and each CONTRADICTED or UNCHECKABLE claim; every claim's
    verdict, VERIFIED included, is in diffgate.json (the MIXED run above checks it)."""
    readme = (ROOT / "integrations" / "gitlab" / "README.md").read_text(encoding="utf-8")
    header = JOB.read_text(encoding="utf-8").split("\ndiffgate:", 1)[0]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = _changelog_entry(changelog, "the GitLab job reports by default")
    for text in (readme, header, entry):
        flat = " ".join(text.replace("#", " ").replace("`", "").split())
        assert "every verdict goes to the job log" not in flat
        assert "each contradicted claim is named in the log" not in flat
        assert "its claim counts and each CONTRADICTED or UNCHECKABLE claim go to the" in flat
        assert re.search(r"every claim's verdict, VERIFIED included, goes to (the )?diffgate\.json", flat)


# #209. GitLab stores only the leading 2700 characters of a longer merge request description in
# CI_MERGE_REQUEST_DESCRIPTION and sets CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED to "true" (docs.gitlab.com,
# "Predefined CI/CD variables reference", both entries, read 2026-10-23). These tests hand the job
# description[:2700] with the flag, as GitLab does, and the whole description as the control.
GITLAB_KEEPS = 2700
FILLER = "The retry helper is described here at the length a long merge request description runs to. "
CUT = ("styxx diffgate: CUT. CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED is 'true': GitLab handed this job only "
       "the leading ")
CUT_NOT_PASS = "so this job does not call the description a pass."
# styxx.diffgate's own sentence split (`_gate`), which the job's cut must use
SPLIT = r"(?<=[.!?])\s+|\n+"


def _prose(n):
    """Whole filler sentences, exactly ``n`` characters (n >= 10), the last one ending '. '."""
    k = (n - 10) // len(FILLER)
    rest = n - k * len(FILLER)
    out = FILLER * k + "Note " + "x" * (rest - 7) + ". "
    assert len(out) == n
    return out


def _gitlab(description):
    """What GitLab hands the job for this description: the variable's value, and the flag's."""
    if len(description) > GITLAB_KEEPS:
        return description[:GITLAB_KEEPS], "true"
    return description, "false"


def _whole_sentences(text):
    """How much of ``text``, a prefix GitLab cut, ends at its last whole sentence: the oracle the
    tests hold the job to, written apart from the job (a filler sentence ends at '. ')."""
    return text.rfind(". ") + 1


def _cut_step():
    """The Python the job runs to cut a description GitLab truncated back to its last whole sentence."""
    for item in _job()["script"]:
        m = re.search(r"^python - <<'STYXX_DESCRIPTION_CUT'\n(.*?)^STYXX_DESCRIPTION_CUT$", item, re.S | re.M)
        if m:
            return m.group(1)
    raise AssertionError("the job has no step that reads CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED")


def _cut_named(stdout, kept, report):
    """The cut is named in the log and in the record, with what was handed and what was read."""
    named = (CUT + f"{GITLAB_KEEPS} characters of the merge request description. The gate reads the leading "
             f"{kept}, up to the last whole sentence, and never the rest")
    assert named in stdout, stdout
    assert stdout.count(CUT) == 1, stdout
    cut = report["description_cut"]
    assert cut["CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED"] == "true"
    assert (cut["characters_handed"], cut["characters_gated"]) == (GITLAB_KEEPS, kept)
    assert f"handed this job {GITLAB_KEEPS} characters" in cut["why"] and f"the leading {kept}," in cut["why"]


@needs_bash
@pytest.mark.parametrize("soft_fail", [None, "false"])
@pytest.mark.parametrize("lead", ["", "All tests pass. "])
def test_a_claim_past_the_cut_is_named_unread_and_never_printed_as_a_pass(mr_repo, soft_fail, lead):
    """#209 (a), the issue's case: the only diff claim, a false "Added 3 tests.", sits past character
    2700. Handed whole, the gate reads it CONTRADICTED. Handed GitLab's prefix, the job printed
    "PASS claims=0" and never said it had read a prefix; it now names the cut and how much it read,
    prints CUT where the gate prints PASS, records the cut, and passes in either mode (the claim is
    unread, not judged). An UNCHECKABLE claim before the cut is still named in the log."""
    work, base, head = mr_repo
    description = lead + _prose(GITLAB_KEEPS + 150) + "Added 3 tests."
    assert description.index("Added 3 tests.") > GITLAB_KEEPS
    whole = _run_job(work, base, head, description, soft_fail=soft_fail)           # the flag unset
    assert whole.returncode == (1 if soft_fail == "false" else 0), whole.stdout + whole.stderr
    assert re.search(rf"(?m)^FAIL +claims={1 + bool(lead)} contradicted=1 uncheckable={int(bool(lead))}\b",
                     whole.stdout), whole.stdout
    assert "[CONTRADICTED:tests_added]" in whole.stdout and "CUT" not in whole.stdout
    assert "description_cut" not in _report(work)

    text, flag = _gitlab(description)
    r = _run_job(work, base, head, text, soft_fail=soft_fail, truncated=flag)
    assert r.returncode == 0, r.stdout + r.stderr
    assert not re.search(r"(?m)^PASS\b", r.stdout), r.stdout
    assert re.search(rf"(?m)^CUT +claims={int(bool(lead))} contradicted=0 uncheckable={int(bool(lead))}\b",
                     r.stdout), r.stdout
    assert "tests_added" not in r.stdout
    assert ("[UNCHECKABLE:tests_pass]" in r.stdout) is bool(lead)
    assert CUT_NOT_PASS in r.stdout and "WARNING" not in r.stdout
    _one_mode_line(r.stdout, soft_fail)
    report = _report(work)
    assert report["verdict"] == "PASS"            # the gate's verdict over the prefix, kept as it wrote it
    assert [c["kind"] for c in report["claims"]] == (["tests_pass"] if lead else [])
    _cut_named(r.stdout, _whole_sentences(text), report)


@needs_bash
@pytest.mark.parametrize("soft_fail", [None, "false"])
def test_a_sentence_the_cut_broke_is_not_judged(mr_repo, soft_fail):
    """#209 (b), found by the triage: GitLab's cut at character 2700 lands inside "Only touches src/
    and tests/.", after "src/". Handed whole, the sentence is VERIFIED. The job gated the prefix
    "Only touches src/" as written and read it CONTRADICTED, and with STYXX_SOFT_FAIL "false" it
    failed the job on a true claim. It now gates whole sentences only."""
    work, base, head = mr_repo
    description = _prose(GITLAB_KEEPS - len("Only touches src/")) + "Only touches src/ and tests/."
    whole = _run_job(work, base, head, description, soft_fail=soft_fail, truncated="false")
    assert whole.returncode == 0 and whole.stdout.startswith("PASS"), whole.stdout + whole.stderr
    assert {c["kind"]: c["verdict"] for c in _report(work)["claims"]} == {"only_touches": "VERIFIED"}
    assert "description_cut" not in _report(work) and "CUT" not in whole.stdout

    text, flag = _gitlab(description)
    assert text.endswith(". Only touches src/") and flag == "true"
    r = _run_job(work, base, head, text, soft_fail=soft_fail, truncated=flag)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CONTRADICTED" not in r.stdout and "only_touches" not in r.stdout
    assert re.search(r"(?m)^CUT +claims=0 contradicted=0 uncheckable=0\b", r.stdout), r.stdout
    assert CUT_NOT_PASS in r.stdout
    _one_mode_line(r.stdout, soft_fail)
    report = _report(work)
    assert report["verdict"] == "PASS" and report["claims"] == []
    assert not any("only touches" in t.lower() for t in report["uncovered_texts"])
    _cut_named(r.stdout, len(text) - len("Only touches src/") - 1, report)


@needs_bash
@pytest.mark.parametrize("soft_fail", [None, "false"])
def test_a_false_whole_sentence_before_the_cut_still_fails_and_the_cut_is_named(mr_repo, soft_fail):
    """A sentence wholly before the cut is a sentence of the description, so its CONTRADICTED stands
    and the job reports or blocks on it as on any FAIL; the log and the record still name the cut."""
    work, base, head = mr_repo
    description = "Added 3 tests. " + _prose(GITLAB_KEEPS + 300)
    text, flag = _gitlab(description)
    r = _run_job(work, base, head, text, soft_fail=soft_fail, truncated=flag)
    assert r.returncode == (1 if soft_fail == "false" else 0), r.stdout + r.stderr
    assert re.search(r"(?m)^FAIL +claims=1 contradicted=1 uncheckable=0\b", r.stdout), r.stdout
    assert "[CONTRADICTED:tests_added]" in r.stdout and CUT_NOT_PASS not in r.stdout
    assert (BLOCKED in r.stdout) is (soft_fail == "false") and (REPORTED in r.stdout) is (soft_fail is None)
    _one_mode_line(r.stdout, soft_fail)
    report = _report(work)
    assert report["verdict"] == "FAIL"
    _cut_named(r.stdout, _whole_sentences(text), report)


def test_the_cut_uses_the_gates_own_sentence_split():
    """The job cuts where styxx.diffgate splits sentences, so every sentence it gates is a sentence
    of the whole description. A reader whose split moves moves this test."""
    src = (ROOT / "styxx" / "diffgate.py").read_text(encoding="utf-8")
    assert re.findall(r'sentences = re\.split\(r"([^"]+)", summary_text\)', src) == [SPLIT]
    assert 're.finditer(r"' + SPLIT + '", text)' in _cut_step()


CUT_CASES = [
    "Fixed v1.2 of the retry loop. Only touches src/ and tests/.\r\nAdded 2 tests!\n\nDone?  Yes.\u00a0Next one."
    "\nModified src/retry.py",
    "One long sentence with no end that GitLab cut in the middle and that has no stop at all",
    "\n\nAdded 1 test.\n```styxx\ntests_added: 3\nonly_touches: src\n```\nAfter the block. e.g. this. Ends.",
    "Wait... what?! Really.\tTabbed. Line\rCR only. \u2028Sep. Last",
]


@pytest.mark.parametrize("description", CUT_CASES)
def test_the_cut_keeps_only_whole_sentences_of_the_description_at_every_cut_point(tmp_path, monkeypatch, capsys,
                                                                                  description):
    """The job's cut step, run on every prefix GitLab could hand it: the text it leaves for the gate is
    a prefix of what it was handed, the gate's split of it is the leading sentences of the gate's split
    of the whole description, nothing longer qualifies, a declaration it leaves is the whole one or
    none, and the record it writes says how much was handed and how much was kept."""
    from styxx.declare import declaration_pass
    code = compile(_cut_step(), "STYXX_DESCRIPTION_CUT", "exec")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED", "true")
    whole = re.split(SPLIT, description)
    declared = declaration_pass(description)[0]
    for cut in range(len(description) + 1):
        handed = description[:cut]
        with open("mr_description.md", "w", encoding="utf-8", newline="") as f:
            f.write(handed)
        exec(code, {"__name__": "__main__"})
        with open("mr_description.md", encoding="utf-8", newline="") as f:
            kept = f.read()
        assert handed.startswith(kept), (cut, kept)
        if kept:
            parts = re.split(SPLIT, kept)
            assert parts == whole[:len(parts)], (cut, kept)
        assert all(m.start() <= len(kept) for m in re.finditer(SPLIT, handed)), (cut, kept)
        assert declaration_pass(kept)[0] in ("", declared), (cut, kept)
        with open("mr_description.cut.json", encoding="utf-8") as f:
            record = json.load(f)
        assert (record["characters_handed"], record["characters_gated"]) == (len(handed), len(kept))
        assert CUT + f"{len(handed)} characters" in capsys.readouterr().out


@pytest.mark.parametrize("flag", [None, "false", "", "no"])
def test_the_cut_step_leaves_a_description_alone_unless_gitlab_says_it_cut_it(tmp_path, monkeypatch, capsys, flag):
    """Without the flag "true" the description is gated as GitLab handed it, nothing is printed, and no
    cut record is left, a stale one from an earlier run included."""
    code = compile(_cut_step(), "STYXX_DESCRIPTION_CUT", "exec")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED", raising=False)
    if flag is not None:
        monkeypatch.setenv("CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED", flag)
    text = "Added 3 tests. Only touches src/"
    (tmp_path / "mr_description.md").write_text(text, encoding="utf-8", newline="")
    (tmp_path / "mr_description.cut.json").write_text("{}", encoding="utf-8")
    exec(code, {"__name__": "__main__"})
    assert (tmp_path / "mr_description.md").read_text(encoding="utf-8") == text
    assert not (tmp_path / "mr_description.cut.json").exists()
    assert capsys.readouterr().out == ""


def test_the_docs_name_the_cut_gitlab_makes_at_2700_characters():
    """GitLab hands the job at most the leading 2,700 characters (#209). The README, the job's header and
    the CHANGELOG entry each say so, name the flag, and say what the job does over a cut."""
    readme = (ROOT / "integrations" / "gitlab" / "README.md").read_text(encoding="utf-8")
    header = JOB.read_text(encoding="utf-8").split("\ndiffgate:", 1)[0]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    entry = _changelog_entry(changelog, "the GitLab job names a merge request description GitLab cut at 2,700")
    for text in (readme, header, entry):
        flat = " ".join(text.replace("#", " ").replace("`", "").split())
        assert "CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED" in flat
        assert "2,700 characters" in flat
        assert "last whole sentence" in flat
        assert "prints CUT where the gate prints PASS" in flat
        assert "description_cut" in flat
        assert "never read" in flat
