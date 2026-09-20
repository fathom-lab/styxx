"""The telescope's published state and the telescope's claims about itself must agree.

`telescope/README.md` opens with *"daily cognometric measurement layer for
fathom.darkflobi.com/scoreboard"*. The public scoreboard fetches
`telescope/data/latest.json` straight from raw.githubusercontent.com. On 2026-09-20 that file was
133 days old, and the workflow that is supposed to refresh it had been reporting success without
measuring anything.

Two independent blockers, either of which alone stops the daily run:

1. **No vendor keys.** `telescope.yml`'s `check vendor keys` step sets `present=false`, the install,
   run and commit steps are all skipped by `if: steps.keys.outputs.present == 'true'`, and the job
   succeeds with a notice. Correct for a fork. On the repository the scoreboard reads from, a green
   tick over an unrefreshed file is the wrong signal.
2. **No corpus.** `run.py` exits at `load_prompts()` unless `telescope/prompts.json` is present, and
   `.gitignore` ignored that path -- a bare `telescope/*.json` filed under *LaTeX build artifacts*,
   meant for stray run output, which caught the held-out corpus too. A fresh checkout has no corpus,
   so a run **with** keys would have died before reaching a model.

Blocker 1 hid blocker 2 for the whole life of the workflow, because the step that would have hit it
never ran. This is the silent-pass shape twice over (`benchmarks/silent_pass/`): the absence of a
measurement reported as a passing check, and the first absence masking the second.

Neither blocker is fixable from a test. What is fixable is the repository saying one thing while
doing another, and that is what these tests hold: `telescope/STATUS.md` must describe the state the
data is actually in, in both directions. If the data goes stale, STATUS.md must say so. If the data
goes fresh, STATUS.md must stop saying so. Whoever restarts the telescope cannot leave the
explanation behind.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TELESCOPE = ROOT / "telescope"
RUN_PY = TELESCOPE / "run.py"
README = TELESCOPE / "README.md"
STATUS = TELESCOPE / "STATUS.md"
LATEST = TELESCOPE / "data" / "latest.json"
WORKFLOW = ROOT / ".github" / "workflows" / "telescope.yml"

# The scoreboard is documented as daily. A week is generous for "the daily job is running".
STALE_AFTER_DAYS = 7


def _in_ci() -> bool:
    return bool(os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"))


def _latest_ts() -> tuple[str, int]:
    """(ts_iso as published, age in days)."""
    doc = json.loads(LATEST.read_text(encoding="utf-8"))
    ts_iso = doc["ts_iso"]
    when = dt.datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
    return ts_iso, (dt.datetime.now(dt.timezone.utc) - when).days


def _corpus_path() -> Path:
    """The corpus path read out of run.py, so renaming the file cannot fool this file."""
    m = re.search(r'p\s*=\s*HERE\s*/\s*"([^"]+)"', RUN_PY.read_text(encoding="utf-8"))
    assert m, "run.py no longer loads its corpus in a form this test can read"
    return TELESCOPE / m.group(1)


# ---- blocker 2: the corpus must at least be committable ---------------------------------------

def test_the_corpus_run_py_requires_is_not_gitignored():
    """`git check-ignore` on the path run.py actually opens.

    Before 2026-09-20 this failed: `telescope/*.json` in the LaTeX-artifacts block ignored
    telescope/prompts.json, so `git add` was a no-op and CI checkouts had no corpus.
    """
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        if _in_ci():
            raise AssertionError(
                "git is unavailable in CI, so the one check on whether the telescope corpus can be "
                "committed did not run. This FAILS rather than skips: a guard that silently does "
                "not run is the defect this file exists to document."
            )
        pytest.skip("not a git checkout; `git check-ignore` cannot be consulted here")
    corpus = _corpus_path().relative_to(ROOT)
    # Plain check-ignore: exit 0 means the path IS ignored, 1 means it is not. `-v` must not be
    # used for the verdict -- it also reports paths matched by a NEGATIVE pattern and returns 0 for
    # them, so `-v` would call an un-ignored path ignored. It is only good for naming the rule.
    ignored = subprocess.run([git, "check-ignore", str(corpus)],
                             cwd=str(ROOT), capture_output=True, text=True).returncode == 0
    rule = subprocess.run([git, "check-ignore", "-v", str(corpus)],
                          cwd=str(ROOT), capture_output=True, text=True).stdout.strip()
    assert not ignored, (
        "%s is the corpus run.py loads, and .gitignore drops it:\n    %s\n"
        "A checkout without it makes every telescope run exit 1 at load_prompts(), which is why "
        "the daily job could not have measured anything even with API keys configured."
        % (corpus, rule))


def test_a_missing_corpus_is_stated_rather_than_left_to_be_discovered():
    corpus = _corpus_path()
    if corpus.exists():
        return
    body = STATUS.read_text(encoding="utf-8")
    assert corpus.name in body and "not in the tree" in body, (
        "%s is absent and telescope/STATUS.md does not say so. Un-ignoring the path does not "
        "create the file; a reader has to be told it still has to be committed." % corpus.name)


# ---- the data and the claims about it ---------------------------------------------------------

def test_status_names_the_measurement_that_is_actually_published():
    """STATUS.md must quote `latest.json`'s own timestamp, so refreshing the data forces
    refreshing the claim."""
    ts_iso, _ = _latest_ts()
    assert ts_iso in STATUS.read_text(encoding="utf-8"), (
        "telescope/data/latest.json publishes ts_iso %s and telescope/STATUS.md does not mention "
        "it. STATUS.md is describing a different run than the one the scoreboard is serving." % ts_iso)


def test_status_and_the_data_agree_about_whether_the_telescope_is_running():
    """Both directions. Stale data needs an acknowledgement; fresh data must not keep one."""
    ts_iso, age = _latest_ts()
    body = STATUS.read_text(encoding="utf-8")
    claims_paused = "The daily run has not happened since" in body
    if age > STALE_AFTER_DAYS:
        assert claims_paused, (
            "telescope/data/latest.json is %d days old and telescope/STATUS.md does not open by "
            "saying the daily run has stopped. README.md calls this a daily measurement layer and "
            "the public scoreboard reads that file." % age)
    else:
        assert not claims_paused, (
            "the telescope is running again (latest data is %d days old) but STATUS.md still opens "
            "with the pause. Delete that section -- a stale explanation is its own wrong claim." % age)


def test_the_readme_sends_a_reader_to_the_status_page():
    assert "STATUS.md" in README.read_text(encoding="utf-8"), (
        "telescope/README.md calls this a daily measurement layer without pointing at STATUS.md, "
        "so a reader has no way to learn it is not currently daily.")


# ---- documented files that nothing maintains ---------------------------------------------------

def test_every_data_file_the_readme_documents_is_written_by_something():
    """`data/timeseries.jsonl` is documented as the long-running per-model trajectory. run.py
    never mentions it. Either something writes a documented file, or STATUS.md says what it is."""
    readme = README.read_text(encoding="utf-8")
    documented = set(re.findall(r"([A-Za-z0-9_]+\.(?:json|jsonl))\s", readme))
    run_py = RUN_PY.read_text(encoding="utf-8")
    status = STATUS.read_text(encoding="utf-8")
    orphans = [n for n in sorted(documented) if n not in run_py and n not in status]
    assert not orphans, (
        "telescope/README.md documents %r as part of the running system, but run.py does not write "
        "them and STATUS.md does not say what they are." % orphans)


# ---- the workflow's green skip ------------------------------------------------------------------

def test_the_workflows_green_skip_is_acknowledged_somewhere_a_reader_looks():
    """The skip is deliberate and annotated in the run log. A run log is not where anyone looks to
    find out whether the scoreboard is current."""
    wf = WORKFLOW.read_text(encoding="utf-8")
    skips_green = "present=false" in wf and "steps.keys.outputs.present == 'true'" in wf
    if not skips_green:
        return
    assert STATUS.exists(), (
        ".github/workflows/telescope.yml reports success without measuring when no TELESCOPE_* "
        "secret is set, and telescope/STATUS.md does not exist to say so.")
    body = STATUS.read_text(encoding="utf-8")
    assert "check vendor keys" in body and "scoreboard" in body, (
        "STATUS.md exists but does not explain the green-skip or what it means for the scoreboard.")
