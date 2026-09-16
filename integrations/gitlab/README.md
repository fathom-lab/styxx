# diffgate on GitLab CI — the merge request description vs the merge request diff

The GitHub Action reads a PR body against a PR diff with no checkout. GitLab hands a merge
request pipeline the same two things another way: the description in
`CI_MERGE_REQUEST_DESCRIPTION`, the diff as the range between `CI_MERGE_REQUEST_DIFF_BASE_SHA`
and `CI_COMMIT_SHA` on the job's checkout. So the door is the CLI, unchanged, in a job of eight
lines. `diffgate.gitlab-ci.yml` next to this file is that job; include it or paste it:

```yaml
include:
  - remote: https://raw.githubusercontent.com/fathom-lab/styxx/main/integrations/gitlab/diffgate.gitlab-ci.yml
```

What it does, line by line: runs on merge request pipelines only; installs the released
`styxx` (floor 7.44.2, the same floor the Action pins and for the same reason — every earlier
release let `only touches <prefix>` verify against an unreadable diff); fetches the diff base so
a shallow clone still has it; writes the description to a file with `printf '%s'`, so the text
never passes through a shell expansion; runs `python -m styxx.diffgate` on that file against
`--base $CI_MERGE_REQUEST_DIFF_BASE_SHA --head $CI_COMMIT_SHA`; keeps the JSON report as a job
artifact whether it passed or failed. The job fails only on a CONTRADICTED claim (the CLI exits
1 on FAIL, 0 on PASS). Add `--strict` to the last line to fail on UNCHECKABLE claims too.

What the CLI prints in the job log when the description lies (the README demo, run locally
against a checkout the same way the job runs it):

```
FAIL  claims=5 contradicted=3 uncheckable=1 uncovered_sentences=0
never read: 0 of 5 sentences — prose outside the closed template set is listed in --out, not judged
  [CONTRADICTED:symbol_added] added lines do NOT define function 'backoff'
  [CONTRADICTED:tests_added] diff adds 1 test functions, claim says 3
  [CONTRADICTED:only_touches] paths outside 'src': ['config/settings.yml', 'tests/test_retry.py']
  [UNCHECKABLE:tests_pass] no --run command supplied; the gate does not take the agent's word for test results
```

`tests/test_gitlab_job.py` parses the job file, runs its `script` lines in a temporary
repository with the four `CI_*` variables set the way GitLab sets them (the release already
installed, so the `pip install` line is skipped), and checks a lying description fails the job
with the lies named and an honest one passes. That is the mechanics of the job, tested here; a
run on GitLab itself is not something this repository has done, and the first one that fails on
a variable name is worth an issue.

Everything the Action's README says about scope applies: the closed template set, "tests pass"
UNCHECKABLE without `--evidence`, a path the diff does not show UNCHECKABLE and never accused
(EXTERNAL-1), the open false-VERIFIED class #101. Rung: this is CI, the enforcement door — the
agent's shell cannot reach the runner — with the usual caveat that a description edited after
the pipeline ran is a new description; GitLab reruns merge request pipelines on new commits, not
on description edits, so gate on push and read the description at merge time.
