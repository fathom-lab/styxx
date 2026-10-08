# diffgate on GitLab CI — the merge request description vs the merge request diff

The GitHub Action reads a PR body against a PR diff with no checkout. GitLab hands a merge
request pipeline the same two things another way: the description in
`CI_MERGE_REQUEST_DESCRIPTION`, the diff as the range between `CI_MERGE_REQUEST_DIFF_BASE_SHA`
and `CI_COMMIT_SHA` on the job's checkout. So the door is the CLI, unchanged, in one short job.
`diffgate.gitlab-ci.yml` next to this file is that job; include it or paste it:

```yaml
include:
  - remote: https://raw.githubusercontent.com/fathom-lab/styxx/main/integrations/gitlab/diffgate.gitlab-ci.yml
```

What it does, line by line: runs on merge request pipelines only; installs the released
`styxx` (floor 7.44.2, the same floor the Action pins and for the same reason — every earlier
release let `only touches <prefix>` verify against an unreadable diff); fetches the diff base so
a shallow clone still has it; writes the description to a file with `printf '%s'`, so the text
never passes through a shell expansion; removes any `diffgate.json` an earlier run left in the
build directory; runs `python -m styxx.diffgate` on that file against
`--base $CI_MERGE_REQUEST_DIFF_BASE_SHA --head $CI_COMMIT_SHA`, writing `diffgate.json`; decides
the job's exit from `STYXX_SOFT_FAIL`, the CLI's exit code (1 on FAIL, 0 on PASS) and that
report; keeps the JSON report as a job artifact whether the job passed or failed.

## It reports by default; blocking is opt-in

The job reports by default, as the GitHub Action does, and the gate's verdicts never fail the
job. The gate's verdict, its claim counts and each CONTRADICTED or UNCHECKABLE claim go to the
job log; every claim's verdict, VERIFIED included, goes to `diffgate.json`. When the gate says
FAIL, the job prints a warning after the gate's output that the job passes anyway, and how to
make it fail. GitLab then shows the job as passed: the Action writes annotations and a job
summary, but this job reports only in its log and the `diffgate.json` artifact, so the merge
request's pipeline status does not show what the gate found. Only STYXX_SOFT_FAIL set to "false"
blocks: the job then fails on a CONTRADICTED claim. Set it wherever GitLab reads CI/CD variables
(the project's or group's settings, the pipeline's top-level `variables:`, or the job's own). In
the settings, mind **Protect variable**: a variable marked Protected reaches a merge request
pipeline only when its source and target branches are both protected branches of the same project
and a maintainer has selected "Allow merge request pipelines to access protected variables and
runners", and never from a fork (GitLab's docs, merge request pipelines, "Control access to
protected variables and runners"). On an ordinary merge request from a feature branch it never
arrives, the job reads it as unset, and it reports. So leave Protect cleared or set the variable
in .gitlab-ci.yml, as here:

```yaml
include:
  - remote: https://raw.githubusercontent.com/fathom-lab/styxx/main/integrations/gitlab/diffgate.gitlab-ci.yml

diffgate:
  variables:
    STYXX_SOFT_FAIL: "false"
```

or, in a pasted copy, uncomment the job's `# STYXX_SOFT_FAIL: "false"` line. The job does not set
the variable itself: in GitLab a job's own `variables:` outrank the pipeline's top-level ones, so
a default written into the job would silently override a pipeline that sets it. Unset reads as
`"true"`.

**Every run that reaches the gate names its mode.** After the gate's output the job prints one
line saying which mode it ran in, on a PASS run too: `styxx diffgate mode: STYXX_SOFT_FAIL is
"false", so a FAIL verdict fails this job.` when it blocks, and `... is not set, so this job
reports and the gate's verdicts do not fail it.` when the variable did not arrive, followed by the
Protected caveat above. A job that should block and says "not set" is missing its variable. A run
that stops before the gate, on a failed install or fetch, prints no mode line and fails the job in
either mode.

**How the value is read**, the way the Action reads its `soft-fail` input: compared
case-insensitively with surrounding whitespace stripped, so `"false"` and `"False "` block and
`"true"`, `" true"` and `"TRUE"` report. Every other value, the empty string included, reports
as `"true"` does, and the job prints a warning naming the value it did not recognise.

**`--strict`.** Add `--strict` to the `python -m styxx.diffgate` line to make an UNCHECKABLE
claim a failure too; the job fails on it only when `STYXX_SOFT_FAIL` is `"false"`.

**Errors are not verdicts.** In either mode the job fails on an install error, on a failed fetch
of the diff base, and on a gate whose report does not match its exit: an exit 1 without a FAIL
report (a head the checkout does not have, say), an exit 0 without a PASS report, or any other
exit. The released CLI writes its report before it returns, so only an error or a CLI that is
not the released one gets there. Reporting swallows the gate's verdicts, not its errors.

**Why it does not block by default.** No kind of accusation the gate still makes has been
measured clearing the 0.95 precision floor the lab set for accusing, and a check that fails
someone else's merge request by default should already have cleared it. Path claims are no
longer accused: EXTERNAL-1 measured 0.23 against a preregistered 0.95 floor
([RESULT](../../papers/closed-model-frontier/RESULT_external1_the_gate_fails_in_the_wild_2026_08_31.md),
receipt `external1_adjudication.json`) and a repaired path accuser measured 0.16 on a held-out
sample ([V14](../../papers/closed-model-frontier/RESULT_v14_naming_the_defects_did_not_save_it_2026_09_01.md),
receipt `v14_adjudication.json`). Of the kinds that still accuse, `only_touches` is the one with
a measured precision, 0.25
([PATH-1](../../papers/closed-model-frontier/RESULT_path1_only_touches_repair_2026_09_17.md),
re-derived by [SCOPE-1](../../papers/closed-model-frontier/RESULT_scope1_ABANDONED_2026_09_18.md),
receipt `scope1_footprint.json`). The CHANGELOG entry "the GitHub Action reports by default"
gives the figures in full.

**Before this change the job failed on every CONTRADICTED claim by default.** A pipeline that
includes the job from `main` reports from its next run; to keep it blocking, set
`STYXX_SOFT_FAIL: "false"` (in .gitlab-ci.yml, or in settings with Protect cleared). A copy
pasted before the change keeps blocking until it is replaced.

## What it prints

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

By default the job then prints its mode line, which names the variable as not set and carries
the Protected caveat, and

```
WARNING: styxx diffgate found failures but STYXX_SOFT_FAIL is on, as it is by default, so the job passes; set STYXX_SOFT_FAIL to "false" to fail it.
```

and passes; with `STYXX_SOFT_FAIL` `"false"` it prints the blocking mode line and that the job
fails, and fails.

## What is tested, and what is not

`tests/test_gitlab_job.py` parses the job file, runs its `script` lines in a temporary
repository with the four `CI_*` variables set the way GitLab sets them (the release already
installed, so the `pip install` line is skipped), and checks: by default a lying description is
reported with the lies named and the job passes; with `STYXX_SOFT_FAIL` `"false"` it fails the
job; ten values of the variable are read as above, each the way the Action's reader reads it;
`--strict` fails the job on an UNCHECKABLE claim only with the opt-in; a gate that errors fails
the job in either mode, and a stale `diffgate.json` is not read as its verdict; a stand-in for the
CLI whose report does not match its exit (exit 0 with no report or a FAIL one, exit 1 with a PASS
one, exit 2) fails the job in either mode; an honest description passes; every run the module
makes, PASS included, prints the one mode line its value calls for; a description with a VERIFIED,
a CONTRADICTED and an UNCHECKABLE claim puts the last two in the log and all three in
`diffgate.json`, and this README, the job's header and the CHANGELOG entry say so; and they carry
the Protected caveat. That is the mechanics of the job, tested here under bash; a run on GitLab
itself is not something this repository has done, and a run there that fails on a variable name or
on the multi-line script step is worth an issue.

Everything the Action's README says about scope applies: the closed template set, "tests pass"
UNCHECKABLE without `--evidence`, a path the diff does not show UNCHECKABLE and never accused
(EXTERNAL-1), the open false-VERIFIED class #101. Rung: this is CI — the agent's shell cannot
reach the runner — and it enforces only with `STYXX_SOFT_FAIL` `"false"`; by default it reports.
The usual caveat holds: a description edited after the pipeline ran is a new description; GitLab
reruns merge request pipelines on new commits, not on description edits, so gate on push and read
the description at merge time.
