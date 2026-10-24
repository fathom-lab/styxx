# diffgate on GitLab CI — the merge request description vs the merge request diff

The GitHub Action reads a PR body against a PR diff with no checkout. GitLab hands a merge
request pipeline the same two things another way: the description in
`CI_MERGE_REQUEST_DESCRIPTION` (at most its leading 2,700 characters: see "A description longer
than 2,700 characters" below), the diff as the range between `CI_MERGE_REQUEST_DIFF_BASE_SHA`
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
never passes through a shell expansion; when GitLab says it cut the description
(`CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED` `"true"`), cuts that file back to its last whole
sentence, keeps the text GitLab handed it in `mr_description.handed.md`, and says so; removes any
`diffgate.json` an earlier run left in the build directory; runs `python -m styxx.diffgate` on
that file against `--base $CI_MERGE_REQUEST_DIFF_BASE_SHA --head $CI_COMMIT_SHA`, writing
`diffgate.json` and keeping what it prints in `diffgate.log`; when the gate FAILs over a cut, runs
it again over `mr_description.handed.md`, writing `diffgate.handed.json`; prints the gate's
output, with CUT where the gate prints PASS over a description GitLab cut; decides the job's exit
from `STYXX_SOFT_FAIL`, the CLI's exit code (1 on FAIL, 0 on PASS) and that report; keeps the JSON
report as a job artifact whether the job passed or failed.

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

**`--strict`.** Add `--strict` to the `python -m styxx.diffgate` line that writes
`diffgate.json` to make an UNCHECKABLE claim a failure too; the job fails on it only when
`STYXX_SOFT_FAIL` is `"false"`. The gate's second run over a cut needs no flag: the job reads only
which claims it withholds or does not read.

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

## A description longer than 2,700 characters

GitLab stores only the leading 2,700 characters of a longer merge request description in
`CI_MERGE_REQUEST_DESCRIPTION`, and sets `CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED` to `"true"`
when it did (GitLab's docs, "Predefined CI/CD variables reference", both entries). The job reads
only that variable, so for a longer description it is handed a prefix. Until #209 it gated the
prefix as if it were the description: a claim past the cut was never read and the job printed
PASS without a word about it, and the sentence the cut broke was gated as written ("Only touches
src/ and tests/." cut after "src/" was read as "Only touches src/", CONTRADICTED). Now, when the
flag is `"true"` (compared case-insensitively, surrounding white space stripped):

- the job cuts what it was handed back to its last whole sentence and the white space after it,
  where the gate itself splits sentences (`.`, `!` or `?` before white space, or a line break),
  so every sentence the gate's templates read is a whole sentence of the description and a
  sentence the cut broke is never read by them;
- it prints one line before the gate's output naming the cut, the characters GitLab handed it
  and the characters the gate reads;
- it prints CUT where the gate prints PASS, and after the mode line one more line saying it does
  not call the description a pass, since a claim past the cut is never read;
- it adds `description_cut` to `diffgate.json`: the flag's value, `characters_handed`,
  `characters_gated` and a `why`. The record's `verdict` stays the gate's, over the prefix;
- a FAIL over a cut is the gate's FAIL over that prefix. Two of the gate's rules read the whole
  text it is handed, not one sentence: PATH-2a's withholding (#121), which reads a sentence as
  both readers end one and some characters anywhere in the text, and the one-`styxx`-block rule.
  So the text the cut drops can change how the gate reads a whole sentence before it, and when the
  gate FAILs over a cut the job runs it again over the text GitLab handed it. A CONTRADICTED claim
  of the cut that this run withholds or does not read is held back: named in the log and under
  `held_back` in `description_cut`, with that run's reason, and the job does not fail on it. A
  FAIL whose every CONTRADICTED claim is held back does not fail the job, in either mode,
  `--strict` too; any other FAIL over a cut reports or blocks as any FAIL does;
- a cut never fails the job by itself, in either mode.

Over the issue's case, a false "Added 3 tests." past character 2,700, the log reads (the `...`
stands for the gate's "no diff-shaped claims" block, its instrument line and the mode line):

```
styxx diffgate: CUT. CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED is 'true': GitLab handed this job only the leading 2700 characters of the merge request description. The gate reads the leading 2639, up to the last whole sentence, and never the rest: a claim past that point is never read and a sentence the cut broke is not judged, so this job prints CUT where the gate prints PASS.
CUT   claims=0 contradicted=0 uncheckable=0 uncovered_sentences=29
never read: 29 of 29 sentences — prose outside the closed template set is listed in --out, not judged
...
styxx diffgate: CUT. Nothing in the leading 2639 characters the gate read failed it, and the rest of the description was never read, so this job does not call the description a pass. diffgate.json keeps the gate's verdict over that prefix and names the cut under "description_cut".
```

What it cannot do: read past the cut, or say how long the description was. A claim the job
should gate belongs in the leading 2,700 characters; a `styxx` declaration block at the top of
the description is one way. The cut back to a whole sentence can also drop a sentence that ended
on the 2,700th character, since the job cannot see what followed it; the line names how many
characters were read. Text past GitLab's cut never reaches the job, so a claim the gate would
withhold, or not read, over the whole description for text there (PATH-2a, a second `styxx`
block) can still be CONTRADICTED over the cut, as it was over GitLab's prefix before #209, and
with `"false"` the job fails on it. GitLab's job-token docs list
`GET /projects/:id/merge_requests/:merge_request_iid` among the endpoints a job token reaches,
which would read the whole description. This job does not call it: the job has not run on GitLab
itself, and by default a job token reads another project's merge requests only when that
project's allowlist admits the job's project. That is left for a later change.

## What is tested, and what is not

`tests/test_gitlab_job.py` parses the job file, runs its `script` lines in a temporary repository
with the four `CI_*` variables set the way GitLab sets them (the release already installed, so the
`pip install` line is skipped), and checks: by default a lying description is reported with the
lies named and the job passes; with `STYXX_SOFT_FAIL` `"false"` it fails the job; ten values of the
variable are read as above, each the way the Action's reader reads it; `--strict` fails the job on
an UNCHECKABLE claim only with the opt-in; a gate that errors fails the job in either mode, and a
stale `diffgate.json` is not read as its verdict; a stand-in for the CLI whose report does not
match its exit (exit 0 with no report or a FAIL one, exit 1 with a PASS one, exit 2) fails the job
in either mode; an honest description passes; every run the module makes, PASS included, prints the
one mode line its value calls for; a description with a VERIFIED, a CONTRADICTED and an UNCHECKABLE
claim puts the last two in the log and all three in `diffgate.json`, and this README, the job's
header and the CHANGELOG entry say so; and they carry the Protected caveat. For #209 the module
hands the job `description[:2700]` with `CI_MERGE_REQUEST_DESCRIPTION_IS_TRUNCATED` `"true"`, as
GitLab does, and the whole description as the control: a claim past the cut is never read and the
run prints CUT, never PASS, names the cut in the log and in `diffgate.json`, and passes in either
mode; a scope sentence the cut broke is not judged; a false whole sentence before the cut still
fails, reported by default and blocking with `"false"`; the white space after the last whole
sentence is kept, so a claim PATH-2a withholds for a character of that white space it withholds
over the cut too; a claim the gate withholds, or does not read, over the text GitLab handed the job
is held back and not failed on, in either mode; a reason to withhold past GitLab's cut never
reaches the job, and the claim fails it with `"false"`; the gate's second run is read as the other is,
a report that does not match its exit failing the job; and over 847 prefixes of 25 descriptions
built to put PATH-2a's or the one-block rule's reason in the text the cut drops, the job's own two
steps never stand on, or fail on, a CONTRADICTED claim that neither the whole description nor the
text GitLab handed gives, where the cut at the start of the white space with nothing held back does
both, and so does the cut after it with nothing held back. The cut step, run on every prefix of
four descriptions, leaves only whole sentences of the description by the gate's own split, as
written and as the CLI reads the file, which the module reads from `styxx/diffgate.py` with the
form of PATH-2a's reason; without the flag it leaves the text alone. The module hands bash the
script as a file. That is the mechanics of the job, tested here under bash; a run on GitLab itself
is not something this repository has done, and a run there that fails on a variable name or on the
multi-line script step is worth an issue.

Everything the Action's README says about scope applies: the closed template set, "tests pass"
UNCHECKABLE without `--evidence`, a path the diff does not show UNCHECKABLE and never accused
(EXTERNAL-1), the open false-VERIFIED class #101. Rung: this is CI — the agent's shell cannot
reach the runner — and it enforces only with `STYXX_SOFT_FAIL` `"false"`; by default it reports.
The usual caveat holds: a description edited after the pipeline ran is a new description; GitLab
reruns merge request pipelines on new commits, not on description edits, so gate on push and read
the description at merge time.
