# SPEC — sworn action v0.1: a report-only action that mints the manifest after the turn and exits zero on every verdict

Fathom Lab · 2026-09-05 · **A spec, not a result.** Frozen in its own commit before any code.
Leg 3, item 4 of `papers/PLAN_the_next_level_2026_09_02.md`, under the plan's own label — *report-only
until the measurement prices FAILED; dogfooded only after the operator merges the workflow* — and
owed by `RESULT_harness_adapters_ship_2026_09_05.md` as "the Action that would consume the GitHub
adapter, whose README owes the fork caveat". It builds on `DESIGN_harness_adapters_2026_09_02.md`
(the two adapters it calls) and on `SPEC_sworn_output_v02_2026_09_02.md` (rule R6, the trust
ladder), and it edits neither. Files: `sworn/action.yml`, `sworn/sworn_action.py`,
`sworn/README.md`, `sworn/examples/sworn.yml`, `tests/test_sworn_action.py`. The root
`action.yml` is diffgate's and is not touched; `styxx/sworn.py`, `styxx/evidence.py` and
`styxx/harness/` are not touched. Nothing is written under `.github/`: no token in this lab can
push there, so the workflow file that would run this action is delivered as an example the
operator copies. This spec makes no numeric claim.

## Why this exists

Every manifest the lab has minted so far prints L1. `harness_pytest.py` is a script on the
author's machine; the Claude Code hooks are by construction the rung R6 calls weak; the JUnit and
GitHub adapters offer L2 only on the caller's declaration and have never been called from a job
that could honestly make it. R6's L2 is "a runner that minted after the agent's turn ended and
that the agent could not write to (a CI job on the base branch)". A pull-request job is the
shape of that sentence: the agent's turn ends when it pushes and opens the pull request; the
pull request's body and its committed documents are the turn's products; a job the base
repository configured then runs the tests, mints the manifest and verifies those documents
against it. This action is that job's body. It is report-only because the measurement that
would price a FAILED span (`DESIGN_sworn_measurement_v2_2026_09_02.md`) has not run, and a gate
without a measured precision is the class of instrument this lab has retired before.

It also narrows one unrepaired attack. A11 of `ATTACKS_sworn_v01_battery_2026_09_02.md`
(post-hoc tagging) is undetectable from bytes; a verifier that reads the pull request body *as
submitted* reads what the author committed to before the runner minted anything, which is the
narrowing the v0.2 spec's unrepaired table names.

## What the job does, in order

In a `pull_request` (or `push`) job whose workflow the operator merged:

1. The workflow checks out the pull request's **head commit** (`ref: ${{ github.event.pull_request.head.sha }}`)
   with full history (`fetch-depth: 0`), installs the project, and runs this action with a
   `command` that runs the project's tests and writes JUnit XML to the path the action exports as
   `$SWORN_JUNIT`.
2. The action reads the event from `GITHUB_EVENT_PATH` and the declarations from its inputs.
   It refuses nothing on the event's text; it decides whether it can run (below) and, if not,
   writes a `DID NOT RUN` summary and exits zero.
3. It runs `command` after the turn — the turn ended at the push — with `SWORN_JUNIT` in the
   environment, under a timeout, capturing exit status and output. The command's failure is
   recorded, never inherited: the action's exit is not the tests' exit.
4. It mints the manifest, after the command returned, by calling the adapters as library code:
   `styxx.harness.junit.mint` over the report the command wrote (when it wrote one) and
   `styxx.harness.github.mint` over the event bytes and a diff taken with git plumbing from the
   checkout (never fetched: the action opens no network connection). Both are minted at the rung
   the job declares, under the declarations the GitHub adapter requires. The two adapter manifests
   are written to the output directory as they are, and one composed manifest is written beside
   them with the layout below; documents are verified against the composed one.
5. It records into the composed manifest's `authored_sha256` the sha256 of every blob the turn
   added or modified (`git diff --name-only --diff-filter=AM` between base and head, bytes read at
   head), so a command that copies a committed file into `$SWORN_JUNIT` yields a report receipt
   that resolves MALFORMED `receipt_author_minted`, end to end.
6. It verifies every sworn document the pull request changed — bytes read at the head commit
   through `styxx.sworn.GitTree`, never from the working tree the command may have altered — and
   the pull request body as submitted, against the composed manifest, with
   `styxx.sworn.verify(raw, name=, manifest=, tree=GitTree(workspace, head), commit=head)`,
   which is what `python -m styxx.sworn verify DOC --repo . --commit HEAD --manifest M` does; it
   issues a verdict receipt per document and writes it LF-only.
7. It writes a job summary: one table row per document with the verdict, the four counts, the
   rung and the manifest's harness string; every non-HELD span listed beneath with its reason,
   receipt and offset; the receipt layout; the reasons for every document it did not verify; the
   `certifies` text of the verifier verbatim. It emits `::notice` lines only — never `::error`,
   never `::warning` — and sets outputs (`manifest`, `receipts-dir`, `out-dir`, `rung`, `verdicts`).
8. It exits zero. On every verdict, on every refusal it reports, on `DID NOT RUN`, on a command
   that failed or timed out, on a report that did not parse. The one nonzero exit is two, for a
   usage error before anything ran: no event path, or an empty `command`.

## The manifest the runner mints, receipt by receipt

A document is verified against one manifest, and `rN` resolves by id inside that manifest and
nowhere else (`styxx.sworn._resolve`). Two adapters each fix their own layout, and their `r1` are
different bytes; verifying a document against both would make one id mean two things and print a
FAILED row for whichever the author did not mean. So the action composes **one** manifest whose
layout is fixed here, per action version, and never renumbered:

| id | bytes | kind | complete | from |
|---|---|---|---|---|
| r1 | the passed count as ASCII digits | `test_report` | true | JUnit adapter r1, id for id — withheld with r2 when the reader parsed no source |
| r2 | the failures count as ASCII digits (harness errors kept apart) | `test_report` | true | JUnit adapter r2, id for id — withheld with r1 |
| r3 | the report bytes, whole | `test_report` | true | JUnit adapter r3, id for id |
| r4 | `load_evidence`'s object as RFC 8785 canonical JSON, one trailing LF — `r4#/totals/errors`, `r4#/outcome`, `r4#/sources/0/producer_guess` are leaves | `test_report` | true | JUnit adapter r4, id for id |
| r5 | the diff between base and head as git printed it | `http_fetch` | true (git printed it whole) | GitHub adapter r1 — omitted when the diff could not be taken |
| r6 | the base sha as ASCII | `harness_note` | true | GitHub adapter r2 |
| r7 | the head sha as ASCII | `harness_note` | true | GitHub adapter r3 |
| r8 | the event name as ASCII | `harness_note` | true | GitHub adapter r4 |
| r9 | the event payload bytes, whole — `r9#/pull_request/number`, `r9#/pull_request/head/repo/fork` are leaves | `harness_note` | true | GitHub adapter r5 |

Every composed entry is the adapter's entry with only its `id` changed: the same bytes, sha256,
kind, completeness, capture time and harness note. The two adapter manifests are written beside
the composed one so the derivation can be checked by anyone, and a test pins it. When the command
wrote no report, r1 to r4 are absent and r5 to r9 keep their ids; when the report did not parse,
r1 and r2 are absent as the JUnit adapter withholds them and r3, r4 keep their ids; when the diff
could not be taken, r5 is absent and r6 to r9 keep their ids. **An absence never renumbers.** A
layout that renumbered on absence would let a document written against it swear to the wrong
bytes with nothing MALFORMED to show for it, which is the A12-shaped attack.

**How the numbering lines up with what the author cited.** The author writes the document before
the runner runs, and cites ids from this table — `r1` for the passed count, `r4#/outcome` for the
reader's verdict word, `r6` for the base sha, `r9#/pull_request/number` — exactly as the lab's own
RESULTs cite `r1`, `r2`, `r5#/passed` against `harness_pytest.py`'s script-fixed positions. The
runner locates nothing and matches nothing: an id the runner did not mint resolves UNRESOLVED
`manifest_id_missing`, the verifier saying it could not see, never an accusation. The README
carries this table under a heading that says what the author may cite, and says that the ids
above r9 exist in no version of this action.

The composed manifest's `harness` string names this action and its ref, the rung as declared
and the declarations it rested on, the fork status, the composition rule, and both adapters'
harness strings verbatim — so the fork caveat and the L1 weakness sentence, which the adapters
print, are printed on every row of every summary this action writes. Its `turn` is
`<repository>#<number>@<head sha>` for a pull request and `<repository>@<head sha>` for a push.

## The rung, declared by the workflow, and the rule for forks

The rung is declared, never detected. The action's inputs are the declarations: `rung` (L1 or
L2), `after-turn-on-base` (the sentence L2 rests on: this job ran after the agent's turn ended, on
a runner and from a workflow file the agent could not write to) and `base-pinned-workflow` (the
workflow file came from the base branch). They reach the adapters unchanged, and the adapters'
refusals are the action's rules:

| event | head repository | declared | minted | why |
|---|---|---|---|---|
| `pull_request` | the base repository | L2 with `after-turn-on-base` | L2 | R6's sentence as declared |
| `pull_request` | a fork, or absent from the payload | L2 without `base-pinned-workflow` | **L1** | the fork rule below |
| `pull_request` | a fork | L2 with `base-pinned-workflow` | L2 | the caller asserted what the bytes cannot show; the summary prints that it did |
| `push` | — | L2 with `after-turn-on-base` | L2 | as the adapter accepts it |
| any | any | L2 without `after-turn-on-base` | **L1** | the declaration L2 rests on was not made |
| any | any | L1 | L1 | as declared |
| `pull_request_target` | any | any | **DID NOT RUN** | this action executes the head's `command`; under that event it would do so with the base's token, which the design forbids |

The action lowers a rung and prints why; it never raises one. The reason is in the summary, in
`run.json`, and in the composed manifest's harness string.

**The fork rule, as the README states it, verbatim:** *On a pull request from a fork, the minting
job is the claimant's. The `pull_request` event runs the workflow file, the action ref, the
command and the code from the pull request's head — every one of them under the pull request
author's control — with a read-only token. The runner did mint after the turn, but "a runner the
agent could not write to" is not true of a job the agent configured, so the manifest declares
rung L1 (weak), never L2, and the job summary says why. L2 is declared only when the head
repository is the base repository, or when the workflow declares `base-pinned-workflow` because
it is one. Do not switch the trigger to `pull_request_target` with a checkout of the head to
change this: that hands the base repository's token to the claimant's code.*

Why the rule is that and not softer: R6 defines L2 by who could write to the minter. On a fork
`pull_request` the claimant wrote the job. A manifest that printed L2 there would print a rung
its minter cannot support, which is the same reason a `sworn/manifest/0.1` file reports
`undeclared` rather than being read as L2. The adapter refuses to print it; the action, which
is the adapter's caller, declares L1 instead and says so, because the alternative — refusing to
mint at all — would leave the documents unverified and the reader with less.

**What L2 means here, and does not.** L2 says the agent did not write the manifest: a job the
base repository configured minted it after the turn. It does not say the numbers are true. The
checked-out code is the agent's and it ran inside the runner; a `conftest.py` in the pull
request can print any count, and the report receipt will carry it faithfully. That is the A7
boundary — the harness lies — which R6 prices at L3 and this version does not reach.
`authored_sha256` catches only the trivial laundering of a committed file into a receipt. And on
a same-repository `pull_request` the workflow file still comes from the head: L2 on such a job
rests on the operator's declaration that the agent cannot write to `.github/workflows` on the
base (branch protection, review), which the action prints as a declaration and checks nowhere.

## The documents

Verified, in this order, each as an inline document at the head commit:

- the pull request body as the event file carries it — UTF-8 bytes, line endings as received,
  no normalisation — under the name `pull_request_body.md`, when it carries the bytes `<sworn`;
- every path the turn added or modified whose name ends in `.md` and whose blob at the head
  commit carries the bytes `<sworn`, bytes read with `GitTree(workspace, head).blob(path)`.

Not verified, and said so in the summary with the path and the reason:

- a changed `*.sworn.json` sidecar: its embedded manifest is the one it was sworn against, and
  `styxx.sworn.verify` refuses a supplied manifest that disagrees with an embedded one; the
  repository's own tests (`tests/test_sworn_dogfood.py` in this tree) re-derive committed
  sidecars at their commits, and that is where a sidecar is checked;
- a changed `.md` that carries no `<sworn` bytes: there is nothing to resolve, and a row of
  UNSWORN for every touched README would bury the rows that matter;
- when the base commit is not in the checkout (a shallow clone, a `push` whose `before` is the
  null sha): changed-file discovery is unavailable, only the body is verified, `authored_sha256`
  stays empty, r5 is absent, and the summary says to set `fetch-depth: 0`.

The head commit must be present (`git cat-file -t`); when it is not — a checkout of the merge
ref, a `pull_request_target` checkout of the base — the action reads DID NOT RUN and says which
`ref:` to check out. `GITHUB_SHA` is never used: on a `pull_request` event it names a merge
commit the author could not have cited.

## What the action refuses, and what reads UNRESOLVED

Nothing on the event's text is a refusal. A document that cites an id the runner did not mint
reads UNRESOLVED `manifest_id_missing`; one that cites `path:` at a file absent from the head
commit reads UNRESOLVED `path_absent`; one whose span is lexically broken reads MALFORMED with
the verifier's reason. All of these are rows in the table, and all exit zero. A `SystemExit`
from the verifier — which inline verification does not raise for any document bytes, and which
is caught anyway — becomes a `REFUSED` row with the message, and exits zero. `DID NOT RUN` is
a summary and exit zero. Exit two is reserved for the job being misconfigured before anything
ran, which is a usage error and not a verdict, exactly as the adapters' exit two.

## The rules, each with its attack

**R1 — Exit zero on every verdict.** No input turns a document verdict into a nonzero exit; the
command's exit status is recorded and never inherited. *Attack:* a workflow author adds
`strict: true` and the action becomes a gate with an unmeasured precision. *Answer:* there is no
such input; the README says the action is report-only until the measurement prices FAILED, in
those words; a test greps the action's source for `::error` and finds none.

**R2 — Event text never touches a shell.** The body and every path go from the event file and
from git plumbing into Python; the only string a shell sees is `command`, which is the
workflow's own configuration and reaches Python through an environment variable, never an
interpolation into a `run:` line. *Attack:* a body containing `$(...)` or a path with spaces
and quotes. *Answer:* the root action's discipline, copied; a test feeds both.

**R3 — No network.** The diff is taken from the checkout with git; the event is a file; the
action imports no `urllib`, `http`, `socket` or `requests`. *Attack:* a fetch that fails or is
served an interstitial, and a verdict minted over it. *Answer:* a test monkeypatches
`urllib.request.urlopen` to raise and the run is unchanged; a test walks the module's imports.

**R4 — Bytes at the commit, never the working tree.** Documents and changed blobs are read at
the head sha through `GitTree`; the command may have rewritten the working tree and its writes
must not enter a verdict. *Attack:* a test that edits a document in place before the action reads
it. *Answer:* a test whose command overwrites the document on disk asserts the verdict is over
the committed bytes.

**R5 — The layout is fixed and absences never renumber.** *Attack:* A12-shaped — a report that
did not parse shifts r3 into r1 and a document swears to the wrong bytes. *Answer:* a test where
the command writes no report asserts r5 to r9 keep their ids and r1 to r4 are absent; a test with
an unparseable report asserts r3 and r4 keep their ids.

**R6 — The rung is the workflow's declaration, lowered with a reason, never raised.** *Attack:*
a fork pull request minted at L2. *Answer:* the GitHub adapter refuses it; the action declares L1
and prints the fork rule; a test with a fork event asserts the rung and the sentence.

**R7 — The verifier and the adapters are untouched.** The action is a caller; nothing under
`styxx/` changes, the purity boundary test stays as it is, and the action lives outside the
package (it is not shipped in the wheel, as `diffgate_action.py` is not). *Attack:* a
convenience helper added to `styxx.sworn` for the action's sake. *Answer:* the diff of this leg
touches no file under `styxx/`.

**R8 — The action commits nothing.** Manifests and receipts are run artifacts in the output
directory; the README shows `actions/upload-artifact`. A receipt cited by a committed document is
history, and a job that rewrote one would be the defect this lab has already paid for. *Attack:*
a `commit-receipts` input. *Answer:* none exists.

**R9 — Every file the action writes is LF-only.** Manifests through `Manifest.write`, receipts
and `run.json` through `styxx.sworn._write_json_lf`, the summary through an explicit
`newline="\n"`. *Attack:* a Windows runner CRLFs a receipt and it hashes differently per
platform. *Answer:* a test reads every output and asserts no CR byte.

## The job summary and the outputs

The summary opens with the action's name and the report-only sentence; then the turn line
(event, repository, number, head, base, fork status); then the rung line — *manifest minted at
rung L2 after the turn by \<harness\>* or *rung L1 — \<reason\>*; then the table:

| document | verdict | held | failed | unresolved | malformed | rung | harness |

with one row per document, the verdict as `styxx.sworn._headline` prints it, and the composed
manifest's harness string in the last cell of every row; beneath it, for every document, one
line per non-HELD span (`VERDICT reason receipt @offset`); then the receipt layout table; then the
list of what was not verified and why; then the reproduction line (`python -m styxx.sworn verify
<out-dir>/documents/<name> --repo . --commit <head> --manifest <out-dir>/sworn.manifest.json`);
then the verifier's `certifies` text verbatim. `run.json` carries the same as JSON: the event,
shas, fork status, declared and minted rung with its reason, the command, its exit status and
whether it timed out, the report path as given to the adapter, the manifest digests, the
authored count, whether discovery was available, one entry per document with its verdict, counts
and receipt path, and one entry per skipped path with its reason.

## What the action does not do, by construction

It does not gate, does not fail a job on a verdict, does not annotate with `::error` or
`::warning`, does not fetch, does not sign, does not verify a signature, does not detect a rung,
does not raise a rung, does not read the working tree for a verdict, does not verify sidecars,
does not commit or push, does not renumber a receipt, does not touch the root action, and does
not write under `.github/`.

## Tests this spec commits to

`tests/test_sworn_action.py` drives `sworn_action.main()` end to end in a temporary git
repository with a fake event file, a command that writes a canned JUnit report to `$SWORN_JUNIT`,
and canned sworn documents committed at the head, with no network: a HELD document, a FAILED
document and an UNRESOLVED document each produce their row and exit zero; the composed manifest's
r1 to r4 equal the JUnit adapter's r1 to r4 and its r5 to r9 equal the GitHub adapter's r1 to r5
with only the id changed; a fork event mints L1 and prints the fork rule; L2 without
`after-turn-on-base` mints L1 with the reason; a `push` event uses `after`; a
`pull_request_target` event and an unknown event read DID NOT RUN; a missing head reads DID NOT
RUN; a command that writes no report leaves r1 to r4 absent and r5 to r9 in place and the report
span UNRESOLVED; an unparseable report leaves r3 and r4 in place; a failing command and a timing
out command exit zero; the body is verified as `pull_request_body.md` with its line endings; a
sidecar is skipped and named; a command that copies a committed file into `$SWORN_JUNIT` reads
MALFORMED `receipt_author_minted`; a command that rewrites a document on disk changes no verdict;
a shallow checkout prints the discovery line; every output is LF-only; the source carries no
`::error`, no `::warning` and no network import; `action.yml` has no `strict` or `soft-fail`
input, passes every input through `env`, and runs `sworn_action.py`; the README opens with the
report-only sentence and carries the fork rule and the layout table; `GITHUB_OUTPUT` carries the
outputs; exit two for an empty command.

## What ships with v0.1

`sworn/action.yml`, `sworn/sworn_action.py`, `sworn/README.md`, `sworn/examples/sworn.yml` (the
workflow the operator copies), `tests/test_sworn_action.py`; a sample run over a fixture
repository committed under `papers/sworn/sworn_action_sample.*` (the script, the composed
manifest, the receipts, the summary and `run.json`), minted with the clock pinned so the sample
is reproducible; a manifest the JUnit adapter minted over the test run; a short sworn RESULT; a
CHANGELOG block. No file under `.github/` and no file under `styxx/`.

## What this spec does not say

That the action has run on GitHub: it cannot until the operator merges the workflow, and no
manifest printing L2 exists until then. That L2 means the tests passed: it means the agent did
not write the manifest. That a document HELD here is true: the verifier's `certifies` text says
what is certified and it is not that. That the fork rule is a repair of anything: it is the
honest rung printed where a higher one cannot be. That the composed layout is the adapters'
layout: r5 to r9 are shifted, on purpose, and the table says so. That the format has been
measured, or that FAILED has a price.

## Owed after v0.1, recorded as owed

1. The operator's merge of `sworn/examples/sworn.yml` into `.github/workflows/`, and the run on
   a runner the author cannot write to that would let a manifest honestly print L2.
2. An adversarial pass against `sworn/sworn_action.py` as committed, with a dated ERRATA here.
3. A way for a document to name which manifest it swears against, so the composition rule can
   be retired and the adapters' own layouts cited directly.
4. A `documents:` input for documents present at head but untouched by the turn.
5. The adversarial pass against `styxx/harness/` that the adapters' RESULT owes; this action
   inherits every defect found there.
6. The measurement, unchanged.

---

*The runner mints after the turn, the author cites ids from a table published before the run,
and the verifier resolves what it can and says where it could not see. Nothing here decides
whether the tests were honest; it decides only that the agent did not write the manifest, and it
prints, on every row, the rung that sentence reaches.*
