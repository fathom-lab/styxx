# styxx sworn — the action

**Report-only until the measurement prices FAILED.** This action exits 0 on every verdict; it
writes a job summary and never fails a job, annotates nothing as an error, and has no `strict` and
no `soft-fail` input. **On a pull request from a fork, the minting job is the claimant's**, and
the manifest this action mints there declares rung L1, never L2 (the rule is stated in full below).

Built to `papers/sworn/SPEC_sworn_action_v01_2026_09_05.md`, frozen before the code. It is the
consumer of the JUnit and GitHub adapters in `styxx.harness`
(`papers/sworn/DESIGN_harness_adapters_2026_09_02.md`) and it edits nothing under `styxx/`. The
root `action.yml` of this repository is diffgate's and is unrelated.

## What it does

In a `pull_request` job, after the agent's turn has ended at the push:

1. runs your test command with `$SWORN_JUNIT` exported, so the tests write a JUnit report;
2. mints, after the command returned, the JUnit adapter's manifest over that report and the
   GitHub adapter's manifest over the event payload and the diff between base and head (taken
   from the checkout with git; no network is opened), both at the rung the workflow declares;
3. composes one manifest from them (layout below) and records the sha256 of every blob the pull
   request added or modified into `authored_sha256`, so a command that copies a committed file
   into `$SWORN_JUNIT` yields a receipt that reads MALFORMED `receipt_author_minted`;
4. verifies the pull request body as submitted and every changed `.md` that carries a `<sworn`
   tag — bytes read at the head commit, never from the working tree — against that manifest, and
   writes one verdict receipt per document;
5. writes the job summary: one row per document with the verdict, the four counts, the rung and
   the manifest's harness string; every non-HELD span with its reason; what was not verified and
   why; and exits 0.

## The workflow the operator merges

Nothing under `.github/` is written by the lab; copy `sworn/examples/sworn.yml` into
`.github/workflows/` yourself. Its essentials:

```yaml
on:
  pull_request:
    types: [opened, edited, synchronize, reopened]   # edited: the body is a document
permissions:
  contents: read
jobs:
  sworn:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha }}   # the commit the author could cite
          fetch-depth: 0                                    # the base must be present too
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: python -m pip install -e ".[test]"
      - id: sworn
        uses: fathom-lab/styxx/sworn@<pinned commit>
        with:
          command: python -m pytest tests -q -p no:cacheprovider --junitxml "$SWORN_JUNIT"
      - uses: actions/upload-artifact@v4
        with:
          name: sworn-${{ github.event.pull_request.number }}-${{ github.event.pull_request.head.sha }}
          path: ${{ steps.sworn.outputs.out-dir }}
```

`GITHUB_SHA` on a `pull_request` event is a merge commit the author could not have named; the
action verifies at `pull_request.head.sha` and reads DID NOT RUN when that commit is not in the
checkout. Pin the action to a commit: `styxx.sworn` is not in any release, and the plan's
cold-start rule is *clone at a pinned commit, not pip install*. By default the action installs
styxx from its own checkout; set `styxx-source: ""` when the job already installed it (this
repository's own example does).

## What the author may cite

The author writes the document before the run and cites ids from this table. The runner locates
nothing: `rN` resolves by id inside the manifest, and an id the runner did not mint reads
UNRESOLVED `manifest_id_missing` — the verifier saying it could not see, never an accusation.
The layout is fixed per action version; **an absence never renumbers**; ids above r9 exist in no
version of this action.

| id | bytes | kind | from |
|---|---|---|---|
| `r1` | the passed count as ASCII digits | `test_report` | JUnit adapter r1, id for id — withheld with r2 when the reader parsed no source |
| `r2` | the failures count as ASCII digits (harness errors kept apart in `r4#/totals/errors`) | `test_report` | JUnit adapter r2, id for id |
| `r3` | the report bytes, whole | `test_report` | JUnit adapter r3, id for id |
| `r4` | `styxx.evidence.load_evidence`'s object as RFC 8785 canonical JSON, one trailing LF; `r4#/totals/errors`, `r4#/outcome`, `r4#/sources/0/producer_guess` are leaves | `test_report` | JUnit adapter r4, id for id |
| `r5` | the diff between base and head as git printed it | `http_fetch` | GitHub adapter r1 — absent when the diff could not be taken |
| `r6` | the base sha as ASCII | `harness_note` | GitHub adapter r2 |
| `r7` | the head sha as ASCII | `harness_note` | GitHub adapter r3 |
| `r8` | the event name as ASCII | `harness_note` | GitHub adapter r4 |
| `r9` | the event payload bytes, whole; `r9#/pull_request/number`, `r9#/pull_request/head/repo/fork` are leaves | `harness_note` | GitHub adapter r5 |

Every composed entry is the adapter's entry with only its `id` changed; the two adapter
manifests are written beside the composed one (`junit.manifest.json`, `github.manifest.json`) so
the derivation can be checked. Examples, written before the run:

```
<sworn r="r1" k="numeric">The runner resolved 139 passed testcases from the report.</sworn>
<sworn r="r2" k="numeric">It resolved 0 failures.</sworn>
<sworn r="r4#/outcome" k="quote">The reader's outcome over the report reads `PASSED`.</sworn>
<sworn r="r5" k="absent">The diff carries no `print(` left behind.</sworn>
<sworn r="r9#/pull_request/number" k="numeric">This is pull request 61.</sworn>
```

The report path enters `r4` as given to the `junit` input (default `<out-dir>/junit.xml`, an
absolute path under the runner's temp directory); give a workspace-relative path there to keep
machine paths out of the receipt.

## The rung: what L2 means here, and does not

The rung is declared by the workflow (`rung`, `after-turn-on-base`, `base-pinned-workflow`) and
handed to the adapters unchanged; the action detects nothing, lowers with a printed reason where
the GitHub adapter would refuse L2, and never raises. **L2 says the agent did not write the
manifest**: a job the base repository configured minted it after the turn. It does not say the
tests passed or that the counts are true — the checked-out code is the agent's and it ran inside
the runner; a `conftest.py` in the pull request can print any count, and the report receipt will
carry it faithfully. That is the boundary rule R6 prices at L3, which this version does not
reach. `authored_sha256` catches only the trivial laundering of a committed file into a receipt.
On a same-repository `pull_request` the workflow file still comes from the head, so L2 there
rests on your declaration that the agent cannot push to `.github/workflows` on the base (branch
protection, review) — printed as a declaration, checked nowhere.

## Pull requests from forks

On a pull request from a fork, the minting job is the claimant's. The `pull_request` event runs
the workflow file, the action ref, the command and the code from the pull request's head — every
one of them under the pull request author's control — with a read-only token. The runner did
mint after the turn, but "a runner the agent could not write to" is not true of a job the agent
configured, so the manifest declares rung L1 (weak), never L2, and the job summary says why. L2
is declared only when the head repository is the base repository, or when the workflow declares
`base-pinned-workflow` because it is one. Do not switch the trigger to `pull_request_target` with
a checkout of the head to change this: that hands the base repository's token to the claimant's
code.

The action reads DID NOT RUN under `pull_request_target` for that reason: it executes the head's
`command`, and under that event it would do so with the base's token.

## What is not verified, and said so

- A changed `*.sworn.json` sidecar: its embedded manifest is the one it was sworn against and
  the verifier refuses a supplied manifest that disagrees; committed sidecars are re-derived at
  their commits by the repository's own tests.
- A changed `.md` without a `<sworn` tag: nothing to resolve.
- Anything when the base commit is not in the checkout (a shallow clone): only the body is
  verified, `authored_sha256` stays empty, `r5` is absent, and the summary says to set
  `fetch-depth: 0`.

## Nothing is committed

Manifests, receipts, the documents as verified, `run.json` and `summary.md` are run artifacts
under `out-dir`; upload them. A receipt cited by a committed document is history, and no input
exists that would write one into the tree.

## Inputs and outputs

Inputs: `command` (required), `rung`, `after-turn-on-base`, `base-pinned-workflow`, `junit`,
`out-dir`, `timeout-minutes`, `python-version`, `styxx-source` — see `action.yml`. Every input
reaches Python through the environment; none is interpolated into a shell line. Outputs:
`manifest`, `receipts-dir`, `out-dir`, `rung`, `verdicts` (JSON object, document name to verdict).

Reproduce a verdict from the artifacts:
`python -m styxx.sworn verify <out-dir>/documents/<name> --repo . --commit <head sha> --manifest <out-dir>/sworn.manifest.json`.

## What this does not say

That it has run on GitHub: it cannot until the operator merges the workflow, and no manifest
printing L2 exists until then. That a HELD document is true — the receipt's `certifies` text says
what is certified. That the fork rule is a repair: it is the honest rung printed where a higher
one cannot be. That FAILED has a price: the measurement has not run.
