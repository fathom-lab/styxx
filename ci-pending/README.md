# CI changes that cannot be pushed from here

Three commits to `.github/workflows/` have been sitting on a local branch since 2026-08-19,
unpushed. GitHub refuses them:

```
! [remote rejected] refusing to allow a Personal Access Token to create or update
  workflow `.github/workflows/diffgate.yml` without `workflow` scope
```

The `fathom-lab` PAT carries `repo` scope only. This is a credential limit, not a code problem,
and it is not something an agent should route around by minting credentials. So the work is parked
here as patches instead of stranded on a branch that could be lost, **and the parts that did not
need a workflow change at all have been extracted and shipped** (see below).

Verified `2026-08-28`: all three still apply cleanly to the workflow files on this branch.

## Landing them

From the repository root, with a token or SSH key that has `workflow` scope:

```bash
git am ci-pending/*.patch && git push
```

Each patch touches `.github/workflows/` only. `git apply --check` passes on all three as of the
date above; if that stops being true, the branch has moved and they need rebasing rather than
forcing.

## What each one fixes

**`e005e94` — diffgate self-gating + nightly-heavy PR paths.**
`diffgate.yml`, `nightly-heavy.yml`, `test.yml`. Makes the diffgate action run against its own
pull requests and gives the heavy nightly suite a PR path, so neither can rot unobserved between
nightly runs.

**`518f7df` — telescope skip path.**
`telescope.yml`. Moves the API-key check *before* `setup-python` and gates the cache post-step, so
a fork without secrets skips cleanly instead of failing in a step that was never going to work.

**`c5d65fc` — the release path tests the tagged ref and binds tag to version.**
`publish.yml`. Two real defects, both documented in the commit message:

* the release path published to PyPI with **zero tests executed on the tagged ref** — `7.35.0`
  shipped while the tests workflow had been red for days;
* `skip-existing: true` made a mistagged release a **silent green no-op**: tag `vX.Y.Z` on a tree
  still versioned `X.Y.(Z-1)` builds the old version, PyPI skips the duplicate, every check is
  green, and nothing shipped. That is SP-1 from `benchmarks/silent_pass/CORPUS.md` wearing a
  release badge.

## What was extracted so it did not have to wait

Two of the guarantees in these patches did not actually require a workflow change, and blocking
them on a credential was a failure of imagination rather than a real dependency.

**`tests/test_ledger.py` no longer depends on `fetch-depth: 0`.** The ledger regeneration
guarantee had *never run in CI*: `actions/checkout` defaults to depth 1, `.git/shallow` exists,
and the test called `pytest.skip`. On every Python version in the matrix, silently. The test now
unshallows its own checkout and, if it cannot, **fails in CI instead of skipping** — locally it
still skips, because a developer with a shallow clone is not the person hiding a defect. The
`fetch-depth: 0` patch is still worth landing (it is faster and more direct), but the guarantee no
longer waits on it.

**`tests/test_version_never_behind_tag.py` catches the mistag from ordinary CI.** It asserts the
declared version is never behind the newest visible tag, which is exactly the silent no-op above.
It is strictly weaker than the `publish.yml` fix — it fires on the next run after a bad tag rather
than before the upload — and it is **not** a substitute for landing that patch.

## What still genuinely needs the scope

The job-graph changes: gating publish on a test job, the diffgate PR path, the nightly-heavy PR
path, and the telescope skip ordering. None of those can be expressed outside `.github/workflows/`,
and no test can stand in for them.
