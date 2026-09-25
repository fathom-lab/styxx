# NOTE — conformance/sworn/ regenerated for 7.48.0, in an environment that reproduces the committed set

Fathom Lab · 2026-09-25 · A release step for styxx 7.48.0 on `release/v7.48.0`. Nothing here is a
result or a measurement. It records why a committed set was regenerated, the check that had to pass
before that was allowed, and exactly what moved.

## Why the set was regenerated

`RESULT_sworn_conformance_v01_ships_2026_09_05.md` ends by saying the set, once sworn to, "is never
regenerated in place". That line was written before anyone had bumped the version under the set.

Commit `7696a94a` (2026-09-18) took a 7.48.0 cut back out of its PR, because in CI the bump broke
exactly one test, `test_the_committed_set_regenerates_to_its_own_digest` (C7). It named the
regeneration as a release step: "Cutting 7.48.0 needs the bump and the regeneration together, done
where the regeneration matches." The italic "Staged for 7.48.0" note at the top of the entries under
7.48.0 in `CHANGELOG.md` says the same, and so does `98536d01`.

These two statements conflict. This NOTE follows `7696a94a`, the later and more specific one, and
leaves the RESULT as written. Two facts bear on it:

- The set had already been regenerated in place after the RESULT. The RESULT binds `85e2d3b9…` at
  `7add6fff`. After that, `set_sha256` changed in 13 commits (`b53fb10c` through `65c98012`, all on
  2026-09-06). All 13 record the same provenance platform and Python as the set regenerated here.
- The RESULT's sworn spans resolve at `7add6fff`, the commit its sidecar names. Neither those
  regenerations nor this one moves them.

## The check before regenerating: 7.47.0 reproduces

**Environment.** Windows 11 (`win32`), CPython 3.12.10 (`py -3.12`). That is the platform and Python
recorded in the committed set's `provenance`. The work ran in a scratch clone, made with
`git clone --shared --no-checkout` from the local repository, with `core.autocrlf=false` and
`core.longpaths=true` set in that clone. So the working tree is byte-identical to the blobs, as on
CI's ubuntu runner. `git ls-files --eol` gave 5750 `w/lf`, 202 `w/-text`, 68 `w/none` and 3 `w/crlf`;
those 3 are files whose blobs are themselves CRLF. `PYTHONPATH` named a stub `torch` module that
raises `ImportError`, because CI installs no torch. `PYTHONDONTWRITEBYTECODE=1` was set. `styxx` was
checked to import from the clone and not from the machine's editable install.

**At `98a5c368`** (origin/main, `__version__ = "7.47.0"`),
`python conformance/sworn/gen_vectors.py --check` exited 0. It regenerated 3620 vectors in 20
families, all of them replaying, with `set_sha256`
`ca5e715ac66ecb4169f7d38a7702badff1fd3498a0249164f152de388a38144d`, which is the committed digest.
It printed "CHECK OK: the set regenerates to its own digest".

**Contrast, same clone and same commit.** The working tree was rewritten with `core.autocrlf=true`,
which made 5322 files CRLF. It also exited 0, with the same digest. So on this machine, at this
commit, the checkout's line endings do not move the set. On 2026-09-05, `ee4771ef` (line endings)
and `7add6fff` (interpreter prose) removed the two dependencies CI had found. `7696a94a` said its environment
"does not reproduce the committed digests even at the unchanged version", but the commit does not
record what that environment was. That failure did not happen here, and a CRLF checkout does not
explain it at this commit.

Docker was not tried. The procedure stops at the earliest environment on its list that reproduces
the set, and this one did.

**How this relates to CI.** The committed set was generated on `win32` with 3.12.10. CI regenerates
it on ubuntu under 3.9–3.12 in C7. That has been the arrangement since `ee4771ef`, which made the two
platforms agree. This NOTE did not observe a CI run on the regenerated set. That run is owed, and it
comes from the release branch's own checks.

## The regeneration at 7.48.0

This was the same environment and the same clone, checked out at `c5bc3083`, the head of
`release/v7.48.0`: the version bump `7d6c6638` plus the challenge repair.

- **Before regenerating**, `--check` exited 1: "set_sha256 drifted: committed ca5e715a…, regenerated
  05e64577…". The drift report named the families `cli`, `gaming`, `receipt_v1` and `rules`, plus
  `blobs.json`. Every count was unchanged.
- **`python conformance/sworn/gen_vectors.py`** (in place) exited 0. The refuse-if-moved guard found
  0 moved vectors. It printed 15 dropped and 15 added.
- **`--check`** then exited 0 ("CHECK OK", `05e64577…`). **`--replay`** exited 0, with 3620 of 3620
  vectors passing across all 20 families.

The seven changed files were copied byte for byte into the release worktree. Each file's sha256 is
equal on both sides. `.gitattributes` has `conformance/** -text`, so the worktree holds the blob
bytes, which are LF. `git diff` shows those 7 files, 348 insertions and 348 deletions, and nothing
outside `conformance/sworn/` apart from this NOTE.

| | committed, 7.47.0 | regenerated, 7.48.0 |
|---|---|---|
| `set_sha256` | `ca5e715ac66ecb4169f7d38a7702badff1fd3498a0249164f152de388a38144d` | `05e64577b04ed5467db4580c621986a3fd67d11f76403d2b6b9f2af62e620a8c` |
| `vectors/cli.json` | `028a51a86b28d1fd02c05f2b9cae23685e16deaac410186b99d83e8a668a696e` | `33e42bdef39ce99da4f85305faeadc84888c92070043a6fdd903686f46b40c9e` |
| `vectors/gaming.json` | `14845eeac2a418585bf08005329986691c68b78c92fe7aa1c2ec68222a72150c` | `9a016be99489344807b4b582b0f29e8eb0b0bda21700d95f190209ba98a444f8` |
| `vectors/receipt_v1.json` | `30901cbd4c1ed309a695656ed54b93a4abaf9bd6b2518f8825ef610525616b31` | `7cbdde14ce4cb1d08987d8e4470e6829a32e3cd33ad87410cf183b8a6ec531a4` |
| `vectors/rules.json` | `704e6b0fefca86a117476e2b91b1138441c9009bf66509968a7a96c581619abd` | `6c182bf72859c06ba85951d6cb2ceffe130cc075c5cd9168648e1f800ac58df2` |
| `blobs.json` | `7a680bd02432107f8b22555fe7cbaa6d3807a2ceddcc2c1fda05710d31afc91e` | `b45cd05e42676089d309b6794b0886a5085eaf4496d906b1682a6573516afbeb` |

These counts did not change: 3620 vectors, 20 families, 3981 blobs. The other 16 family files are
byte-identical.

## What moved, and why

**Fifteen vectors, all in mode `receipt_check`:**

- `cli`, 1: `TestCLI::test_canon_render_verify_check_round_trip`
- `gaming`, 7: `TestGamingLensFromTheAttackPass::test_a_tampered_receipt_fails_and_never_crashes_or_refuses`
- `receipt_v1`, 4: `TestVerdictReceipt`: `test_the_receipt_is_content_addressed_and_re_derivable`,
  `test_a_tampered_verdict_or_document_fails_re_derivation` (×2) and
  `test_a_receipt_from_another_verifier_build_is_reported_not_hidden`
- `rules`, 3: `TestRules::test_r9_the_v1_receipt_re_derives_without_its_coverage_block`

**Why they moved.** Each of these vectors takes a verdict receipt as an input blob. A receipt's
`digest` is sha256 over the JCS of the receipt, leaving out only `digest`, `timestamp`, `coverage`
and `coverage_sha256` (`styxx/sworn.py:1824`). The `verifier` block is therefore inside the digest,
including `verifier.styxx_version`, which is stamped from `styxx/_version.py` at `styxx/sworn.py:1816`.

**The blobs.** Each of the 15 new receipt blobs equals a committed one once `verifier.styxx_version`
and `digest` are set aside. There are 14 distinct contents on each side, equal as sets. One content
appears twice on each side, and the two copies differ only in the digest they carry.

- 5 blobs have a digest that re-derives from their own content, on both sides.
- The other 10 are receipts that the tests tamper with on purpose, so their digest is not meant to
  re-derive. They carry the digest of the receipt they were cut from, and that digest moved with it.
  One of them has `verifier` set to null and differs only in its digest. Another is a sidecar
  receipt whose `commit` was edited after issue. Its digest re-derives on both sides once the
  commit it was issued at is put back.
- The 15 blobs carry 6 distinct digests on each side. Each committed digest maps to exactly one
  regenerated digest.

**The vectors.** A vector's id is sha256 over `{mode, inputs}`, and the receipt is one of the
inputs, so these 15 vectors get new ids. They do not move. Each dropped vector has an added
counterpart that is equal to it in sources, rules, mode, expect block and every other input. No
expected outcome changed.

**The other files:**

- `blobs.json`: 15 blobs out, 15 in.
- `observer.json`, which is outside `set_sha256`: 15 rows move to the new ids with their contents
  unchanged as a multiset. No other row changed.
- `index.json`: the four family digests, the blobs digest, `set_sha256`, and
  `provenance.styxx_version` (7.47.0 → 7.48.0). The rest of `provenance` is unchanged: `python`
  3.12.10, `platform` win32, `sworn_sha256` `7e4602472010497a…`, both `sources_sha256`, and
  `pytest_summary` "350 passed".

### A correction to the staged note

The staged note in the CHANGELOG, and the messages of `7696a94a` and `98536d01`, all say the bump
invalidates the set because the set pins `provenance.styxx_version`. That is not what moves the
digest.

- `provenance` is outside `set_sha256`. The set rule is "index minus set_sha256 minus provenance"
  (`conformance/sworn/gen_vectors.py:54`, applied at `:482`).
- Recomputing the committed index's digest with only `provenance.styxx_version` changed to 7.48.0
  gives `ca5e715a…`, unchanged.
- The digest moved because of `verifier.styxx_version` inside the 15 receipt digests above.

So editing `provenance` could not have repaired C7. The CHANGELOG entry text is left as written,
because the release keeps its entries whole. This NOTE is where the correction is recorded.

## What was not touched

Only two things changed: the seven files under `conformance/sworn/` and this NOTE. These did not
change:

- no committed receipt, certificate, sworn document, sidecar, capsule, charon log, PREREG, RESULT or
  ANALYSIS;
- the 60 `papers/**/*.sworn-receipt.json` stamped 7.47.0, which stay as issued;
- the RESULT that binds `85e2d3b9…` at `7add6fff`, which is not edited.

## Tests, in the release worktree with the regenerated set

These ran in `C:/Users/heyzo/clawd/wt/release-748`, whose working tree is CRLF under
`core.autocrlf=true`, using `py -3.12 -m pytest -q -p no:cacheprovider` with the same stub-torch
`PYTHONPATH`.

- `tests/test_sworn_conformance.py`: 72 passed, 0 skipped. That includes C7, which ran the
  regeneration and found `05e64577…`.
- All 26 `tests/test_sworn*.py` files: 1186 passed, 0 failed, 0 skipped, in 229.60 s.

## What this does not say

- It does not say CI passes on the regenerated set. That was not observed here.
- It does not say the set is more correct than it was. The set pins what this verifier does. A
  version string moved, and no verdict, span, refusal or expected outcome moved with it.
- It does not say 7.48.0 is released. The regeneration is a step toward that release.
