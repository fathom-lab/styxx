# NOTE: how the styxx 7.48.1 Zenodo record was published (2026-10-06)

This note says what the 7.48.1 receipts in this directory record, how they were made, and what they
do not show. It is written from the receipts and from the two scripts' logs. No receipt is edited.

**The record.** styxx 7.48.1 is Zenodo record 23200977, DOI 10.5281/zenodo.23200977, a new version
of the software concept 10.5281/zenodo.19758618, made from 7.48.0's record 23042251
(10.5281/zenodo.23042251). Zenodo's read-back puts it last in the concept's version list
(`relations.version`: index 2, `is_last` true). It holds three files:

| file | bytes | md5 |
|---|---|---|
| `styxx-v7.48.1-source-bundle.zip` | 275578927 | 438ed1bd56ffb8cda4e95c82ec7fe20c |
| `styxx-7.48.1-py3-none-any.whl` | 8043925 | 52cf80c432493b64ddff70cf9f513cb5 |
| `styxx-7.48.1.tar.gz` | 8593254 | e418930c1defb173dffbf37ba077db6a |

The zip is `git archive` of tag `v7.48.1` (commit b42942186015848daa32e40b95970f4e370a3016) run
with `core.autocrlf=false`. The wheel and sdist are the files PyPI serves for 7.48.1. Each file's
sha256 is in both receipts. The related identifiers are 7.48.0 (isNewVersionOf), advisory
GHSA-h5xv-4344-f62r (isDocumentedBy), the spec v1.0 10.5281/zenodo.19746215 (isSupplementTo), the
Fathom research series 10.5281/zenodo.19326174 (isPartOf), the GitHub release and the PyPI page
(isSupplementTo), and the tree at the tagged commit (isDerivedFrom).

**When.** Times are UTC. The deposit script last wrote the draft receipt at 2026-10-07T02:12:14Z.
The publish script recorded its publish at 02:12:35Z by the local clock, and Zenodo records the
record created at 02:12:37.718Z and modified at 02:12:38.089Z. In the lab's time zone (UTC-4) that
is 22:12 on 2026-10-06, the release date, which is also the metadata's `publication_date`. The
record's notes give PyPI's upload time as 2026-10-07T00:11Z.

**The files in this directory.**
- `zenodo-draft-receipt-software-v7.48.1.json`: the deposit script's receipt, status
  `draft_ready_unpublished`.
- `zenodo-deposit-receipt-software-v7.48.1.json`: the publish script's receipt, status
  `published_and_verified`.
- `zenodo-metadata-software-v7.48.1.json`: the `metadata.json` both scripts read, as sent.
- `zenodo-metadata-software-v7.48.1-as-published.json`: the deposition's metadata as Zenodo returned
  it after the publish.
- `zenodo-record-software-v7.48.1-readback.json`: the public record as Zenodo returned it after the
  publish.

**The two scripts.** `scripts/zenodo_deposit_software_v7_48_1.py` makes the draft and stops.
Publish, edit and discard are not on its request allowlist, and each request that changes anything
is pinned to the one record it may touch. `scripts/zenodo_publish_software_v7_48_1.py` publishes
only the draft id the draft receipt names, and only when `--confirm` repeats that id. The deposit
receipt pins the sha256 of both scripts (4a313e96... and 43c8d56f...), of the draft receipt
(1397d92c...) and of `metadata.json` (1c5f6ac3...). The committed git blobs of those four files have
those sha256. A Windows checkout with `core.autocrlf=true` holds them with CRLF line endings, and
those working-copy bytes hash differently.

**The checks before the draft.** Offline, the deposit script checked:
- the three files against pinned sizes, sha256 and md5;
- the zip against the tag's tree (6043 of 6043 blobs byte-identical);
- the wheel's 353 files outside `.dist-info`, and the sdist's 614 files besides the 8 its build
  generates, against the tag's blobs;
- the wheel and sdist names, sizes and sha256 against PyPI's file list for 7.48.1;
- `metadata.json`: its creators, keywords, licence, language, access right and upload type against
  7.48.0's record as published, and its opening paragraph and spec relation against 7.48.0's as
  corrected;
- every number, date, `<code>` span, commit id and pull request number in the description against
  the tag (CHANGELOG `[7.48.1]`, `zenodo/MANIFEST.json`, `CITATION.cff` and the tree);
- the title, description, notes and keywords against the charter words;
- that the public text holds none of six unreleased-content markers. The markers live in a local
  file that is not committed, and the draft receipt records only its sha256 and the count.

Then, from Zenodo:
- The concept's latest version was 23042251 (7.48.0), as the lab's receipts say, and the concept
  held no unpublished draft.
- `actions/newversion` on 23042251 made draft 23200977. The three files it inherited from 7.48.0
  were deleted and the three new ones uploaded.
- `metadata.json` was sent, and each of its twelve fields read back the same. Zenodo added no key
  beyond `doi`, `prereserve_doi` and `imprint_publisher`.
- Zenodo's md5 and size for each file matched the local bytes.

The licence, sent as `mit`, reads back as `mit-license`. This deposit script expected that, so,
unlike 7.48.0's, it did not stop at read-back.

**The checks before and after the publish.** The publish script ran every offline check again. It
checked that the draft receipt is ready and names this concept, version and commit, and that
`metadata.json`, the three files and the deposit script are the ones the receipt pins. It then
re-read the draft from Zenodo and checked:
- the draft is unpublished, in concept 19758618, and its reserved DOI is its own;
- every metadata field equals `metadata.json`, the licence included (7.48.0's publish script did not
  check the licence);
- the draft's metadata holds no key that was not sent, other than the three Zenodo adds;
- the draft holds exactly the three files, each with the local md5 and size;
- the concept's latest version was still 23042251.

It sent one `actions/publish`, which Zenodo answered with HTTP 202. Reading the deposition and the
public record back, it checked: state done; DOI 10.5281/zenodo.23200977; concept 19758618; version
7.48.1; the title and every metadata field as sent; the three files with the local md5 and size; and
the concept's latest version now 23200977. The deposit receipt lists these as
`checks_before_publish` and `checks_after_publish`, all true, and as `concept_latest_after_publish`.

**The DOI.** Just after the publish, doi.org answered 10.5281/zenodo.23200977 with HTTP 302 to
`https://zenodo.org/doi/10.5281/zenodo.23200977`, not to the record page. The receipt records
`points_at_record: false`, and the log marks the line LATER ("DataCite registration may lag"). A
later check, not part of the receipts: at 2026-10-07T02:18Z, a request without credentials that
followed that redirect reached `https://zenodo.org/records/23200977` (HTTP 200) in two hops. 7.48.0's
DOI 10.5281/zenodo.23042251 takes the same two hops.

**Reproducing the zip.** `git -c core.autocrlf=false archive --format=zip --prefix=styxx-7.48.1/
v7.48.1` gives md5 438ed1bd56ffb8cda4e95c82ec7fe20c on git 2.52.0.windows.1. That run was at
2026-10-07T02:17Z, after the publish, and is not part of the receipts. The record's description says
other git and zlib versions can write a zip with different bytes and the same file contents.

**Running the scripts from this repository.** As committed under `scripts/`, neither script runs on
its own. Both read and write files in their own directory:
- `bundle/`, holding the three files;
- `metadata.json` (committed here as `zenodo-metadata-software-v7.48.1.json`);
- `excluded_markers.json`, which is not committed;
- `pypi-sources-7.48.1.json` and `datacite-relations-7.48.1.json`, PyPI's file list and DataCite's
  resource types as fetched, which are not committed;
- their receipts.

They also read a clone that holds the tag (`--repo`, which defaults to the lab's local checkout, as
in 7.48.0's script). They were run from a working directory outside the repository. Dry runs of both
scripts and an offline self-test against a fake Zenodo (81 checks passed) also ran there, and their
outputs were written before the draft receipt. Neither those outputs nor the two logs are committed.

**The token and the markers.** Both receipts give the token source as `--token-file ([ZENODO]
zenodo_token)`, with no path and no value. The markers file appears in a receipt only as a sha256 and
a count. Neither of the two exposures that the 7.48.0 note records is repeated in these files.

**What these files do not show.**
- Who ran either script, the command lines beyond what the logs print, or who authorized the
  publish. The draft receipt's `publish_step` names the operator, after reading the draft. The
  publish was recorded 21 seconds after the draft receipt was written, and nothing here shows whether
  anyone read the draft in the browser in between.
- That the deposited tree is free of stale text. The record's description lists the stale or wrong
  lines it knows of in the tree at the tag. One of them is `zenodo/MANIFEST.json` naming 7.48.0 as
  the concept's latest version, which the change that adds this note updates.
- Anything merged after the tag. None of it is in the record.
- Use of the record. The read-back's `stats` are Zenodo's counters at 02:12:38Z.

Zenodo lets a published record's metadata be edited in place and keeps no public prior revision. Any
later edit is therefore committed here as a new dated file, and these files stay as they are.
