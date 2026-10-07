# NOTE: how the styxx 7.49.0 Zenodo record was published (2026-10-07)

This note says what the 7.49.0 receipts in this directory record, how they were made, and what they
do not show. It is written from the receipts and from the two scripts' logs. No receipt is edited.

**The record.** styxx 7.49.0 is Zenodo record 23221755, DOI 10.5281/zenodo.23221755, a new version
of the software concept 10.5281/zenodo.19758618, made from 7.48.1's record 23200977
(10.5281/zenodo.23200977). Zenodo's read-back puts it last in the concept's version list
(`relations.version`: index 3, `is_last` true). It holds three files:

| file | bytes | md5 |
|---|---|---|
| `styxx-v7.49.0-source-bundle.zip` | 276179193 | b6f174e73d8b83182edf0f2b5e4745d0 |
| `styxx-7.49.0-py3-none-any.whl` | 8101703 | 3eafa8d101253cfe304eab200d03ae8e |
| `styxx-7.49.0.tar.gz` | 8761703 | 60dbcf0d480220da3de29ed2810759cd |

The zip is `git archive` of tag `v7.49.0` (commit c6e00da02673ec6a2e84919f17f56a2e4435a8c9) run
with `core.autocrlf=false`. The wheel and sdist are the files PyPI serves for 7.49.0. Each file's
sha256 is in both receipts. The related identifiers are 7.48.1 (isNewVersionOf), advisory
GHSA-3g8h-qcfm-25xw (isDocumentedBy), the spec v1.0 10.5281/zenodo.19746215 (isSupplementTo), the
Fathom research series 10.5281/zenodo.19326174 (isPartOf), the GitHub release and the PyPI page
(isSupplementTo), and the tree at the tagged commit (isDerivedFrom).

**When.** Times are UTC. The deposit script last wrote the draft receipt at 2026-10-07T19:47:18Z.
The publish script recorded its publish at 19:47:40Z by the local clock, and Zenodo records the
record created at 19:47:39.455Z and modified at 19:47:39.909Z. In the lab's time zone (UTC-4) that
is 15:47 on 2026-10-07, the release date, which is also the metadata's `publication_date`. The
record's notes give PyPI's upload time as 2026-10-07T12:43Z.

**The files in this directory.**
- `zenodo-draft-receipt-software-v7.49.0.json`: the deposit script's receipt, status
  `draft_ready_unpublished`.
- `zenodo-deposit-receipt-software-v7.49.0.json`: the publish script's receipt, status
  `published_and_verified`.
- `zenodo-metadata-software-v7.49.0.json`: the `metadata.json` both scripts read, as sent.
- `zenodo-metadata-software-v7.49.0-as-published.json`: the deposition's metadata as Zenodo returned
  it after the publish.
- `zenodo-record-software-v7.49.0-readback.json`: the public record as Zenodo returned it after the
  publish.

**The two scripts.** `scripts/zenodo_deposit_software_v7_49_0.py` makes the draft and stops.
Publish, edit and discard are not on its request allowlist, and each request that changes anything
is pinned to the one record it may touch. `scripts/zenodo_publish_software_v7_49_0.py` publishes
only the draft id the draft receipt names, and only when `--confirm` repeats that id. Both were
adapted from 7.48.1's. The deposit receipt pins the sha256 of both scripts (fbd21187... and
4c7cb855...), of the draft receipt (d3a3771f...) and of `metadata.json` (aaf1d0b9...). The
committed git blobs of those four files have those sha256. A Windows checkout with
`core.autocrlf=true` holds them with CRLF line endings, and those working-copy bytes hash
differently.

**The checks before the draft.** Offline, the deposit script checked:
- the three files against pinned sizes, sha256 and md5;
- the zip against the tag's tree (6094 of 6094 blobs byte-identical);
- the wheel's 354 files outside `.dist-info`, and the sdist's 622 files besides the 8 its build
  generates, against the tag's blobs;
- the wheel and sdist names, sizes and sha256 against PyPI's file list for 7.49.0;
- `metadata.json`: its creators, keywords, licence, language, access right and upload type against
  7.48.1's record as published, after checking that it agrees with 7.48.1's public read-back; its
  opening paragraph and its spec and series relations against 7.48.1's as published; and the
  resource types of the spec, the series and 7.48.1 against what DataCite records;
- every number, date, `<code>` span (121), commit id (5), hash prefix and pull request number in the
  description against the tag: CHANGELOG `[7.49.0]` and, where a check allows it, `[7.48.1]`,
  `zenodo/MANIFEST.json`, `CITATION.cff` and the tree;
- the release paragraph's summary and the five sections condensed from CHANGELOG `[7.49.0]`
  against how that section states them (25 checks), and that the condensed sections hold no other
  statement but headings, labels, joins and statements the script checks elsewhere;
- the list of stale lines against checks of the tree at the tag, one or more for each of its
  eleven lines;
- the title, description, notes and keywords against the charter words;
- that the public text holds none of six unreleased-content markers. The markers live in a local
  file that is not committed; the draft receipt records its file name, its sha256 and the count, and
  no marker.

Then, from Zenodo:
- The concept's latest version was 23200977 (7.48.1), as the lab's receipts say, and the concept
  held no unpublished draft.
- `actions/newversion` on 23200977 made draft 23221755. The three files it inherited from 7.48.1
  were deleted and the three new ones uploaded.
- `metadata.json` was sent, and each of its twelve fields read back the same. Zenodo added no key
  beyond `doi`, `prereserve_doi` and `imprint_publisher`.
- Zenodo's md5 and size for each file matched the local bytes.

The licence, sent as `mit`, reads back as `mit-license`, which the deposit script expected.

**The checks before and after the publish.** The publish script ran every offline check again. It
checked that the draft receipt is ready, names this concept, version and commit, and was made from
23200977, the version `metadata.json` names as the one before this, and that `metadata.json`, the
three files and the deposit script are the ones the receipt pins. It then re-read the draft from
Zenodo and checked:
- the draft is unpublished, in concept 19758618, and its reserved DOI is its own;
- every metadata field equals `metadata.json`, the licence included;
- the draft's metadata holds no key that was not sent, other than the three Zenodo adds;
- the draft holds exactly the three files, each with the local md5 and size;
- the concept's latest version was still 23200977.

It sent one `actions/publish`, which Zenodo answered with HTTP 202. Reading the deposition and the
public record back, it checked: state done; DOI 10.5281/zenodo.23221755; concept 19758618; version
7.49.0; the title and every metadata field as sent; the three files with the local md5 and size; and
the concept's latest version now 23221755. The deposit receipt lists these as
`checks_before_publish` (22) and `checks_after_publish` (9), all true, and as
`concept_latest_after_publish`.

**The DOI.** Just after the publish, doi.org answered 10.5281/zenodo.23221755 with HTTP 302 to
`https://zenodo.org/doi/10.5281/zenodo.23221755`, not to the record page. The receipt records
`points_at_record: false` and `points_at_zenodo_doi_page: true`, a field 7.49.0's publish script
adds, and the log marks the line LATER. A later check, not part of the receipts: at
2026-10-07T19:51Z, a request without credentials that followed that redirect reached
`https://zenodo.org/records/23221755` (HTTP 200) in two hops, 7.48.1's DOI took the same two hops,
and Zenodo's public API gave 23221755 (7.49.0) as the concept's latest version.

**Reproducing the zip.** `git -c core.autocrlf=false archive --format=zip --prefix=styxx-7.49.0/
v7.49.0` gives md5 b6f174e73d8b83182edf0f2b5e4745d0 on git 2.52.0.windows.1. That run was at
2026-10-07T19:51Z, after the publish, and is not part of the receipts. The record's description says
other git and zlib versions can write a zip with different bytes and the same file contents.

**Running the scripts from this repository.** As committed under `scripts/`, neither script runs on
its own. Both read and write files in their own directory, and the publish script imports the
deposit script from there:
- `bundle/`, holding the three files;
- `metadata.json` (committed here as `zenodo-metadata-software-v7.49.0.json`);
- `excluded_markers.json`, which is not committed;
- `pypi-sources-7.49.0.json` and `datacite-relations-7.49.0.json`, PyPI's file list and DataCite's
  resource types as fetched, which are not committed;
- their receipts.

They also read a clone that holds the tag (`--repo`, which defaults to the lab's local checkout, as
in 7.48.1's script). They were run from a working directory outside the repository. Dry runs of both
scripts (the deposit script's: 134 checks passed; the publish script's: 137 passed and 1 skipped,
since the draft receipt did not exist yet) and an offline self-test against a fake Zenodo (120
checks passed) also ran there, and their outputs were written before the draft receipt. Neither
those outputs nor the two logs are committed, and neither are the helpers that built or checked the
inputs (the metadata builder, the bundle maker and checker, the PyPI and DataCite fetchers, and the
self-test), as for 7.48.1.

**The token and the markers.** Both receipts give the token source as `--token-file ([ZENODO]
zenodo_token)`, with no path and no value. The markers file appears in a receipt as its file name, a
sha256 and a count, and no marker. Neither of the two exposures that the 7.48.0 note records is
repeated in these files.

**What these files do not show.**
- Who ran either script, or who authorized the publish. Neither the receipts nor the logs record it,
  and this note does not state it. The draft receipt's `publish_step` text asks that the draft be
  read at its `draft_url` before the publish script runs, and says the receipt does not record
  whether anyone did. The publish was recorded 22 seconds after the draft receipt was written.
  What these files record in between is the publish script's own re-check of the draft against the
  local files and `metadata.json`.
- That the deposited tree is free of stale text. The record's description lists eleven stale or
  wrong lines it knows of in the tree at the tag. By topic:
  1. two entries of CHANGELOG `[7.49.0]`, kept whole from `[Unreleased]`: the PATH-2a entry says it
     is not released, and where it says `pip install styxx` (7.48.0), 7.48.1 was already on PyPI
     when PATH-2a merged; #201 made stale, before the cut, the Action entry's sha256 for
     `styxx/diffgate.py` and the sha256 both entries and `action.yml` give for the reader the
     overlay runs over;
  2. G-P1: the PATH-2a entry and `web/gate/README.md` leave to the operator what the operator's
     amendment of 2026-10-06 decided;
  3. `web/gate/`: its README's opening paragraph and drift section, `py_side.py`, and the headers of
     `diffgate.js` and `bookmarklet_src.js` speak of PATH-2a or 7.48.0 as not yet released, and the
     bookmarklet's panel text calls its port 7.48.0's;
  4. the header comment of `.github/workflows/diffgate.yml`, which says the instrument is the
     released package;
  5. `sworn/README.md` and `sworn/action.yml`, which give 7.48.0 as the release to install, one
     both advisories name as affected;
  6. `styxx/_data/LEADERBOARD.md` (and the root copy) and two comments in `styxx/__init__.py`, which
     still carry the ordinal priority claim for Baseline-019 that `styxx/critique.py`'s docstring
     withdrew;
  7. `README.md`, which labels the research-series concept DOI 10.5281/zenodo.19326174 as
     always-latest (D3 and D4 in `zenodo/MANIFEST.json`);
  8. `CITATION.cff`, whose preferred citation, which GitHub's citation prompt renders, is the
     position paper 10.5281/zenodo.19777921, older than its 2026-06-21 scope erratum (D2);
  9. the EXTERNAL-1 packet, which still carries the id leak;
  10. `zenodo/README.md`, which describes the 7.48.0 flow only;
  11. `zenodo/MANIFEST.json`, which names 7.48.1 as the concept's latest version. The change that
      adds this note updates it; the other ten are not changed here.
- Anything merged after the tag. None of it is in the record.
- Use of the record. The read-back's `stats` were read just after the publish; its `version_*`
  counters are 0.

Zenodo lets a published record's metadata be edited in place and keeps no public prior revision. Any
later edit is therefore committed here as a new dated file, and these files stay as they are.
