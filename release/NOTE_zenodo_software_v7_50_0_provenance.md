# NOTE: how the styxx 7.50.0 Zenodo record was published (2026-10-23)

This note says what the 7.50.0 receipts in this directory record, how they were made, and what they
do not show. It is written from the receipts, from the deposit script's log and from the outputs of
the dry runs. No log of the publish run was kept with the kit; what this note says of the publish
comes from the deposit receipt. No receipt is edited.

**The record.** styxx 7.50.0 is Zenodo record 23594959, DOI 10.5281/zenodo.23594959, a new version
of the software concept 10.5281/zenodo.19758618, made from 7.49.0's record 23221755
(10.5281/zenodo.23221755). Zenodo's read-back puts it last in the concept's version list
(`relations.version`: index 4, counted from 0, `is_last` true). It holds three files:

| file | bytes | md5 |
|---|---|---|
| `styxx-v7.50.0-source-bundle.zip` | 277268802 | 37ce89258987104eb06ba1556b7e4b33 |
| `styxx-7.50.0-py3-none-any.whl` | 8208487 | 13d709d1c1419a8ad60fc7520f22ebd3 |
| `styxx-7.50.0.tar.gz` | 9041414 | 9192f9a80ecaa7f7ed8ee082b96b1ae6 |

The zip is `git archive` of tag `v7.50.0` (commit bebdadf1562c742f53f2114cad99b4aba0738cdf) run
with `core.autocrlf=false`. The wheel and sdist are the files PyPI serves for 7.50.0. Each file's
sha256 is in both receipts. The related identifiers are 7.49.0 (isNewVersionOf), the spec v1.0
10.5281/zenodo.19746215 (isSupplementTo), the Fathom research series 10.5281/zenodo.19326174
(isPartOf), the GitHub release and the PyPI page (isSupplementTo), and the tree at the tagged commit
(isDerivedFrom). No security advisory accompanies 7.50.0, so the record has no isDocumentedBy row;
7.49.0's and 7.48.1's records each had one.

**When.** Times are UTC. The deposit script wrote the draft receipt at 2026-10-23T11:57:59Z. The
publish script recorded its publish at 13:04:08Z by the local clock, and Zenodo records the record
created at 13:04:07.852Z and modified at 13:04:08.233Z. In the lab's time zone (UTC-4) that is 09:04
on 2026-10-23. The metadata's `publication_date` is 2026-10-23: the tag's date
(2026-10-23T06:31:33-04:00) and the day PyPI records the upload (2026-10-23T10:32Z, which the
record's notes give). CHANGELOG.md's `[7.50.0]` heading dates the cut 2026-10-22, and the notes say
that too.

**The files in this directory.**
- `zenodo-draft-receipt-software-v7.50.0.json`: the deposit script's receipt, status
  `draft_ready_unpublished`.
- `zenodo-deposit-receipt-software-v7.50.0.json`: the publish script's receipt, status
  `published_and_verified`.
- `zenodo-metadata-software-v7.50.0.json`: the `metadata.json` both scripts read, as sent.
- `zenodo-metadata-software-v7.50.0-as-published.json`: the deposition's metadata as Zenodo returned
  it after the publish. It differs from what was sent only in the licence (`mit` read back as
  `mit-license`) and in the three keys Zenodo adds (`doi`, `prereserve_doi`, `imprint_publisher`).
- `zenodo-record-software-v7.50.0-readback.json`: the public record as Zenodo returned it after the
  publish.

**The two scripts.** `scripts/zenodo_deposit_software_v7_50_0.py` makes the draft and stops.
Publish, edit and discard are not on its request allowlist, and each request that changes anything
is pinned to the one record it may touch. `scripts/zenodo_publish_software_v7_50_0.py` publishes
only the draft id the draft receipt names, and only when `--confirm` repeats that id. Both were
adapted from 7.49.0's. The publish script's docstring names the script it was adapted from as
`scripts/zenodo_publish_software_v7_50_0.py`, its own name; the script that published 7.49.0 is
`scripts/zenodo_publish_software_v7_49_0.py`. The line is not corrected here: a corrected blob would
not have the sha256 the deposit receipt pins. The deposit receipt pins the sha256 of both scripts (5643c518... and
8d2986c1...), of the draft receipt (e9d26303...) and of `metadata.json` (12f5af55...). The
committed git blobs of those four files have those sha256. A Windows checkout with
`core.autocrlf=true` holds them with CRLF line endings, and those working-copy bytes hash
differently.

**The checks before the draft.** Offline, the deposit script ran 90 checks, all passed (its dry run
ran the same 90 and one more, that the socket guard refuses connections). They cover:
- the three files against pinned sizes, sha256 and md5;
- the zip against the tag's tree (6164 of 6164 blobs byte-identical);
- the wheel's 357 files outside `.dist-info`, and the sdist's 640 files besides the 8 its build
  generates, against the tag's blobs;
- the wheel and sdist against PyPI's file list for 7.50.0, as fetched;
- `metadata.json`: its creators, licence, language, access right and upload type against 7.49.0's
  record as published, after checking that it agrees with 7.49.0's public read-back; its opening
  paragraph and its spec and series relations against 7.49.0's as published; the resource types of
  the spec, the series and 7.49.0 against what DataCite records; its 12 keywords against
  `pyproject.toml` at the tag (7.49.0's record carries the 15 of `pyproject.toml` at v7.49.0); its
  creators against `CITATION.cff`'s authors at the tag;
- the description: its title against the `[7.50.0]` headline; its second paragraph, which quotes the
  `[7.50.0]` section's opening paragraph without its last sentence, word for word; its list of the
  section's 17 entry headings, in order; every number and date in it against CHANGELOG `[7.50.0]`,
  the tag message, the earlier sections or the tree; every `<code>` span (66) against the tree, an
  ancestor commit or the CHANGELOG; the one commit id it names; and every pull request number;
- the list of stale lines against checks of the tree at the tag, one or more for each of its
  twelve lines;
- the title, description, notes and keywords against the charter words;
- that the public text holds none of six unreleased-content markers. The markers live in a local
  file that is not committed; the draft receipt records its file name, its sha256 and the count, and
  no marker;
- that the allowlist refuses publish, edit and discard, and refuses nine probes of changing
  requests that are not pinned to the one record they may touch.

Then, from Zenodo, in 15 requests:
- The concept's latest version was 23221755 (7.49.0), as the lab's receipts say, and the concept
  held no unpublished draft.
- `actions/newversion` on 23221755 made draft 23594959. The three files it inherited from 7.49.0
  were deleted and the three new ones uploaded (the zip's upload took 639.42 seconds).
- `metadata.json` was sent, and each of its twelve fields read back the same. Zenodo added no key
  beyond `doi`, `prereserve_doi` and `imprint_publisher`.
- Zenodo's md5 and size for each file matched the local bytes.

**The checks before and after the publish.** The publish script's dry run, offline, ran once before
the draft existed (94 checks passed, 1 skipped because the draft receipt did not exist yet) and once
after it (105 passed), validating the draft receipt; it sent no request. The real run re-read the
draft from Zenodo and checked:
- the draft is unpublished, in concept 19758618, and its reserved DOI is its own;
- every metadata field equals `metadata.json`, the licence included;
- the draft's metadata holds no key that was not sent, other than the three Zenodo adds;
- the draft holds exactly the three files, each with the local md5 and size;
- the concept's latest version was still 23221755.

It sent one `actions/publish`, which Zenodo answered with HTTP 202. Reading the deposition and the
public record back, it checked: state done; DOI 10.5281/zenodo.23594959; concept 19758618; version
7.50.0; the title and every metadata field as sent; the three files with the local md5 and size; and
the concept's latest version now 23594959. The deposit receipt lists these as
`checks_before_publish` (22) and `checks_after_publish` (9), all true, and as
`concept_latest_after_publish`. It records seven requests, one of them the publish.

**The DOI.** Just after the publish, doi.org answered 10.5281/zenodo.23594959 with HTTP 302 to
`https://zenodo.org/doi/10.5281/zenodo.23594959`, not to the record page. The receipt records
`points_at_record: false` and `points_at_zenodo_doi_page: true`, as 7.49.0's did. A later check, not
part of the receipts: at 2026-10-23T15:49Z, a request without credentials that followed the
redirects reached `https://zenodo.org/records/23594959` (HTTP 200) in two hops from the version DOI
and in two from the concept DOI 10.5281/zenodo.19758618, and Zenodo's public API gave 23594959
(7.50.0) as the concept's latest version.

**Reproducing the zip.** `git -c core.autocrlf=false archive --format=zip --prefix=styxx-7.50.0/
v7.50.0` gives md5 37ce89258987104eb06ba1556b7e4b33 on git 2.52.0.windows.1. That run was at
2026-10-23T15:46Z, after the publish, and is not part of the receipts. The record's description says
other git and zlib versions can write a zip with different bytes and the same file contents.

**Running the scripts from this repository.** As committed under `scripts/`, neither script runs on
its own. Both read and write files in their own directory, and the publish script imports the
deposit script from there:
- `bundle/`, holding the three files;
- `metadata.json` (committed here as `zenodo-metadata-software-v7.50.0.json`);
- `excluded_markers.json`, which is not committed;
- `pypi-sources-7.50.0.json` and `datacite-relations-7.50.0.json`, PyPI's file list and DataCite's
  resource types as fetched, which are not committed;
- their receipts.

They also read a clone that holds the tag (`--repo`, which defaults to the lab's local checkout, as
in 7.49.0's script). They were run from a working directory outside the repository. An offline
self-test against a fake Zenodo (114 checks passed) also ran there. Neither the dry-run outputs, the
self-test's output nor the deposit log are committed, and neither are the helpers that built or
checked the inputs (the metadata builder, the bundle maker and checker, the PyPI and DataCite
fetchers, the self-test and the draft-receipt writer), as for 7.49.0. A read-only review of the
draft, which found it unpublished and the version DOI not yet registered, was kept with the kit and
is not committed either.

**The token and the markers.** Both receipts give the token source as `--token-file ([ZENODO]
zenodo_token)`, with no path and no value. The markers file appears in the draft receipt as its file
name, a sha256 and a count, and no marker.

**What these files do not show.**
- Who ran either script, or who authorized the publish. Neither the receipts nor the deposit log
  record it, and this note does not state it. The draft receipt's `publish_step` text asks that the
  draft be read at its `draft_url` before the publish script runs, and says the receipt does not
  record whether anyone did. The publish was recorded 66 minutes and 9 seconds after the draft
  receipt was written. What these files record in between is the publish script's own re-check of
  the draft against the local files and `metadata.json`.
- That the deposited tree is free of stale text. The record's description lists twelve stale or
  wrong lines it knows of in the tree at the tag. By topic:
  1. `CITATION.cff`, whose version and release date the cut did not move from 7.49.0 and 2026-10-07,
     and whose identifier note names 7.48.0's version DOI beside the concept DOI;
  2. `README.md`, whose 91 tag-pinned links name v7.49.0;
  3. `web/gate/`: its README's opening paragraph and drift section, `py_side.py`, and the headers of
     `diffgate.js` and `bookmarklet_src.js` speak of PATH-2a or 7.48.0 as not yet released, and the
     bookmarklet's panel text calls its port 7.48.0's;
  4. the header comment of `.github/workflows/diffgate.yml`, which says the instrument is the
     released package;
  5. `sworn/README.md` and `sworn/action.yml`, which give 7.48.0 as the release to install, one
     both advisories name as affected;
  6. `README.md`, which labels the research-series concept DOI 10.5281/zenodo.19326174 as
     always-latest (D3 and D4 in `zenodo/MANIFEST.json`);
  7. `CITATION.cff`, whose preferred citation, which GitHub's citation prompt renders, is the
     position paper 10.5281/zenodo.19777921, older than its 2026-06-21 scope erratum (D2);
  8. the EXTERNAL-1 packet, which still carries the id leak;
  9. `zenodo/README.md`, which describes the 7.48.0 flow only;
  10. `zenodo/MANIFEST.json`, which names 7.49.0 as the concept's latest version;
  11. the `[7.49.0]` section, kept as released, which carries the wording 7.49.0's record lists as
      stale; of the files that record names, `styxx/_data/LEADERBOARD.md` and the two comments in
      `styxx/__init__.py` no longer carry the Baseline-019 claim, and the rest stand;
  12. `papers/frequency-resonance/NOTE_zoo_d0_screen_plan_2026_10_12.md`, whose section 15 says two
      files stay untracked that are committed beside it.

  The change that adds this note updates line 10, the version and release date of line 1, and the
  links of line 2. Line 1's identifier note, which is still true of 7.48.0, and the other nine are
  not changed here.
- Anything merged after the tag. None of it is in the record.
- Use of the record. The read-back's `stats` were read just after the publish; its `version_*`
  counters are 0.

Zenodo lets a published record's metadata be edited in place and keeps no public prior revision. Any
later edit is therefore committed here as a new dated file, and these files stay as they are.
