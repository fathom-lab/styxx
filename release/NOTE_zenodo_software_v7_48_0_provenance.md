# NOTE: how the styxx 7.48.0 Zenodo record was published (2026-09-29)

The two receipts in this directory disagree about how the deposit was published. This note says what
happened. Neither receipt is edited.

**The draft.** `scripts/zenodo_deposit_software_v7_48_0.py` created deposit 23042251 as a new
version of the software concept 10.5281/zenodo.19758618 and uploaded three files. It then stopped at
metadata read-back: Zenodo stores the licence `mit` as `mit-license`, and the script compares the
two strings. `zenodo-draft-receipt-software-v7.48.0.json` records `status:
stopped_at_metadata_readback`, `published: false`, and a `publish_step` of "the operator, in the
browser, after reading the draft". The last of these did not happen.

**The publish.** On 2026-09-29 the operator told the agent to publish the draft itself and to make
sure the record joined the existing software chain rather than starting a new one. The agent (a
Claude Code session on the lab machine) ran `scripts/zenodo_publish_v7_48_0.py --token-file <path
not recorded> --draft-id 23042251`. Before publishing, the script re-read the draft from Zenodo and
checked six things: the draft was unpublished; it sat in concept 19758618; its version was 7.48.0;
it carried both related identifiers, 19746215 (isSupplementTo) and 19326174 (isPartOf); it held
exactly three files, each md5-equal to the local bytes; and its title, description and notes
contained no charter word. It did not check the licence. The script then called
`actions/publish`, and Zenodo recorded 2026-09-29T15:46:04Z. The script wrote
`zenodo-deposit-receipt-software-v7.48.0.json`, which has the publish script's shape.

**Running the scripts from this repository.** As committed under `scripts/`, neither script runs
on its own. Both expect sibling files in their own directory: `metadata.json`, `bundle/` (the three
files) and, for the publish script, `zenodo-draft-receipt-v7.48.0.json`. They were run from a
working directory outside the repository that held all three. The local bundle was deleted after
publishing to free disk. The deposited zip can be rebuilt from the tag: `git -c core.autocrlf=false
archive --format=zip --prefix=styxx-7.48.0/ v7.48.0` gives md5 1145aff5e0fe5f532ee2f7e8ab01f9fe.
The wheel and sdist are the files PyPI serves for 7.48.0.

**What the record held, and what was later edited.** The metadata as published is
`zenodo-metadata-software-v7.48.0.json`. Zenodo lets a published record's metadata be edited in
place without changing its DOI or files, and it keeps no public prior revision of the text. Any such
edit is therefore committed here as a new dated file beside the original, which stays as it is.

**Two exposures, not repaired.** The draft receipt names the local path of the token file (not its
value). The deposit script's `EXCLUDED_MARKERS` names two local branch names. Both are already in
history. The receipt pins the script's sha256, so editing the script would break a receipt.
Removing either needs a force-push, which is the operator's call. Later deposit scripts should read
their markers from a gitignored file and record a token source without a path.
