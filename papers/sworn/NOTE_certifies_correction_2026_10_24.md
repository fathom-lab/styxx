# NOTE — the receipt's certificate sentence said more than the verifier checks; schema /v2 corrects it

Fathom Lab · 2026-10-24 · A correction, on the route the operator chose for it. No committed receipt is edited.
Every count below is sworn to `papers/sworn/certifies_census_2026_10_24.json`, which
`papers/sworn/certifies_census.py` writes from git objects and from `styxx.sworn` (`--check` regenerates it).
Line numbers in `styxx/sworn.py` are at `2e98debc`, the commit this step was cut from.

## What the sentence said

Every verdict receipt `styxx.sworn` issued under `styxx.sworn.verdict-receipt/v1` carries a `certifies` field,
the verifier's statement of its own boundary (`_CERTIFIES`, `styxx/sworn.py` l.277-283). It
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/v1_opening" k="quote">opens `the spans the author bound were checked against bytes the author did not write`</sworn>,
a clause this NOTE withdraws, then names where the check ran (the commit or manifest the document names, the rung the manifest declares)
and what it is not (a claim that the document is correct or that the right sentences were bound, a check of
write time or of any signature).

## Why that was too wide

The verifier never sees who wrote a receipt. It sees the manifest the harness minted, and its only
authorship check reads that manifest:

- an `rN` receipt is refused when the manifest records it under an author-side `kind_of_source`, or lists
  its sha256 in `authored_sha256` (l.1228);
- a `path:` or `prereg:` receipt is refused when a manifest is given and lists its sha256 there (l.1265);
  with no manifest nothing about authorship is consulted, and every such span already prints
  "authorship unchecked";
- `authored_sha256` is the harness's record of what the agent produced in one turn. The code says what
  follows, in its own comment: "a file the agent committed under a name the harness never digested is still
  not caught" (l.1262-1264), and `SPEC_tree_channel_authorship_v01_2026_09_06.md` says so under "What this
  spec does not say";
- an UNRESOLVED span was not checked at all, and the sentence opened with "the spans … were checked".

The adapters' own documents already drew the line the sentence did not: `sworn/README.md`
("`authored_sha256` catches only the trivial laundering of a committed file into a receipt") and
`integrations/claude-code/sworn-hooks/README.md` ("What this does NOT attest — That the agent did not write
a file").

In this lab the gap is the ordinary case.
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/sidecars" k="numeric">The 58 committed sidecars</sworn>
carry
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/path_spans" k="numeric">1,261 spans that name a committed file</sworn>
(`path:` spans) over
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/path_targets" k="numeric">137 distinct files</sworn>,
and
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/path_spans_on_speaker_touched_files" k="numeric">1,261 of those spans rest on a file whose git history names one of the lab's five agent names as author or committer</sworn>.
A HELD `path:` span here has been a consistency check between the speaker's sentence and files the
speaker, or the operator, committed. That says nothing about whether those files are true; it says what
the receipt certified.

## Where the sentence is

<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/files_carrying_the_v1_clause_at_base" k="numeric">81 tracked files carry it</sworn>
at `2e98debc`:
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/v1_receipts_carrying_the_v1_sentence" k="numeric">71 verdict receipts carry it byte for byte</sworn>
(<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/by_class_at_base/sworn_receipt_json" k="numeric">66 of them</sworn>
are `*.sworn-receipt.json` files and
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/by_class_at_base/dryrun_receipt" k="numeric">5 are dry-run receipts</sworn>),
and the rest are the sealed capsule that embeds one, two action `run.json` files and two summaries, one
sworn RESULT and its sidecar that paraphrase it, the frozen v0.1 SPEC, and the two verifiers. Of the
oldest receipt schema, `/v0`,
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/v0_receipts_at_base" k="numeric">0 receipts are committed</sworn>.

## Why it was not corrected in place

`certifies` is inside the digest of a /v1 receipt and inside the core `verify_receipt` compares. A new
wording in the verifier would have made every committed /v1 receipt stop re-deriving: a receipt is history
too.

## The route

- Receipts issue as `styxx.sworn.verdict-receipt/v2`; only `schema` and `certifies` differ from /v1. The
  /v2 sentence, in full:

  > the spans the author bound that read HELD or FAILED were checked, each by its kind, against the
  > receipt it names (its bytes, or for a hash span on a manifest entry that carries none, the sha256 the
  > manifest records), at the commit and manifest the receipt records and at the rung the manifest
  > declares; the only authorship check is against the manifest: an rN receipt is taken as not the
  > author's when the manifest records it under a kind_of_source from the closed external vocabulary and
  > does not list its sha256 in authored_sha256, and a path: or prereg: receipt when no manifest is given
  > or the manifest does not list its sha256 there; authored_sha256 is the harness's record of what the
  > agent produced in the turn the manifest names, so bytes the author wrote that the harness never
  > digested, such as a file committed in another turn or under a name the harness never saw, are not
  > caught; an UNRESOLVED span was not checked — NOT a claim that the author did not write the receipt's
  > bytes, NOT a claim that the document is correct, NOT a claim that the right sentences were bound, NOT
  > a check that the tags were written at write time, NOT a check of any signature, and only as
  > trustworthy as the harness that minted the manifest and the tree that served the commit (git history
  > from the CLI; a snapshot or in-memory tree on its caller's word)

  (`_CERTIFIES` in `styxx/sworn.py` is that text on one line; the /v1 text stays beside it as
  `_CERTIFIES_V1`, marked withdrawn, loaded and never emitted.)
- A /v1 receipt keeps its digest over its bytes as committed and is compared on the fresh core with
  `schema` and `certifies` held at the /v1 strings, so it re-derives exactly when it did; one whose
  `certifies` was edited and re-digested still fails. A /v0 receipt is checked as before.
- `styxx/_data/sworn_verify.js` carries the same two strings and `coreAs`, the same projection.
- The conformance set's cores stay pinned at /v1:
  <sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/conformance_v1_core_vectors_unchanged_since_base" k="numeric">2,069 core vectors are unchanged</sworn>;
  a /v2 set beside it, `conformance/sworn/receipt_v2/`, pins the two new strings.
- The capsule's layer 2, the research capsule's L4 and the challenge's re-issue learned the projection, so
  a committed /v1 receipt still seals, still checks and still re-issues. The action sample that prints the
  sentence is history; a new one, `sworn_action_sample_2026_10_24`, is written beside it.
- The module docstring's invariant and the sworn row of `papers/INDEX.md` are narrowed to the same
  boundary, and `scripts/surface_lint.py` lists the /v1 opening as withdrawn on shipped surfaces.

## What did not change

<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/committed_receipts_modified_or_deleted_since_base" k="numeric">0 committed receipts, capsules, action samples or dry-run receipts were modified or deleted</sworn>.
Of the
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/sidecar_receipts" k="numeric">58 receipts beside a committed sidecar</sworn>,
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/sidecar_receipts_verified" k="numeric">58 re-derive</sworn>
under the `/v2` verifier at the commits they name.

## What this does not say

- That a `path:` span means more under /v2. The check is the same; the sentence now says what it is.
- That authorship is checked now. Nothing new is checked. What would check more is owed: `kind_of_tree`
  (`FINDING_tree_channel_states_no_kind_2026_09_06.md`, remedy (a)), or a harness that records the agent's
  writes across turns. Both are the operator's.
- That the files the census counts are wrong. It counts who touched them, not what they hold.
