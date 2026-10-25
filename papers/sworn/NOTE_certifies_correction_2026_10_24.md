# NOTE — the receipt's certificate sentence said more than the verifier checks; schema /v2 corrects it

Fathom Lab · 2026-10-24 · A correction, on the route the operator chose for it. No receipt committed on main is
edited. Every count inside a sworn span below is sworn to `papers/sworn/certifies_census_2026_10_24.json`, which
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
  with no manifest nothing about authorship is consulted. Every resolved span of these two forms prints
  "authorship unchecked", with a manifest or without: nothing beyond that membership was tested;
- `authored_sha256` is the harness's record of what the agent produced in one turn. The code says what
  follows, in its own comment: "a file the agent committed under a name the harness never digested is still
  not caught" (l.1262-1264), and `SPEC_tree_channel_authorship_v01_2026_09_06.md` says so under "What this
  spec does not say". The comment's "name" is loose: the membership is tested on the bytes' sha256, never
  on a name, so the bytes the harness digested, committed under a name it never saw, are refused. What is
  not caught is bytes the harness never digested, such as a file written in another turn, or by a shell
  command or a tool the harness does not digest;
- an UNRESOLVED span was not compared at all, and the sentence opened with "the spans … were checked".

The adapters' own documents already drew the line the sentence did not: `sworn/README.md`
("`authored_sha256` catches only the trivial laundering of a committed file into a receipt") and
`integrations/claude-code/sworn-hooks/README.md` ("What this does NOT attest — That the agent did not write
a file").

In this lab the gap is the ordinary case.
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/sidecars" k="numeric">The 58 committed sidecars</sworn>
carry, at `2e98debc`,
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/path_spans" k="numeric">1,261 spans that name a committed file</sworn>
(`path:` spans) over
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/path_targets" k="numeric">137 distinct files</sworn>,
and
<sworn r="path:papers/sworn/certifies_census_2026_10_24.json#/path_spans_on_speaker_touched_files" k="numeric">1,261 of those spans rest on a file whose git history names one of the lab's five agent names as author or committer</sworn>.
A HELD `path:` span here has been a consistency check between the speaker's sentence and files the
speaker, or the operator, committed. That says nothing about whether those files are true; it says what
the receipt certified. The spans of this NOTE are the same case: each is a `path:` span on the census, a
file a speaker committed, so what a reader can rely on is that `papers/sworn/certifies_census.py --check`
regenerates the census from git objects and from `styxx.sworn`.

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

  > the spans the author bound that read HELD or FAILED were checked, each by its kind, against the receipt
  > it names (its bytes, or for a hash span on a manifest entry that carries none, the sha256 the manifest
  > records), at the commit and manifest this receipt records, and an rN span at the rung the manifest
  > declares (a path: or prereg: span carries no rung and is counted as committed); the only authorship test
  > reads the manifest: an rN receipt is taken as not the author's when the manifest records it under a
  > kind_of_source from the closed external vocabulary and does not list its sha256 in authored_sha256, and
  > a path: or prereg: receipt when no manifest is given or the manifest does not list its sha256 there (its
  > span's provenance reads authorship unchecked: nothing beyond that membership was tested);
  > authored_sha256 is the harness's record of what the agent produced in the turn the manifest names,
  > tested on the bytes' sha256 and never on a name, so bytes the author wrote that the harness never
  > digested, such as a file written in another turn, or by a shell command or a tool the harness does not
  > digest, are not caught; a MALFORMED span was refused before any comparison, the authorship refusals
  > among them, and an UNRESOLVED span was not compared — NOT a claim that the author did not write the
  > receipt's bytes, NOT a claim that the document is correct, NOT a claim that the right sentences were
  > bound, NOT a check that the tags were written at write time, NOT a check of any signature, and only as
  > trustworthy as the harness that minted the manifest and the tree that served the commit (git history
  > from the CLI; a snapshot or in-memory tree on its caller's word)

  (`_CERTIFIES` in `styxx/sworn.py` is that text on one line; the /v1 text stays beside it as
  `_CERTIFIES_V1`, marked withdrawn, loaded and never emitted.)
- A draft of the /v2 sentence, committed on the branch that carries this NOTE and reviewed before merge,
  attached "at the rung the manifest declares" to every span. Only an `rN` span has a rung: a `path:` or
  `prereg:` span has none in its provenance, and the receipt's `rungs` block counts it as `committed`, so
  under a manifest declaring L3, a rung the verifier refuses, a `path:` span still reads HELD beside an
  `rN` span that reads UNRESOLVED. The sentence above is the corrected one. The receipts the draft was printed
  in, this NOTE's and the new action sample's, were issued again before merge; no receipt on main carries
  the draft.
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

## What else moved

- The `/v1` conformance set's `set_sha256`, and the ids of its `receipt_check` vectors whose input receipts
  name the verifier's build, moved, as they do at every edit of `sworn.py`; no core vector and no expected
  outcome moved (`conformance/sworn/README.md`).
- `same_verifier_build` reads false on every committed receipt an earlier `sworn.py` issued. It is advisory,
  outside the digest and outside the compare.
- The 14 sworn lines of `papers/charon/charon.log.jsonl` that recorded
  `counts.receipt_check.same_verifier_build` true carry that build-relative flag inside the core charon
  compares, so under any edited `sworn.py` they read SKEW under charon's SPEC E5 (the line moved and the
  verifier moved), and the stranger's ferry-log step (`python -m styxx.stranger`) read FAIL on this branch.
  Their receipts still verify, and the flag is the only value that moved in each. On the operator's decision
  of 2026-10-25, `papers/charon/AMENDMENT_charon_same_verifier_build_flag_2026_10_25.md` reads such a line
  MOVED_VERIFIER when that flag alone moved, from true to false, while the `sworn.py` the line names is not
  the installed one and the receipt still names it; under the same build it stays DRIFT, and beside any other
  moved value it reads what E5 reads. The 14 lines now read MOVED_VERIFIER, the ferry-log step reads PASS,
  `papers/checksum/STRANGER.md` says so beside its step 2, and no committed line was edited.

## What this does not say

- That a `path:` span means more under /v2. The check is the same; the sentence now says what it is.
- That authorship is checked now. Nothing new is checked. What would check more is owed: `kind_of_tree`
  (`FINDING_tree_channel_states_no_kind_2026_09_06.md`, remedy (a)), or a harness that records the agent's
  writes across turns. Both are the operator's.
- That the files the census counts are wrong. It counts who touched them, not what they hold.
