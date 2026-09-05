# SPEC — sworn conformance vectors v0.1: the tests become bytes a second verifier can be held to

Fathom Lab · 2026-09-05 · **A spec, not a result.** Frozen in its own commit before any code.
Leg 3, item 2 of `papers/PLAN_the_next_level_2026_09_02.md`, and owed item 2 of
`SPEC_sworn_output_v02_2026_09_02.md`. It ships under the plan's own label — *the precondition
for any second verifier; no claim* — and makes no claim of its own: not that the format is
covered, not that a verifier agreeing with these vectors is correct, not that any second verifier
exists. Module additions: `styxx/sworn.py` gains one pure class (`SnapshotTree`);
`conformance/sworn/` holds the recorder, the generator, the replay and the generated set;
`tests/test_sworn_conformance.py` replays the set. A prototype census shaped the rules below; it
is not committed and none of its numbers is written here. The RESULT that follows the set
carries every count as a sworn span.

## Why this exists

`styxx.sworn` is the only verifier of sworn output. Its verdict is a pure function of bytes —
`tests/test_sworn.py` and `tests/test_sworn_attacks.py` pin that — but the pins live in Python
assertions that only this verifier can be run against. Item 5 of the plan is a browser verifier
"held to the vectors for `rN` and embedded blobs only", and a verifier can only be held to
something that exists as bytes: a document, what it was verified against, and the exact core the
verdict must reproduce, addressed by one digest so that two people arguing about a verdict are
arguing about the same bytes.

The verdict receipt cannot be that target. Its `digest` covers `verifier.sworn_sha256` and
`verifier.decisions` (sworn.py, `_RECEIPT_OUTSIDE_DIGEST` excludes only digest, timestamp and the
coverage pair), so it moves with every build of the verifier; the committed v0.2 RESULT receipt
already carries a `sworn_sha256` this checkout does not reproduce. What a second verifier can
reproduce is what `verify_receipt` compares for a `/v1` receipt: the core minus `verifier` minus
`coverage`. That is the object this spec pins.

The plan's row says the vectors are "generated from `tests/test_sworn.py`". That file alone does
not exercise R1's positive case (`rN#/pointer` HELD), R5 at rung L2, R6's L2, `rung_unknown` and
`manifest/0.1` cases, or R9's v1 re-derivation; those live only in
`tests/test_sworn_attacks.py::TestRules`. **This spec widens the source list to both files** and
records the widening in the index. Everything the set knows about the format, it learned from
tests the builder wrote.

## What a vector is

One recorded call into one of the verifier's entrypoints — its inputs as bytes, and the outcome
the verifier produced — content-addressed by a descriptor that names every blob by sha256:

```
{
 "id": "<sha256(utf8(jcs({"mode": mode, "inputs": inputs})))>",
 "family": "lexer",
 "sources": ["tests/test_sworn.py::TestLexer::test_..."],
 "rules": ["R2", "A4"],
 "mode": "inline" | "sidecar" | "canon" | "load" | "manifest" | "receipt_check",
 "requires": [] | ["manifest"] | ["tree"] | ["manifest", "tree"],
 "inputs": {
   "name": "d.md",
   "commit": null | "<40 or 64 lowercase hex>" | <the JSON value the caller passed>,
   "document": "<blob sha256>" | null,
   "sidecar": "<blob sha256>" | null,
   "manifest": "<blob sha256>" | null,
   "receipt": "<blob sha256>",
   "tree": null | {"snapshot_commit": "<hex>" | null,
                   "handle_commit": "<hex>" | null,
                   "entries": {"<path>": {"mode": "100644" | "100755" | "120000" | "040000",
                                          "size": <int>, "sha256": "<blob sha256>" | null}}}
 },
 "expect": {
   "outcome": "core" | "refused" | "sidecar" | "accepted" | "manifest" | "check",
   "core_sha256": "<sha256 of the JCS text of the core; that text is the blob of the same id>",
   "document_verdict": ..., "counts": {...}, "sworn_total": n, "unresolved": n, "rungs": {...},
   "spans": [{"at": n, "receipt": ..., "kind": ..., "verdict": ..., "reason": ...}],
   "floor": {"sworn_total": n, "narrative_sentences": n | null, "sentence_share": x | null},
   "refusal": {"where": ..., "code": ..., "match": "<substring of this verifier's message>"},
   "sidecar_sha256": "<sha256(utf8(jcs(sidecar)))>", "sidecar": "<blob sha256>",
   "manifest": {"digest": ..., "spec": ..., "rung_status": [state, rung], "intact": bool},
   "check": {"status": ..., "digest_match": bool, "verdict_reproduces": bool}
 }
}
```

Which `inputs` keys a mode carries: `inline` — name, commit, document, manifest, tree;
`sidecar` — name, commit, sidecar, manifest, tree; `canon` — name, commit, document, manifest;
`load` — sidecar; `manifest` — manifest; `receipt_check` — receipt, document, sidecar, manifest,
tree. Which `expect` keys an outcome carries: `core` — the core group and `floor`; `refused` —
`refusal`; `sidecar` — `sidecar_sha256` and `sidecar`; `accepted` — nothing more; `manifest` —
`manifest`; `check` — `check`. `expect.spans`, `counts`, `rungs` and `document_verdict` are a
human-readable projection of the core; the pinned number is `core_sha256`, and the core's JCS
text is kept as a blob so a second verifier can diff on a mismatch instead of guessing.

The set is: `conformance/sworn/index.json` (one digest over everything), `vectors/<family>.json`
(one file per family, vectors sorted by id), `blobs.json` (every byte object the vectors name,
keyed by sha256, base64, sorted), and `observer.json` (outside the digest, see C4).

## The rules, each with its attack

**C1 — the core follows the code, not the prose.** The pinned core is the verifier's output minus
`verifier` minus `coverage`: `schema`, `format`, `document`, `commit`, `manifest_digest`,
`spans`, `counts`, `sworn_total`, `unresolved`, `document_verdict`, `document_malformed`, `rungs`,
`certifies` — thirteen keys, in the order `verify()` emits them; `core_sha256 =
sha256(utf8(jcs(core)))`. R9 of the v0.2 spec lists the content-addressed core without `schema`
and `format` and with `verifier` inside it; the code emits `schema` and `format` in the core and
puts `verifier` inside the receipt's digest. The vectors follow the code and say so here rather
than choose silently. The receipt's own `digest` is never a vector's number, because it is
build-bound.
*Attack:* a second verifier that reproduces every span verdict and differs only in the
`certifies` sentence, or in one `detail` field, and calls that conformance. *Answer:* both are
inside the core by the code, so `core_sha256` moves and the vector fails; the core blob shows the
reader exactly which bytes differed. A verifier that will not carry the boundary sentence
verbatim does not reproduce this verifier's receipts and must not say it does.

**C2 — inputs are bytes, in six modes.** `inline` is `verify(raw, name=, manifest=, tree=,
commit=)`; `sidecar` is `verify(sidecar=, …)`; `canon` is `to_sidecar(raw, name, commit,
manifest)`; `load` is `load_sidecar(obj)`; `manifest` is `Manifest.from_dict(obj)`;
`receipt_check` is `verify_receipt(receipt, raw | sidecar, manifest=, tree=)`. A document is its
bytes. A sidecar, a receipt and a manifest are JSON texts that parse to the object the caller
passed — a manifest is `Manifest.core()` plus `digest` when the object carried a declared digest,
so that `intact()` is reproduced exactly, a tampered digest included. A tree is a snapshot with
modes (C10), taken after the call at the commit the verdict resolved against, with the handle's
own commit recorded beside it, because `verify()` overwrites the handle's commit with the
document's and the difference is a pinned behaviour. `commit` is the JSON value the caller
passed, so a refusal of a malformed commit is a vector too.
*Attack:* capturing a manifest through `to_dict()`, which re-digests and so cannot carry a
declared digest that disagrees with the bytes. *Answer:* `core()` plus the declared digest; the
tampered-manifest tests re-derive `manifest_integrity` on replay or the generator refuses.

**C3 — the clock and the git dates are pinned.** At generation `sworn._now` returns
`2026-09-01T00:00:00Z` and `GIT_AUTHOR_DATE` / `GIT_COMMITTER_DATE` carry the same instant, so
every manifest a test mints without fixed dates and every commit a fixture repository makes has
the same digest and the same object id on every run and every platform. The index records the
clock. *Attack:* a set whose digest never stabilises because `Manifest("h", "t")` reads the wall
clock into `minted_at` and `captured_at`, which sit inside `manifest_digest`, which sits inside
the core. *Answer:* the seam is pinned, and pinning it changes no verdict —
`test_the_verdict_is_a_function_of_bytes_not_of_cwd_or_clock` already asserts that the digest
does not depend on `_now`.

**C4 — the floor is pinned; the observer is not.** `coverage.sworn_total`,
`coverage.narrative_sentences` and `coverage.sentence_share` are a pure function of the document
bytes (the splitter literal is pinned by `test_the_coverage_splitter_is_diffgate_s_own_literal`)
and travel in `expect.floor` inside the set digest. `diff_claim_sentences`, `diff_claim_share`,
the count of `unsworn_claims`, `claimdetect_version`, and `verify_receipt`'s
`coverage_reproduces` and `same_verifier_build` are the observer's or the build's, and travel in
`observer.json`, outside the set digest, keyed by vector id. *Attack:* STRUCT-1 changes its
version string and every vector fails. *Answer:* R9 took the observer out of the receipt digest
for exactly this reason; the set follows.

**C5 — nothing is dropped silently.** An input the set cannot carry is listed, never omitted:
a manifest holding a value no JSON text can represent (the NaN test); a tree whose embeddable
blobs exceed one mebibyte or that is the repository this set lives in (the committed-v0-receipt
test resolves against the whole tree at a historical commit, and a whole repository is not a
vector); any input or outcome that no JSON text or JCS can serialise. Each is listed under
`index.unvectored.skipped` with its test id and reason. Every reason in `REASONS` that no vector
produces is listed under `index.unvectored.reasons`; every verdict in `VERDICTS` that no vector
produces under `index.unvectored.verdicts`. The replay test asserts those lists are exactly
`manifest_spec_unknown` and `git_unavailable` (the one reason the verifier declares and never
emits; the one reason only a missing git binary produces) and `WITHHELD` (no producer).
`receipt_too_large` and `not_a_blob` are produced by `TestSnapshotTree` (C10), so they are in the
set. *Attack:* a second verifier that passes every vector and believes it has met every reason.
*Answer:* the lists, and a test that pins them.

**C6 — a moved core refuses regeneration.** The generator loads the committed set before it
writes. A vector already in the set whose `core_sha256`, `refusal.code`, `sidecar_sha256`,
`manifest.digest` or `check` differs from what the run produced is a moved core: the generator
prints the vector's id and sources and exits 1 without writing anything, in the manner of
`papers/sworn/reissue_receipts_v1.py`. A moved core is a finding about the verifier, never a
reason to rewrite the set. The same id observed with two different outcomes in one run is
refused as nondeterminism. Vectors are only ever added (a new test) or dropped with notice (the
generator prints every id the run no longer produces); the set digest changes only in a commit
that says why. Every vector is replayed through `styxx.sworn` inside the generator before the
set is written, and a vector that does not replay is refused.
*Attack:* regenerating the set after a verifier change and committing the new digest as if
nothing happened. *Answer:* the refusal, the printed ids, and the rule that the RESULT which
swears to a set digest is never re-sworn over a regenerated set in place.

**C7 — one digest pins every byte transitively; provenance sits outside it.**
`set_sha256 = sha256(utf8(jcs(index minus set_sha256 minus provenance)))`. The index names every
family file by sha256 and the blob store by sha256; every vector names its blobs by sha256; every
blob hashes to its key. `index.provenance` records which bytes generated the set — the verifier's
content hash (sha256 over `styxx/sworn.py` with CRLF normalised to LF, the corpus doctrine), the
sources' content hashes, the interpreter, the platform and the package version — and is
excluded from the digest, because the set pins the verifier's behaviour and not its bytes: an
edit to `sworn.py` that moves no core leaves `set_sha256` where it was, and an edit that moves
one is refused by C6. `gen_vectors.py --check` regenerates into a temporary directory and exits
1 if `set_sha256` differs.
*Attack:* a comment edit to the verifier, or a release bump of `styxx/_version.py`, invalidating
the set and forcing a re-swear of the RESULT that names the digest. *Answer:* neither is inside
the digest; the RESULT names behaviour.

**C8 — `requires` says what a verifier needs.** `[]` when no manifest and no tree were given;
`manifest` when a manifest was; `tree` when a tree handle was, even an empty one (a handle that
names no commit yields `no_commit`, no handle yields `no_repository`, and the difference is
pinned). The legend in the index also names `git` (only a live object store reproduces it —
no vector in v0.1 carries it) and `observer` (never required; C4). A second verifier for `rN` and
embedded blobs must pass every vector whose `requires` is a subset of `{manifest}`, which is the
whole fuzz family and every sidecar, load and manifest refusal shape, and may skip vectors with
`tree`, reporting the skipped count per family. *Attack:* a browser verifier claiming
conformance while skipping the tree vectors silently. *Answer:* the skipped count is part of
what it must print; this spec says which vectors it must not skip.

**C9 — layout, names and pinning.** `conformance/sworn/` holds `recorder.py`, `gen_vectors.py`,
`replay.py`, `README.md`, `index.json`, `blobs.json`, `observer.json` and `vectors/<family>.json`;
`conformance/__init__.py` and `conformance/sworn/__init__.py` make the recorder importable as a
pytest plugin. No file under `conformance/` wears `.sworn.json`, `.sworn-receipt.json` or
`.certificate.json` (`tests/test_sworn_eol.py` and `styxx/charon.py` claim those), and none has
`result` in its name (`scripts/rigor_gate.py` and `tests/test_protocol_v2v3.py` claim that). The
whole directory is `-text` in `.gitattributes`, in the same commit as this spec, because this
box has `core.autocrlf=true` and the vectors carry CRLF, lone CR, BOM, NUL and invalid UTF-8
inside base64 while the JSON around them is LF. Every JSON the set writes goes through
`open(…, newline="\n")`. Families are named by the test class that produced the lowest source
id: `lexer`, `canonical`, `fuzz` (the seeded corpus, in full, uncapped — a capped corpus with the
same seed would be a different set), `receipts`, `tree`, `numeric`, `quote_hash_absent`,
`document`, `coverage`, `receipt_v1`, `invariants`, `cli`, `doctrine`, `worked_examples`,
`sidecar_hardening`, `receipt_hardening`, `gaming`, `attacks`, `rules`, `snapshot`; a class the
table does not know is `other`. *Attack:* a vector file swept into the sworn-artifact test or the
ferry log by its suffix. *Answer:* the test that no such name exists.

**C10 — the verifier stays pure; git enters through the recorder.** `styxx.sworn.SnapshotTree`
is a tree snapshot with modes at one commit — `{path: {mode, size, sha256, bytes}}` — that
reproduces every reason `GitTree` can return except `git_unavailable`, without a git binary:
`no_commit`, `commit_absent` (the handle names a commit the snapshot was not taken at, or one
that is not a full lowercase hex id), `path_absent`, `not_a_blob` (a tree or a symlink),
`receipt_too_large` (an entry whose bytes were not embedded because they exceed the cap), and
`prereg_not_in_tree`. It imports nothing, calls nothing, sits in section 3 of `sworn.py` beside
`MemoryTree`, and carries `from_memory`. Everything that reads git — `ls-tree -r -t -l`,
`cat-file` — lives in `conformance/sworn/recorder.py`, which the verifier never imports.
*Attack:* the four GitTree-only reasons carried as `requires: ["git"]` vectors that every
second verifier skips. *Answer:* the snapshot, and a test that `SnapshotTree.from_memory`
replays every MemoryTree vector to the same core.

**Refusals carry no codes in the verifier, and this spec adds none.** A refusal vector records
`where` (the entrypoint called), `code` (a name from the table the generator carries, one per
`SystemExit` site in `sworn.py`) and `match` (a substring of this verifier's own message, from
the same table). A second verifier is held to `code`; this verifier is checked on `match`. A
refusal message the table does not know makes the generator refuse. The table is in the index
under `refusal_codes`.

**Rule tags.** Each source test carries the v0.2 rules and battery rows it pins (`R1`–`R9`,
`A1`–`A12`), from a table in the generator; a vector's `rules` is the union over its sources.
The index carries `rule_contract`, one sentence per rule saying what a positive and a negative
vector for it look like, and the replay test holds one predicate per rule and asserts both are
present.

## What the set does not do, by construction

It proposes no span and reads no prose: every document in it was written by a test. It does not
run against a working tree: every `tree` is a snapshot. It does not gate: the replay is a test
that fails, not a verifier that refuses. It does not carry `path:` or `prereg:` receipts against
this repository's own history: that is the `committed` family, owed. It does not pin the
observer, the receipt's `digest`, `timestamp`, `verifier`, or anything that names a build.

## Tests this spec commits to

`tests/test_sworn.py::TestSnapshotTree`: every `MemoryTree` vector in `TestReceipts` replays
through `SnapshotTree.from_memory` to the same `(verdict, reason)`; a `040000` entry and a
`120000` entry are `not_a_blob`; a handle commit that differs from the snapshot commit is
`commit_absent`; a handle commit of `None` is `no_commit`; an entry over the cap with no bytes is
`receipt_too_large`; a prereg digest is found among the embedded blobs and not among the
unembedded ones; on the `git_repo` fixture a snapshot built in the test by `git ls-tree` agrees
with `GitTree` and `MemoryTree` on every reason.

`tests/test_sworn_conformance.py`, with no skip: (1) every blob in `blobs.json` hashes to its
key; (2) every family file hashes to its index entry, the blob store hashes to its entry, and
`set_sha256` re-derives from the index; (3) every vector's id re-derives from its own mode and
inputs; (4) every vector replays through `styxx.sworn` — cores by `core_sha256` and floor,
refusals by `match`, sidecars by `sidecar_sha256` and byte-exact `render`, manifests by digest
and rung status, checks by the three fields — parametrised per family, and nothing is skipped;
(5) every rule in the contract has a positive and a negative vector; (6) every reason and every
verdict is either produced by some vector or listed as unvectored, and the unvectored lists are
exactly the ones C5 names; (7) every tracked file under `conformance/` resolves `text: unset`
and holds no CR byte; (8) no file under `conformance/` wears a suffix another sweep claims;
(9) the committed set regenerates to its own digest (`gen_vectors.py --check`, exit 0).
`tests/test_sworn.py`, `tests/test_sworn_attacks.py` and `tests/test_sworn_eol.py` pass with no
verdict changed.

## What ships with v0.1

The spec (this commit, with the `.gitattributes` line). `SnapshotTree` and its tests. The
recorder, the generator, the replay, the README, the generated set, the replay test. A short
sworn RESULT that binds the set's counts and its digest at the commit that carries them, and
nothing else. A CHANGELOG entry. No INDEX row: `papers/sworn/` already has one.

## Owed after v0.1, recorded as owed

1. A `committed` family from `tests/test_sworn_dogfood.py`: the sworn documents in this tree
   with partial snapshots of only the blobs their receipts resolved and `partial: true`, and the
   v0 receipts from history for `receipt_check` — valuable to item 5, outside the plan row's
   wording, and larger than this set should be at v0.1.
2. A `refusal` attribute on the verifier's `SystemExit`, message-preserving, if item 5's
   verifier turns out to need codes at the source rather than a table.
3. The `verdict_core()` split of `verify()` — the seam item 5 ports line by line — deferred here
   because the vectors do not need it and a split that reorders the receipt's emitted keys would
   move committed receipt bytes on the next re-issue.
4. A lone surrogate in a manifest string reaching `provenance` makes `issue_receipt` raise
   where `_safe_text` guards every `detail`: no test exercises it, the recorder would list it as
   unrepresentable, and it is a row for the next attack pass, not a repair here.
5. The Python-versus-JavaScript semantics the vectors pin and item 5 must implement: Unicode
   `\d` and `\w` in the number grammar, ASCII `\s` in the bytes splitter, `str(Decimal)` in
   `detail`, a JSON reader that keeps number text, duplicate keys and NaN/Infinity, half-even
   decimal quantisation past double precision, base64 with validation, lone-surrogate detection.
   The vectors are the list; the port is item 5.

## What this spec does not say

That agreement on these vectors makes a verifier correct: it makes a verifier agree with this
one on inputs two test files chose. That the vectors cover the format: they cover what the tests
exercise, and the tests were written by the builder, which is the weakest attacker there is.
That the fuzz family is adversarial: it is a seeded random walk over a fixed list of atoms. That
any second verifier exists, or that one is easier to write because of this. That any number in
it is a measurement of anything.

---

*A verifier that only its author has run is a verifier whose behaviour lives in one head. These
vectors move that behaviour into bytes, under one digest, so that the next verifier — in a
browser, in another language, by another hand — can be shown where it disagrees, byte by byte,
before anyone is asked to trust it.*
