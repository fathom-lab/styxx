# The sworn conformance set

Built to `papers/sworn/SPEC_sworn_conformance_vectors_v01_2026_09_05.md`, under the plan's own
label: **the precondition for any second verifier; no claim.** Every count is in `index.json` and
in the sworn RESULT that names this set's digest; none is written here.

## What is here

| file | what it is |
|---|---|
| `index.json` | one digest over everything (`set_sha256`), the family files by sha256, the blob store by sha256, the core definition, the refusal table, the rule contract, the unvectored lists, and provenance outside the digest |
| `vectors/<family>.json` | one file per family, vectors sorted by id; a family is named by the test class that produced it |
| `blobs.json` | every byte object a vector names, keyed by sha256, base64; every blob hashes to its key |
| `observer.json` | outside the digest (C4): the observer's numbers (`diff_claim_*`, `claimdetect_version`) and the build's (`coverage_reproduces`, `same_verifier_build`), keyed by vector id |
| `recorder.py` | the pytest plugin that records every call into `styxx.sworn` while the sources run; the clock and the git dates are pinned here; git is read here and nowhere in the verifier |
| `gen_vectors.py` | runs the sources under the recorder, folds the records into vectors, replays every vector, refuses a moved core, writes the set |
| `replay.py` | replays one vector through `styxx.sworn`; the reference for what a second verifier does, entrypoint by entrypoint |

The generated files are never hand-edited. `tests/test_sworn_conformance.py` replays the committed
set with nothing skipped and fails, rather than skips, on any vector that does not reproduce.

## What a vector is

One recorded call into one of the verifier's entrypoints, its inputs as bytes and the outcome the
verifier produced, addressed by `id = sha256(utf8(jcs({"mode": mode, "inputs": inputs})))`. The
modes: `inline` (`verify(raw, name=, manifest=, tree=, commit=)`), `sidecar` (`verify(sidecar=, …)`),
`canon` (`to_sidecar(raw, name, commit, manifest)`), `load` (`load_sidecar(obj)`), `manifest`
(`Manifest.from_dict(obj)`), `receipt_check` (`verify_receipt(receipt, raw | sidecar, manifest=, tree=)`).

The pinned number of a `core` outcome is `core_sha256 = sha256(utf8(jcs(core)))`, where the core is
the verifier's output minus `verifier` minus `coverage`: thirteen keys, `schema`, `format`,
`document`, `commit`, `manifest_digest`, `spans`, `counts`, `sworn_total`, `unresolved`,
`document_verdict`, `document_malformed`, `rungs`, `certifies`. The core's JCS text is a blob of the
same sha256, so a mismatch can be diffed key by key. `expect.spans`, `counts`, `rungs` and
`document_verdict` are a readable projection of that blob, not a second pin. `expect.floor` is
the pure half of coverage and is inside the digest; the observer's half is not.

A refusal is `{where, code, match}`: a second verifier is held to `code`; this verifier is checked
on `match`, a substring of its own message. The table is `index.refusal_codes`; a code with an
empty `where` is produced by no vector and is listed under `index.unvectored.refusal_codes`.

A tree is a snapshot with modes, `{snapshot_commit, handle_commit, entries: {path: {mode, size,
sha256}}}`, replayed through `styxx.sworn.SnapshotTree`, which reproduces every reason `GitTree`
can return except `git_unavailable` without a git binary. `handle_commit` is what the handle
carried before the call; `verify()` overwrites it with the document's commit, and the difference
is a pinned behaviour.

## How to consume it

1. Read `index.json`; recompute `set_sha256 = sha256(utf8(jcs(index minus set_sha256 minus
   provenance)))` and compare.
2. For each family file, compare its bytes' sha256 to `index.families[name].sha256`; the same for
   `blobs.json`.
3. For each vector, resolve every blob it names, check each hashes to its key, and re-derive `id`.
4. Build the manifest from its blob (`Manifest.from_dict` semantics, the declared `digest`
   included when present), the tree from `inputs.tree` (`SnapshotTree` semantics), and run the
   entrypoint `mode` names.
5. Compare: cores by `core_sha256` and `floor`; refusals by `code`; sidecars by `sidecar_sha256`
   and a byte-exact `render`; manifests by `digest`, `spec`, `rung_status`, `intact`; checks by
   `status`, `digest_match`, `verdict_reproduces`.
6. `requires` says what a verifier needs. A verifier for `rN` and embedded blobs must pass every
   vector whose `requires` is a subset of `{manifest}` and may skip vectors carrying `tree`,
   printing the skipped count per family. No vector in v0.1 carries `git`.

`python conformance/sworn/gen_vectors.py --replay` does steps 3 to 5 through `styxx.sworn` and
prints pass, fail and skip counts per family.

## How to regenerate it

```
python conformance/sworn/gen_vectors.py            # regenerate in place; refuses a moved core
python conformance/sworn/gen_vectors.py --check    # regenerate in memory; exit 1 if set_sha256 differs
```

The generator loads the committed set before it writes. A vector already in the set whose
`core_sha256`, `refusal.code`, `sidecar_sha256`, `manifest.digest` or `check` differs from what the
run produced is a moved core: the generator prints its id and sources and exits 1 without writing
anything. A moved core is a finding about the verifier, never a reason to rewrite the set. Vectors
are only ever added (a new test) or dropped with notice (the generator prints every id the run no
longer produces); the set digest changes only in a commit that says why, and a RESULT that swears
to a set digest is never re-sworn over a regenerated set in place.

The set is generated with `sworn._now` pinned to `index.clock` and `GIT_AUTHOR_DATE` /
`GIT_COMMITTER_DATE` at the same instant, so every manifest a test mints and every fixture commit
has the same digest on every run and every platform. Pinning the clock changes no verdict.

The whole directory is `-text` in `.gitattributes`: the vectors carry CRLF, lone CR, BOM, NUL and
invalid UTF-8 inside base64 while the JSON around them is LF, and this is generated on a box with
`core.autocrlf=true`.

## What passing means, and does not

Agreement on these vectors makes a verifier agree with `styxx.sworn` on the inputs two test files
chose. It does not make a verifier correct, it does not mean the format is covered (the vectors
cover what the tests exercise, and the tests were written by the builder, the weakest attacker
there is), and it does not mean a second verifier exists. The fuzz family is a seeded random walk
over a fixed list of atoms, carried in full because a capped corpus with the same seed would be a
different set; it is not adversarial.

`index.unvectored` is the honest remainder: the reasons in `REASONS` no vector produces, the
verdicts in `VERDICTS` no vector produces, the refusal codes no vector produces, and every recorded
call the set could not carry, with its test id and the reason. Nothing is dropped silently.

## What is owed

A `committed` family from `tests/test_sworn_dogfood.py` (the sworn documents in this tree with
partial snapshots); a `refusal` attribute on the verifier's `SystemExit` if a second verifier needs
codes at the source; the `verdict_core()` split of `verify()`; and the Python-versus-JavaScript
semantics the vectors pin and a port must implement — Unicode `\d` and `\w` in the number grammar,
ASCII `\s` in the bytes splitter, `str(Decimal)` in `detail`, a JSON reader that keeps number text,
duplicate keys and NaN/Infinity, half-even decimal quantisation past double precision, base64 with
validation, lone-surrogate detection. The SPEC's "Owed after v0.1" section is the list of record.
