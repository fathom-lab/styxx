# SPEC — receipt binding: a certificate names the bytes it swore to, and the audit says where those bytes went

Fathom Lab · 2026-09-04 · **A spec, not a result.** Frozen in its own commit before any code.
Leg 3, item 1 of `papers/PLAN_the_next_level_2026_09_02.md`, the one thing
`AUDIT_the_whole_program_2026_09_01.md` §ADVANCE named: *bind receipts by digest, not by
filename*. It makes no numeric claim. Every number in it is a count the census below will
produce, and until that census is committed no sentence here may be quoted as a finding.

## Why this exists

An OATH certificate records three things — a document's digest, a verifier's digest, and a
dictionary `receipts_sha256` of *basename → sha256* — and then binds every VERIFIED token to a
receipt by that basename. The digest is recorded and never used to resolve: `certify_doc`
keys the dictionary by `rp.name`, and `corpus_audit._resolve_receipts` looks for a file of
that name beside the document, then for one of that name anywhere under the root whose content
matches. The binding is to a *file*, and a file can be rewritten in place.

It was, three times in two days (`a receipt is history too`, REPLICATIONS.md §corpus audit):
`corpus_census.json` rebuilt under `CORPUS_STATE_2026_08_31`, `external1_summary.json`
regenerated under `RESULT_external1_the_gate_fails_in_the_wild`, and the same pin carrying a
stale exception list. Each time the certificate went on asserting a verdict over bytes that no
longer existed, and each time the only detector was CI noticing that the certificate had stopped
reproducing — which reports *the certificate is wrong* when the truth was *the receipt moved*.
The audit cannot tell those two apart, and neither can a reader, because the certificate does
not say which bytes it meant and the audit does not look for them.

This spec makes the certificate say it, and makes the audit look. It changes no verdict and
re-issues no certificate. All 213 tracked certificates stay exactly as committed; the census is
a receipt written beside them, not over them.

## The rules, each with its attack

**R1 — the certificate names bytes.** Every certificate `certify_doc` issues from this version
on carries a block:

```
"receipt_binding": {
  "schema": "styxx-oath/receipt-binding/v1",
  "content_rule": "sha256 over bytes with CRLF normalised to LF",
  "head": "<commit the working tree was at, or null>",
  "all_receipts_committed": true | false,
  "receipts": [
    {"name": "x_result.json",
     "path": "papers/arc/x_result.json" | null,
     "raw_sha256": "<sha256 of the bytes as read>",
     "content_sha256": "<sha256 of the bytes with CRLF->LF>",
     "blob": "<git blob id the working bytes equal at HEAD, or null>",
     "committed": true | false}
  ]
}
```
`receipts_sha256` stays exactly as it is — the same dictionary, the same raw hashes — because
readers, `corpus_audit`, `test_certificate_reproduces.py` and every Charon OATH line read it.
`content_sha256` is the binding that survives a platform: the corpus doctrine
(`_receipt_sha_matches`, `charon._content_sha256`) already compares receipts modulo newlines,
and this block records that identity at mint time instead of recovering it at audit time.
`committed` is true only when the receipt's content bytes equal the blob at `head:<path>`;
an untracked or modified receipt records `committed: false` and `blob: null`, and
`all_receipts_committed` is false.
*Attack:* a certificate minted over an uncommitted receipt that is edited before the commit
lands — the exact shape of the three breakages. *Answer:* the certificate now says so in its own
bytes; a reader or an audit sees `committed: false` without any history. The CLI flag
`--require-committed` refuses to issue (exit 2, the names printed) for a caller who wants the
gate; the default reports, because a gate this lab has not measured is a gate this lab does
not ship.

**R2 — resolution on the working tree is unchanged.** Beside the document by name, then
anywhere under the root by name and content. Nothing about how a verdict is computed today
moves; a live verdict still comes from the working tree, and the pinned audit line in
REPLICATIONS.md must reproduce character for character after this ships.
*Attack:* a "by-digest" resolver that also accepts a same-named file with different content when
the digest matches nothing, to keep the document examinable. *Answer:* refused, for the reason
`_resolve_receipts` already records — a `*_result.json` from another experiment would certify
this one's claims. Digest resolution only ever *adds* a historical match; it never loosens the
working-tree rule.

**R3 — history gives every citation one of five cells.** For a certificate C, its *issuing
commit* I(C) is the last commit in `HEAD`'s history that changed C's bytes
(`git log -1 --format=%H -- <C>`); a certificate that is untracked, or modified in the working
tree, has no issuing commit. For each receipt name n with recorded digest d in C:

| cell | meaning | how it is decided |
|---|---|---|
| `same` | the bytes the certificate swore to are in the working tree | R2 resolves n and the content matches d |
| `at_issue` | the bytes existed at I(C) and were changed or removed after | some path with basename n in the tree at I(C) has blob content matching d |
| `elsewhere` | the bytes were committed, but not at I(C) | some commit ≠ I(C) that touched a path with basename n has blob content matching d; the record says `before` or `after` I(C) |
| `unbacked` | no commit ever held the bytes | every blob under every path with basename n across `HEAD`'s history fails to match d |
| `unrecoverable` | history cannot answer | no issuing commit (dirty or untracked certificate), no repository, a shallow clone, or git absent |

A `same` citation is also checked against I(C) and the record carries `at_issue_too: true|false`,
so a reader can see whether the working bytes are the issued bytes or a later identical rewrite.
Matching is modulo newlines everywhere, as R1's content rule and the existing corpus doctrine
say; the record carries which normalisation matched (`raw`, `lf`, `crlf`).
*Attack:* a cosmetic commit that rewrites every certificate (a reformat, a key reorder) becomes
I(C) for the whole corpus, so a receipt regenerated between the real issue and the reformat
reads `at_issue` for the wrong reason. *Answer:* it cannot — at the reformat commit the tree
holds the regenerated bytes, which do not match d, so the original bytes are found `elsewhere:
before`, and the census prints I(C) on every record so the reader can see a corpus-wide I(C)
and ask why. A certificate rewritten in place *is* a re-issue by this spec's definition; the
bytes changed.

**R4 — the audit re-derives over the sworn bytes, and says whether the certificate stands.**
For a certificate whose citations all sit in `same` or `at_issue`, the audit re-runs the current
verifier over the document's bytes at I(C) and the receipts' bytes at I(C), and records
`stands_over_sworn_bytes: true|false` — whether the verdict *class* over the bytes the
certificate actually swore to equals the recorded class. Any other citation pattern records
`null`, never a guess. This is the field that separates the plan's two phrases:

- *receipt regenerated under a certificate* = at least one citation `at_issue` or `elsewhere`,
  and `stands_over_sworn_bytes: true`;
- *certificate wrong* = `stands_over_sworn_bytes: false` (the verifier does not reproduce the
  verdict over the certificate's own bytes; the read≠write submission source is the known
  instance), or any citation `unbacked` (the certificate swore to bytes nobody can fetch).

*Attack:* reading `stands_over_sworn_bytes: false` on a `same`-only certificate as a binding
defect. *Answer:* it is not one; it is the verifier having moved, which `verdict_changed` already
reports and Charon calls SKEW. The census prints the two side by side and adjudicates neither.

**R5 — history is optional, and its absence is printed.** `corpus_audit` gains `--history
auto|on|off` (default `auto`: on when the root is inside a non-shallow git repository with
`git` on the path). When off or unavailable the audit prints one line
`binding: history unavailable (<reason>)` beneath the uncovered line, and every binding field is
`null`. The first line of the audit — the one REPLICATIONS.md pins — is byte-identical whether
history is on or off. CI's `replications.yml` checks out at depth 1, so CI prints the
unavailable line; the census in R6 is the receipt, run on a full clone and committed.
*Attack:* a shallow clone reporting every citation `unbacked`, which would read as 213 wrong
certificates. *Answer:* a shallow clone has no history to search; the reason is printed and no
cell is filled. `unbacked` requires a full history and says so in its record.

**R6 — the census.** `papers/closed-model-frontier/receipt_binding_census.py` runs over every
tracked `*.certificate.json` (`git ls-files`, 213 today, staging copies included as their own
rows because they are tracked) and writes `receipt_binding_census_result.json`: per citation the
cell and its evidence (path, commit, blob, normalisation), per certificate I(C) and
`stands_over_sworn_bytes`, and corpus totals with *issuing commit unrecoverable* as its own
cell in both tables. It rebuilds nothing under `papers/` but its own result; it is subject to
the rule it measures, so the RESULT that describes it swears to the result's leaves and the
result is never regenerated in place after that RESULT is committed.
*Attack:* the census writing into a file a committed certificate cites. *Answer:* its one output
is new and cited by nothing; `test_receipt_binding_census_writes_only_its_own_file` asserts the
output path against every `receipts_sha256` key in the corpus.

**R7 — the verifier stays pure; git lives in one module.** `styxx/receipt_binding.py` holds
every subprocess call (`git rev-parse`, `log`, `ls-tree`, `cat-file`); it has no dependency on
`styxx.certify` and `styxx.certify` calls it only to fill R1's block, catching every failure into
`head: null`. `styxx.corpus_audit` calls it for R3–R5. No verdict, count, ledger row or
`epistemics_summary` field is computed differently. `tests/test_certify_by_digest.py` asserts
that a certificate issued with git unavailable differs from one issued with it only in the
`receipt_binding` block.
*Attack:* GitPython or dulwich as a dependency. *Answer:* refused; the package's runtime
dependencies do not grow for an audit feature, and the module degrades to `null` without git.

**R8 — no corpus re-issue.** The plan says it, the audit says it, and R1's block is
forward-only. The census reports the corpus as it is. A future re-issue of the corpus under R1
is its own cycle with its own blast radius (every Charon OATH line moves), and this spec does
not pre-empt it.

## What the cells are expected to show, stated so it can be wrong

Recorded here before the census runs, as the falsifiable part. From the prototype on four
certificates: the recorded digests are CRLF hashes of LF blobs (every match will be `crlf`);
cross-directory citations (`PAPER_frame_locality`, the read≠write submission source) resolve
nowhere beside the document and must be found by name in the tree at I(C); most citations will be
`same`. The interesting cells are the small ones: `elsewhere: after` should contain
`external1_summary.json` (regenerated, then the certificate re-issued — the re-issue moved I(C)
past the regeneration, so the *current* certificate's bytes are `same` and the old one's are
gone); `unbacked` should contain the arXiv staging receipts if they were minted over files that
were never committed at that path; `unrecoverable` should be zero on a full clone. If the
census contradicts any of this, the RESULT says so first.

## Tests this spec commits to

`tests/test_certify_by_digest.py`, on a temporary git repository built in the test (never the
corpus): (1) a certificate issued over committed receipts records `committed: true` and a `blob`
that `git hash-object` reproduces; (2) over a modified receipt records `committed: false`,
`blob: null`, and `--require-committed` exits 2 naming it; (3) regenerating a receipt after
issue and committing yields `at_issue` for that citation, `same` for the rest, and
`stands_over_sworn_bytes: true` when the regeneration did not move the verdict; (4) a receipt
whose sworn bytes were never committed yields `unbacked`; (5) a certificate modified in the
working tree yields `unrecoverable` for every citation; (6) a receipt committed only after
the certificate yields `elsewhere: after`; (7) a CRLF and an LF copy of the same receipt produce
the same `content_sha256` and match the same recorded digest with the normalisation named;
(8) with git absent (`PATH` emptied) the certificate differs only in `receipt_binding`;
(9) `--history off` leaves the pinned first line byte-identical to `--history on`;
(10) the census's only output path is cited by no certificate. `tests/test_corpus_audit.py`
and `tests/test_certificate_reproduces.py` pass unchanged.

## Owed after this spec, recorded as owed

- A corpus re-issue under R1, so that every certificate carries its own binding block (R8).
- A blob-id search for receipts that were renamed, which the basename search of R3 cannot see;
  such a receipt reads `unbacked` today and the census's limits section says so.
- The same binding for capsules (`styxx.capsule` embeds receipt bytes and so does not have this
  defect for the document it seals, but its `receipts_sha256` inherits the certificate's).
- The read≠write submission source decision, which this spec measures and does not make.

## What this spec does not say

That a certificate is now immutable, tamper-proof, or self-verifying. That a `same` citation
proves the receipt was ever committed at I(C) (`at_issue_too` says that, or does not). That the
census's cells are a measurement of anything but this corpus on this day. That a receipt's
*content* is what it claims — binding says which bytes were sworn to, not whether they were
true. That any of this has been run outside this lab.

---

*The three breakages had one shape: a certificate that could not say which bytes it meant, and
an audit that could not look for them. After this the certificate says, the audit looks, and
the corpus is measured as it is rather than rewritten to pass.*
