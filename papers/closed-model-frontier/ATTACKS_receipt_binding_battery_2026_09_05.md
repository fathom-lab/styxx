# ATTACKS — receipt binding, the battery: three lenses, fifty-two constructions, thirty-six that broke something

Fathom Lab · 2026-09-05 · **A record, not a result.** The adversarial pass over
`SPEC_oath_receipt_binding_2026_09_04.md` (frozen at 4d75edf), the module and audit changes at
1791527, the tests, the first census and its result, and the prose that shipped with them. Four
attackers were launched (workflow `wf_28464677-1f3`); three returned — *binding semantics*,
*every sentence*, *does this repair the defect it names* — and the fourth, the *code as code* lens,
died on the session usage limit before returning anything, as did 95 of the 108 skeptic votes
that were to verify each BROKEN finding. The 14 votes that completed are recorded below; every
other finding was adjudicated by the builder, which the reader should weigh. The attackers and
the builder are the same model family in different sessions; no human attacked this.

Every attack was a construction the attacker ran — a temporary git repository, a command against
the worktree, a digest compared — against a quoted sentence or rule. **BROKEN** means the sentence
was false or the code did the wrong thing on the construction; **HELD** means the construction
failed to break it. The repairs landed in the commit after this file's; the frozen SPEC was not
edited, and carries a dated ERRATA instead.

## Lens 1 — does this repair the defect it names (15 attacks, 10 BROKEN)

| id | target | outcome | what happened | repair |
|---|---|---|---|---|
| A1 | R1, the *Attack* sentence: "the exact shape of the three breakages" | **BROKEN** (3 votes confirm) | Replayed from history: `corpus_census.json` was born in the same commit as `CORPUS_STATE`'s certificate (4033777) and rewritten 8.5 h later (5cdb349); `external1_summary.json` was committed one commit before its certificate (f015f70 → 414db45) and rewritten two commits later (584ff10). Neither was "edited before the commit lands"; both were a later commit rewriting an already-landed receipt with no certify run. R1 cannot see that moment. | ERRATA: R1 gates the shape not yet observed; R3/R4 are the instrument for the observed one |
| A2 | R3/R4 replayed over the CORPUS_STATE breakage | HELD | at the breakage commit the citation reads `at_issue`, stands `true` | — |
| A3 | the same over external1 | HELD | as A2 | — |
| A4 | "Why this exists": the audit cannot tell receipt-moved from certificate-wrong | **BROKEN** (design) | at the external1 breakage the DOCUMENT had moved too (`document_sha256` recorded 5320be2c…, live 8259c722…); the audit had no document cell and would have printed "receipt regenerated, stands" over a document that was not the sworn one | a `document` cell (same / at_issue / moved / unrecoverable) on every record, checked before any re-derivation |
| A5 | "three times in two days … each time the certificate went on asserting" | **BROKEN** (sentence; 1 vote confirms) | the third instance was a stale prose pin in REPLICATIONS.md, not a certificate over a rewritten receipt | ERRATA; CHANGELOG reworded to "twice in two days, and once in June" |
| A6 | R3 `at_issue` = "changed or removed after" | **BROKEN** (defect; 2 votes confirm) | under root `papers/ancient-question-program` the audit printed `at-issue 12` for eleven receipts sitting unchanged in the working tree of other arcs — the cell was root-relative and the meaning column was not | `classify_citation` now scans the working tree at the repository root for the sworn bytes and reads `same` with a note; the binding line prints `over N/M certificates` |
| A7 | R6 "never regenerated in place" and the test that was to enforce it | **BROKEN** (design; 1 vote refutes as a claim R6 does not make) | the first census overwrote its committed result silently; the test asserted citation-absence, which fails the moment a RESULT cites the file | the census refuses (exit 2) to overwrite a tracked result and takes `--out`; the test now proves the refusal on a temporary repository and reads the corpus by the census's own population rule |
| A8 | census `head` names 4d75edf, whose tree holds none of the code that ran | **BROKEN** (sentence; 1 vote refutes as a claim the field does not make) | the first census ran from an uncommitted working tree and named its parent | `provenance.code` blob ids and `code_committed_at_head` on every census; the CHANGELOG says head is the parent of the census's own commit |
| A9 | every `same` citation's `path` was the absolute path of this checkout | **BROKEN** (defect; 1 vote refutes the prose reading) | the bytes a RESULT would swear to depended on where the clone sat | repository-relative paths everywhere; the first result was removed from the tree before anything swore to it |
| A10 | R1 `head` dangles after a rebase | HELD | nothing reads `head` to resolve; the blob ids survive a rebase | — |
| A11 | R5 names `replications.yml` as the depth-1 workflow | **BROKEN** (sentence) | `replications.yml` never runs the corpus audit; `test.yml` is the depth-1 workflow, and it runs pytest, not the audit | ERRATA; CHANGELOG: CI never runs the audit; the shallow reason is pinned by a test on a clone the test builds |
| A12 | CHANGELOG: CAPSTONE's receipt "present-and-changed since 2026-08-27" | **BROKEN** (sentence) | regenerated at 10907dd on 2026-06-10, seventeen minutes after the issuing commit 7e70cb4; 2026-08-27 is when the counters made it visible | CHANGELOG reworded |
| A13 | REPLICATIONS: "the receipt moved; the certificate was right" | **BROKEN** (sentence) | `stands_over_sworn_bytes` says the verifier reproduces the class over the sworn bytes; it says nothing about whether the receipt's content was true, and the SPEC forbids that reading | "stands over the bytes it swore to", in both places |
| A14 | CHANGELOG's account of the two repaired certificates | HELD | — | — |
| A15 | R2/R5: the pinned line reproduces; no verdict moved | HELD | — | — |

## Lens 2 — the binding semantics (16 attacks, 12 BROKEN; 27 constructions in temporary repositories)

| id | target | outcome | what happened | repair |
|---|---|---|---|---|
| B-01 | `unbacked` when the sworn bytes sit on a side branch merged `-s ours` | **BROKEN** (defect) | default history simplification prunes a merge that is TREESAME to its first parent; the commit holding the bytes was never listed | `git log --full-history` |
| B-02 | `unbacked` when the sworn bytes are AT a merge commit | **BROKEN** (defect) | `--name-only` prints no paths for a merge without `-m` | `-m` beside `--full-history`; the blob set de-duplicates the per-parent repeats |
| B-03 | a receipt with a non-ASCII name reads `unbacked` | **BROKEN** (defect) | `core.quotepath` octal-escapes and quotes the path, so the basename never matched | `-z`, NUL-record parsing |
| B-04 | `result[1].json` reads `unbacked` | **BROKEN** (defect) | `[1]` is a character class in a `:(glob)` pathspec | every metacharacter in the basename becomes `?`; the Python basename check rejects over-matches |
| B-05 | a receipt with MIXED newlines matches no reading | **BROKEN** (defect) | raw / all-LF / all-CRLF renderings cannot reproduce a mixed file's raw digest | a fourth reading, `content`, for certificates carrying their own `content_sha256`; a stated limit for legacy receipts |
| B-06 | R4: a `same`-only certificate yields `null` when the document has no blob at I(C) | **BROKEN** (sentence) | the frozen R4 named no null case | ERRATA; `stands_reason` on every null |
| B-07 | `elsewhere` relation is `before` or `after` | **BROKEN** (sentence) | `unrelated` exists (a parallel branch merged later) | ERRATA |
| B-08 | `at_issue` while another directory under the root still holds the sworn bytes | **BROKEN** (sentence) | `_resolve_receipts` resolves the beside-the-doc file first and reports drift; the copy elsewhere was never consulted | the A6 repair reads it `same` with the other path |
| B-09 | "such a receipt reads `unbacked` today" (renamed receipts) | **BROKEN** (sentence) | renamed after a commit reads `elsewhere` or `at_issue`; only a rename before the first commit reads `unbacked` | census limits and this file say so |
| B-10 | a MISSING_DOC certificate gets no cells | **BROKEN** (defect) | the early return preceded the binding | binding computed before the missing-document and no-receipts returns |
| B-11 | a `receipts_sha256` key differing from the file only by case | **BROKEN** (defect) | basename comparison was case-sensitive on a `core.ignorecase` repository; the recorded `path` in the certificate's own block was ignored | recorded path first; case-folded comparison when the repository ignores case |
| B-12 | zero receipts: `head` null with no note, `stands` true vacuously | **BROKEN** (defect) | `bind_at_mint` had no path to open the repository; `sworn_bytes_at_issue` re-derived over nothing | the document directory anchors the repository; `note: no receipts`; `stands_reason: no citations` |
| B-13 | R3's cosmetic-commit answer | HELD (for receipts; see ES-03 for the document) | — | — |
| B-14 | "a certificate rewritten in place is a re-issue" | HELD | — | — |
| B-15 | the `at_issue`/`same` split in the test | HELD | — | — |
| B-16 | "matching is modulo newlines everywhere" | HELD | — | — |

## Lens 3 — every sentence (21 attacks, 14 BROKEN)

| id | target | outcome | what happened | repair |
|---|---|---|---|---|
| ES-01 | reading: "false on a same-only certificate is the verifier having moved" | **BROKEN** (defect) | a document edited after issue, receipts in place, verifier unchanged: stands `false` with no verifier movement | the document is checked at I(C) against `document_sha256` before any re-derivation; the reading now says "with every byte in place" |
| ES-02 | R3 cosmetic-commit paragraph, receipt side | HELD | — | — |
| ES-03 | the same paragraph, document side | **BROKEN** (design) | a cosmetic rewrite of the certificate moved I(C) past a document edit and flipped stands true → false | ES-01's repair; ERRATA |
| ES-04 | R4's two phrases are not exhaustive: `elsewhere` fell in neither | **BROKEN** (design) | "regenerated" required stands true, which R4 made impossible for `elsewhere` | R4 re-derives over `elsewhere` bytes too (the blob is known); ERRATA |
| ES-05 | I(C) "changed C's bytes" vs the parenthetical command "touched C's path" | **BROKEN** (sentence) | a rename touches the path without changing the bytes; the code implements the command | ERRATA: the command is the definition |
| ES-06 | `at_issue` meaning column | **BROKEN** (sentence) | as A6 | as A6 |
| ES-07 | CHANGELOG census bullet vs the result | HELD | — | — |
| ES-08 | `unrecoverable` row lists no-repository / shallow / git-absent | **BROKEN** (sentence) | those are R5's whole-audit unavailable state; no citation ever receives the cell for them | ERRATA |
| ES-09 | CAPSTONE dates | **BROKEN** (sentence) | as A12 | as A12 |
| ES-10 | R5 pinned line | HELD | — | — |
| ES-11 | "CI … prints `binding: history unavailable (shallow clone)`" | **BROKEN** (sentence) | as A11 | as A11 |
| ES-12 | the six predictions | **BROKEN** (sentence) | four wrong or half-wrong: working-tree matches read `raw` not `crlf`; six recorded digests are LF hashes; `elsewhere: after` could not hold `external1_summary.json`; `unbacked` held no arXiv receipt | ERRATA table |
| ES-13 | R6: "the RESULT that describes it swears to the result's leaves" — no RESULT existed | **BROKEN** (sentence) | the CHANGELOG stood in for a RESULT | `RESULT_oath_receipt_binding_2026_09_05.md`, sworn |
| ES-14 | "12 tests on a temporary repository"; test (3) vacuous with one receipt and a FAILED fixture | **BROKEN** (sentence) | 13 tests, one over the corpus; the fixture certified OATH-FAILED | two receipts, a HELD fixture, the count corrected |
| ES-15 | the census-output test globbed `papers/**` and missed two tracked certificates | **BROKEN** (defect) | POSITIONING and the read≠write source live outside `papers/` | population = `git ls-files`, as the census |
| ES-16 | "null for the 3 arXiv anc/ copies whose source.md does not exist" | **BROKEN** (sentence) | the certificate's own `document` field names a file that exists at I(C) and matches; the audit was looking by the certificate's basename | the document is also sought by the certificate's `document` field |
| ES-17 | R7 purity | HELD | — | — |
| ES-18 | forbidden words | HELD | — | — |
| ES-19 | docstring: `on` vs `auto` differ | **BROKEN** (sentence) | they were identical | `--history on` exits 2 when history is unavailable; the docstring says so |
| ES-20 | R1 `head` and `committed` semantics | HELD | — | — |
| ES-21 | "all 213 certificates stay exactly as committed" | HELD | — | — |

## Lens 4 — the code as code: not run

The attacker died on the usage limit with nothing returned. The builder re-read
`styxx/receipt_binding.py` against the lens's own list: the `cat-file --batch` reader now
handles a missing blob, a short read and a dead process by closing and raising
`RepoUnavailable` rather than hanging; `Repo.rel` on another drive is caught everywhere it is
called (`rel_or_none`); the `-z` change removed the last text-mode parse; nothing in the three
files uses a construct newer than Python 3.9. That is a builder's reading, not an attack, and
the next battery should run the lens.

## What the battery did not do

It did not attack `styxx/charon.py`'s reading of the changed audit (every OATH line's
`verifier.modules` now moves, which Charon calls SKEW and which this leg expected). It did not
run on Linux. It produced no number about anything but this branch.

---

*Thirty-six sentences and rules broke, twelve of them in code. The instrument that was built to
say where a certificate's bytes went could not, before this pass, say where the document went,
could not see past a merge, could not read a non-ASCII name, and printed its own checkout's paths
into the file it meant to be sworn to. All of that is repaired in the next commit; the sentences
that were wrong are struck in the SPEC's ERRATA rather than rewritten.*
