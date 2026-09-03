# SPEC — Charon v0.1: the ferry log — every verdict this lab ever issued, re-derived from bytes, chained, and printed with the size of the receipt set it was bought with

Fathom Lab · 2026-09-02 · An engineering spec (not a measurement prereg): frozen before any code
exists. Module: `styxx/charon.py`. Log schema `styxx.charon/log/v0`, entry schema
`styxx.charon/entry/v0`. Successor to `RECON_state_2026_09_01.md`, whose finding this spec is
built around and whose recommendation it adopts verbatim.

## What Charon is

The Styx is the river; Charon is the ferryman; nothing crosses unseen. Charon is **the log of
crossings**: an append-only, hash-chained record in which every line is a verdict that was
*re-derived from bytes* at the moment it was written, over the three verdict-bearing artifacts
this lab already produces — a sworn document at a named commit, a capsule, an OATH certificate
with its receipts. Charon reproduces verdicts. It never adjudicates one, never accuses anyone,
and never fetches anything.

Three things make it more than a list.

1. **Every line is a pure function of bytes the lab committed.** A stranger with the package and
   the repository at the named commits re-derives the whole log with one command and gets the
   same lines, or a named reason why not.
2. **The size of the receipt set is on the line.** The 2026-09-01 dogfood showed an OATH verdict
   on fixed document bytes moving from FAILED to HELD as the author supplied more receipts,
   because the verifier value-matches over every leaf. Charon cannot stop that and does not try;
   it prints `receipts.n` and the digest of every receipt beside every HELD, so a verdict bought
   with volume is visible in the log rather than hidden in it. *A HELD entry means: this document
   matched the receipt set its author chose to embed. A larger set makes HELD strictly easier.*
3. **A verifier that moved reads as SKEW, never as drift.** The audit's mechanism M7 — every new
   verifier field silently partitions the corpus — becomes a status: when re-derivation under the
   installed verifier disagrees with the line AND the verifier's own bytes differ from the ones
   the line names, that is SKEW, a fact about the instrument; when the verifier is the same build
   and the verdict moved, that is DRIFT, a fact about the bytes. The two were one word in every
   corpus audit before this.

## The entry

The content-addressed core, digested under RFC 8785 JCS (`styxx.attestation.jcs`) into `entry_id`:

```json
{"schema": "styxx.charon/entry/v0",
 "seq": 1,
 "prev": null,
 "kind": "sworn | capsule-oath | capsule-diffgate | oath-certificate",
 "subject": {"name": "RESULT_x.md", "path": "papers/…/RESULT_x.md", "sha256": "<of the artifact bytes>"},
 "at": {"commit": "<40-hex or null>", "manifest_digest": "<or null>"},
 "receipts": {"n": 4, "sha256": ["<sorted, unique>"]},
 "verifier": {"styxx_version": "7.47.0", "module": "styxx.sworn", "module_sha256": "<of the verifier's bytes>"},
 "verdict": "SWORN-HELD",
 "verdict_class": "HELD | FAILED | UNSWORN | PASS | FAIL",
 "counts": {},
 "floor": 0.1667,
 "rungs": {"committed": 9, "undeclared": 4},
 "reproduced": true}
```

Outside the digest: `entry_id`, `timestamp` (volatile, disclosed as unsealed), `note`.

- `subject.sha256` is over the primary artifact's bytes: the inline sworn document as committed
  at the named commit; the capsule file; the OATH-certified document.
- `at.commit` is the sidecar's commit for a sworn entry, null for a capsule (its bytes are inside
  it), and the certificate's own commit is NOT recorded for an OATH entry in v0.1 — resolution
  follows `styxx.corpus_audit` (working tree, drift flagged), because digest resolution at the
  issuing commit is a separate leg of the plan and Charon does not pre-empt it.
- `receipts.sha256`: for a sworn entry, the `resolved_sha256` of every span plus every manifest
  receipt; for a capsule, the embedded receipts' digests (v0.1) or the summary and diff bindings
  (v0.2); for an OATH entry, the certificate's `receipts_sha256` values. Sorted, unique, so two
  ingests of the same artifact produce the same core.
- `verdict` is the artifact's own verdict string, re-derived; `verdict_class` is the
  HELD/FAILED dichotomy the corpus auditor already defines (`verdict_class`), extended with
  UNSWORN for sworn documents and PASS/FAIL for diffgate capsules.
- `reproduced` records whether the artifact's OWN recorded verdict (its committed receipt, its
  embedded certificate or gate) matched the re-derivation at ingest. `false` is an entry, not a
  refusal: Charon logs what it saw.
- `prev` is the previous entry's `entry_id`; `seq` is dense from 1. The log's `head` is the last
  `entry_id`.

## The log

JSON Lines, LF, one entry per line, first line a header `{"schema": "styxx.charon/log/v0",
"name": …, "created": …}`. Append-only by contract: `ingest` refuses to write if the existing
lines do not chain, and never rewrites a line. A log is a file anyone can copy; its `head` is
the one string that summarises it.

## Ingest — reproduce, never adjudicate

`python -m styxx.charon ingest --log L [--repo R] PATH…`

| artifact | how the verdict is re-derived | needs |
|---|---|---|
| `*.sworn.json` (sidecar) | `styxx.sworn.verify(sidecar, tree=GitTree(repo, sidecar.commit))` | the repository, at the commit |
| `*.capsule.html` | `styxx.capsule.verify_capsule(path)` — a pure function of the file | nothing |
| `*.certificate.json` | `styxx.corpus_audit.audit_document(path)` — the working-tree receipts, drift flagged | the tree beside it |

Anything else is refused by name. A sworn entry whose commit is absent, a certificate whose
document or receipts are missing: written as an entry with `verdict` `UNRESOLVED` and
`verdict_class` `UNRESOLVED`, `reproduced: false`, and the reason in `counts.reason` — the
verifier declining to see, never an accusation, and never silently skipped (a population defined
by what survived the fetch is the defect this lane has catalogued nine times).

## Verify — the log re-derived, line by line

`python -m styxx.charon verify --log L [--repo R]` recomputes every `entry_id` and every `prev`,
then re-derives every entry from bytes under the installed verifier and assigns one status:

| status | meaning |
|---|---|
| `REPRODUCED` | same `verdict_class`, same verifier `module_sha256` |
| `MOVED_VERIFIER` | same `verdict_class`, different verifier bytes — informational |
| `SKEW` | different `verdict_class`, different verifier bytes — the instrument moved |
| `DRIFT` | different `verdict_class`, same verifier bytes — the bytes moved |
| `UNRESOLVED` | bytes unavailable now (commit absent, file gone) — never an accusation |
| `TAMPER` | `entry_id` does not re-derive from its core, or `prev`/`seq` do not chain |

Totals are printed by status and by kind. Exit status is 0 for every status except TAMPER,
because SKEW and DRIFT are facts about instruments and bytes that the log exists to show, while
TAMPER is the log not being what it says it is.

## The page — offline, zero scripts

`python -m styxx.charon page --log L --out index.html` renders the log as one static file with no
JavaScript and no external request: the head hash; totals by status and kind; one row per entry
with `seq`, kind, name, verdict, `receipts.n`, rungs, verifier build, `entry_id`, and the exact
command that re-derives it; and a footer that states what the page proves — that these lines are
in this log — and what it does not: that any verdict is *true*, that anyone in particular wrote
any line, or that a receipt set was not chosen to pass. The palette is the lab's terminal:
black, matrix green, cyan, monospace.

## What Charon does not do, by construction

- **Adjudicate.** It calls the three verifiers and writes what they return.
- **Sign.** No key material, no PKI, no timestamping authority. A re-mint over different bytes is
  a different honest entry; who wrote a line is not proven.
- **Prevent receipt shopping.** It prints `receipts.n` and the digests so it can be seen.
- **Resolve receipts by digest at the issuing commit.** That is the plan's leg 3 item 1; Charon
  v0.1 records the corpus auditor's drift flags and will carry the digest resolution when it
  ships.
- **Fetch.** The repository at a commit and the files on disk are its whole world.
- **Say "immutable", "blockchain", "tamper-proof" or "self-verifying".** The log is append-only
  by contract and re-derivable by construction; a chain is a chain of hashes and nothing more.

## The name

"Charon" is this lab's name for the ferryman of its own river. Whether the name collides with a
shipping project in the attestation ecosystem is a question for the prior-art survey the sworn
spec already owes; until it runs, the module ships under this name inside `styxx.` and the
spec records the check as owed.

## What ships with v0.1

`styxx/charon.py`; `tests/test_charon.py` (entry determinism across timestamps; chain tamper;
SKEW versus DRIFT under a monkeypatched verifier hash; UNRESOLVED on an absent commit; every
write LF; the page contains every entry id and no `<script`); the first log,
`papers/charon/charon.log.jsonl`, over every sworn sidecar, every capsule and every tracked
certificate in the tree; `papers/charon/charon_verify_result.json` from `verify`; a sworn
RESULT over that receipt; `papers/charon/index.html`.

---

*The river already had oaths, capsules and certificates. It did not have a log of crossings a
stranger could re-derive, or a line that said how many receipts a verdict cost. Now it does, and
the line that says it is itself a pure function of the bytes.*

---

## ERRATA — 2026-09-02, after the adversarial pass

This section is appended; nothing above it is edited. The pass is
`ATTACKS_charon_v01_battery_2026_09_02.md`; the schemas move to `styxx.charon/log/v1` and
`styxx.charon/entry/v1`, and the v0 log written before the pass stays in history at `a5cf9ec`.

**E1 — "frozen before any code exists" is corrected to "committed before the module was
committed."** The record shows the spec commit six minutes before the module commit, which is
consistent with the code existing first. The weaker sentence is the one the bytes support.

**E2 — the chain's guarantee is narrower than the Verify table implied.** A forger who rebuilds
`seq`, `prev` and `entry_id` down the file produces a chain that checks. The chain binds order
and content **to the head**; a rebuilt or truncated log is a different log with a different head.
`verify` and `status` take `--expect-head`, `HEAD_MISMATCH` joins the status vocabulary, and
without an expected head a report establishes internal consistency only.

**E3 — the header is chained.** Entry 1's `prev` is `sha256("styxx.charon/log/v1\n" + jcs(header))`,
domain-separated so a header can never masquerade as an entry. An edited header is TAMPER.

**E4 — capsule verdicts are re-derived, not copied.** `derive_capsule` re-runs the pure function
the capsule verifier runs and puts that verdict on the line; the embedded string is
`counts.recorded_verdict`.

**E5 — `verify` compares the whole core.** REPRODUCED is renamed `SAME_LINE` and requires every
compared field to match; `fields_changed`, `subject_moved` and `receipts_moved` print on every row.

**E6 — the verifier is a path, not a file.** `verifier.modules` carries every module the
derivation ran through, Charon included, each with its bytes; `verifier.digest` is over that list.
SKEW detection is bounded by the hashed set and a change outside it reads as DRIFT.

**E7 — the receipt set on an OATH line is the RESOLVED one**, hashed from the bytes handed to the
verifier; the certificate's cited digests travel beside it as `receipts.cited`.

**E8 — `reproduced` is three-valued.** `true`/`false`/`null`; for a sworn line it is
`styxx.sworn.verify_receipt`, not a string comparison; `null` means no recorded verdict exists.

**E9 — the shipped verdict-class vocabulary** is HELD, FAILED, UNSWORN, OATH-HELD, OATH-FAILED,
PASS, FAIL, UNRESOLVED. The entry table above names a shorter set; the OATH prefix is kept so an
OATH class is never confused with a sworn one.

**E10 — `subject.sha256` for a sworn line is the sidecar as presented at ingest**, not the
document at the named commit; `at.document_at_commit` answers that question separately and is
false for seventeen of the eighteen sworn lines, because a document is committed with its sidecar.

**E11 — the population is a script.** `papers/charon/build_log.py` enumerates it and the header
carries the rule. The arXiv staging certificates enter as UNRESOLVED lines rather than as
absences. The one exclusion is the sworn RESULT that describes the log: a snapshot cannot contain
its own description, the rule the corpus census already pays.

**E12 — scope.** Charon covers three verdict-bearing formats. The thirty-four `*.seal.json`
artifacts in the tree have no deriver and appear on no line; "the record over three formats" is
the claim, not "every verdict".

**E13 — a malformed line is TAMPER, never a traceback**, and duplicates are permitted: a line is
a crossing, not an artifact, so every count is of lines and `distinct_subjects` is reported beside
them.

**E14 — the name.** Three shipping projects are called `charon` (a distributed-validator client,
an account-management service, the strongSwan IKE daemon). None is in the attestation ecosystem;
two are verification-adjacent command-line tools. The module keeps the name inside `styxx.`.
