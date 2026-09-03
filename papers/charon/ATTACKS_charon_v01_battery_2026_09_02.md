# ATTACKS — the charon battery: three attackers, forty-five findings, and the four sentences that were false

Fathom Lab · 2026-09-02 · **An adversarial pass, not a result.** No preregistration covers it and
no bar was frozen before it. It is the pass the standing rule requires — *no instrument is
announced before an adversarial pass* — run against `styxx/charon.py` as committed at `6b61f12`,
read-only on the repository, with every repro executed against copies under a scratch directory.
Three attackers worked in parallel on three lenses: the chain and the bytes; verdict semantics;
the page and the outward claims. Every repair below is in the commit this document ships with,
and every attack that is **not** repaired is named here with what the instrument now prints
instead.

## What was false, and is now repaired

Four sentences the instrument published about itself did not survive.

**1. "A line cannot be removed or reordered without the chain saying so."** False for anyone who
recomputes the suffix — nine lines of Python, no secret required. Reorder two lines, drop thirty
from the middle, or rewrite every receipt count, then rebuild `seq`, `prev` and `entry_id` down
the file: the chain check finds nothing. What is actually true is narrower and is now what the
module says: **the chain binds order and content to the head**, and a rebuilt or truncated log is
a different log with a different head. `verify --expect-head` and `status --expect-head` now take
the head a reader was given outside the log, `HEAD_MISMATCH` is a status, and the page prints the
command with the head in it. Without an expected head the report establishes internal consistency
and nothing more, which the report now says in its own `note`.

**2. "Every line is a verdict re-derived from bytes."** For capsules it was not. The capsule
verifier computed a live re-derivation and then returned the *embedded* verdict, so a capsule
whose embedded verdict had been edited would have carried that edit onto the line, into the class
card, and into the held count. Both capsule kinds now re-run the same pure function the verifier
runs — `certify_doc` over the embedded document and receipts, `gate_diff_text` over the embedded
summary and diff — and the line carries **that** verdict, with the embedded string beside it as
`counts.recorded_verdict`. A test forges an embedded verdict and asserts it never reaches the line.

**3. "`verify` re-derives every line."** It compared the verdict class and one module's bytes.
A line whose receipt set had grown, whose subject digest had moved and whose commit had changed
came back REPRODUCED as long as the class held — the exact receipt-shopping surface the log
exists to expose. `verify` now compares the **whole core** field by field and reports
`fields_changed`, `subject_moved` and `receipts_moved` on every row. The status that used to be
called REPRODUCED is now `SAME_LINE`, because the old word collided with the artifact-level
`reproduced` flag and the RESULT used both senses sixteen lines apart.

**4. "A verifier that moved reads as SKEW, never as drift."** True only of the one file the line
hashed. An OATH line's verdict depends on `styxx.corpus_audit` as much as on `styxx.certify`; a
diffgate line's on `styxx.diffgate` and `styxx.evidence`; every line's shape on `styxx.charon`
itself, which was hashed nowhere. A change to any of them moved verdicts under an unchanged
digest and `verify` would have called it DRIFT — *the bytes moved* — which is the opposite of the
truth. Every line now carries `verifier.modules`, the full derivation path with each module's
bytes, digested together; SKEW detection is bounded by that set and the module says so.

## Repaired, and smaller

| # | attack | now |
|---|---|---|
| CB-03 | ingest onto a headerless log appended a header **after** the entries, corrupting it | refused; a missing header is a chain problem, not a crash |
| CB-05 | the header was outside the chain: name, creation date and the `certifies` sentence could be rewritten with the head unchanged | the first line's `prev` is the header's domain-separated digest, so an edited header is TAMPER; the page prints the header's own sentence, not the module's |
| CB-06 | a BOM, a JSON array, a non-JSON line or a missing key raised out of the reader | each is a reported problem keyed by line number; a malformed line is TAMPER and is rendered as one |
| CB-14 | an artifact outside the repository was logged with an absolute path | refused by name; no absolute path is ever written into a line |
| CB-15 | `verify` on a missing log printed a clean zero | refused |
| CB-16 | one bad line marked every later line TAMPER with no way to see which broke | `chain_broken_at_line`, and later lines are labelled `UNVERIFIABLE_AFTER_BREAK` |
| CB-13 | a CRLF re-save had the same head with different bytes | the report carries `file.eol` and `file.file_sha256` |
| CB-09 | duplicate ingests inflated every count silently | `distinct_subjects` is reported beside `entries` and the spec says a line is a crossing, not an artifact |
| CB-10 | the population lived only in prose | the header carries it, and `papers/charon/build_log.py` **is** it |
| VS-5 | a sworn line's `reproduced` compared one string; a receipt file containing only `{"document_verdict": "SWORN-HELD"}` satisfied it | `styxx.sworn.verify_receipt` runs: digest, verdict re-derivation and build, with the result in `counts.receipt_check` |
| VS-6 | a sworn document that resolved nothing was still a green HELD in the held count | `receipts.vacuous`, excluded from every held count |
| VS-7 | an OATH line listed the certificate's **cited** digests as "the set it was reproduced against" | the resolved bytes are hashed onto the line; the cited set travels beside them |
| VS-8 | UNRESOLVED could be manufactured by deleting the artifact, and SKEW hid moved bytes | `subject_moved` and `receipts_moved` print on every row whatever the status |
| CB-07 / VS-4 | `subject.sha256` was documented as the document at the named commit; it is the sidecar as presented | documented correctly, and `at.document_at_commit` now answers the question directly — it is **false** for seventeen of eighteen sworn lines, because a document is committed with its sidecar, not before it |
| VS-14 | re-deriving sworn lines read every blob of the tree into one buffer per commit and died at 13 GB | `styxx.sworn.GitTree` sizes with `--batch-check` first and streams the bodies it keeps |
| PC-01 / PC-02 / VS-10 | the page's "re-derive it yourself" commands did not re-derive their lines | a `derive` subcommand prints the exact core a line carries, and that is what every row now prints |

## Not repaired, and why — the instrument prints instead

- **A forger who rebuilds the whole chain produces an honest-looking log.** Only a head pinned
  outside the log catches it. There is no signature here and the module says so; `--expect-head`
  is the whole defence and it depends on a reader having been given a head by some other channel.
- **Every prefix of a log is a valid log.** Same answer, same limit.
- **The header is chained but not signed.** An attacker who rebuilds everything rewrites it too.
- **UNRESOLVED can be manufactured** by removing bytes. It is never an accusation and it is never
  evidence of stability; `subject_moved` prints beside it when the subject still exists.
- **Duplicate lines are permitted.** A line is a crossing. Counts are of lines, and
  `distinct_subjects` sits next to them.
- **JCS key ordering** is `styxx.attestation.jcs`: ASCII keys and finite doubles. A count key
  outside ASCII is refused at ingest rather than canonicalised differently from RFC 8785.
- **`seal.json` artifacts are not covered.** The tree carries thirty-four of them, from three
  arcs, and Charon has no deriver for them. They are absent from every line and the claim is now
  "the record over three verdict-bearing formats", not "every verdict".

## What the attackers could not break

The chain mechanics do what they claim within their stated bound: an interior edit, an interior
removal, a duplicated line, a blank line before the header and a changed field inside the digested
core are each TAMPER, and every entry id re-derives from its own core. The page carries no script,
no inline handler, no `javascript:` URL and no external reference — asserted on the bytes before
they are written, not by inspection afterwards. The writes are LF on Windows. The sworn spans of
the documents in this arc re-derive. And the two numbers the log exists to print — the size of the
receipt set behind a verdict, and whether the instrument that issued it is the instrument you are
holding — were correct on every line the attackers checked by hand.

## What this battery does not say

That Charon is safe. Three attackers ran for one session against a module written the same day by
the agent that wrote the attackers' brief; that is the weakest adversary there is, and it still
found four false sentences. That the repairs are complete: each is pinned by a test, and a test
pins behaviour, not absence of defects. That the name is free — the survey found three shipping
projects called `charon` (a distributed-validator client, an account-management service, and the
strongSwan IKE daemon), none in the attestation ecosystem and two of them verification-adjacent
command-line tools; the module keeps the name inside `styxx.` and the collision is now recorded
rather than owed.

---

*The instrument said four things about itself that were not true. It now says narrower things
that are, and prints the difference where a reader will see it. The next attacker should not be
us.*
