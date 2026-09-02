# RESULT — charon v0.1 ships: the lab's whole record, re-derived from bytes, in one chained log a stranger can re-run

Fathom Lab · 2026-09-02 · Spec: `SPEC_charon_v01_2026_09_02.md`, frozen before any code. Module:
`styxx/charon.py`. Tests: `tests/test_charon.py`. **This document is itself sworn**: every count
is bound to `charon_verify_result.json` or to a line of `charon.log.jsonl` at the commit the
sidecar names, and its own sidecar cannot be in the log it describes — the log is a snapshot at
its commit and a line about the snapshot would have to be its own successor (the corpus-state
capsule's rule, inherited).

## What Charon is, in one paragraph

<sworn r="prereg:88b3fa3b1762730b49beaa0fc612cf73e3bb28f54623ea030526b2ed32bec55b" k="quote">The spec says of the ferryman that `It never adjudicates one, never accuses anyone`</sworn>.
Charon calls the three verifiers this lab already has — `styxx.sworn` at the commit a sidecar
names, `styxx.capsule`, and `styxx.corpus_audit` over `styxx.certify` — and writes what they
return as one line each: a JCS-digested core chained to the line before it, carrying the size and
digests of the receipt set the verdict was reproduced against, the verifier's own bytes, and
whether the artifact's own recorded verdict matched the re-derivation. `verify` re-runs every
line under the installed verifier and names what happened to it: REPRODUCED, MOVED_VERIFIER,
SKEW (the verdict moved and the verifier's bytes moved), DRIFT (the verdict moved under the same
build), UNRESOLVED (bytes unavailable, never an accusation), TAMPER (the chain broke). The page is
one static file with no script.

## The first log

<sworn r="path:papers/charon/charon.log.jsonl" k="hash">At the commit this document names the log's bytes hash to 228460be35365b88906815e0041be19b7cc0df14d8040118577e90616a0a9102.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/head" k="quote">Its head is `7eec6b2fbd9921bdcf777ba1257c7f313001d0817ff3dbfcb10c05aedad19b0b`.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/entries" k="numeric">It carries 240 lines.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/by_kind/sworn" k="numeric">Of these, 18 are sworn documents at the commits their sidecars name</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/by_kind/capsule-oath" k="numeric">10 are OATH capsules</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/by_kind/capsule-diffgate" k="numeric">2 are diffgate capsules</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/by_kind/oath-certificate" k="numeric">and 210 are OATH certificates re-certified over their receipts</sworn>
(every tracked certificate, the arXiv `anc/` staging copies excluded).
<sworn r="path:papers/charon/charon_verify_result.json#/by_status/REPRODUCED" k="numeric">Under the installed verifier every one of the 240 re-derived: REPRODUCED.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/by_status/TAMPER" k="numeric">The chain check found 0 tampered lines.</sworn>
<sworn r="path:styxx/charon.py" k="hash">The verifier's own bytes hash to ade9001d5e91cb3d68ba572e709f1ee6d4377f83147aad5b11846f1c6718c591.</sworn>

## What the log says that no corpus audit said

**The receipt-set size is on the line.**
<sworn r="path:papers/charon/charon_verify_result.json#/receipts_n/max" k="numeric">The largest receipt set behind any line is 21.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/receipts_n/held_total" k="numeric">Of the 227 lines whose class is held or passing</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/receipts_n/held_with_10_or_more" k="numeric">10 rest on ten or more receipts.</sworn>
Those ten are not accused of anything. They are the lines a reader should read with the
2026-09-01 dogfood in mind — an OATH verdict on fixed bytes moved from FAILED to HELD as receipts
were added — and the log now makes that reading possible without re-deriving anything.

**Reproduced at ingest is a column, and it already has content.**
<sworn r="path:papers/charon/charon_verify_result.json#/reproduced_at_ingest/sworn/reproduced" k="numeric">All 18 sworn documents reproduced their committed receipts.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/reproduced_at_ingest/capsule-oath/not_reproduced" k="numeric">6 of the OATH capsules did not reproduce their embedded verdict under the installed verifier</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/reproduced_at_ingest/capsule-diffgate/not_reproduced" k="numeric">nor did 1 of the diffgate capsules</sworn>,
and the reason is on each line: a later verifier appended a coverage suffix to the verdict string
(`OATH-HELD, 5 uncovered` against an embedded `OATH-HELD`), and the retired path-claim branch
changed the gate record. The instrument moved since those capsules were minted, and
`styxx.capsule` compares verdict strings where the corpus auditor compares classes — a repair
owed to `capsule.py` and not made here. Charon's job was to make it visible, not to hide it by
patching the verifier in the same commit.
<sworn r="path:papers/charon/charon_verify_result.json#/reproduced_at_ingest/oath-certificate/not_reproduced" k="numeric">3 certificates did not reproduce their recorded verdict.</sworn>
Two are the drift flags the corpus audit already carries:
<sworn r="path:papers/charon/charon.log.jsonl#L86" k="quote">the capstone whose receipt changed, `"receipt_changed": ["mind_v0_validation.json"]`</sworn>,
and the sycophancy finding whose verdict moved —
<sworn r="path:papers/charon/charon.log.jsonl#L114" k="quote">recorded `"recorded_verdict": "OATH-HELD"`</sworn>
<sworn r="path:papers/charon/charon.log.jsonl#L114" k="quote">and re-certified `"verdict": "OATH-FAILED, 1 uncovered"`</sworn>.
The third is new, because the corpus audit walks `papers/` and this log walked every tracked
certificate:
<sworn r="path:papers/charon/charon.log.jsonl#L33" k="quote">a staged arXiv copy at `"path": "arxiv/read_neq_write/source.md"`</sworn>
<sworn r="path:papers/charon/charon.log.jsonl#L33" k="quote">re-certifies `"verdict": "OATH-FAILED, 10 uncovered"`</sworn>
where it recorded OATH-HELD. A submission package whose certificate no longer holds under the
current verifier is the M7 mechanism reaching a document the corpus audit never reached, and it
is now a line with a number on it.

## What this does not say

That any verdict in the log is true: a line says a verifier re-derived it, nothing more. That the
log is immutable, tamper-proof or self-verifying: it is append-only by contract and re-derivable
by construction, and a chain of hashes is a chain of hashes. That anyone in particular wrote any
line: nothing is signed. That receipt shopping is prevented: it is printed. That the six
capsules are wrong: the verifier moved, and the repair belongs to `capsule.py`. That the arXiv
staging copy's paper is wrong: its certificate no longer reproduces, which is a fact about the
certificate and the verifier, and the package's owner decides what follows. That the name is
free: the survey that would price "charon" against the attestation ecosystem is owed with the
sworn survey.

## Owed

The `capsule.py` class-comparison repair, preregistered like every verifier change. Digest
resolution at the issuing commit for OATH entries (leg 3 item 1 of the plan), after which Charon
stops recording the corpus auditor's drift flags and starts recording recoveries. A second log
kept by someone who is not this lab.

---

*The river had oaths, capsules and certificates. It now has a log of crossings that a stranger
re-derives with one command, and every line says how many receipts the crossing cost.*
