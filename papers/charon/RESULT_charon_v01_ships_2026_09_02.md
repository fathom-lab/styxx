# RESULT — charon v0.1 ships: the lab's record over three formats, re-derived from bytes, in one chained log a stranger can re-run

Fathom Lab · 2026-09-02 · Spec: `SPEC_charon_v01_2026_09_02.md`, committed before the module was
committed, with a dated ERRATA section appended after the adversarial pass. Battery:
`ATTACKS_charon_v01_battery_2026_09_02.md`. Module: `styxx/charon.py`. Tests:
`tests/test_charon.py`. **This document is itself sworn**, and it is the one artifact the log
excludes: a snapshot cannot contain its own description, which is the rule the corpus census
already pays. Everything else tracked in the tree is a line.

## What Charon is

Charon calls the three verifiers this lab already has — `styxx.sworn` at the commit a sidecar
names, `styxx.capsule`, and `styxx.corpus_audit` over `styxx.certify` — and writes what they
return as one line each: a canonical core chained to the line before it, carrying the receipt set
the verdict was reproduced against, every module on the derivation path with its bytes, and
whether the artifact's own recorded verdict matched. It reproduces; it never adjudicates, never
accuses, never fetches, never signs. `verify` re-derives every line under the installed verifier
and names what happened: SAME_LINE, MOVED_VERIFIER, SKEW (the line moved and the instrument's
bytes moved), DRIFT (the line moved under the same build), UNRESOLVED (bytes unavailable, never
an accusation), HEAD_MISMATCH, TAMPER.

## The log

<sworn r="path:papers/charon/charon.log.jsonl" k="hash">At the commit this document names the log's bytes hash to 74ee3ffd95b95f9181563c00b680c1e69eeae7fff8a10aba0244ad5ffd37f783.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/head" k="quote">Its head is `bcffbebc4152374f4b3301ec07aa05fd26bea39f2c31bf748f4a91507d179dc6`.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/entries" k="numeric">It carries 243 lines</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/distinct_subjects" k="numeric">over 243 distinct subjects</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/malformed_lines" k="numeric">of which 0 are malformed.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/by_kind/oath-certificate" k="numeric">213 are OATH certificates</sworn> —
every certificate `git ls-files` reports, the arXiv staging copies included —
<sworn r="path:papers/charon/charon_verify_result.json#/by_kind/sworn" k="numeric">18 are sworn documents at the commits their sidecars name</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/by_kind/capsule-oath" k="numeric">10 are OATH capsules</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/by_kind/capsule-diffgate" k="numeric">and 2 are agent-handoff capsules.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/by_status/SAME_LINE" k="numeric">Under the installed verifier 243 lines re-derived to the same core</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/by_status/TAMPER" k="numeric">and the chain check found 0 tampered lines.</sworn>
That is a determinism check and not a stability result: `verify` ran under the same builds as
`ingest`, on the same tree, minutes later. Any other outcome would have been a defect in Charon.
<sworn r="path:styxx/charon.py" k="hash">Charon's own bytes hash to 10b633c74f4e2b2b99287b66aed05cda4403c689d38509009fb03f7603f0f9eb</sworn>,
and that digest is inside every line's `verifier.modules`, so a Charon that counts differently
tomorrow reads as a moved instrument rather than as moved bytes.

The population is not a sentence. It is `papers/charon/build_log.py`, which anyone can re-run.

## What the log says that no corpus audit said

**The receipt set is on the line, and it means different things by kind.**
<sworn r="path:papers/charon/charon_verify_result.json#/receipts_n/max" k="numeric">The largest receipt set behind any line is 21.</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/receipts_n/held_total" k="numeric">Of the 227 lines whose class is held or passing</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/receipts_n/held_with_10_or_more" k="numeric">10 rest on ten or more receipts</sworn>:
<sworn r="path:papers/charon/charon_verify_result.json#/receipts_n/by_kind/oath-certificate/held_10_or_more" k="numeric">8 of them OATH certificates</sworn>
<sworn r="path:papers/charon/charon_verify_result.json#/receipts_n/by_kind/sworn/held_10_or_more" k="numeric">and 2 sworn documents.</sworn>
Only the first eight are the shape the 2026-09-01 dogfood warned about: for an OATH verdict the
verifier value-matches over every leaf, so a larger set makes HELD strictly easier, and those
eight are the lines a reader should read with that dogfood in mind. For a sworn document the
author named the leaf, so volume buys nothing; for a handoff capsule the count is bindings, not
receipts. The log prints the number and the kind, and the page prints the distinction rather than
leaving a reader to assume one rule.

**Seven lines record that their artifact did not reproduce at ingest, or had nothing to reproduce
against.**
<sworn r="path:papers/charon/charon_verify_result.json#/reproduced_at_ingest/oath-certificate/false" k="numeric">3 certificates did not reproduce — by verdict or by receipt set</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/reproduced_at_ingest/oath-certificate/null" k="numeric">3 more had no document to certify at all</sworn>,
<sworn r="path:papers/charon/charon_verify_result.json#/reproduced_at_ingest/capsule-diffgate/false" k="numeric">and 1 handoff capsule did not reproduce its gate record.</sworn>
Each carries its reason on its own line.

- <sworn r="path:papers/charon/charon.log.jsonl#L3" k="quote">The read≠write submission source at `"path": "arxiv/read_neq_write/source.md"`</sworn>
  <sworn r="path:papers/charon/charon.log.jsonl#L3" k="quote">re-certifies `"verdict": "OATH-FAILED, 10 uncovered"`</sworn> where its certificate
  recorded OATH-HELD. Its document bytes and all its receipt digests match what the certificate
  says; the verifier moved. This is the canonical arXiv and Zenodo submission source, not a
  staging copy, and it sits outside the directory the corpus audit is run over in
  `REPLICATIONS.md` — which is why no audit had reached it.
- <sworn r="path:papers/charon/charon.log.jsonl#L59" k="quote">The universal-mind capstone lists `"receipt_changed": ["mind_v0_validation.json"]`</sworn>
  and is therefore certified over an incomplete receipt set. Its verdict class did not move.
- The black-box sycophancy finding re-certifies OATH-FAILED where it recorded OATH-HELD, with no
  receipt missing or changed. Both of the last two are drift flags the corpus audit already
  carried; Charon puts them on a line with the rest.
- <sworn r="path:papers/charon/charon.log.jsonl#L96" k="quote">The handoff capsule's divergence is recorded as `gate.claims not reproduced: embedded`</sworn>, and
  the diverging field is the explanation string attached to its `tests_pass` claim, which grew
  when the evidence channel shipped. Not the retired path-claim branch — an earlier draft of this
  document said that, and the bytes do not support it.
- The three certificates with nothing to certify are arXiv staging copies whose document was
  renamed at submission. They are UNRESOLVED lines rather than absences, because a population
  defined by what survived the walk is the defect this lane has catalogued nine times.

**Ten OATH capsules reproduce that did not reproduce a commit ago.**
<sworn r="path:papers/charon/charon_verify_result.json#/reproduced_at_ingest/capsule-oath/true" k="numeric">All 10 now carry reproduced=true.</sworn>
Six of them read as failures until this cycle because `styxx.capsule` compared verdict *strings*
where the corpus auditor compares *classes*, and a later verifier appends a coverage suffix to
the string. The comparison is now on classes, the string difference is reported as advisory, and
a forged class still fails — pinned by `tests/test_capsule_uncovered_suffix.py`. That is a
loosened check, stated as one: it removes a false failure and keeps the true one.

## What the adversarial pass changed

Three attackers ran against the module before any announcement.
<sworn r="path:papers/charon/ATTACKS_charon_v01_battery_2026_09_02.md" k="hash">The battery's bytes hash to 3aed126737ea249b806dcb2771ff486c693c20fd0809e0061ddefa764a4ba163.</sworn>
Four sentences the instrument published about itself were false: that the chain catches removal
and reordering (it catches them only against a head pinned outside the log); that every line's
verdict is re-derived (capsule verdicts were copied from the embedded record); that `verify`
re-derives a line (it compared a class); and that a moved verifier reads as SKEW (it saw one file
of several on the derivation path). All four are repaired in the commit this document names, the
spec carries a dated errata for each, and the attacks that are **not** repaired — a fully rebuilt
chain, a truncated prefix, a manufactured UNRESOLVED — are named in the battery with what the
instrument prints instead.

## What this does not say

That any verdict in the log is true: a line says a verifier re-derived it. That the log is
immutable, tamper-proof or self-verifying: it is append-only by contract and re-derivable by
construction, and a chain of hashes is a chain of hashes. That a rebuilt or truncated log is
detectable without the head: it is not, and `--expect-head` is the whole defence. That anyone in
particular wrote any line: nothing is signed. That receipt shopping is prevented: it is printed.
That this is every verdict the lab has issued: the tree carries thirty-four `*.seal.json`
artifacts with no deriver here, absent from every line. That the read≠write paper is wrong: its
certificate no longer reproduces under the current verifier, which is a fact about the
certificate and the instrument, and what follows is the package owner's decision. That the
attacker was independent: the builder wrote the attackers' brief.

## Owed

Digest resolution at the issuing commit for OATH lines, after which Charon records recoveries
rather than the corpus auditor's drift flags. A deriver for `seal.json`. A second log kept by
someone who is not this lab, over artifacts we did not choose. The name survey is discharged in
the errata: three shipping projects carry the name, none in the attestation ecosystem.

---

*The river had oaths, capsules and certificates. It now has a log of crossings that a stranger
re-derives with one command, in which every line says how many receipts the crossing cost and
which instrument ferried it — and which says, in its own header, that a chain of hashes proves
order and nothing else.*
