# RESULT — receipt binding ships: every tracked certificate now says where its bytes went, and the corpus is measured as it is

Fathom Lab · 2026-09-05 · Spec: `SPEC_oath_receipt_binding_2026_09_04.md`, frozen before any
code, with a dated ERRATA appended after the pass. Battery:
`ATTACKS_receipt_binding_battery_2026_09_05.md`. Module: `styxx/receipt_binding.py`. Tests:
`tests/test_certify_by_digest.py`. Census: `receipt_binding_census.py` →
`receipt_binding_census_result_2026_09_05.json`, the third run, from code committed at its own
head, which this document swears to and which is never regenerated in place. The second run,
`receipt_binding_census_result.json`, stays as committed; this document was first sworn against it
and re-sworn against the third after the document cell was repaired (ERRATA, second note). **This document is itself sworn**, at the commit its sidecar names, and
every number in it is bound to a leaf of the census or to a blob at that commit. It is a result
about this corpus on this day, not a claim about verification in general.

## What shipped

An OATH certificate recorded each receipt's digest and bound by basename; a receipt rewritten in
place invalidated every certificate citing it, and the audit reported *the certificate is wrong*
when the truth was *the receipt moved*. Leg 3, item 1 of the plan closes that: a certificate
issued from now on carries a `receipt_binding` block naming the content digest and the committed
blob of every receipt; the audit, given history, gives every citation one of five cells and the
document its own, re-derives the verdict over the bytes at the issuing commit, and prints a
`binding:` line beneath the verdict line that REPLICATIONS.md pins. No verdict moved. No
certificate was re-issued.
<sworn r="path:styxx/receipt_binding.py" k="hash">At the commit this sidecar names the module hashes to f3436793167cd2f1e24b8eca933b307363c911d24e9cb0cbb4eb431be99cebbb.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census.py" k="hash">The census script hashes to 48a20a7639511e3ecce7eb551cb588d839064b804e3c2e4fceda737e7ace9fc4.</sworn>

## What the census found

The population is every certificate `git ls-files` returns, staging copies included.
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/population/n" k="numeric">There are 213 of them.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/n" k="numeric">Between them they cite 631 receipts.</sworn>

**Receipts.**
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/same" k="numeric">630 citations read `same`: the bytes the certificate swore to are in the working tree.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/same_at_issue_too" k="numeric">Every one of those 630 is also present, byte for byte modulo newlines, in the tree at the certificate's issuing commit.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/at_issue" k="numeric">1 citation reads `at_issue`.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/elsewhere" k="numeric">0 read `elsewhere`.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/unbacked" k="numeric">0 read `unbacked`.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/unrecoverable" k="numeric">0 read `unrecoverable`.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/certificates/issuing_commit_unrecoverable" k="numeric">0 certificates have an unrecoverable issuing commit.</sworn>
The one `at_issue` citation is
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/examples/at_issue/0" k="quote">`papers/ancient-question-program/CAPSTONE_universal_mind_2026_06_10.certificate.json`</sworn>,
whose `mind_v0_validation.json` was rewritten seventeen minutes after the issuing commit in June
and has been reported present-and-changed by the audit since the counters of late August. Its
sworn bytes sit at the issuing commit, and the certificate stands over them:
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/certificates/regenerated_and_standing" k="numeric">1 certificate in the corpus is the plan's *receipt regenerated under a certificate*.</sworn>
Whether the receipt's content was true is not a question this audit answers.

**Documents.** The battery found that the document moves too, and the audit now looks — by the
certificate's own `document` name anywhere in the tree, since the second note of the ERRATA.
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/documents/same" k="numeric">205 certificates' documents are the sworn bytes.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/documents/at_issue" k="numeric">8 documents were edited after their certificate issued; the sworn document is recovered from the issuing commit.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/documents/moved" k="numeric">0 documents are recoverable from neither the working tree nor the issuing commit.</sworn>

**Standing.** The current verifier was re-run over the document and receipt bytes at each
issuing commit.
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/certificates/stands_over_sworn_bytes/true" k="numeric">211 certificates stand over the bytes they swore to.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/certificates/stands_over_sworn_bytes/false" k="numeric">2 do not.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/certificates/stands_over_sworn_bytes/null" k="numeric">0 could not be re-derived.</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/certificates/not_standing_same_only" k="numeric">Both of the 2 that do not stand have every receipt and the document in place</sworn>
— the verifier moved, which Charon calls SKEW and which is not a binding defect. They are
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/examples/not_standing/0" k="quote">`arxiv/read_neq_write/source.certificate.json`</sworn>,
the submission source Charon found re-certifying FAILED where it recorded HELD, and
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/examples/not_standing/1" k="quote">`papers/closed-model-frontier/FINDING_behavioral_sycophancy_blackbox_2026_06_09.certificate.json`</sworn>,
the entry `KNOWN_VERDICT_DRIFT` has carried since August. Neither is new; both are now placed.

**Newlines.** The spec predicted every match would read `crlf`. On the blob side it nearly did:
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/blob_normalisation/crlf" k="numeric">624 committed blobs match their recorded digest only after CRLF is restored,</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/blob_normalisation/raw" k="numeric">and 6 match as stored, because six recorded digests are LF hashes.</sworn>
On the working-tree side it did not:
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/normalisation/raw" k="numeric">625 working files match as read,</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/normalisation/lf" k="numeric">5 only after CRLF is stripped,</sworn>
<sworn r="path:papers/closed-model-frontier/receipt_binding_census_result_2026_09_05.json#/citations/normalisation/crlf" k="numeric">and 1 only after it is restored.</sworn>
The corpus's digests are not uniformly anything; the record says which reading each one needed.

## What the battery changed

Fifty-two constructions, thirty-six that broke a sentence or a rule, twelve in code — a merge
that hid a sworn blob, a non-ASCII name that read as absent, a document that had moved under a
certificate the audit called standing, eleven receipts read `at_issue` while sitting unchanged
in the working tree of another arc, a census that printed this checkout's absolute paths into the
file it meant to be sworn to. All are repaired in the commit this document is sworn at; the
sentences that were wrong are struck in the ERRATA rather than rewritten, and the six predictions
the spec made before the run are scored there. The code lens of the battery did not run.

## What this does not say

That a certificate is immutable, tamper-proof or self-verifying. That the receipts' contents are
true — binding says which bytes were sworn to. That the two certificates that do not stand are
wrong in any sense but the verifier's having moved since they issued. That the ten documents
edited after issue were edited improperly — the audit records the fact and places the sworn
bytes. That any of this has been run outside this lab, or on Linux. That the census is a
measurement of anything but this corpus at one commit, on a full clone, on one day.

---

*Before this, a certificate could not say which bytes it meant and the audit could not look for
them. Now the certificate says, the audit looks, and the corpus turns out to be almost entirely
where it said it was: one receipt moved and one certificate stands over it; eight documents moved
and their certificates stand over them; two certificates do not stand, and every byte of theirs
is in place. The next reviewer can re-run the census and get these numbers or file a divergence.*
