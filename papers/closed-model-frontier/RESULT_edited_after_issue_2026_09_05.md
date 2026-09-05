# RESULT — edited after issue: the documents that moved under their certificates, and the census laid beside Charon's log

Fathom Lab · 2026-09-05 · Script: `edited_after_issue_census.py`, reading the third receipt-binding
census (`receipt_binding_census_result_2026_09_05.json`) and Charon's committed log, writing one
receipt, `edited_after_issue_census_result.json`, which this document swears to. **This document is
itself sworn**, at the commit its sidecar names. It is a result about bytes: which lines a
document lost or gained after its certificate was issued, and where the two instruments the lab
has for the same question agree. It says nothing about whether any number was true.

## The question

The receipt-binding census found certificates whose document no longer hashes to the
`document_sha256` they recorded — the document was edited after issue, and the sworn document was
recovered from the issuing commit. A reader of the published certificate beside such a document
needs two things: which lines the certificate examined are gone, and which lines with numbers
arrived that the certificate never examined. The live corpus audit re-certifies the working
document; this census says what the *published* certificate no longer describes.

## What it found

<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/documents_at_issue" k="numeric">8 certificates carry a document edited after issue,</sworn>
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/with_working_document" k="numeric">and all 8 have a working document to diff against.</sworn>
Between them the edits
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/lines_removed_total" k="numeric">removed 9 lines</sworn>
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/lines_added_total" k="numeric">and added 82.</sworn>

**Lines the certificates had examined.**
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/ledger_rows_removed_total" k="numeric">4 ledger rows sat on removed lines,</sworn>
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/verified_rows_removed_total" k="numeric">all 4 of them VERIFIED tokens,</sworn>
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/documents_with_verified_rows_removed" k="numeric">in 2 certificates</sworn>
— both over one document,
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/documents/0/certificate" k="quote">`papers/SYNTHESIS_connection_of_minds_2026_08_01.certificate.json`</sworn>
and its arXiv staging copy: the tokens
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/documents/0/ledger_rows_on_removed_lines/tokens/0/token" k="quote">`0.071`</sworn>
and
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/documents/0/ledger_rows_on_removed_lines/tokens/1/token" k="quote">`0.057`</sworn>
on line
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/documents/0/ledger_rows_on_removed_lines/tokens/0/line" k="numeric">25</sworn>
of the sworn document are not in the working one.

**Lines the certificates never examined.**
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/numbers_added_total" k="numeric">26 numbers sit on lines added after issue,</sworn>
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/documents_with_numbers_added" k="numeric">in the same 2 certificates.</sworn>
The other six edits added lines without numbers — the back-pointers the whole-program audit asked
for — and one removed a line without a ledger row.

**The live audit.**
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/live_class_equals_recorded" k="numeric">For all 8, the current verifier over the working document reproduces the recorded verdict class;</sworn>
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/totals/live_class_differs" k="numeric">0 differ.</sworn>
So no verdict moved; what moved is what the published certificate describes. The 26 numbers are
examined by the live audit and by nothing a reader of the certificate file can see.

## The census beside Charon

Charon reproduces a certificate over the working tree at ingest; the census re-derives over the
bytes at the issuing commit. Laid side by side:
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/charon/charon_oath_lines" k="numeric">213 OATH lines in the log,</sworn>
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/charon/census_certificates" k="numeric">213 rows in the census,</sworn>
every line matched to a row.
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/charon/categories/both_reproduce" k="numeric">207 both reproduce.</sworn>
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/charon/categories/charon_not_reproduced_census_stands" k="numeric">1 line Charon could not reproduce at ingest stands over its sworn bytes</sworn>
— the capstone whose receipt was regenerated after issue.
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/charon/categories/charon_unresolved_census_stands" k="numeric">3 lines Charon left UNRESOLVED stand over their sworn bytes</sworn>
— the arXiv staging copies, whose certificate sits beside no document and names one that lives
under `papers/`.
<sworn r="path:papers/closed-model-frontier/edited_after_issue_census_result.json#/charon/categories/both_not_verifier_moved" k="numeric">2 fail in both,</sworn>
with every byte in place: the verifier moved, which Charon calls SKEW. The census adds to the log
exactly what the log could not do — re-derive at the issuing commit — and contradicts it nowhere.

## What this does not say

That the removed tokens were wrong or the added numbers are unsupported: the live audit holds
over the working document, and this census reads bytes, not truth. That an edit after issue is
improper — the back-pointer edits were asked for by the audit. That the two certificates over
`SYNTHESIS_connection_of_minds` should be re-issued: that is a decision, and the census places the
facts for it. That the line-level diff sees a number changed within a line; it does not. That any
of this was run outside this lab.

---

*Of the documents that moved under their certificates, one lost two verified tokens and gained
twenty-six unexamined numbers; the rest gained a footer. The ferry log and the census agree on
every line, and the census places the six the log could only flag.*
