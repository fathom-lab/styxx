# CANNED — the JUnit adapter's manifest, verified through the command line

Fathom Lab · 2026-09-05 · **A canned run, not a result.** The manifest embedded in this
document's sidecar was minted by `python -m styxx.harness junit papers/sworn/canned_harness_junit.xml --rung L1 --turn canned_harness_junit`
over a report written by hand for this purpose: three passing testcases in one suite and a
collection error in a second, so that the reader's separation of failures from harness errors is
visible in the receipts. Every sentence below is bound to that manifest and verified with
`python -m styxx.sworn verify ... --manifest ...`; nothing here is a measurement of anything.

<sworn r="r1" k="numeric">The reader resolved 3 passed testcases.</sworn>
<sworn r="r2" k="numeric">It resolved 0 failures.</sworn>
<sworn r="r4#/totals/errors" k="numeric">The collection error is kept apart as 1 harness error.</sworn>
<sworn r="r4#/outcome" k="quote">With a harness error on the report the reader's outcome reads `FAILED`.</sworn>
<sworn r="r4#/sources/0/format" k="quote">The reader sniffed the format `junit` from the bytes, never from the extension.</sworn>

The rung is L1 because the report was written and the manifest minted by the same hand on the
same machine: declared by the caller, printed beside every verdict above, and checked by nothing.
