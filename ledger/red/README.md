# red ledger: the mutation ledger of the reader

a second chain under the protocol: every text in it was planted by a script from a sealed list; nothing here was sent to anyone.

this folder holds a SECOND hash-chained ledger under the protocol in `../PROTOCOL.md`, written by a copy of darkflobi's gate (the gate's code as copied at 2026-11-08T22:46:08Z, digests in STATS.json `gate_copy_sha256`: the chain code `ledger.py` and the reader `helpers/numerals.py` byte-equal to the live ones; the live `gate.py` moved to the v1 send schema the same evening, so these are v0 send lines and darkflobi's newest are v1; its own salt, a config of synthetic ids, shadow) while a script planted numeral forms into it from a sealed per-batch list. it is not darkflobi's chain and says nothing about his traffic. what it measures: the gate's own numeral reader (`helpers/numerals.py` over the pinned snapshot `3437340b`) against texts whose one numeral form is known in advance; the misses are counts with denominators that move when the reader is repaired.

- `ledger.jsonl` — the red chain as the copy wrote it, every line, byte for byte (nothing withheld: the lines carry hashes and counts; the texts are the planter's and sit in the batch files).
- `batches/<id>.json` — the sealed list of one batch (20 texts, each one form); `batches/<id>.results.json` — the planter's record of the verdict lines. the seal: `red batch <id> sha256 <sha256 of the batch file>` went through the gate before every text of the batch, so its line precedes them; a batch file changed after sealing fails that check.
- `forms.json` — the catalogue (25 forms: 9 counted by design, 6 excluded by design, 10 open question); `RED_PREREG.md` — the kill gates, written before the first batch (sha256 `658bab0adc2e25b118672a0b3112afb913fffc7cdfdb463231e31a6fcb29d256`, also on the chain's started line); `RED_AMENDMENT_2026_11_08.md` — the dated amendment after the first five batches (sha256 `409d0808e7efddc2bfec90c6f4dfd745d6d971ef1d96bca06d7cc4df81e57b5f`): the probe disclosure, the void-batch rule, the unattributed-line rule, the digit-free split, the gate_sha256 note, the planter guards.
- `anchors/<date>.txt` — the protocol's anchor text for the red chain's last line of a day, posted to NO chat: a file, stamped with OpenTimestamps (`.ots`), committed here; `anchors/index.json` — each proof's state.
- `index.html` — re-derives the chain in the browser with `../verify.js` (the same verifier as the main page), re-runs the seal checks, and shows the misses, the exclusions and the open-question totals; `STATS.json` — the builder's counts.

## the counts (built 2026-11-08T23:32:57Z from head `fc22d2ee00b17f7a`, 106 lines)

- send lines 105 = 5 seal + 100 planted + 0 unattributed; planted forms counted here: 100 in 5 batches (0 void: none).
- seal lines 5 of 5 re-derive (the batch file's sha256 is the digest the seal names, the seal line precedes every text of the batch, every text re-hashes onto its line; texts contiguous at seal+1..seal+20 in file order: 5 of 5); seal lines whose sha fragment the reader counted: 0 of 5 (kill gate c: held; the seal text also carries the token `sha256`, a second designed exclusion, so a positive would need `unbound_shapes`, kept in STATS.seals.detail).
- counted by design: 36 of 36 counted; 0 of 36 missed; 0 of 36 extra. missed: none. extra: none.
- excluded by design: reader exclusions held 18 of 18 (year, date, version, sha fragment, list marker: texts with a digit); digit-free texts handed to no reader 6 of 6 (the gate's pre-filter `_ANY_DIGIT`, not a reader rule; total 0 by construction). not held 0 of 24: none.
- open question (40 plantings), observed totals per form:
  - `accounting_sign`: total 1: 4 of 4 — V15_ACCOUNTING_SIGN off by decision (#242); the amount inside the parentheses is a plain token — read in part; the flag is off
  - `bare_magnitude_words`: total 1: 4 of 4 — V15_MAGNITUDE_BARE off by decision (#242); the mantissa alone is a token at the pin — read in part; the flag is off
  - `clock_time`: total 2: 3 of 4; total 3: 1 of 4 — V15_FORM_NOISE off by decision (#242); no rule at the pin names a clock time — reads outside any rule
  - `code_span`: total 1: 4 of 4 — disclosed unread by the v0.13 band (numerals inside code) — counted though disclosed unread
  - `ordinal`: total 0: 4 of 4 — disclosed unread by the v0.13 band (letter-glued numbers) — unread, as disclosed
  - `scientific`: total 0: 4 of 4 — disclosed unread by the v0.13 band (scientific notation) — unread, as disclosed
  - `sentence_final`: total 1: 3 of 4; total 2: 1 of 4 — V15_SENTENCE_FINAL off by decision (#242); the v0.13 band's trailing-period counts it meanwhile — read in part; the flag is off
  - `suffix_magnitude`: total 0: 4 of 4 — V15_MAGNITUDE_BARE off by decision (#242); the $-led case is V15_MAGNITUDE, on in v2, unreachable at the pin — unread; the flag is off
  - `unicode_digits`: total 1: 4 of 4 — disclosed unread by the v0.13 band (non-ASCII digits) — counted though disclosed unread
  - `url_numeral`: total 1: 4 of 4 — no rule names a url at the pin or in v2 — reads outside any rule
- kill gate (a): red digests found in darkflobi's published ledger: 0 of 106 lines checked.
- kill gate (b): 0 misses of 36 counted-by-design plantings among 100 planted forms: a null result with its denominators, not a failure of the instrument. the counted-by-design and excluded-by-design classes were checked against the pinned extract_numbers outside the gate before the catalogue froze (BUILD_LOG_3c.md section 2; one smoke text of 22:46Z went through the copy and its chain was deleted before the started line): the null result is the gate agreeing with that probe, not a recall measurement over unseen forms. what can move here is the open-question totals, a form added at a later started line without a probe, and any later reader
- gate_sha256 on the lines: seqs 1-106 carry 20192b8d6f0db61e… = the copy's *.py + helpers/*.py PLUS the LIVE plugin folder as it stood at 22:51Z (no STYXX_GATE_PLUGIN_DIR set; moves when the live plugin is redeployed). a change of this digest between two red lines is RED_AMENDMENT rule 6 (the plugin folder it folds in), dated, not a change of the gate code; the copy-only digest (no plugin folder) is 7c8bddf45e3a4fb1…
- the chain: 106 lines (send 105, started 1); head `fc22d2ee00b17f7a21172288cb30758d2113d7abcd9d3357d387ad4797bfb019`; file sha256 `40430e31d4f07df12a496edaaec040b4a50f4a39fe42f3f2c2bf68680dad6d68`; both implementations (the copy's ledger.py, the published verify.js) re-derive it to the same head.
- anchors: 1 (0 in a bitcoin block, 1 pending, 0 not stamped); the chain's own anchor line for a day is written after 00:00 new york of the next day (anchor-line first, then any new batch). the anchor text follows protocol v0 and names no ledger: the file path `anchors/<date>.txt` under ledger/red/ and the head hex are the binding.
- `sent_sha256` differs from `checked_sha256` on allow_marked lines (64 of 105 send lines) because the gate appends its marker trailer to the sent text; the count is over the checked text, and the seal check uses `checked_sha256`.
- vectors 10 of 10 under the same verifier (the verifier's conformance to the format, not a property of the red chain)

## check it without the page

    node ../check_ledger.cjs ledger.jsonl
    sha256sum batches/<id>.json            # the digest the seal text names
    printf 'red batch <id> sha256 <that digest>' | sha256sum   # = checked_sha256 of the seal line, which precedes the batch

## what this is not

- not darkflobi's traffic, and not a measure of the live gate's refusals: synthetic texts, a synthetic destination, shadow.
- not a truth check: the reader counts numerals; the planted sentences claim nothing.
- not a stranger's chain: the same lab, the gate's code as copied at 2026-11-08T22:46:08Z, the same box. it shows that the format re-derives under the same verifier and the same ten vectors for a second writer; a chain written elsewhere by someone else would show more.
- not a recall measurement of the reader: the counted-by-design and excluded-by-design classes were checked against the pinned extract_numbers outside the gate before the catalogue froze (BUILD_LOG_3c.md section 2; one smoke text of 22:46Z went through the copy and its chain was deleted before the started line): the null result is the gate agreeing with that probe, not a recall measurement over unseen forms. what can move here is the open-question totals, a form added at a later started line without a probe, and any later reader.
- not a wall-clock proof that the list preceded the results: the seal precedes the texts in chain order, and the chain existed before one bitcoin block per day (pending tonight); a chain restarted before its first stamp leaves no trace outside (one smoke chain was deleted tonight before the started line). a batch list stamped before planting would be the stronger seal; not built.
- not the header's own words: the header's `certifies` sentence is the gate code's fixed one, inherited by the copy; the sentence that says what this chain is stands at the top of this file and of the page.
- the open-question totals are observed, not graded: the lab's later reader (certify.py v0.15, PR #242) ships those forms behind flags that are OFF by a recorded decision, or names them as unread; a repair that moves a count is an amendment with a date.
