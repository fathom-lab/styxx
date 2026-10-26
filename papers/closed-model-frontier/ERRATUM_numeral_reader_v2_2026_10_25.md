# ERRATUM — certify's numeral reader v2, after its two reviews: two defects fixed in the reader, thirteen sentences of the NOTE corrected

Fathom Lab · 2026-10-25 · An erratum to `papers/closed-model-frontier/NOTE_numeral_reader_v2_2026_10_25.md`, which
is not edited: its sworn receipt is committed, and a receipt is history too. Two reviews of branch
`feat/certify-suffixed-numbers` at 73d480fa, one for correctness and one for what the documents claim, found two
defects in the reader and sentences in the NOTE that the code contradicts. Their reports, scripts and raw outputs are
in the operator's product kit, outside the tree, under `clawd/wt/_product_2026_10_24/certify_suffix/`
(`REVIEW_correctness.md`, `REVIEW_claims.md`, `review_correctness/`); the re-runs after the fix are under `fix/`
there. The reader is fixed; the NOTE's sentences are corrected below. Every count inside a sworn span is sworn to
`papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json`, which the census script
wrote on a clean tree after the fix. The result the NOTE swears to is not regenerated.

## What changed in the reader

**The time and id leaf class is wider.** Cited against a whole exchange response, a magnitude after a currency
symbol verified made-up numbers on timestamp leaves the class did not name: `BTC market cap is $1.79T` bound OKX's
`ts` (milliseconds since the epoch), `$1.8T` bound Binance WebSocket's event time `E` and a kline's open time, and
`$1.79B` bound a seconds `ts` and an `updated` key. A leaf is now never a magnitude hit when a snake or camel part of
its last path segment is `time(s)`, `timestamp(s)`, `datetime`, `epoch`, `date(s)`, `at`, `id(s)`, `firstid`,
`lastid`, `tradeid`, `ts`, `mts`, `unix`, `created`, `updated` or `modified`, or ends in `time`, `stamp` or `date`
(`createtime`, `servertime`, `birthdate`); when the whole segment is one of Binance's single-letter time and
trade-id keys `E T O C t F L f`, compared case-sensitively (lower-case `c` is Binance's last price and binds); or
when the leaf is a cell of an array of arrays, as in a kline or a chart series, whose cells carry no name. The
excluded leaves are recorded on the row, as before.

**The accounting forms ship off.** `($x)` and `- $x` read as negative on common post shapes. The product red
team's own probe `Wallet balance: 0.5 BTC ($33,706)` read `-33,706` and went from VERIFIED before v2 to ABSTAIN;
so did `1 ETH ($2,431)`, `**BTC** - $67,412.50` and `Bitcoin (BTC) - $67,412.50`, and `BTC ($67.4k)`, a bullet
indented four spaces, an emoji or a table pipe before ` - $67.4k` read a price as a loss. Both forms now sit behind
their own flag, `V15_ACCOUNTING_SIGN`, shipped off, and read the positive amount, as before v2. `-$x`, `$-x` and
`−$x` stay the sign.

**The certificate says more of what it does not read.** The `uncovered_policy` of a certificate issued from here on
names the leaf class and what it leaves open, the window arithmetic, the condition that the currency symbol be glued
to the digits, the lakh misreading, the two sentence-final exceptions and the cost of the accounting flag being off.

## What did not move

Under the shipped flags after the fix, of the
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/certificates" k="numeric">240 committed certificates whose documents and receipts resolve</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/default/certificates_moved" k="numeric">0 re-derive differently</sworn>
from the reader before v2, with
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/default/rows_added" k="numeric">0 rows added</sworn>
and
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/default/rows_removed" k="numeric">0 rows removed</sworn>;
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/default/capsules_ok" k="numeric">12 committed capsules verify</sworn>
of
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/default/capsules" k="numeric">13</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/default/capsules_stop_verifying" k="numeric">0 stop verifying</sworn>,
and of the
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/default/charon_lines" k="numeric">257 charon lines</sworn>,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/default/charon_moved_verifier" k="numeric">257 read MOVED_VERIFIER</sworn>
and
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/default/charon_skew" k="numeric">0 read SKEW</sworn>.
The `REPLICATIONS.md` audit line still re-derives:
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/configurations/default/audit_line" k="quote">`corpus papers: 208 certificates | HELD 200  FAILED 8  unresolved 0  verdict-drift 1  receipt-drift 0  incomplete 1  receipt-changed 1`</sworn>.
The new flag, on alone,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/only_V15_ACCOUNTING_SIGN/certificates_moved" k="numeric">moves 0 certificates</sworn>:
the corpus holds no accounting negative and no magnitude after a currency symbol, which is why neither fix moves a
committed output. Every count the NOTE swears to has the same value in the result after the fix as in the result it
swears to (`tests/test_certify_numerals_census.py::test_every_count_the_note_swears_to_re_derives_after_the_fix`),
so the operator's open decision costs what the NOTE says: the bare magnitude arm, on alone,
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/only_V15_MAGNITUDE_BARE/charon_skew" k="numeric">turns 58 charon lines SKEW</sworn>,
and the sentence-final rule
<sworn r="path:papers/closed-model-frontier/oath_v15_numeral_reader_census_result_after_review.json#/summary/configs/sentence_final/class_flips" k="numeric">flips 11 committed OATH-HELD certificates to OATH-FAILED</sworn>.

The two committed vector sets replay unchanged (`conformance/numerals/gen_vectors.py --check`). A set written
before the accounting flag existed replays it at the value of the flag it was split from, as `vectors_v2.json`
already replays the bare arm; without that rule, 9 vectors of each set would not replay. The shipped default after
the fix is pinned in a new set, `conformance/numerals/vectors_v2_default_after_review.json`, with the reviews'
cases from a new texts file, `conformance/numerals/texts_review.py`; `texts.py` is not edited.

## The NOTE's sentences, corrected

1. "a leaf whose path names a time or an id (`openTime`, `lastId`, `updated_at`) is never a hit". The class was a
   fixed list, and common timestamp names were not on it. The list above is the class now, and it is still a list: a
   timestamp under another name can bind. In 2026 a millisecond epoch, about 1.79e12, lies inside the window of
   `$1.8T`, 1.75T to 1.85T, and a seconds epoch inside that of `$1.8B`. Against a whole response a sibling field
   inside the window binds too: a wrong `$66k`, `$68.1k` and `$66.8k` bind `lowPrice`, `highPrice` and `openPrice`,
   `$1.83M liquidations` binds the trade `count`, and a price of `$18.2k` binds the base-asset `volume`. In the
   correctness review's sweep, 5 of 149 wrong `$X.Yk` prices verify against a whole 24h ticker, before the fix and
   after it, and 0 of 149 against a receipt holding `lastPrice` alone. The mitigation is a narrow receipt, holding
   only the fields the post cites. `tests/test_certify_numerals_v2_review.py` pins the closed binds and the
   residual both.
2. "`($1.2k)` and `- $1.2k` read negative". Under the shipped flags they read positive. The cost, on the review's
   cases: an accounting-format loss binds a positive leaf (`BTC PnL ($5k) today` against a `pnl` of 5000 is
   VERIFIED), as the unsuffixed `($5,000)` did before v2. `V15_ACCOUNTING_SIGN = True` restores the reading.
3. "Still not read and not counted: a magnitude no currency symbol leads". The condition is that the symbol is
   glued to the digits. `$ 67.4k`, `€ 5m`, `US$ 1.2B`, `67.4k$`, `5m€`, `USD 67.4k` and `67.4k USD` read nothing and
   are not counted, so a made-up `$ 99.9k` or `99.9k USD` still passes unmarked. `$67.4 k` reads `67.4`, the
   mantissa alone.
4. Word magnitudes. Under the shipped flags a word magnitude no currency symbol leads reads its mantissa, as before
   v2: `a 1.2 billion users` reads `1.2`. The NOTE listed `1.2 billion` only under the bare flag.
5. "A numeral before a sentence period is still the band's `trailing-period`." Two exceptions: a
   thousands-grouped numeral reads its leading group as a value beside a `partial-extraction` item (`SL 66,200.`
   reads `66`, `all 23,247.` reads `23`), unchanged from before v2; and a magnitude before a period is read, by
   design (`BTC is at $99.9k.` is a row).
6. The certificate's scope said lakh and crore are "not read and not counted". They are misread, as before v2:
   `12,34,567` reads `12` and `34,567`, each a value.
7. The ellipsis rule (a numeral followed by an ellipsis is ABSTAIN) withdraws a VERIFIED as well as an accusation:
   `the drift rate 0.0666…` against a 0.0666 leaf was VERIFIED before v2 and is ABSTAIN `v15-ellipsis-truncated`
   now, and `BTC at $67.4k… waiting` is ABSTAIN with a right price.
8. "Counted, never valued: a decimal comma or dotted thousands". A three-decimal euro price, `Super E10 costs
   €1.789`, was VERIFIED before v2 and is a counted `dotted-groups` item now. The rule chose this; the cost is a
   vector in the set after the fix.
9. "word-led version strings". Only after `version` or `ver.`. With `V15_FORM_NOISE` on, `opus 4.8 and gemini 3.1`
   and `Top 3` still read their numerals; of the product kit's form-noise examples the flag clears `14:00` and `1/`
   only.
10. "Not a change to the obligation ladder". It changes in two places. A bare magnitude is never obligated, which
    is dormant while `V15_MAGNITUDE_BARE` is off and live the moment it is on; and the ellipsis rule adds an ABSTAIN
    branch beside the notation classes, which is live.
11. "by re-deriving every committed certify output". The census covers the committed `*.certificate.json` and
    `CERT_*.json` files whose documents and receipts resolve, outside `anc/` paths, the committed capsules and the
    charon log. It does not cover the certificates under `anc/` or the `*.seal.json` files. The correctness review
    re-derived the seals from their sibling certificates' receipts and found none moved; the claims review found the
    `anc/` documents either byte-identical to documents inside the census or no longer at their recorded sha.
12. "62 false accusations on model sizes". 60 were model sizes and 2 were `10k` held-out counts.
13. "`conformance/numerals/vectors_v2.json` ... is not edited". Its bytes, and those of `vectors_v2_default.json`,
    were converted from CRLF to LF once, in d49e9427: only CR bytes were removed, and every id, expectation and
    `set_sha256` is unchanged.

## Still owed, named

The rule as built said its owed items are named in the NOTE; these were not. A-18: `ETH $2,431` against 2431.27
is ABSTAIN while `ETH at $2.4k` against the same leaf is VERIFIED, so a coarser short form binds where the fuller
integer does not (both are vectors in the set after the fix). A receipt-side suffixed string (`{"price": "67.4K"}`)
is not a number, so `$67.4k` against it is ABSTAIN. Lakh and crore grouping (above). The letter-glued spans the
design counted. Count-binding stem misses, trigger-word polysemy, fenced-code skipping, the A-17 label exclusion
(`Cycle N`, relevant under the sentence-final flag) and A-4(d) (counting magnitude-suffixed drops as band items).
Two more from the reviews: the `magnitude.coarse` annotation is wrong for a one-digit mantissa with zeros (`10`,
`500`, `0.05`); nothing reads it, and a fix would move committed vectors. And the capsule's numeral-reader advisory
compares the reader's schema and names the master flag only, so two v2 certificates minted under different flag
sets draw no advisory (their rows still differ, and are reported); under the shipped reader 10 of the 13 committed
capsules print it while still verifying (the census result's `configurations.default.capsules.rows`).

## Re-derive

    py -3.12 -B papers/closed-model-frontier/oath_v15_numeral_reader_census.py      # writes the result after the fix
    py -3.12 -B conformance/numerals/gen_vectors.py --check
    py -3.12 -B -m pytest tests/test_certify_numerals_v2_review.py tests/test_certify_numerals_v2_vectors.py \
        tests/test_certify_numerals_census.py
    py -3.12 -B -m styxx.sworn check papers/closed-model-frontier/ERRATUM_numeral_reader_v2_2026_10_25.sworn-receipt.json \
        papers/closed-model-frontier/ERRATUM_numeral_reader_v2_2026_10_25.sworn.json --repo .
