# DEFECT — uncovered lines: the decimals the verifier never read

Fathom Lab · 2026-09-01 · **No preregistration, and therefore no headline finding.** This is a
defect report. It describes a mechanism in `styxx/certify.py`, gives a minimal reproduction,
sizes the affected corpus, and states what the size does and does not license. Nothing here is
scored against a frozen gate, so nothing here is a result.

Verifier under examination: `styxx/certify.py`, sha256
`774a0d233d0d5572ded6aee748130978e533c4a62d3ca02205d6327347872bd6`.

## The mechanism

`_NUM` is the doc-side number extractor — the function that decides which numerals in a paper
enter the ledger and get sworn against a receipt. It is four alternatives, and every one of them
ends with the same lookahead:

```python
_NUM = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w.])|(?<![\w.])[-+]?\d+\.\d+(?![\w.])"
                  r"|(?<![\w.])[-+]?\.\d+(?![\w.])|(?<![\w.])[-+]?\d+(?![\w.])")
```

`(?![\w.])` was written to stop the extractor from biting into version strings and dotted
identifiers. It also matches the ordinary English period. A decimal that ends a sentence is
followed by `.`, the lookahead fails, and the regex backtracks through every shorter prefix —
each of which is then followed by a word character, so each fails too. The other three
alternatives cannot rescue it: the bare-decimal form needs a `.` not preceded by `[\w.]`, and
the integer forms are blocked by their own lookbehind against the digits around them.

The span produces **zero matches**. This matters more than it sounds. The token is not
CONTRADICTED, and it is not ABSTAIN — ABSTAIN is a recorded, reviewable act of the instrument
declining to swear. The token is **absent**: it never enters the ledger, never appears in the
counts, and leaves no trace in the certificate that it existed. The verdict is computed over
what was extracted, and nothing in the artifact says extraction was partial.

## Minimal reproduction

One receipt, `{"precision": 0.55}`, and two documents differing by a single character.

| document | certify output |
|---|---|
| `The gate reached a precision of 0.55.` | `OATH-HELD  verified=0 abstained=0 contradicted=0` |
| `The gate reached a precision of 0.55`  | `OATH-HELD  verified=1 abstained=0 contradicted=0` |

Both are OATH-HELD. The first certifies with **zero tokens examined**; deleting the period
certifies the same sentence with one. At the regex level:

```
>>> _NUM.findall("precision of 0.55.")   ->  []
>>> _NUM.findall("precision of 0.55")    ->  ['0.55']
```

A reader who sees OATH-HELD on the first document learns nothing except that the empty set held.

## Corpus census

Scope: every document under `papers/` carrying a sibling `*.certificate.json` — **208
documents**. An uncovered token is a span matching the decimal alternative
`(?<![\w.])[-+]?\d+\.\d+` that is immediately followed by a period followed by whitespace or
end-of-line, measured after the same SHA/date/version scrubbing `extract_numbers` applies. Each
counted span was asserted to yield no `_NUM` match.

| quantity | value |
|---|---:|
| certified documents scanned | 208 |
| documents with ≥ 1 uncovered decimal | **89 (42.8%)** |
| uncovered tokens | **168** |

Nearly half the certified corpus contains at least one number its own certificate never looked
at. For scale only: the 208 certificates carry 8,200 ledger entries between them, so the 168
sit at roughly 2% of the numeric surface. That percentage is soft and is offered as an order of
magnitude rather than a rate — those certificates were produced by **15 distinct verifier
builds**, so their ledgers are not a single homogeneous baseline. The 89/208/168 figures are
not soft: they are one scan, one definition, over the documents as they stand today.

A looser definition — a decimal followed by a period followed by anything that is not a digit,
which catches `0.55.)` and `0.55.*` as well — gives **98 documents (47.1%) and 187 tokens**. The
stricter sentence-ending figure is reported as the headline because it is the unambiguous case;
the looser one is reported so the stricter one is not mistaken for a ceiling.

This census has no committed receipt JSON. The certify repair and its receipts belong to a
separate work item, and this report deliberately writes no receipts and re-certifies nothing.
The definition above is stated in full so the numbers can be recomputed rather than trusted.

## The convergence

The defect was not found by reading the regex. It was found because a document this lab had
already published contained a sentence that later work withdrew, and the certificate did not
know.

`papers/SYNTHESIS_connection_of_minds_2026_08_01.md` certifies **OATH-HELD**, 81 VERIFIED, 13
ABSTAIN, 0 UNGROUNDED, 94 ledger entries. Its ledger is ordered by line, and it runs

```
... line 25, line 26, line 32, line 33 ...
```

Line 27 is missing. Line 27 is the only uncovered token in the document, and it is this:

> gemma-2-2b, the *highest*-isometry target in the battery (RSA 0.955, above even the
> same-family anchor), reads at exactly chance 0.014.

That sentence, unscoped, says gemma is unreadable. `b31v2_result.json` (2026-08-01) then read
the same model at 0.7857 and `b34v3_result.json` (2026-08-03) at 0.5714; the cliff was the
linear map class, not the mind. The retraction was written into the following paragraph but
never marked on the sentence itself, and the sentence ships verbatim in both arXiv copies
(`papers/arxiv/connection-of-minds/main.tex:53` and
`papers/arxiv/connection-of-minds/submission/main.tex:53`). It is struck in place as of today,
with a scope erratum at the head of §2.

Of the 168 uncovered tokens in the corpus, the one in the document that most needed a reader was
the number in the sentence that was wrong. We claim no mechanism for that and it is almost
certainly coincidence — 89 documents is a wide net. It is recorded because it is what made the
defect visible, not because it is evidence of anything.

The convergence has a second half, and this one is not coincidence. In the marked-up file the
struck original still extracts nothing — `0.014.~~` is still period-terminated — while the
corrected restatement beside it, where the same number is followed by a space, extracts
`0.955` and `0.014` normally. Same number, same document, same verifier: invisible in the
sentence that was wrong, visible in the sentence that corrects it. The instrument's coverage
was decided by punctuation.

## What this does not mean

**No verdict in the corpus is invalidated, and none should be re-read as suspect.** OATH-HELD
means everything the instrument examined held, and that is still exactly true of all 208
documents. Not one uncovered token has been shown to be a false claim. This report does not
allege a single wrong number anywhere in the corpus, and readers should not infer one — the
withdrawn sentence at SYNTHESIS line 27 was found by other means and is the only claim named
here.

What is wrong is narrower and worse than a wrong number. **The corpus reported on a subset while
reading as though it covered the whole.** The certificate's purpose is to tell a stranger what
was checked. It named a verdict, a count, and a per-token ledger, and it did all of that
truthfully about the numerals it saw, while giving the stranger no way to learn that 168
numerals across 89 documents were never candidates. Absence of an entry looked identical to
absence of a number.

## The family

This is the fourth instance today of one shape.

- **Mention versus use** — the verifier cannot separate a numeral that asserts something from a
  numeral that points somewhere, so a residue of twenty-one pointers — section cross-references,
  a DOI, a `file.py:89`-style source citation, the year inside a venue name — is refused as
  unbound claims. Recorded in `ANALYSIS_base_rate_ceiling_2026_09_01.md`, whose certificate reads
  OATH-FAILED with exactly those 21 UNGROUNDED; sized in `oath_mention_use_census.json`.
- **The handed target** and **the extraction term** — named elsewhere in today's work. They are
  cited here by name only. This document did not re-derive them and attaches no receipt for
  them; treat both as UNVERIFIED at this reference.
- **Uncovered lines** — this report.

The shape is not that the instrument lies. In all four the instrument reports what it examined,
accurately. The shape is that **what it did not examine has been invisible** — no count, no
flag, no line in the artifact. Two of the four over-include and two under-include, and the
direction turns out not to be the interesting part. What they share is that the report's silence
about its own coverage reads, to anyone who did not write it, like completeness.

That is a claim about four cases found in one day, not about the instrument in general. Whether
this shape is the dominant defect class, or whether four is what a day of looking hard at
anything produces, is not established here and would need a preregistration to become a finding.

## What is not established

No claim that the 168 uncovered tokens contain errors, at any rate. No rate of error among them
— none has been checked. No claim that 89/208 generalises beyond `papers/` as it stands today,
or to any other corpus, or to prose written after a fix. No claim about the other three defects
beyond their names. No repair is proposed here and none is evaluated: a repair changes what
`_NUM` matches, which changes every ledger, and that belongs behind a preregistration with a
recall gate, not behind a defect report. This document re-certifies nothing.

---

*A verifier that swore to 8,200 numbers and never saw 168 more was not lying about the 8,200.
It simply had no way to say "and there were others". Everything we build next has to say that.*
