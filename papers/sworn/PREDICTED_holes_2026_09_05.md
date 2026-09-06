# Predicted holes in the differential grammar, written before the mutation study reports

Fathom Lab · 2026-09-05 · Companion to `SPEC_mutation_coverage_v01_2026_09_05.md`. **Written and
committed before the mutation catalogue was run**, so that the study can be scored against a
prediction rather than explained after the fact.

A mutation study tells you which behaviours your harness cannot see. It does not tell you *why*,
and a miss list read after the run is easy to rationalise into a story that was never predicted.
So this is the prediction, made by reading `conformance/sworn/differential.py` — the generator, not
the results — and asking what it structurally cannot produce.

## The structural reason there are holes at all

`_manifest()` builds a Python **dict** and hands it to `sworn.Manifest.from_dict(...)`. It is never
serialised to JSON text and re-parsed. So the strict JSON *parser* — the one that refuses a BOM,
refuses NaN and Infinity, and decides what a duplicate key means — is reached only through the
**receipt payload bytes**, and those come from a fixed list of ten literals in `_manifest`.

That list is the whole aperture. Every JSON-parser behaviour not exercised by one of those ten
byte strings is invisible to this harness, no matter how many cases it runs. 150000 draws from a
list of ten reaches the same ten things 150000 times.

## The predictions

Each is a behaviour a mutation could target. **HOLE** predicts the mutation is missed; **SEEN**
predicts it is caught.

| # | behaviour | prediction | why |
|---|---|---|---|
| P1 | BOM-prefixed JSON payload refusal | **HOLE** | already confirmed missed by the ad-hoc probe; no payload literal begins with `EF BB BF` |
| P2 | NaN / Infinity in a JSON payload | **HOLE** | no payload literal contains either token |
| P3 | invalid UTF-8 *inside a receipt payload* | **HOLE** | all ten payload literals are valid UTF-8; only the *document* reaches invalid bytes |
| P4 | surrogate escapes (`\udXXX`) in a payload | **HOLE** | no payload literal contains a `\u` escape at all |
| P5 | astral characters in a payload | **HOLE** | same |
| P6 | deep nesting / recursion limits in the payload parser | **HOLE** | the deepest literal is `{"a": {"b": [1,2,3]}}` |
| P7 | `captured_at` / `minted_at` validation | **HOLE** | both are the same constant `2026-09-01T00:00:00Z` in every manifest ever generated |
| P8 | receipt-id grammar *on the manifest side* | **HOLE** | manifest ids are always `r1`–`r4`; the odd id forms live only in the span's receipt string, where they fail to resolve before any id logic runs |
| P9 | `authored_sha256` with more than one entry, or a malformed one | **HOLE** | it is always `[]` or a single well-formed digest |
| P10 | duplicate keys in a JSON payload | SEEN | `{"dup": 1, "dup": 2}` is in the list |
| P11 | JSON Pointer walking, `~0`/`~1` unescaping | SEEN | pointer forms are drawn in the span receipts and payloads have nested structure |
| P12 | line slices `#L1`, `#L1-L3` | SEEN | `line one\nline two\nline three\n` is in the list |
| P13 | the short-needle bound | SEEN | confirmed caught by the ad-hoc probe, 35 disagreements |
| P14 | the span code-point cap | SEEN | confirmed caught, 260 disagreements |
| P15 | signed-zero folding in the decimal path | SEEN | confirmed caught, 6 disagreements; `{"neg": -0.0, "e": 1e5}` is in the list |
| P16 | very large integers in the decimal path | SEEN | `{"big": 9…9}` with 400 nines is in the list |
| P17 | document-level fence balancing and UTF-8 refusal | SEEN | the document generator reaches both; the census counts 15702 and 1524 |

Nine holes predicted, eight behaviours predicted visible. **The prediction that matters is P1–P9
sharing one cause**: they are not nine independent weaknesses, they are one — the payload literal
list — wearing nine faces. If the study's miss list is dominated by payload-level hazards and its
caught list by document-level and adjudicator-level ones, the diagnosis is a single aperture, and a
single change to `_manifest` closes most of it.

If instead the misses scatter across regions with no common cause, this prediction is wrong and the
harness is weak in a way reading it did not reveal — which is worth strictly more than being right.

## What this document does not claim

That the list is complete: it is what one reading of one generator suggested, and the completeness
critic in the catalogue workflow exists because this list is not to be trusted as exhaustive. That
a HOLE is a bug in the format or in either implementation — it is a place where a bug would be
invisible to *this harness at this size*. That closing them is free: every payload literal added is
a change to the instrument, and by M6 a changed instrument is a second run and a second file, never
an improved number written over the old one.
