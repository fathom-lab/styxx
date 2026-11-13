# GUESSES (written before any vector was run)

Written after the checker was built from RECEIPT_v0.md and receipt-v0.schema.json alone, and before
receipt_v0_vectors.json was opened. Each entry: where in the text, the two readings, what I chose.
A column "settled by a vector?" is filled in AFTER the first run, in the section at the end.

Already not a guess: 7.2's Unicode pin. Go 1.25.6's `unicode` tables reproduce both digests the text
states (L/N set 4ca0e4c9..., Nd set ef10fe9b...), checked at every start of the checker. So the
host tables ARE 15.0.0 here and I carry no tables of my own.

| id | where | reading A | reading B | chose |
|----|-------|-----------|-----------|-------|
| G1 | 6.13, 13.5 JSON nesting depth in receipt bytes | no limit (parse any depth) | a fixed limit (host recursion, e.g. ~1000) answered as receipt_not_json | A: no limit |
| G2 | 3.3 which code wins when a value has several faults | walk in first-insertion order, all keys of an object checked for non_ascii_key before its values; depth checked on entering a container | walk in sorted key order, or key then its value pairwise | A |
| G3 | 3.3 a KEY holding a lone surrogate | `non_ascii_key` (its code point is above U+007E) | `lone_surrogate` | non_ascii_key |
| G4 | 3.1 `-0` as a value for the canonical form | an integer, canonical `0` | not admitted / kept as `-0` | `0` (11.3 says `-0` for `0` is the same receipt) |
| G5 | 7.4 detail of `needle_count`, `needle_empty` | `{}` (text names none) | some count | `{}` |
| G6 | detail where the text names none: `empty_span` (7.1b), `anchor_out_of_range`, `receipt_not_json`, `pointer_unresolvable`, `pointer_ambiguous`, `kind_of_source_unknown`, `receipt_author_minted`, every UNRESOLVED | `{}` | something | `{}` (8: "{} when nothing is said") |
| G7 | 7.3, 7.5, 7.7, 7.8 "Detail: {...}" after "HELD, or FAILED" | the same detail on HELD rows as on FAILED rows | detail only on FAILED, `{}` on HELD | same detail on HELD |
| G8 | 7.5 narrowing when k = 0 (receipt is only LF bytes) | narrows only if receipt < 16 bytes | slice compared with an empty full-content slice | A, as the parenthesis says |
| G9 | 2.3d "every open one not yet marked is marked nesting" | an open declaration already marked `hidden_commitment` keeps that mark | it becomes `nesting` | keeps `hidden_commitment` ("not yet marked") |
| G10 | 7.1b / 7.4 "ASCII whitespace" | the six bytes listed in 7.1b (SP TAB LF VT FF CR) also for `needle_empty` | Python's str.isspace or another set | the six bytes |
| G11 | 5.4 rebuilt core when `receipts` / `authored_sha256` / `rung` are missing or null | core carries `{}`, `[]`, `null` (the values "read as") | core carries the key as written / omits it | the read values |
| G12 | 5.6 `digest` present but not a string, or hex in upper case | not intact (not equal) | refuse, or compare case-insensitively | not intact |
| G13 | 6.3 `rung` a non-string (number, object) | `rung_unknown` | manifest_form / integrity | rung_unknown (a float rung already fails 6.2) |
| G14 | 6.5 entry without `sha256` | `manifest_integrity` | other | manifest_integrity |
| G15 | 7.3 `leaf` of an oversized number | its `receipt` string cut to 80 code points | uncut | cut to 80 (the rule says the first 80 code points of the text) |
| G16 | 11.3 schema maxItems 0 on `attestations` vs refusal `attestations_not_empty` listed AFTER `receipt_shape` | the shape step does not enforce maxItems on attestations, so `attestations_not_empty` is reachable | full schema: non-empty attestations is `receipt_shape` and `attestations_not_empty` is dead | A (13.15 says every refusal code but `type` is expected by some vector) |
| G17 | 11.3 a receipt check handed no document | `nothing_to_verify`, after the profile step | refuse first | after profile |
| G18 | 11.3 where a manifest refusal (5.5) lands when re-deriving | returned as that refusal code after `document_mismatch` | DOES-NOT-REDERIVE | the refusal code |
| G19 | 11.3 const / enum with numbers | equal only when both written as integers (`300.0` fails `const 300`) | JSON Schema value equality | integers only; `-0` counts as integer 0 and meets `minimum 0` |
| G20 | 11.3 `pattern` on a string with a lone surrogate | never matches the schema's ASCII patterns | error | never matches |
| G21 | 7.3 printed value with leading zeros (`007`) | plain notation drops them (`7`) | kept | dropped |
| G22 | 7.2 grammar details | fraction needs at least one digit after `.`; grouped form needs at least one `,ddd` group | `1.` admitted (it cannot reach: trailing `.` is stripped) | A |
| G23 | 11.1 a name that is missing / not a string, a commit that is not a string | `name_form` / `commit_form` | nothing_to_verify | name_form / commit_form |
| G24 | output | the canonical receipt printed is `{"attestations":[],"core":...,"digest":...}` with no `verifier` | with a verifier object | no verifier (9.4: optional, outside the digest) |
| G25 | 4.3 a line number too large for a machine integer (`L99999999999999999999`) | passes the grammar, then `anchor_out_of_range` | receipt_form | passes, out of range |
| G26 | 7.6 a hex run next to a non-ASCII letter | counts (only ASCII `A-Z a-z 0-9 _` glue) | does not | counts |
| G27 | 6.12 array token `-` or `00` | pointer_unresolvable | other | pointer_unresolvable |
| G28 | 2.1 `at` for invalid UTF-8 | offset where Go's decoder first fails (first byte of the maximal ill-formed prefix) | offset of the last good byte + 1 (same thing) | Go decoder |
| G29 | 3.2 depth | the 33rd nested container is `depth`; scalars inside the 32nd are fine | count scalars too | containers only |
| G30 | 6.13 duplicate detection | keys compared after unescaping (`"a"` and `"a"` are the same key) | compared as written | after unescaping |
| G31 | 2.5 / 8 `claim` and `claim_sha256` when the document HAS canonical text | given for every row (all declarations are paired then) | | as A |
| G32 | 7.1g code point format | `U+%04X` upper case | | as A (schema pattern ^U\+[0-9A-F]{4,6}$) |
| G33 | vector runner | the shape of the vector file had to be learned before the runner could be written: I printed key paths and value kinds only (no values) | | see PROGRESS.md |

## Settled or not by the vectors (filled after the first run)

First run: 2193 of 2193 passed (first_run.txt), so no guess had to be changed. A pass shows that my
readings fit the vectors. It does not show that the vectors pinned each one. To find out, every guess
that has a real alternative was built as a switch (`RV0_ALT=Gn`, checker/alt.go) and the whole file
was run with only that switch on. If the alternative fails at least one vector, the vectors decide
that guess. If it passes all 2193, the vectors leave the guess open. Raw lines: alt_runs.txt.

Controls, deliberately wrong rules that check the harness can fail: C1 ties round half-up: 11 fail.
C2 7.3a off: 8 fail. C3 short-needle floor off: 155 fail.

| id | alternative run | vectors failing | settled by a vector? |
|----|-----------------|-----------------|----------------------|
| G1 | JSON depth limit 1000 -> receipt_not_json | 0 | NO, open |
| G2 | canonical-form faults checked in sorted key order, key then value | 0 | NO, open |
| G3 | lone-surrogate key -> lone_surrogate | 0 | NO, open (both readings pass the jcs vectors) |
| G4 | `-0` kept as `-0` | 2 (1 jcs, 1 recheck) | YES |
| G5 | needle_count detail carries a count | 12 (9 conformance, 3 hand) | YES, `{}` |
| G6 | anchor_out_of_range detail `{receipt}` | 5 hand | YES for anchor_out_of_range only; the other codes in G6 were not switched |
| G7 | `{}` detail on HELD rows | 423 | YES, HELD rows carry the detail |
| G8 | k = 0 compared like any other slice | 0 | NO, open |
| G9 | an open hidden_commitment declaration becomes nesting | 7 conformance | YES, it keeps hidden_commitment |
| G10 | needle_empty by Unicode whitespace | 0 | NO, open |
| G11 | rebuilt core omits missing keys / keeps null | 4 hand | YES |
| G12 | digest compared case-insensitively | 1 hand | YES |
| G13 | non-string rung -> manifest_integrity | 1 hand | YES, rung_unknown |
| G14 | not switched (no other reading I could write down that 6.5 allows) | - | not tested |
| G15 | oversized-number leaf uncut | 1 hand | YES, cut to 80 |
| G16 | full schema (maxItems 0 on attestations) | 2 recheck | YES, attestations_not_empty is reachable |
| G17 | nothing_to_verify first in a recheck | 0 | NO, open (no recheck vector lacks a document) |
| G18 | manifest refusal while re-deriving -> DOES-NOT-REDERIVE | 1 recheck | YES, the refusal code |
| G19 | const/enum numbers by value (`300.0` == `300`) | 0 | NO, open |
| G20 | not switched | - | not tested |
| G21 | printed leading zeros kept | 1 hand | YES, dropped. My first alternative also added a zero to `0.5` and failed 217 vectors. That was a bug in the alternative, not a verdict, and it was fixed and run again |
| G22 | not switched (the other reading cannot be reached) | - | - |
| G23 | null name -> nothing_to_verify | 1 hand | YES, name_form |
| G24 | output only | - | - |
| G25 | huge line number -> receipt_form | 1 hand | YES, anchor_out_of_range |
| G26 | non-ASCII letters glue hex runs | 1 hand | YES, only ASCII glues |
| G27, G28, G31, G32 | not switched | - | - |
| G29 | scalars count toward depth | 1 jcs | YES, containers only |
| G30 | duplicate keys compared as written | 0 | NO, open. 13.15 says "escaped manifest key spellings ... are barely drawn" |

Score: of the 24 guesses that were switched, 16 are settled by at least one vector and 8 stay open
(G1, G2, G3, G8, G10, G17, G19, G30). For the 8 open ones my checker and the lab's could differ on
inputs that no vector draws.
