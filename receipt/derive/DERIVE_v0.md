# DERIVE v0

Status: DRAFT, unsigned, nothing frozen. The builder's repair, dated 2026-11-13, of the designer's draft of the same date (`DERIVE_v0_DRAFT.md`, kept in the lab's notes). It answers the 30 findings of three critics, one by one, in section 19. It stands beside `RECEIPT_v0.md`, which is itself a draft. Appendix A restates every v0 rule this text depends on, so that the text can be implemented from itself.

Two implementations exist, `derive.py` and `derive.js` beside this file. One agent wrote both, with this text open. Their agreement shows the text was computed the same way twice. They are not independent of each other or of the lab. Section 18 lists where they are known to disagree.

A fifth KIND for the span grammar of RECEIPT v0, named `derive`. A span prints one number. The author names a manifest entry and a **derivation**: a record of fixed shape that selects rows from that entry's bytes and names one of five operations built into the checker. The checker recomputes the number from the rows and compares it with the printed number by v0's rounding rule. Its answer also carries what the author cannot declare: how many rows there were, how many were selected, what each selection test did, what the selection would give without each test, what values the tested fields actually hold, whether rows repeat, and what else sat beside the rows in the same file.

`derive` is not the reserved kind `exec`. Nothing an author writes is run. A derivation has no operator the author can combine, no variable, no operation inside another, no loop, no pattern, no regular expression and no author-supplied function. It selects among five functions this text defines, and section 12 states their cost.

Read sections 15 and 16 before relying on a HELD derive span. HELD never means true.

## 1. Terms

- **derive block**: a fenced region of the document that carries derivations (2.3).
- **derivation**: one JSON object of the closed shape of section 3, known by a NAME.
- **row source**: the part of the receipt bytes the rows come from (section 4). **row**: one element, member or line of it. **row key**: the member name of a row that came from a JSON object.
- **subject**: what a test reads from a row: a member, a value at a pointer, the row key, or the row itself (3.4).
- **test**: one comparison of a subject with literals. **selection**: the rows on which every test of `where` is true.
- **exact value**: the result of the operation as a fraction of two integers (7.2).
- **checking function**: the function of section 13 that answers one derive span. It is the unit the vectors test.

## 2. The claim form

2.1 **The span.** `<sworn r="rN#NAME" k="derive">inner</sworn>`. The opener grammar of v0 2.3d is unchanged. For KIND `derive` the REF must be exactly `r`, a positive decimal integer without a leading zero, `#`, and a NAME. A NAME is a lowercase ASCII letter followed by at most 31 of `a-z 0-9 _`. Every other REF is MALFORMED `receipt_form`, detail `{receipt}`: no fragment, a JSON Pointer, a line range, an empty fragment, and every `path:` or `prereg:` head. So no derive span can reach a resolver that has a repository branch (finding S7). For the other four kinds v0 4.3 and 4.4 are unchanged.

2.2 **The inner text** is a number in prose, under A.3: exactly one digit-bearing token. "23 of 100" is two spans.

2.3 **The derive block.** A fenced region (A.4) whose opening delimiter line is exactly the nine bytes `` ```derive ``, optionally followed by one CR. Its **content** is every byte after that line's LF and before the first byte of the closing delimiter line. No other fenced region is a derive block. v0 2.3a already skips fenced regions, so nothing inside a block is a tag.

2.4 **The block table.** Take the derive blocks in document order. For each, check in this order: it is the 17th derive block; its content is over 65,536 bytes; its content does not read under A.1; the value is not an object; then, for each member in the order written: its name is not a NAME; its name is already in the table (from this block or an earlier one; a name written twice in one block is caught here); it would be the 257th derivation. The first of these found is the **block defect**. Otherwise each member is added to the table as NAME to value. A member's value is not examined here.

2.5 If there is a block defect, every derive span that reaches it in step 9e is MALFORMED `derive_block_malformed`, detail `{block_line}`: the 1-based line number of the defective block's opening delimiter line. Spans of other kinds are not affected.

2.6 A derivation no span names is never examined, except as the target of a `within` (4.5). A derive block in a document with no derive span changes nothing.

## 3. The derivation

3.1 A derivation is a JSON object with a canonical form under A.2: no repeated key at any level, every key in U+0000..U+007E, no unpaired surrogate, no `NaN` or infinity, no number with a fraction or an exponent, and every integer written as `0` or an optional `-`, a non-zero digit and digits, with |n| <= 2^53 - 1. The bound is decided on the digits as written: more than 16 digits is out before any conversion, so a 5,000-digit literal is `derivation_form` and never a crash (S1, S5). `-0` has no canonical form. Its canonical text is at most 4,096 bytes. Its keys, all required unless marked:

    op "count"      rows, where, op                 [within]
    op "share"      rows, where, match, op          [within]
    op "percent"    rows, where, match, op          [within]
    op "sum"        rows, where, of, op             [within]
    op "mean"       rows, where, of, op             [within]

3.2 `rows` is `{"format": "json", "at": POINTER}` or `{"format": "jsonl"}`, no other keys. POINTER is the empty string (the whole value) or a JSON Pointer under A.5, at most 1,024 bytes and 16 tokens.

3.3 `where` is an array of zero or more tests. `match` is an array of one or more tests. Together they hold at most 8.

3.4 A **test** is an object with exactly two members: one subject and one relation.

    subject     "field": S         S a string of at most 256 bytes: a member name of the row
                "at": P            P a non-empty JSON Pointer (A.5), at most 1,024 bytes and 16 tokens,
                                   walked from the row
                "row": "key"       the row key (not with format jsonl)
                "row": "value"     the row itself
    relation    "eq": LIT
                "in": [LIT, ...]   1 to 16 literals, no two with the same canonical text
                "ge": OLIT         subject >= literal
                "le": OLIT         subject <= literal
                "present": B       B true or false; only with a "field" or "at" subject

LIT is a string of at most 256 bytes, `true`, `false`, an integer, or a decimal `{"dec": D}`. OLIT is a string of at most 256 bytes, an integer, or a decimal. D is a string: an optional `-`, then `0` or a non-zero digit followed by at most 17 digits, then optionally `.` and 1 to 18 digits. So `{"dec": "0.001"}` selects a learning rate and `{"dec": "0.5"}` is a threshold (F10). A decimal is written as a string so that the derivation keeps a canonical form.

Every byte limit in this text is measured on the UTF-8 of the decoded string, never on its escaped form (F5): a field name written as 129 `é` escapes is 258 bytes.

A row key subject with format `jsonl` is `derivation_form`: JSON Lines rows never have a key, and the document alone shows it (G13).

3.5 `of` is an object with exactly one member: `"field": S`, `"at": P` or `"row": "value"`.

3.6 `within`, when present, is a NAME (4.5).

3.7 **Checking a derivation**, first that applies:
  1. The value is not an object, or has no canonical form: MALFORMED `derivation_form`.
  2. Its canonical text is over 4,096 bytes: `derivation_form`. (Size comes before `op`, so a 5,000-byte derivation with op `median` is `derivation_form`; G07.)
  3. `op` is missing or not a string: `derivation_form`.
  4. `op` is not one of the five: MALFORMED `op_unknown`, detail `{op}`, the whole string. It is at most 4,096 bytes and has a canonical form by step 1, so nothing is cut (S5).
  5. Any other breach of 3.1 to 3.6: `derivation_form`, detail `{}`. Which rule was broken is for a message outside the digest.

`op` is compared by exact string equality with the five names before anything is looked up. `constructor`, `toString` and `__proto__` are `op_unknown` like any other unknown name (S4).

## 4. The row source

4.1 The receipt bytes are the whole bytes of the manifest entry (v0 6.10). Over 16,777,216 bytes: MALFORMED `rows_too_large`, detail `{bytes, cap}`.

4.2 **`json`.** The bytes must be well-formed UTF-8 and must read under A.1; otherwise MALFORMED `receipt_not_json`, detail `{}`. A.1 counts depth and values while reading. A 33rd container is `receipt_not_json`. A 1,000,001st value is MALFORMED `receipt_too_many_values`, detail `{cap}`. Whichever fault the reader meets first in reading order decides. Then the pointer is walked (A.5): a token that is not a member is `pointer_unresolvable`, a member written twice on the path is `pointer_ambiguous`, an array token that breaks A.5's index rule or is past the end is `pointer_unresolvable`, and a walk into a scalar is `pointer_unresolvable`. The detail of all three is `{at}`. Then, in order: the value reached is not an array or an object: MALFORMED `rows_not_container`, detail `{type}`; it holds more than 100,000 elements or members: MALFORMED `rows_too_many`, detail `{rows, cap}`; it is an object in which some member name occurs twice: MALFORMED `rows_ambiguous`, detail `{}`. The row cap is checked before repeated names (G12). In an array the elements are the rows, in order, without keys. In an object the members are the rows, in the order written; the row key is the member name and the row is the member value.

4.3 **`jsonl`** (JSON Lines). The number of lines is the count of LF bytes, plus one if the bytes are non-empty and do not end in LF. More than 100,000 lines: MALFORMED `rows_too_many`, detail `{rows, cap}`, decided from that count before anything else is read. Then the bytes must be well-formed UTF-8; otherwise MALFORMED `receipt_not_jsonl`, detail `{line}`, the line holding the first byte of the first ill-formed sequence. Then split at every LF; if the bytes end with LF, the empty piece after it is not a line, and zero bytes are zero rows. Each line, in order, must read under A.1 as one value with nothing around it but space, TAB and CR. A line that is empty or holds only those bytes, or that does not read, is `receipt_not_jsonl` with its 1-based number. The value count of A.1 runs across all lines, and its 1,000,001st value is `receipt_too_many_values`. Each line's value is a row without a key.

4.4 `type`, here and below, is one of `null`, `boolean`, `string`, `number`, `non_finite` (the tokens `NaN`, `Infinity`, `-Infinity`), `array`, `object`. `row`, in a detail, is the 0-based position of a row in its source.

4.5 **`within`.** It names another derivation in the table. MALFORMED `within_form`, detail `{within}`, when any of these holds: it names the derivation itself; the name is not in the table; the named derivation fails 3.7; the named derivation has a `within` of its own; or its `rows` has a different canonical text. Only the named derivation's `where` is used. Its tests are applied to every row with the rest (5.2). The **within selection** is the rows on which all of them are true.

4.6 **`row_key_unavailable`.** When the rows have no key (an array) and a test of `where`, `match` or the within `where` has a `row: key` subject: MALFORMED `row_key_unavailable`, detail `{}`. This is decided after the container is reached and before any row is visited, whatever the rows hold (G13).

## 5. Selection

5.1 **The subject's value** for one row.
- `"field": S`: if the row is an object with a member named S (the same code points after escapes are decoded; nothing normalised), that member's value. If that name occurs twice in the row: MALFORMED `field_ambiguous`, detail `{row, subject}`. If the row is not an object or has no such member, the subject is **absent**.
- `"at": P`: walk P from the row as A.5 walks, except that a token that is not a member, an index that is not valid or is past the end, and a walk into a scalar each make the subject absent. A member written twice on the path is `field_ambiguous`.
- `"row": "key"`: the row key. `"row": "value"`: the row.

A name is looked up only in the row's own parsed members. Nothing inherited from a host object is ever a member: a row without a member named `constructor` has no such member (S4).

5.2 **Every test reads every row.** The rows are visited once, in order. On each row, in this order: every distinct subject is looked up, in the order of its first appearance in `where`, `match`, `of` and the within `where`; then every test of `where` is applied, then every test of `match`, then `of` is read, then every test of the within `where`. No test is skipped because an earlier one was false. A row on which every test of `where` is true is **selected**. A selected row on which every test of `match` is true is **matched**. The first MALFORMED met in this order on the first row that raises one ends the span. So `field_ambiguous` is raised only for a name some subject reads, and it comes before any test error on the same row (G05).

The draft stopped at the first false test. That made `where_steps` depend on the order the tests were written (C12) and made the leave-one-out values of 8.3 impossible to compute. The cost is that a type fault on a row the selection excludes is now MALFORMED.

5.3 **The relations**, v being the subject's value:
- `present`: with `true`, true when the subject is not absent; with `false`, when it is absent. A member whose value is JSON null is present. A `present` test counts no no_value.
- For `eq`, `in`, `ge` and `le`: if the subject is absent or v is JSON null, the test is false and counts one **no_value** (8.2).
- `eq` with a string literal: true when v is a string of the same code points. With `true` or `false`: when v is that boolean. With an integer or a decimal: when v is a JSON number whose exact value (7.1) equals the literal's. Otherwise false. A string is never equal to a number. `NaN` and the infinities equal nothing.
- `in`: true when `eq` is true for at least one literal. If v is a number and the set holds at least one integer or decimal, v is converted (7.1) whatever the order of the set.
- `ge`, `le` with a string literal: v must be a string. With an integer or a decimal: v must be a JSON number. Otherwise MALFORMED `where_type_mismatch`, detail `{row, type, literal_type}`, where `literal_type` is `string` or `number`. Numbers are compared by exact value, strings by 5.4.

5.4 **String order** is by code point, position by position, a proper prefix being smaller. A host whose strings are UTF-16 must not use its built-in comparison: U+FF5E is below U+1F600 by code point and above it by code unit.

5.5 **The field guard.** After the visit, every `field` or `at` subject of any test (including `present` tests, F8), of `of`, and of the within `where` must have been present in at least one row of the source, selected or not. The first that was present in none, in the order where, match, of, within, is MALFORMED `field_never_present`, detail `{subject, rows_total}`, with the subject object whole (S8). The guard does not apply to a source with zero rows. A MALFORMED raised during the visit comes before it. An honest "no row has X" is the `absent` kind's job, over an entry marked complete.

5.6 Nothing else selects rows. There is no negation, no strict inequality, no "or" other than `in`, no test of what an array holds, no prefix, substring or pattern, no row position, no arithmetic, and no comparison with another field.

## 6. The five operations

N is the number of rows, k the number selected.

- `count`: the value is k. Zero is a value.
- `share`: m is the number of matched rows; the value is m / k. k = 0: MALFORMED `division_by_zero` (10.2).
- `percent`: the value is 100 m / k. k = 0: the same.
- `sum`: on every row, `of` is read. Absent or JSON null: on a selected row, that row counts as no_value. A JSON number: it is converted (7.1), and on a selected row it is added and the row is **used**. Anything else, on any row and a string that spells a number included, is MALFORMED `field_not_numeric`, detail `{row, type}`. If no selected row was used: MALFORMED `no_rows_used` (10.2). The value is the total.
- `mean`: as `sum`; the value is the total divided by the number of rows used. A mean of N integers is a fraction over N, so a printed mean that no N integers can give FAILS (GRIM; vectors D247 and D248).

For `count`, `share` and `percent` every selected row is used. There is no `min`, `max`, `median`, `difference` or `ratio` (section 15).

## 7. Numbers

7.1 **A row number** is a JSON number token, read as text: sign, integer digits I, fraction digits F (possibly none), exponent e (0 if none). Let C be I followed by F with leading zeros removed. If C is empty the value is 0, whatever the exponent. Otherwise, if e without its leading zeros has more than 4 digits, the number is out of range. Otherwise let q = e minus the length of F, remove the trailing zeros of C, and add their count to q. The number is in range when q >= -18 and (length of C) + q <= 18; its value is then C x 10^q with its sign. Every bound is decided on these lengths before any integer is built, so a token of 10,000 digits costs a scan and nothing more (S5). So the numbers a checker computes with are the multiples of 10^-18 below 10^18 in size. Out of range: MALFORMED `number_out_of_range`, detail `{row, number}`, the token's first 80 characters (a token is ASCII). A number is converted only when a test compares it with an integer or a decimal, or when `of` reads it. A number that is never converted is never refused. A JSON string is never a number.

**This domain is narrower than the `numeric` kind's (F7).** v0 7.3 compares a receipt number up to adjusted exponent 320, so the same token `1e-19` is a number for `numeric` and `number_out_of_range` here. The reason: `sum` and `mean` add up to 100,000 values exactly, which needs one fixed scale and a stated integer width. A derive checker needs no arbitrary exponent arithmetic.

7.2 **The exact value** is a fraction P/Q, Q > 0: k/1 for `count`; m/k for `share`; 100m/k for `percent`; T/10^18 for `sum`, T being the integer total in units of 10^-18; T/(u x 10^18) for `mean`, u the rows used. No binary floating point is used anywhere.

**Widths (S6).** A row value is below 10^36 in units, a total below 10^41, a printed number below 10^55 at its own scale, and the products 7.4 and 7.5 form are below 2^260. A checker needs exact integers of at least 260 bits. Both implementations use arbitrary-precision integers: Python's `int` and JavaScript's `BigInt`.

7.3 **The printed number** is taken as v0 7.2 and 7.3 take it (A.3): the token without grouping commas and without `%`, with U+2212 read as `-`; f is its count of digits after the point. `28.6%` is 28.6: no conversion. f above 18: MALFORMED `places_cap`, detail `{fractional_digits, cap}`. More than 37 digits before the point: MALFORMED `printed_too_long`, detail `{integer_digits, cap}` (S6). A wrapping 256-bit checker would otherwise read the 78-digit token 2^256 + 23 as 23 (vector D040).

7.4 **Rounding.** Let n = |P| x 10^f, w the integer quotient of n by Q, and r the remainder. If 2r > Q, or 2r = Q and w is odd, add 1 to w. The rounded value is w written in decimal with exactly f digits after the point (at least one before it), with `-` in front when P < 0 and w is not 0. The value is an **exact tie** when 2r = Q. This is v0 7.3's rule (ties to even, a zero result unsigned) stated for a fraction. The author does not declare the rounding. The printed number is written by the same rule (as P'/10^f, P' its signed digits), so `+5` is `5` and `-0.00` is `0.00`.

7.5 **Comparison**, first that applies:
- The two strings differ: FAILED `value_mismatch`.
- P is not 0 and w is 0: FAILED `rounds_to_zero` (v0 7.3a).
- The op is `share` or `percent`, the exact value is below its maximum (1, or 100), and w equals the maximum at f places: FAILED `rounds_to_whole` (F02). So 999 of 1,000 printed `1.00` or `100%` fails, and 1,000 of 1,000 holds.
- The derivation has `within` and some selected row lies outside the within selection: FAILED `not_within` (F04).
- Otherwise HELD.

Every HELD and FAILED row carries `printed_vs_exact`: `above`, `below` or `equal`, comparing the printed value P'/10^f with P/Q exactly. 9,496 of 10,000 printed `0.95` holds, and the row says `above` (C07).

7.6 **The percent sign is bound (F03).** A printed token that carries `%` with an op other than `percent` is MALFORMED `unit_mismatch`, detail `{printed_token, op}`. The other direction stays allowed: "40 percent" in words with op `percent` holds. The `%` is part of the span's own printed token. Checking it reads no prose.

## 8. What the row shows: the denominator rule and the repairs

Every count below is computed by the checker. The author can declare none of them. All of them are inside the digest.

8.1 **`rows`**:

    total            N: every element, member or line of the row source, whatever it holds
    selected         k
    used             count, share, percent: k; sum, mean: the selected rows that had a number
    matched          share, percent: m
    no_value         sum, mean: selected rows whose `of` was absent or null
    within           with within: the rows the within selection holds
    outside_within   with within: selected rows outside it
    duplicate_rows   array and jsonl sources: rows whose text equals an earlier row's (8.5); null for objects

8.2 **`where_steps`**: one `{passed, no_value}` per test of `where`, in order. `passed` is the number of rows, of all N, on which that test is true. `no_value` is the number on which its subject was absent or null (for `eq`, `in`, `ge`, `le`). Neither depends on the order of the tests (C12). **`match_steps`** (share, percent) is the same for each test of `match`, counted over the selected rows.

8.3 **`without`** (leave-one-out, F06): one entry per test of `where`, in order. `selected` is the number of rows on which every `where` test except this one is true. `matched` (share, percent) is how many of those match. `used` (sum, mean) is how many of those had a number. `derived` is the exact value over those rows, in lowest terms, or null where it is undefined. For C01, the derivation that drops UNSURE answers holds at 0.29 (23/80), and its row says that without that test the value is 23/100 (vector D302).

8.4 **`census`** (F01): one entry per distinct subject of an `eq` or `in` test, in the order of first appearance in `where` then `match`. Each entry is computed over all N rows:

    subject                the subject object
    distinct               how many distinct values occur
    absent                 rows where the subject is absent
    values                 at most 16 entries {type, text, rows}: by rows, most first; then by type
                           name by code point; then by the full text by code point
    other                  N - absent - the rows the shown entries count
    literals_never_seen    the literals of every eq or in test on this subject, each once by canonical
                           text, in the order written, that equal no row's value under 5.3

A value's identity in the census is its type and its text. For a string the text is the string. For a number or a non-finite token it is the token as written, so `1` and `1.0` are two entries, though the integer literal 1 is seen in both. It is `true`, `false` or `null` for those values, and empty for every array and every object, which are counted as one entry per type. The shown `text` is at most 64 code points, with each unpaired surrogate replaced by `?`. A conversion that would be out of range counts as unequal in `literals_never_seen` and raises nothing there. For C06 (a count of status `FAILED` over runs whose failures read `FAIL`), the count holds at 0 and the row shows `PASS` 144, `FAIL` 19, and `FAILED` never seen (vector D289). A literal that never occurs is reported, not refused: an honest zero has one too.

`ge` and `le` subjects have no census.

8.5 **`duplicate_rows`** (F09). For an array, a row's text is its bytes as written, from its first to its last character. For JSON Lines it is the line without leading and trailing space, TAB and CR. Rows whose text equals an earlier row's are counted. Values that differ only in spelling are not duplicates (`1` and `1.0`, a changed timestamp, a reordered object). For an object source the count is null: member names are already unique, and repeated member values are ordinary.

8.6 **`selected_span`** (F11): `first_row` and `last_row`, the positions of the first and last selected rows in source order, and for an object source `first_key` and `last_key`, their keys (shown text as in 8.4). All are null when nothing is selected. For ids `r1` to `r130`, `ge r1, le r50` selects 77 rows, and `last_row` 129 shows the range ran to `r130` (vector D319).

8.7 **`source`** (F05):

    format              json or jsonl
    receipt_bytes       the size of the entry's bytes
    pointer_tokens      the number of POINTER tokens (0 for jsonl)
    parent_members      the member or element count of the container holding the target; null
                        when POINTER is empty or the format is jsonl
    siblings            at most 16 {at, type, size}: the parent's other members or elements that
                        are arrays or objects, in the order written; at is the member name (shown
                        text) or the index in decimal
    sibling_containers  how many such siblings there are in all
    complete            the manifest entry's `complete` (true, false or null)

A pointer to a pre-filtered `kept` beside an `all` of 100 now shows `all` and its size (C02, vector D313). A file filtered before it was captured shows nothing (16.2).

8.8 **Surfaces.** A surface that shows a derive verdict to a person:
- shows the counts with it, in the words "k of N rows selected", for `share` and `percent` "m of k matched", for `sum` and `mean` "u of k used, x without a value", and with `within` "o selected rows outside NAME";
- shows provenance with every verdict: `kind_of_source`, rung, harness, `complete` and the receipt size, from the row (F07);
- renders the derivation as a fixed template sentence beside the author's sentence, such as `share of rows where field "actual" = true, among rows where field "pred" = true` (F12). The template reads the derivation object and nothing else, and the comparison with the author's prose stays the person's;
- says "printed rounds up" when `printed_vs_exact` is `above`, and "rounds down" when it is `below`;
- shows each census entry and each literal never seen, and the value without each `where` test;
- shows every string from a derivation, a row or a detail as text, never as markup, and escapes it for the medium (S9; vector D323 carries a field named `<img src=x onerror=alert(1)>`);
- never uses the words "raw" or "recomputed from the record" (F07). The profile sentence is "recomputed from rows of the named entry", and nothing stronger, unless the bytes are signed by a party other than the author (rung L3, reserved in v0).

The surface text of the two implementations (`surface` in the result object, 13.3) follows these rules. It is outside every digest.

## 9. Order of checks for one derive span

First that applies ends the span.

  a. v0 7.1a: a mark from the span scan (outside the checking function).
  b. The inner bytes are all ASCII whitespace (space, TAB, LF, VT, FF, CR), or empty: MALFORMED `empty_span`.
  c. The inner text has more than 300 code points: MALFORMED `length_cap`, detail `{code_points, bytes, cap}`.
  d. It is the 257th or a later derive span, counted in `at` order: MALFORMED `derive_span_cap`, detail `{cap}`.
  e. In order: REF fails 2.1 (`receipt_form`); U+202D or U+202E in the inner text (`directional_override`, detail `{code_point}`, the first found); A.3 (`number_count`, `number_grammar`, detail `{digit_bearing_tokens}`); `places_cap`; `printed_too_long`; the block table (`derive_block_malformed`); NAME not in the table (`derivation_missing`, detail `{name}`); 3.7 (`derivation_form`, `op_unknown`); 7.6 (`unit_mismatch`); 4.5 (`within_form`). No answer up to here depends on receipt bytes.
  f. Resolution, v0 section 6, as for a `numeric` span without a fragment: UNRESOLVED with v0's reason (`manifest_absent`, `manifest_integrity`, `rung_unknown`, `manifest_id_missing`, `manifest_bytes_absent`), or MALFORMED `kind_of_source_unknown` or `receipt_author_minted`, detail `{}`. The entry must carry bytes.
  g. `rows_too_large`; the budget's bytes check (12.2); the row source, section 4; `row_key_unavailable` (4.6); the budget's rows check. A MALFORMED from this step carries neither `provenance` nor `resolved_sha256`.
  h. The visit (5.2): `field_ambiguous`, `number_out_of_range`, `where_type_mismatch`, `field_not_numeric`; then the field guard; then `division_by_zero` or `no_rows_used`. A MALFORMED from this step carries both.
  i. The comparison, 7.5.

## 10. The result

10.1 The span row is v0 section 8's, with `kind` = `derive` and `receipt` = the REF as written. For HELD and FAILED its `detail` has exactly these keys:

    printed_token      as v0 7.3
    printed            the printed number, written by 7.4
    fractional_digits  f
    rounding           "ROUND_HALF_EVEN"
    derived            {"num", "den"}: the exact value in lowest terms, den > 0, both decimal strings
    derived_rounded    7.4
    exact_tie          true or false
    printed_vs_exact   above, below or equal (7.5)
    derivation         the derivation object, in canonical form
    derivation_sha256  sha256 of its canonical text
    rows               8.1
    where_steps        8.2
    match_steps        8.2, share and percent only
    without            8.3
    census             8.4
    selected_span      8.6
    source             8.7

10.2 **`division_by_zero` and `no_rows_used`** (F4) carry the same detail without the five value keys (`rounding`, `derived`, `derived_rounded`, `exact_tie`, `printed_vs_exact`). The `without` entries in it may hold values: dropping a test may make the selection non-empty, and the row shows that.

10.3 Every other reason's detail is the one this text gives where it names the reason. A reason given none has `{}`. UNRESOLVED rows have `{}`.

10.4 Every value a derive row carries has v0's canonical form (A.2), so no derive input can make the verdict core uncanonical and reach v0 11.2 (S1). Integers in a row are counts below 2^53. Every exact number is a decimal string.

10.5 The verdict core, its digest and the document verdict are v0 section 9's, with one change: when at least one row has KIND `derive`, `profile` is `rn-derive/v0-draft`. That id stands for everything `rn-inline/v0-draft` stands for, plus this: each `derive` span that reads HELD or FAILED was recomputed by one of five built-in operations from rows of the manifest entry it names, selected as the derivation in the document says, and the rounded result was compared with the printed number. A document with no derive span is answered exactly as v0 answers it, profile included. A checker that knows only v0 answers `kind_unknown` for a derive span and `profile_not_checkable` for a receipt of this profile; both are the intended closed answers.

## 11. The closed lists added

MALFORMED, from the document's bytes alone: `derive_span_cap`, `places_cap`, `printed_too_long`, `derive_block_malformed`, `derivation_missing`, `derivation_form`, `op_unknown`, `unit_mismatch`, `within_form`. (`empty_span`, `length_cap`, `receipt_form`, `directional_override`, `number_count`, `number_grammar` are reached as in v0.)

MALFORMED, from the derivation and the bytes it names: `rows_too_large`, `derive_budget`, `receipt_not_jsonl`, `receipt_too_many_values`, `rows_not_container`, `rows_ambiguous`, `rows_too_many`, `row_key_unavailable`, `field_ambiguous`, `where_type_mismatch`, `number_out_of_range`, `field_not_numeric`, `field_never_present`, `division_by_zero`, `no_rows_used`. (v0's `receipt_not_json`, `pointer_unresolvable`, `pointer_ambiguous`, `kind_of_source_unknown` and `receipt_author_minted` are reached as in v0.)

**Capacity reasons** (S5) are a named group inside MALFORMED: `derive_span_cap`, `places_cap`, `printed_too_long`, `rows_too_large`, `rows_too_many`, `receipt_too_many_values`, `derive_budget`. A surface says "over the checker's limits" for these, not "wrong". They stay MALFORMED, so the document verdict stays `SWORN-FAILED` and padding a file cannot turn a FAILED into an UNRESOLVED.

FAILED: `value_mismatch`, `rounds_to_zero` (as v0), `rounds_to_whole`, `not_within`.

UNRESOLVED: v0's list, nothing added. Refusals of the document (v0 section 11): nothing added. Refusals of the checking function: section 13.

A checker that raises anything outside these lists (an exception, a crash, a host error) on any input is non-conformant (S5).

## 12. Limits, cost and memory

12.1 Limits:

    receipt bytes behind one derive span      16,777,216     rows_too_large
    JSON values read from one receipt         1,000,000      receipt_too_many_values
    containers deep (receipt, line, block)    32             receipt_not_json, receipt_not_jsonl, derive_block_malformed
    rows in one row source                    100,000        rows_too_many
    derive blocks in a document               16             derive_block_malformed
    bytes of one block's content              65,536         derive_block_malformed
    derivations in a document                 256            derive_block_malformed
    derive spans in a document                256            derive_span_cap
    canonical text of one derivation          4,096 bytes    derivation_form
    tests in where and match together         8              derivation_form
    literals in one in                        16             derivation_form
    a string literal, a field name            256 bytes      derivation_form
    POINTER, an at subject                    1,024 bytes, 16 tokens   derivation_form
    a derivation integer                      |n| <= 2^53 - 1, at most 16 digits   derivation_form
    a decimal literal                         18 digits each side of the point     derivation_form
    a row number                              below 10^18, a multiple of 10^-18, exponent of at most 4 digits   number_out_of_range
    digits after the point, printed           18             places_cap
    digits before the point, printed          37             printed_too_long
    budget units per document                 67,108,864     derive_budget

12.2 **The document budget (S3).** A span's cost is the size of its entry's bytes, unless an earlier derive span of the same document already read that entry, plus N x (T + 1), where T is the number of tests in `where`, `match` and the within `where`, plus 1 if there is `of`. The bytes part is checked after `rows_too_large` and before reading. The rows part is checked after the row source and `row_key_unavailable`. A span whose part would take the document's total over 67,108,864 units is MALFORMED `derive_budget`, detail `{budget, spent_before, cost}`, and spends nothing for that part. A span that fails while reading has spent its bytes part. Every checker reaches the same step, so a server's limit is the text's limit. A checker must read each distinct entry once per document; the answers are the same either way. The checking function takes the units spent before it and whether its entry was read before, and returns the units spent after it (13.3).

12.3 **Cost of one span.** The receipt is read once, and the JSON reader's work is linear in its bytes. Each row is visited once. On a row there are at most 17 subject lookups (8 tests of the derivation, 8 of the within `where`, and `of`), and the literals compared come from two derivations of at most 4,096 canonical bytes each, so fewer than 8,192 literal bytes are compared per row. The census and the leave-one-out are counters kept in the same pass. Measured on this machine (builder notes `test_run_2.out`): a 3.7 MB source of 100,000 rows with 8 tests took 2.75 s in `derive.py` and 0.77 s in `derive.js`; 16 MiB of `[` stopped in 0.01 s and 0.09 s; 8,000,000 string escapes took 2.40 s and 1.02 s.

12.4 **Memory (S2).** Memory is the parsed receipt plus counters. The parse holds at most 1,000,000 values and 16 MiB of text. Measured at exactly 1,000,000 values (builder notes `memory_2.out`): 499,998 small objects took a Python heap peak of 500.2 MB and a Node maximum resident set of 718.4 MB; 1,000,000 flat numbers took 178.8 MB and 259.4 MB. A phone browser tab may not survive the first. Lowering the value cap changes answers, so a surface cannot lower it on its own (18.4).

12.5 The checker reads no clock, opens no connection, runs no program, reads no file but its inputs, and writes nothing (v0 11.4).

## 13. The checking function, the claim file, and the command line

13.1 The span scan (v0 2.3) and the manifest steps (v0 sections 5 and 6) are v0's. The **checking function** is everything from 9b on for one derive span, given:
- the derive blocks of the document, in order: each `{line, content}`, the content as a string;
- the span: `{receipt, inner, ordinal}`, `ordinal` being its 1-based count among derive spans (default 1);
- the resolution of its REF by v0 section 6: `{"verdict": "RESOLVED", "entry": {kind_of_source, complete, harness, rung}}`, or `{"verdict": "UNRESOLVED", "reason"}`, or `{"verdict": "MALFORMED", "reason"}` with a reason v0 6.6 to 6.8 can give;
- the budget: `{spent_before, entry_read_before}` (default 0, false);
- the entry's bytes.

13.2 The **claim file** is these inputs, except the bytes, as one JSON object with keys `blocks`, `span`, `resolution` and optionally `budget`, read under A.1. The checking function **refuses** it, emitting no row, when it is over 8,388,608 bytes (`claim_too_large`); when it is not well-formed UTF-8, does not read, holds a repeated key or an unpaired surrogate anywhere, has a missing or extra key, a value of the wrong type, a `rung` outside `L1`, `L2`, `undeclared`, an UNRESOLVED reason outside v0's five, a negative integer, or an ordinal or line below 1 (`claim_form`). Integers in the claim file are written without sign, fraction or exponent, and have at most 15 digits.

13.3 **The result object** is `{profile, row, spent, surface}`: `profile` is `derive/v0-draft`; `row` is the span row without the scan's fields (`at`, `start`, `end`, `claim_sha256`, `claim`); `spent` is the budget units after this span; `surface` is the text of 8.8, empty unless the row is HELD or FAILED. Only `row` enters a verdict core. A refusal is `{"refusal": code}`. Both are printed as canonical JSON (A.2).

13.4 **The command line.** `python derive.py CLAIM_FILE RECEIPT_FILE` and `node derive.js CLAIM_FILE RECEIPT_FILE` read the two files the user names, and no path written inside either file. They print the result object and exit 0, or 2 on a refusal. `--batch VECTOR_FILE` prints one `{"id", "out"}` line per vector. Both files are importable: `derive.py` as a module, and `derive.js` through `module.exports` in Node and `globalThis.DeriveV0` in a browser.

## 14. Implementers

These rules exist because a port from the earlier text would get each one wrong (S4, S5, S10). Each has a vector.

- **Never the host's default JSON reader.** Read JSON by A.1. In JavaScript, `JSON.parse` rounds `9007199254740993` to 2^53, reads `1e400` as a value that writes as `null`, reads `1.0000000000000000001` as 1, keeps the last of repeated keys silently, and throws on `NaN` (vectors D217, D194, D256, D221). Python's `json` module has no depth or value count while reading.
- **Decode UTF-8 fatally, without stripping a byte order mark.** A BOM-prefixed `[1]` is `receipt_not_json` (D145). CESU-8 surrogate bytes are not UTF-8 (D146).
- **Objects are lists of pairs**, looked up by exact string equality after escapes are decoded (`"a"` and `"a"` are one name; D188, D161). In JavaScript use `Map`, never a plain object. Then a row has a member named `constructor` only when it was written (D222), and a member named `__proto__` is an ordinary member (D224).
- **Closed names are compared, never looked up.** `op`, a relation name and a subject name are compared by exact equality with the closed lists before any dispatch, and are never used as a key into a host object, attribute or module (D079 to D081).
- **An array token must match the index grammar before use**: `0`, or a non-zero digit and at most 8 more digits. `length` is not an index (D158).
- **Bounds are decided on text before conversion** (3.1, 7.1, 7.3). No integer is built from an unbounded digit string. Python 3.12 refuses to convert more than 4,300 digits by default, so a checker that converts first crashes there.
- **Code points, never UTF-16 units**, for every length, every truncation and every comparison (5.4, 8.4). Never slice a JavaScript string by index to cut text.
- **No floating point** in any value. A checker needs exact integers of 260 bits (7.2).
- **Freeze rule (S10).** No relation whose cost or result depends on the structure of a literal may be added: no regular expression, glob, locale-aware or case-insensitive comparison, ever. If `prefix` is added later, it is a code-point prefix whose cost is linear in the literal.

## 15. What this cannot express, on purpose

- **A join of two row sources.** "21 of 85 path accusations upheld" needs a field from one file and an answer from another.
- **Rows spread over several byte objects.** One span names one manifest entry.
- **One operation inside another**, and **a claim of two printed numbers.** There is no ratio of two sums, no difference of two shares (a "points over baseline" figure) and no mean of group means. `within` links two spans' selections; it does not compute across them.
- **min, max, median, difference, ratio.** `max` and `min` stay out because "best seed" is itself a cherry-pick (F10). `op_unknown` answers all five.
- **Variance, an interval bound, a p-value, anything with a root or a logarithm.**
- **Pattern, prefix or substring tests; a test of what an array holds; a function of several fields of one row; negation; a strict inequality; a row's position.**
- **CSV and JUnit XML.** Out. CSV leaves the header and the quoting optional and every cell is text. XML needs a second reader.
- **Rows that cannot be handed over.** The checker needs the bytes.
- **A function the author defines, in any form.**

## 16. What a HELD derive span does not show

v0 section 12 applies in full. In addition:

1. **That the rows are anyone's but the author's (F07).** If the author's own script wrote the rows, a HELD derive span is a consistency check one step further back than a `numeric` span, and no more. RECEIPT v0 counts `tool_stdout` as external, so an author's script printing rows passes the authorship test. Rows synthesised from a stored 0.95 hold exactly as real rows do (C09, vector D336). The only signal is provenance, which a surface must show.
2. **That the rows are all the rows.** N is the size of the named container. Runs never logged, files never handed over, and rows deleted or filtered before the bytes were captured are not counted. `source` shows only containers beside the target in the same file.
3. **That the selection is the right one.** The author chose it. `without` and the census show what each test did and what the tested fields hold; they do not say which selection was fair.
4. **How many derivations were tried (F08).** One derivation from a garden of 105, published alone, holds (C10, vector D337). `derivation_sha256` is in the row so that a ledger outside this checker, sealed before the receipt was captured, can bind it. Nothing in this checker does.
5. **That the operation answers the question, or that a field is the quantity the sentence names.** A precision can be printed where the prose says recall (C04). The template sentence of 8.8 shows what was computed. The checker never reads the prose.
6. **What a selection means.** "Keys E1-000 to E1-099 are the accusation arm" is a fact about the lab's recipe. The bytes do not say it.
7. **That a zero is not a misspelt literal.** A misspelt field name is now MALFORMED (5.5). A misspelt literal still selects nothing and a count of 0 holds; the census shows the values that do occur and names the literal that never does.
8. **That rows are not near-duplicates.** Exact repeats are counted (8.5). A repeat with a changed timestamp is not.
9. **Which side is wrong when it FAILS.** The rows may be the wrong ones.
10. **That a FAILED on an exact tie is an error.** A tool that rounds halves up prints the other neighbour. `exact_tie` is in the row so that this case can be told from a different number.
11. **That skipped rows did not matter.** `sum` and `mean` hold over the rows that had a value; `rows.no_value` says how many did not (C05).
12. **That another checker agrees.** Section 18.

## 17. Rules taken from elsewhere

- **RECEIPT v0**: the span grammar, the manifest, the strict JSON reader, the pointer walk, the printed-number grammar, ties to even, the rounding floor, the canonical form, the four verdict words. Restated in Appendix A.
- **JSON Lines**: one value per line; a blank line is not a value.
- **SQL** (general knowledge, not fetched): an aggregate skips NULL and a comparison with NULL is not true. Here the skipped rows are counted in the answer.
- **Vega-Lite's aggregate vocabulary**: the count of objects is kept apart from the count of valid values.
- **GRIM**: a mean of N integers is a fraction over N.
- **XBRL Calculations 1.1**: a closed arithmetic relation with the rounding rule in the specification and a code per failure. Its interval rule is not taken (18.2).
- **CEL**: the cost of every construct is stated. **JsonLogic**: a computation written as inert data.
- **`styxx.protocol` v4** (the lab's own): an aggregate over an empty population refuses; an unknown aggregate refuses.
- **AggChecker and Scrutinizer**: recomputing a printed aggregate from the underlying table. Here the author hands over the query and nothing is guessed.
- **Multiverse and specification-curve analysis** (general knowledge): reporting the value under neighbouring choices. `without` is the smallest such report.

None of these was fetched or re-read for this repair. The designer's draft's prior-art scout is the source.

## 18. Open before freeze

1. **RECEIPT v0 is not frozen.** Appendix A restates the rules derive depends on as v0 reads on 2026-11-13. A change there must be copied here.
2. **Ties.** 7.4 keeps ties to even. A share that is an exact tie, printed by a half-up tool, FAILS. Not measured: how many printed shares are exact ties.
3. **Unicode version (inherited from v0 13.1).** A.3's letter, number and digit classes depend on a Unicode version this text does not pin, and **the two implementations disagree today**. Python 3.12.10 has Unicode 15.0.0 and Node v24.13.0 has Unicode 16.0. Over every code point, 5,004 are letters or numbers to Node and not to Python, and 80 of those are decimal digits (builder notes `unicode_class_diff.out`). The inner text `5 U+10D40` holds in `derive.py` and is `number_count` in `derive.js` (`unicode_divergence.out`). No vector carries such a code point. Pinning a version needs a class table in the text.
4. **Memory at the caps** is too large for a phone tab (12.4). A lower value cap changes answers and must be the text's choice.
5. **Every test reads every row** (5.2). A header row or a "n/a" in a field the selection excludes is now MALFORMED. Not measured on the lab's real files.
6. **`within` is one level** and checks containment only. A pair of spans can still be printed as "k of n" without `within`. A surface could refuse to render such a pair, but that would mean reading the prose between them.
7. **`difference` of two derivations over one source** (F10's last step) is not in. It would need two printed numbers per span or a span that names two derivations.
8. **Where the derivation lives.** It is a fenced block in the document. A block inside an HTML comment is invisible when rendered and still counts. The row carries the derivation either way.
9. **Over the size limit is MALFORMED here.** Today's `numeric` code answers a 64 MiB limit with UNRESOLVED `receipt_too_large`.
10. **`provenance` on MALFORMED rows** follows v0 6.14 by analogy (9g, 9h). v0 does not yet say it for a fifth kind.
11. **Disclosure.** The row carries the derivation, census values (64 code points each) and sibling names, so text from a private row source appears in the receipt.
12. **The vectors' reach is measured once, by their author.** 347 vectors, partial expectations written from this text; 25 mutants and 2 controls. Run 1 missed 2 mutants (raw control characters in strings, the exponent-digit bound), and vectors were added for both. On the final code (run 3, `mutants_3.out` and `mutants_3_j02b.out`, with the control-character mutant rebuilt for the rewritten JavaScript string reader), 23 are caught by expectation, 2 crash (a visible failure), none is missed, and neither control is caught (builder notes `mutants.out`, `mutants_2.out`, `mutants_3.out`). Most mutants are caught by 1 to 3 vectors. The mutation catalogue was written by the same agent who wrote the vectors. No outside mutation catalogue exists.
13. **The checking function, not the document.** No code here runs the span scan or the manifest. The order of 9a and 9f with v0's steps is stated, not tested.

## 19. Repairs, finding by finding

Every fatal or high finding is repaired or its feature is cut. "Partial" means the finding's repair was taken and the case it cannot reach is stated in section 16.

| finding | weight | what was done |
|---|---|---|
| F01 misspelt literal proves a zero | high | Repaired: `census` with `literals_never_seen` (8.4). The member-name census proposed for `present` tests is **cut**, made unnecessary by extending the guard to `present` (F8). |
| F02 rounding hides a missed threshold | high | Repaired: FAILED `rounds_to_whole`; `printed_vs_exact` in every row; the surface says "rounds up" (7.5, 8.8). The threshold in prose stays out of scope. |
| F03 % not bound | high | Repaired: MALFORMED `unit_mismatch` (7.6). |
| F04 k of n not nested | high | Repaired: `within`, FAILED `not_within`, `rows.within` and `rows.outside_within` (4.5, 7.5). |
| F05 filter moved into the bytes | high | Partial: `source` with siblings and `complete` (8.7). Upstream filtering stays invisible (16.2). |
| F06 dropped rows, order dependence | high | Repaired: `without` (8.3); every test reads every row, so `where_steps` no longer depend on order (5.2). |
| F07 summary rows indistinguishable | high | No result repair exists. Text: provenance on every surface, the words "raw" and "recomputed from the record" banned, profile sentence kept narrow (8.8, 16.1). |
| F08 garden of forking paths | medium | Text only: 16.4. `derivation_sha256` is there for an outside ledger. |
| F09 duplicated rows | medium | Repaired: `rows.duplicate_rows` (8.5). Near-duplicates stay invisible (16.8). |
| F10 vocabulary too narrow | medium | Partial: `mean`, decimal literals and `at` subjects added, each covered by the census and `without`. `difference`/`ratio` not added (18.7). `max`/`min`/`median` **cut**. CSV and JUnit XML **cut**. |
| F11 code-point id ranges | low | Repaired: `selected_span` (8.6). |
| F12 precision printed as recall | medium | Repaired at the surface: the template sentence (8.8), outside the digest. |
| F1 text not implementable alone | high | Repaired: Appendix A restates every v0 rule used. v0 itself is unfrozen (18.1). |
| F2 no vectors | high | Repaired: `vectors/derive_v0_vectors.json`, 347 vectors with inputs, rule and expectation. |
| F3 error precedence | medium | Repaired: G07 size before op (3.7); G12 row cap before repeated names (4.2); G13 jsonl row key is `derivation_form`, and the array case is decided before the visit (3.4, 4.6); G05 lookups by subject on every row (5.2). Each has a vector. |
| F4 detail of two reasons | medium | Repaired: 10.2. |
| F5 depth and byte units | medium | Repaired: the outermost container counts one (A.1); bytes are the UTF-8 of the decoded string (3.4). |
| F6 float readers, 64-bit sums | medium | Repaired: vectors for 2^53 + 1, `1.0` against 1, `2.50`, `NaN`, repeated and escaped-repeated keys, and a sum whose unreduced numerator exceeds 2^63 (D250). |
| F7 number domain differs | low | Repaired: named in 7.1 with its reason. |
| F8 present unguarded | low | Repaired: the guard covers `present` (5.5). X07a of the draft is now `field_never_present`. |
| S1 2^53 breaks the core | high | Repaired: 2^53 - 1, decided on digits (3.1); 10.4. |
| S2 a gigabyte per span | high | Repaired: value cap counted while reading, depth counted while reading, memory restated from measurement (12.4). Still large: 18.4. |
| S3 no document budget | high | Repaired: `derive_budget` in units, not time (12.2). |
| S4 host reader and plain objects | high | Repaired: section 14 and its vectors. |
| S5 refusal, crash and verdict blur | medium | Repaired: bounds on text first; `op` carried whole; a crash is non-conformant; capacity reasons named (11). |
| S6 200 bits too few | medium | Repaired: `printed_too_long` at 37 digits; widths stated (7.2, 7.3). |
| S7 path: and prereg: | medium | Repaired: only `rN#NAME` (2.1), with a vector for `path:../../x#n`. |
| S8 sketch departs from text | low | Repaired: subjects carried whole in details; the JSON Lines line count comes before any split. The sketch's 26 examples were not turned into vectors; the vectors were written from this text. |
| S9 author strings in a browser | low | Repaired: 8.8, vector D323. |
| S10 pressure toward patterns | low | Repaired: the freeze rule (14). |

**Answers that change against the designer's draft.** X07a (`count` where a field is `present`, 0 of 163) is now MALFORMED `field_never_present`. X15 (`mean`) is now an operation. X19's detail says `literal_type` `number`, not `integer`. `where_steps` count every row on which a test is true, not the rows still in after the earlier tests. `field_never_present` and `field_ambiguous` carry `subject`, not `field`. Every HELD and FAILED detail has six more keys (`printed_vs_exact`, `without`, `census`, `selected_span`, `source`, and `rows.duplicate_rows`), so no digest from the draft's sketch equals one from this text.

## 20. What this text is not

It is a lab draft of a profile. It is not a specification anyone has adopted, and it does not claim the idea is new: each part exists elsewhere (section 17), and what the scouts did not find in one place is only the combination. It checks a number an author hands over against rows the author names. It does not find claims, does not judge whether a statistic is the right one, and does not detect false statements in prose. HELD on a derive span means the printed number re-derives from the named bytes under the named selection and operation. It means none of the things section 16 and v0 section 12 list.

## Appendix A. The v0 rules derive depends on, restated

A.1 **The strict JSON reader** (v0 6.13, with depth and a value count added).
- Input is well-formed UTF-8 (Unicode Table 3-7: no overlong form, no surrogate, nothing above U+10FFFF). A byte order mark is not stripped. U+FEFF is not whitespace, so a document starting with one does not read.
- Whitespace is space, TAB, LF and CR only.
- A value is an object, an array, a string, a number, `true`, `false`, `null`, or one of the bare tokens `NaN`, `Infinity` and `-Infinity`. The bare tokens are values and never numbers.
- A string holds no raw character below U+0020. Its escapes are `\" \\ \/ \b \f \n \r \t` and `\uXXXX` with four hex digits of either case. A `\u` escape of a high surrogate immediately followed by a `\u` escape of a low surrogate is one code point. Any other surrogate escape is kept as an unpaired surrogate.
- A number is RFC 8259's: an optional `-`, `0` or a non-zero digit and digits, an optional `.` and one or more digits, and an optional `e` or `E`, an optional sign and one or more digits. It keeps its token text. It is never converted while reading.
- An object keeps its members as pairs, in order. A name written twice (compared after escapes are decoded) keeps both pairs. A lookup by that name answers the last value and knows it was repeated.
- **Depth is counted while reading**: opening a container while 32 are open does not read. The outermost container counts as one.
- **Values are counted while reading**: each value counts once when its first character is reached; member names do not count. Reaching the 1,000,001st value ends the read with the value-cap answer of the caller.
- After the top value, only whitespace may follow.

A.2 **The canonical form** (v0 3.1 to 3.3). RFC 8785 restricted to null, true, false, integers with |n| <= 2^53 - 1, strings, arrays and objects. No number with a fraction or an exponent. Object keys are strings of code points U+0000..U+007E, sorted by code point, with no duplicates. Strings escape `"` and `\` as `\"` and `\\`, U+0008, U+0009, U+000A, U+000C and U+000D as `\b \t \n \f \r`, and other code points below U+0020 as `\u00xx` in lowercase hex; everything else is written raw. A string with an unpaired surrogate has no canonical form. There is no whitespace between tokens. Containers nest at most 32 deep, the outermost counting as one. sha256 is over the UTF-8 bytes of the canonical text, written as 64 lowercase hex digits.

A.3 **The printed number** (v0 7.2 and 7.3). Cut the inner text into maximal tokens of letters and numbers (Unicode General_Category L* and N*), the characters `_ . , + - % / :`, U+2212, U+00B1, and the 26 dash code points U+00AD U+058A U+05BE U+1400 U+1806 U+2010 U+2011 U+2012 U+2013 U+2014 U+2015 U+2E17 U+2E1A U+2E3A U+2E3B U+2E40 U+2E5D U+301C U+3030 U+30A0 U+FE31 U+FE32 U+FE58 U+FE63 U+FF0D U+10EAD. A token is digit-bearing if it holds a character of General_Category Nd. Not exactly one digit-bearing token: `number_count`. Strip every trailing `.`, `,` and `:` from it. The rest must be, whole: an optional sign (`-`, `+` or U+2212), then either ASCII digits (plain, or grouped in threes with `,` after a leading group of one to three) with an optional `.` and ASCII digits, or `.` and ASCII digits; then an optional `%`. Otherwise `number_grammar`. Detail for both is `{digit_bearing_tokens}`, the tokens as found. `printed_token` is the stripped token, otherwise as written. The printed value drops the grouping commas and the `%`, and reads U+2212 as `-`.

A.4 **Fences** (v0 2.2). Lines end at LF only. A fence delimiter is a line that begins with zero to three spaces and then three or more backticks. Delimiters pair in order: the first opens, the second closes. A fenced region runs from the first byte of its opening delimiter line through its closing delimiter line. An odd number of delimiters gives the document no spans (`unbalanced_fences`).

A.5 **JSON Pointer** (v0 4.3 and 6.12). The empty string names the whole value. Otherwise the pointer starts with `/`, and every `~` is followed by `0` or `1`. Tokens are split on `/`, then `~1` becomes `/` and then `~0` becomes `~`. In an object a token must be a member name, and a name written twice is ambiguous. In an array a token must be `0` or a non-zero digit followed by at most 8 digits, and below the length. A scalar cannot be walked.

## Appendix B. Files

- `derive.py`, `derive.js`: the checking function and command line, in Python (standard library only) and dependency-free JavaScript.
- `vectors/derive_v0_vectors.json`: 347 vectors. Each has `id`, `rule`, `claim_b64` (or `claim_parts`), `receipt` (base64 parts with repeat counts) and `expect`, a partial expectation written from this text.
- `vectors/make_vectors.py`: writes the vector file. It imports neither implementation, and it refuses to overwrite a vector file whose content would change unless it is given `--force`.
- `tests/test_derive_v0.py`: runs every vector through both implementations, checks every expectation in both, requires byte-identical output, fails (does not skip) without node, scans both sources for forbidden names, and times six hostile inputs.
