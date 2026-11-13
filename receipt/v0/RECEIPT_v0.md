# RECEIPT v0

Status: DRAFT, unsigned, nothing frozen. Three rules here change a verdict against today's code (D1, D2, D3 in `DECISIONS_v0.md`); they were taken in draft by the lead on 2026-11-13 and wait for the operator's signature.

What an author writes, what a harness mints, what a checker returns, and the digest two checkers must agree on. This text describes one checker the lab has (`styxx/sworn.py` at `6064871b`) as rules that do not mention it. It is a lab draft: it has not been implemented from the text by anyone, and nobody outside the lab emits or checks these objects.

Section 12 says what a HELD verdict does not show. Read it before relying on one.

## 1. Terms

- **document**: a byte string an author wrote. The checker reads its bytes, never a rendering of it.
- **span**: a marked run of the document, `<sworn r="REF" k="KIND">inner</sworn>`. The inner bytes are the **claim**.
- **REF**: names the bytes the claim rests on. In v0 only `rN` (an entry in a manifest) is resolved.
- **KIND**: how the claim and the bytes are compared: `numeric`, `quote`, `hash` or `absent`.
- **manifest**: an object a harness mints, never the author, listing byte objects by id with where each came from.
- **receipt bytes**: the byte object a manifest entry names. (A manifest calls its entries "receipts"; the object this text defines is the **verdict receipt**.)
- **span row**: the checker's answer for one span. **verdict core**: all rows and the document verdict. **digest**: sha256 of the core's canonical form.
- **checker**: a pure function of (document bytes, document name, manifest or none, commit label or none). No clock, no network, no file system, no repository, no model.

sha256 is always over exact bytes and written as 64 lowercase hex digits. Nothing is ever normalised before hashing: no line-ending change, no Unicode normal form, no trimming.

## 2. The span grammar

2.1 **UTF-8.** A document that is not well-formed UTF-8 (no overlong forms, no surrogates, nothing above U+10FFFF) has no spans. `document_malformed` is `{"reason": "invalid_utf8", "at": n}`, n the offset of the first byte of the first ill-formed sequence. `canonical_sha256` is null and there are no rows.

An empty byte string is a document: it has no spans, its canonical text is empty, and it is `UNSWORN` with `canonical_sha256` = `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

2.2 **Fences.** Lines end at LF (0x0A) only. A fence delimiter is a line that begins with zero to three spaces (0x20) and then three or more backticks (0x60). Delimiters pair in order: first opens, second closes, third opens. A fenced region runs from the first byte of an opening delimiter line through the last byte of its closing delimiter line, its LF included. An odd number of delimiters gives no spans and `document_malformed` = `{"reason": "unbalanced_fences", "delimiter_lines": [...]}`, the 1-based line numbers of every delimiter; `canonical_sha256` is null and there are no rows. Tilde fences, blockquoted fences and indented code are not fences.

2.3 **The scan.** Keep a position p (from 0), a stack of open declarations and a comment end c (0 at the start). At each p take the first case that applies:

  a. p is inside a fenced region: move p to the region's end.
  b. The bytes at p are `<!--` and p >= c: set c to just past the next `-->` found from p+4, or to the document's length if there is none; advance p by 4. Scanning continues inside the comment.
  c. The byte at p is a backtick: let n be the length of the maximal backtick run at p. Look on the same line, after the run, for the first maximal backtick run of length exactly n. If there is one, move p past it (an inline code span; nothing inside it is a tag). If not, move p past the opening run only.
  d. The byte at p is `<` and a **candidate** starts at p: `<`, an optional `/`, the letters s, w, o, r, n in either ASCII case, and then a byte that is not `A-Z a-z 0-9 _ -` (the end of the document counts). Then, first that applies:
     - An **opener** starts at p: exactly `<sworn r="` REF `" k="` KIND `">`, lowercase, single spaces, double quotes, where REF and KIND are each zero or more bytes other than `"`, `<`, `>`, CR and LF. A declaration `{at: p, receipt: REF, kind: KIND}` opens. If any declaration is already open, every open one not yet marked is marked `nesting` and so is the new one; otherwise, if p < c, the new one is marked `hidden_commitment`. Push it; move p past the opener.
     - The bytes at p are exactly `</sworn>`: if a declaration is open, pop the innermost; its **inner bytes** run from the end of its opener to p; if they are zero bytes and its mark is none or `hidden_commitment`, mark it `empty_span`. If none is open, record a declaration `{at: p, receipt: null, kind: null}` marked `stray_closer`, with `raw` = `</sworn>`. Advance p by 8.
     - Otherwise record a declaration `{at: p, receipt: null, kind: null}` marked `tag_syntax`, with `raw` = the bytes from p up to and including the next `>`, or up to but not including the next LF, or to the end, whichever comes first; move p there.
  e. Otherwise advance p by 1.

2.4 At the end, every declaration still open whose mark is none or `hidden_commitment` is marked `unclosed`.

2.5 **Canonical text.** If no declaration is marked `tag_syntax`, `nesting`, `stray_closer` or `unclosed`, the canonical text is the document with every opener and every paired closer deleted, and each declaration's `start` and `end` are byte offsets into the UTF-8 canonical text of its inner bytes (end exclusive). Example: in `é<sworn r="r1" k="numeric">n 7</sworn>`, `at` is 2 (é is two bytes), `start` is 2 and `end` is 5. Otherwise there is no canonical text: `canonical_sha256` is null and every row's `start` and `end` are null.

2.6 Every declaration yields exactly one span row; rows are ordered by `at`. A paired closer yields no row. Text outside spans is narrative and is never judged.

## 3. The canonical form

3.1 The canonical text of a value is RFC 8785 (JCS) restricted to: null, true, false, integers with |n| <= 2^53 - 1, strings, arrays, and objects. No number with a fraction or an exponent is admitted anywhere. The bound is chosen so that every integer a binary double rounds onto (2^53 and beyond) is outside it. A checker that reads JSON decides on the number as written, never on a binary value it was converted to: `9007199254740993` is `int_range` and `1.0` is `float`, whatever a host's parser turns them into.

3.2 Objects: keys are strings of code points U+0000..U+007E, sorted by code point, no duplicates. Strings: `"` and `\` escaped as `\"` and `\\`; U+0008, U+0009, U+000A, U+000C, U+000D as `\b \t \n \f \r`; other code points below U+0020 as `\u00xx` with lowercase hex; everything else raw (U+007F, U+2028 and U+2029 included). Strings are never normalised. No whitespace between tokens. Containers nest at most 32 deep, the outermost counting as one.

3.3 A value outside 3.1 and 3.2 has no canonical form. The codes: `float` (a non-integral number, or any floating value such as 1.0), `int_range`, `non_ascii_key`, `lone_surrogate` (a string holding an unpaired UTF-16 surrogate), `depth`, `type` (anything else).

3.4 Every digest in this text is sha256 over the UTF-8 bytes of a canonical text.

## 4. REF

4.1 Split REF at its first `#` into head and fragment. A REF with `#` and an empty fragment is `receipt_form`.

4.2 **`rN`**: the head is `r` followed by a positive decimal integer without a leading zero (`r1`, `r12`; not `r0`, `r01`, `R1`).

4.3 **Fragment**, optional. If it starts with `/` it is a JSON Pointer (RFC 6901) in which every `~` is followed by `0` or `1`; tokens are split on `/` and `~1` then `~0` are unescaped. Otherwise it must be `L<a>` or `L<a>-L<b>`, positive decimal integers without leading zeros, b >= a: a 1-based inclusive line range. Anything else is `receipt_form`.

4.4 **Recognised, never resolved in v0.** `path:` followed by a target, and `prereg:` followed by exactly 64 hex digits (either case), each with the same optional fragment, pass the grammar and resolve as UNRESOLVED `no_repository` (7.1 still applies its earlier steps). A `path:` target is non-empty, does not start with `/` or `:`, has no empty, `.` or `..` segment between `/`, and contains none of: `\`, `*`, `?`, `[`, `]`, U+0000..U+0020, U+007F, U+0085, U+00A0, U+1680, U+2000..U+200A, U+2028, U+2029, U+202F, U+205F, U+3000. A target that breaks this is `receipt_form`.

4.5 Any other REF is `receipt_form`.

## 5. The manifest

5.1 A manifest is a JSON object. Its **core** has exactly seven keys:

    spec             "sworn/manifest/0.2"
    harness          string: who minted it
    turn             string: the unit of work it covers
    minted_at        string: the harness's clock, a statement
    authored_sha256  array of sha256 strings, sorted, lowercase: every byte object the author produced in the turn, as the harness saw it
    receipts         object: id -> entry
    rung             "L1", "L2" or null

An entry is `{id, sha256, kind_of_source, captured_at, complete}` plus optional `bytes` (standard base64 with padding, RFC 4648 section 4) and optional `harness_note` (string). `complete` is a boolean: true says the bytes are the whole of the thing captured.

5.2 `kind_of_source` is from a closed vocabulary. External: `tool_stdout`, `tool_stderr`, `file_read`, `http_fetch`, `harness_note`, `test_report`, `attestation`. Author-side: `agent_output`, `agent_file_write`, `agent_message`. `attestation` names the shape of the bytes (an envelope a harness recorded); no signature is verified.

5.3 `rung` is what the harness declares about itself and nothing checks it. `L1`: a local hook that shares a file system and a shell with the author. `L2`: a runner that minted after the turn ended and that the author could not write to. null resolves as `undeclared`.

5.4 **Manifest digest** = sha256 of the canonical text of the core. A manifest file may carry it as an eighth key, `digest`. The core used for the digest is rebuilt by the checker: the seven keys only (any other top-level key is ignored and is not covered), `authored_sha256` rebuilt as below, `receipts` exactly as given. The verdict core's `manifest_digest` is this recomputed digest, or null when no manifest was given or its core has no canonical form.

`authored_sha256` is rebuilt in three steps that every checker can share: each entry must be 64 hex digits in either case (otherwise the manifest is refused, `manifest_authored_shape`); `A`-`F` become `a`-`f` and nothing else changes; the entries are sorted by code point, duplicates kept. Example: `["BB…B", "aa…a", "aa…a"]` (64 digits each) is rebuilt as `["aa…a", "aa…a", "bb…b"]`, and its digest differs from that of `["aa…a", "bb…b"]`.

5.5 **Reading a manifest.** A manifest is handed to a checker as bytes and parsed as in 6.13, with one difference: a key written twice in any object of the manifest refuses the whole manifest (`manifest_form`). Example: `{"receipts": {"r1": {…}, "r1": {…}}, …}` is refused; it is never read first-wins or last-wins. A checker then **refuses** (section 11), in this order: a manifest that is not an object, or whose `spec` is not `sworn/manifest/0.2` (`manifest_spec`); whose `receipts` is not an object (`manifest_receipts_shape`); whose `authored_sha256` is not an array of 64-hex-digit strings (`manifest_authored_shape`); whose `harness`, `turn` or `minted_at` is missing or not a string, or that holds a duplicated key (`manifest_form`). A missing or null `receipts` is read as `{}`, a missing or null `authored_sha256` as `[]`, a missing `rung` as null.

5.6 A manifest is **intact** when its core has a canonical form and its `digest`, if present and not null, equals the recomputed digest. A `digest` key whose value is null is treated as absent. Example: a manifest with `"digest": null` and an otherwise sound core is intact, and its `manifest_digest` in the verdict core is the recomputed digest. A manifest whose core has no canonical form (an unpaired surrogate in any string, a `NaN`, a non-integer number, an integer outside 3.1) is not intact whether or not it carries a digest.

## 6. Resolving an `rN` REF

Take the first case that applies. "UNRESOLVED x" and "MALFORMED x" end the span with that verdict and reason.

6.1 No manifest was given: UNRESOLVED `manifest_absent`.
6.2 The manifest is not intact: UNRESOLVED `manifest_integrity`.
6.3 `rung` is neither null, `L1` nor `L2`: UNRESOLVED `rung_unknown`. (`L3`, signed and verified, is reserved; nothing in v0 verifies a signature, so nothing may declare it.)
6.4 The id is not a key of `receipts`: UNRESOLVED `manifest_id_missing`.
6.5 The entry is not an object, or its `sha256` is not 64 lowercase hex digits: UNRESOLVED `manifest_integrity`. The entry's own `id` is not read: the key in `receipts` names the entry. Example: `"receipts": {"r1": {"id": "r2", …}}` cited as `r1` resolves as `r1`.
6.6 `kind_of_source` is not a string: MALFORMED `kind_of_source_unknown`.
6.7 `kind_of_source` is author-side, or the entry's `sha256` is in `authored_sha256`: MALFORMED `receipt_author_minted`.
6.8 `kind_of_source` is not in the external vocabulary: MALFORMED `kind_of_source_unknown`.
6.9 From here the span has `provenance` = `{"form": "rn", "harness": <manifest harness>, "rung": "L1" | "L2" | "undeclared", "kind_of_source": <the entry's>}` and `resolved_sha256` = the entry's `sha256`.
6.10 If the entry has a `bytes` key, its value must be a string that decodes as strict base64 and hashes to the entry's `sha256`; any other value, null included, is UNRESOLVED `manifest_integrity`. Only an entry without the key has no bytes: then, if KIND is not `hash`, UNRESOLVED `manifest_bytes_absent`. **Strict base64** is the RFC 4648 section 4 alphabet (`A-Z a-z 0-9 + /`), a length that is a multiple of 4, `=` only as one or two final padding characters, and nothing else: no whitespace, no line breaks, no URL-safe alphabet, nothing after the padding. Pad bits that are not zero are ignored. Examples: `QQ==` and `QR==` both decode to `A`; `QQ`, `QQ=`, `Q Q==`, `QQ==QQ==` and `QQ==` followed by LF are `manifest_integrity`; the empty string decodes to zero bytes.
6.11 **Line fragment.** The receipt bytes are split on LF only (CR is kept). The number of lines is the count of LF bytes, plus one if the bytes are non-empty and do not end in LF. A range reaching past the last line is MALFORMED `anchor_out_of_range`. The slice runs from the first byte of line a to the end of line b, without b's terminating LF.
6.12 **Pointer fragment.** The receipt bytes must parse as JSON (6.13), else MALFORMED `receipt_not_json`. Walk the tokens from the root: in an object the token must be a key (else `pointer_unresolvable`) and must not be a key that occurs twice in that object (`pointer_ambiguous`); in an array it must be `0` or a decimal integer without a leading zero, below the length; a scalar cannot be walked. Failures are MALFORMED `pointer_unresolvable`. The value reached is the **leaf**.
6.13 **JSON, read strictly.** UTF-8 without a byte order mark; RFC 8259, plus the three bare tokens `NaN`, `Infinity`, `-Infinity`, which parse (so that one bad leaf does not hide a file) and are never numeric. Every number keeps its own digits and exponent exactly; it is never converted to binary floating point. A repeated key keeps its last value and is remembered as repeated.
6.14 Which rows carry `provenance` and `resolved_sha256`:

    with both      HELD; FAILED; MALFORMED absent_over_partial from 7.1j; MALFORMED receipt_not_json,
                   leaf_not_scalar, leaf_not_numeric from 7.3; MALFORMED leaf_not_string, short_needle from 7.5
    with neither   every UNRESOLVED; every MALFORMED from section 2 and 7.1b to 7.1h; MALFORMED from 6.6 to 6.8
                   (kind_of_source_unknown, receipt_author_minted), 6.11 (anchor_out_of_range) and
                   6.12 (receipt_not_json, pointer_unresolvable, pointer_ambiguous)

So `receipt_not_json` appears both ways: with provenance when 7.3 parses a whole receipt or a line slice, without when 6.12 cannot parse the receipt to walk a pointer.

## 7. The kinds

7.1 **Order of checks for one declaration.** First that applies ends it.

  a. It carries a mark from section 2: MALFORMED with that reason; `detail` is `{"raw": ...}` when it has a raw, else `{}`.
  b. The inner bytes are all ASCII whitespace (space, TAB, LF, VT, FF, CR): MALFORMED `empty_span`.
  c. The inner text has more than 300 code points: MALFORMED `length_cap`, detail `{code_points, bytes, cap: 300}`.
  d. KIND is `exec`: MALFORMED `kind_reserved`. KIND is not one of the four: MALFORMED `kind_unknown`. Detail `{kind}`.
  e. REF fails section 4: MALFORMED `receipt_form`, detail `{receipt: REF}`.
  f. REF has a fragment and KIND is `absent`: MALFORMED `absent_over_partial`; KIND is `hash`: MALFORMED `hash_over_partial`. Detail `{receipt: REF}`.
  g. The inner text contains U+202D or U+202E: MALFORMED `directional_override`, detail `{code_point: "U+202D"}` (the first one found). The isolates U+2066..U+2069 and the embeddings are legal.
  h. The kind's form check on the inner text alone (7.2, 7.4, 7.6): MALFORMED with the kind's reason. No receipt has been opened yet, so no MALFORMED up to here depends on evidence.
  i. Resolution (section 6, or 4.4).
  j. KIND is `absent`: the entry's `complete` is not a boolean: UNRESOLVED `manifest_no_completeness`; it is false: MALFORMED `absent_over_partial`, detail `{complete: false}`, carrying provenance and `resolved_sha256`.
  k. The kind's comparison: HELD, or FAILED with a reason; `detail` as the kind says.

7.2 **numeric, form.** Cut the inner text into maximal tokens of these characters: letters and numbers (Unicode General_Category L* and N*), `_ . , + - % / :`, U+2212, U+00B1, and the 26 dash code points U+00AD U+058A U+05BE U+1400 U+1806 U+2010 U+2011 U+2012 U+2013 U+2014 U+2015 U+2E17 U+2E1A U+2E3A U+2E3B U+2E40 U+2E5D U+301C U+3030 U+30A0 U+FE31 U+FE32 U+FE58 U+FE63 U+FF0D U+10EAD. A token is digit-bearing if it holds a decimal digit (General_Category Nd). Not exactly one digit-bearing token: `number_count`. Strip every trailing `.`, `,` and `:` from it; the rest must be, whole: an optional sign (`-`, `+` or U+2212), then either ASCII digits (plain, or grouped in threes with `,` after a leading group of one to three) with an optional `.` and ASCII digits, or `.` and ASCII digits; then an optional `%`. Otherwise `number_grammar`. Detail for both: `{digit_bearing_tokens: [...]}`, the tokens as found. So `r1`, `v0.1`, `1e-5`, `3/4`, `2026-09-01` and a non-ASCII digit are MALFORMED, never a number.

7.3 **numeric, comparison.** The printed value is the token without commas and without `%`, U+2212 read as `-`; f is its count of digits after the point. `42%` is 42: no conversion. The receipt value is the leaf (6.12), or else the line slice (6.11) or the whole receipt bytes parsed as JSON (not JSON: MALFORMED `receipt_not_json`). The slice must itself be a JSON value: `r1#L1` on a line `p95: 240` is MALFORMED `receipt_not_json`; there is no token search inside a line. An object, array or null: MALFORMED `leaf_not_scalar`, detail `{leaf_type: "object" | "array" | "null"}`. A string, a boolean, `NaN`, an infinity, or a number whose adjusted exponent is above 320: MALFORMED `leaf_not_numeric`, detail `{leaf}`.

**The receipt number.** From the JSON number take the **coefficient**: its integer and fraction digits joined, with leading zeros removed (if every digit is zero, one 0 remains). Its **exponent** is the written exponent (0 if none) minus the number of fraction digits. Its **adjusted exponent** is exponent + (number of coefficient digits) - 1. Trailing zeros stay in the coefficient. Examples: `0.4211` is coefficient 4211, exponent -4, adjusted -1; `1.50` is 150, -2, 0; `100e1` is 100, 1, 3; `0.001e322` is 1, 319, 319 (compared, so FAILED against `0`); `0.000e323` is 0, 320, 320 (compared, so HELD against `0`); `0.0001e324` is 1, 320, 320 (FAILED against `1`); `1e321` and `0e400` have adjusted exponents 321 and 400 and are `leaf_not_numeric`.

Both values are rounded to f fractional digits, ties to even, in exact decimal arithmetic, and written in plain notation with exactly f fractional digits and a zero result unsigned. Equal strings: HELD; else FAILED `value_mismatch`. Detail: `{printed_token, printed, receipt, receipt_rounded, fractional_digits: f, rounding: "ROUND_HALF_EVEN"}`, where:

- `printed_token` is the digit-bearing token after its trailing `.`, `,` and `:` are removed, otherwise as written (sign, grouping commas, `%` and U+2212 kept): `42.` gives `42`, `+42` gives `+42`, `42%` gives `42%`, `−3` gives `−3`.
- `printed` is the printed value written by the same rule as `receipt_rounded` (plain notation, exactly f fractional digits, unsigned zero): `.5` gives `0.5`, `+42` gives `42`, `1,234.5` gives `1234.5`, `−3` gives `-3`.
- `receipt` is the receipt number's coefficient and exponent in the scientific string form of the General Decimal Arithmetic specification: a `-` if the number is negative (`-0` included); then, when the exponent is at most 0 and the adjusted exponent at least -6, plain notation with the point placed so that exactly -exponent digits follow it (zeros added in front as needed); otherwise the first coefficient digit, a `.` and the remaining digits if there are any, `E`, `+` or `-`, and the absolute adjusted exponent. Examples: `0.4211`, `1.50`, `1E-7`, `1.00E+3` (from `100e1`), `1E+318` (from `0.0001e322`), `0E+320` (from `0.000e323`), `-0`.
- `leaf`, in `leaf_not_numeric`, is the first 80 code points of a text, with each unpaired surrogate then replaced by U+003F (`?`): for a string leaf, its own text (`"\ud800x"` gives `?x`); for a boolean, `True` or `False`; `NaN`, `Infinity`, `-Infinity`; for an oversized number, its `receipt` string as above (`1E+321`, `0E+400`). 13.2 records what is unsettled about it.

7.3a **The rounding floor (D2, draft).** If the strings are equal, the receipt value is not zero and its rounded string is zero, the span is FAILED `rounds_to_zero`, with the same detail. A receipt that is exactly zero holds against any printed zero.

7.4 **quote and absent, form.** The **needle** is the content of the one inline code span inside the inner bytes (2.3c's rule: a backtick run closed by an equal run on the same line), bytes verbatim, untrimmed. Not exactly one such span: `needle_count`. A needle of zero bytes or only ASCII whitespace: `needle_empty`.

7.5 **quote, comparison.** With a pointer leaf: the leaf must be a JSON string (else MALFORMED `leaf_not_string`, detail `{}`); a string leaf holding an unpaired surrogate (from a `\ud800` escape) has no UTF-8 form and is MALFORMED `leaf_not_string`, detail `{note: "leaf is not encodable as UTF-8"}`; otherwise the haystack is its UTF-8 bytes; the floor below does not apply. Otherwise the haystack is the line slice, or the whole receipt bytes, and **the short-needle floor** applies: a needle under 16 bytes is MALFORMED `short_needle`, detail `{needle_bytes, minimum_bytes: 16}`, unless the REF has a line fragment that **narrows**. It narrows when the slice differs from the full-content slice, or when the whole receipt is under 16 bytes. The full-content slice is lines 1 to k of the receipt, k being the number of lines after all trailing LF bytes are removed (with k = 0 nothing narrows except the under-16 clause). HELD if the needle occurs in the haystack, else FAILED `needle_missing`. Detail: `{needle_bytes, haystack_bytes, occurrences}`, occurrences counted without overlap from the left.

7.6 **hash, form.** Find every maximal run of hex digits in the inner text. A run next to `A-Z a-z 0-9 _` (on either side) is not a run: it is neither counted nor listed. Exactly one run must be 64 long and none may be 32, 40, 96 or 128 long; otherwise MALFORMED `digest_form`, detail `{hex_runs: [lengths of the runs, in order]}`. Examples: `sha256 <64 hex>` has the one run 64 (`a256` is glued to `h`), so it passes; `sha256 <64 hex> and <32 hex>` is `digest_form` with `hex_runs` `[64, 32]`; a 32-digit run glued to `g` does not count.

7.7 **hash, comparison.** The 64-digit run, lowercased, against the entry's `sha256` (which 6.10 has checked against the bytes when the entry carries them; when it carries none, this is the manifest's word). HELD, or FAILED `digest_mismatch`. Detail: `{stated, receipt_sha256}`.

7.8 **absent, comparison.** No fragment (7.1f), `complete` true (7.1j), bytes present (6.10). The haystack is the whole receipt bytes. FAILED `needle_present` if the needle occurs, else HELD. Detail: `{needle_bytes, haystack_bytes, complete: true}`. No short-needle floor: a short needle that is absent from a complete object is the stronger statement.

## 8. The span row

    at               byte offset of the declaration in the document
    start, end       byte offsets of the inner bytes in the UTF-8 canonical text, or null (2.5)
    receipt, kind    REF and KIND as written, or null (stray closer, malformed tag)
    claim_sha256     sha256 of the inner bytes exactly as written, or null when start is null   (D3, draft)
    claim            the inner text when it is at most 300 code points, else null; null when start is null   (D3, draft)
    verdict          HELD | FAILED | UNRESOLVED | MALFORMED
    reason           null for HELD; otherwise one code from section 10
    detail           an object, as sections 2 and 7 say; {} when nothing is said
    resolved_sha256  present from 6.9 on
    provenance       present from 6.9 on

All numbers in a row are integers; every decimal value is a string. A HELD row always has `resolved_sha256` and `provenance`.

## 9. The verdict core, the digest, the receipt

9.1 The core has exactly thirteen keys. Its `schema` is `receipt/v0-draft`.

    schema             "receipt/v0-draft"
    profile            "rn-inline/v0-draft" (9.5)
    format             "sworn/0.1" (the span grammar of section 2)
    document           {name, inline_sha256, canonical_sha256}
    commit             null, or 40 or 64 lowercase hex digits: a label the caller hands in; v0 resolves nothing at it
    manifest_digest    5.4
    spans              the rows, ordered by at
    counts             {HELD, FAILED, UNRESOLVED, MALFORMED}
    sworn_total        the number of rows
    unresolved         counts.UNRESOLVED, beside the verdict
    document_verdict   9.2
    document_malformed null, or the object of 2.1 or 2.2
    rungs              how many rows stood on each rung: an object holding only the keys that occur ({} when there are
                       no rows); a row with provenance counts under its provenance.rung (L1, L2 or undeclared); every
                       row without provenance counts under "unresolved", MALFORMED rows included (6.14). Example: one
                       HELD row on L2 and one tag_syntax row give {"L2": 1, "unresolved": 1}. A legacy projection
                       also counts rows resolved through a repository under "committed".

9.2 **Document verdict (D1, draft).** First that applies: `document_malformed` is not null: `SWORN-FAILED`. No rows: `UNSWORN`. Any FAILED or MALFORMED row: `SWORN-FAILED`. No HELD row: `SWORN-UNCHECKED`. Otherwise `SWORN-HELD`. So held needs at least one HELD row, and a document whose every span is UNRESOLVED is not held.

9.3 **digest** = sha256 of the canonical text of the core. It covers nothing else. In particular it does not cover which build of which checker ran.

9.4 **The verdict receipt** is `{"core": ..., "digest": ..., "attestations": [], "verifier": {...}}`. `verifier` is optional provenance about the checker that issued it (a name, a version, a hash of its source); a checker reading a receipt never lets it change a verdict. `attestations` is the one named slot for "who else attests these bytes". In v0 it is an empty array and a receipt with anything in it is refused: v0 specifies no signature.

9.5 **The profile id** replaces a sentence. `rn-inline/v0-draft` stands for: the spans that read HELD or FAILED were compared, each by its kind, against the bytes a manifest entry names (or, for a `hash` span on an entry without bytes, against the sha256 the manifest records); the manifest is the one whose digest the core records; the only authorship test is that the entry's `kind_of_source` is external and its sha256 is not in `authored_sha256`; MALFORMED spans were refused before the kind's comparison ran (some only after their receipt bytes were opened, section 10) and UNRESOLVED spans were not compared. `legacy-projection/v0-draft` marks a core projected from a receipt issued before v0 that a v0 checker cannot re-derive: a row was resolved through a repository, or a row has offsets and no claim. No v0 checker emits it.

9.6 Left out of v0 on purpose: any coverage or share-of-sentences number; the sidecar form of a document; manifests at `sworn/manifest/0.1`; receipts at the older schemas; rung `L3`; a timestamp.

## 10. The closed lists

Span verdicts: `HELD`, `FAILED`, `UNRESOLVED`, `MALFORMED`.

Document verdicts: `SWORN-HELD`, `SWORN-FAILED`, `SWORN-UNCHECKED`, `UNSWORN`.

Document-level reasons: `invalid_utf8`, `unbalanced_fences`.

MALFORMED, from the document's bytes alone: `tag_syntax`, `nesting`, `stray_closer`, `unclosed`, `empty_span`, `hidden_commitment`, `length_cap`, `kind_unknown`, `kind_reserved`, `receipt_form`, `absent_over_partial`, `hash_over_partial`, `directional_override`, `number_count`, `number_grammar`, `needle_count`, `needle_empty`, `digest_form`.

MALFORMED, from the declaration and the object it names: `kind_of_source_unknown`, `receipt_author_minted`, `anchor_out_of_range`, `receipt_not_json`, `pointer_unresolvable`, `pointer_ambiguous`, `leaf_not_scalar`, `leaf_not_numeric`, `leaf_not_string`, `short_needle`. (`absent_over_partial` is also reached here, by 7.1j.)

UNRESOLVED, the checker could not see the evidence; never an accusation: `manifest_absent`, `manifest_integrity`, `rung_unknown`, `manifest_id_missing`, `manifest_bytes_absent`, `manifest_no_completeness`, `no_repository`.

FAILED, the comparison ran and did not pass: `value_mismatch`, `rounds_to_zero`, `needle_missing`, `needle_present`, `digest_mismatch`.

Outside the profile, found only in legacy projections: `no_commit`, `commit_absent`, `path_absent`, `not_a_blob`, `receipt_too_large`, `git_unavailable`, `prereg_not_in_tree`.

Refusals (section 11): `nothing_to_verify`, `name_form`, `commit_form`, `manifest_spec`, `manifest_receipts_shape`, `manifest_authored_shape`, `manifest_form`; from the canonical form `float`, `int_range`, `non_ascii_key`, `lone_surrogate`, `depth`, `type`; when checking a receipt `receipt_shape`, `receipt_digest`, `attestations_not_empty`, `profile_not_checkable`, `document_mismatch`.

## 11. What a checker must refuse

A refusal is not a verdict: nothing is emitted and the caller gets a code.

11.1 Before checking a document, in order: no document at all (`nothing_to_verify`; an empty byte string is a document, 2.1); a document name that is not a string of Unicode scalar values, such as a name holding an unpaired surrogate the way a POSIX file name decoded with surrogateescape does (`name_form`); a commit label that is neither null nor 40 or 64 lowercase hex digits (`commit_form`); a manifest 5.5 rejects (`manifest_spec` for one that is not an object or not at `sworn/manifest/0.2`, then `manifest_receipts_shape`, `manifest_authored_shape`, `manifest_form`, in that order).

11.2 A core that has no canonical form is never emitted; the code is 3.3's. Given 11.1 and 5.6, none can arise: the name is refused before checking, and a manifest without a canonical form is not intact, so none of its strings reaches a row. The rule stays as a guard.

11.3 **Checking a receipt** against a document and a manifest, in order: the receipt's bytes do not parse under 6.13, or any of its objects has a key written twice, or it does not match the schema (`receipt_shape`; a receipt carrying `document_verdict` twice is refused, never read first-wins or last-wins); `attestations` is not empty (`attestations_not_empty`); `digest` is not the digest of `core` (`receipt_digest`); the profile is not `rn-inline/v0-draft` (`profile_not_checkable`); the document's sha256 is not `core.document.inline_sha256` (`document_mismatch`). Then the checker runs sections 2 to 9 itself, with the name and commit label the core records, and answers `REDERIVES` when the digest of its own core equals the receipt's digest and `DOES-NOT-REDERIVE` otherwise. `REDERIVES` says the receipt is what this checker computes. It does not say the document held; the document verdict is a different word in a different place.

11.4 A checker never: reads a clock, opens a network connection, runs a program, reads a file other than its inputs, consults a model, proposes spans, or repairs an input.

## 12. What a HELD verdict does not show

1. **That the sentence is true.** HELD says one comparison passed: a printed number equals a named number at the printed precision; a quoted string occurs in named bytes; a stated hash equals the sha256 the manifest records, and, when the entry carries bytes, the hash of those bytes; a string does not occur in bytes a harness called complete. The words around the number are not read. "Latency fell to `r1`'s 240" holds if `r1` says 240, whether or not anything fell.
2. **That the bytes are not the author's.** The only authorship test reads the manifest (6.7). Bytes the author wrote that the harness never digested, such as a file written in another turn or by a tool the harness does not watch, are not caught.
3. **That the manifest is honest.** Nothing is signed. The rung is declared, not verified. A manifest minted by the author's own code passes every rule here.
4. **That the right sentences were bound.** The author chooses what to mark. A document can be `SWORN-HELD` on one trivial span beside any number of unbound claims. `sworn_total` and `unresolved` are counts of what was marked, not of what mattered.
5. **That the receipt bytes mean what the sentence suggests.** A number at a JSON pointer is a number at a JSON pointer. Whether that leaf is the quantity the sentence talks about is not checked.
6. **That the span was written before the evidence was seen**, or at any particular time.
7. **That anything was checked, unless a row reads HELD or FAILED.** MALFORMED and UNRESOLVED rows were not compared. `SWORN-UNCHECKED` exists so that this case has its own word.
8. **That another checker agrees.** As of this draft no checker implements this text. Its rules were read from one implementation. A second, in JavaScript, was ported by the lab with the first open; it covers inline documents with a manifest, its committed agreement report is on an older vector set, and it implements none of D1 to D3.
9. **That any evidence bytes were read.** A `hash` span on an entry without `bytes` compares the sentence with the manifest's own string; nothing else is read. A document whose every HELD row is such a span reads `SWORN-HELD` with no evidence byte opened. D1 does not change this.

## 13. Open before freeze

Written as the code behaves today; each is an accident or an unpinned choice that should be decided, not inherited.

1. **Unicode version.** 7.2's letter, number and digit classes depend on a Unicode version this text does not pin. The lab's two implementations state the class two ways (a regex word class in one, L and N categories in the other). The Python half was measured once on Python 3.12.10 with Unicode 15.0.0: its word class equals L, N and `_`, and its digit class equals Nd, with no code point in the difference either way. The JavaScript half is unmeasured, and so is any other Unicode version.
2. **`leaf` text.** 7.3 states the rule as the code behaves. Unsettled: a string spelling `True` and the boolean are indistinguishable (proposed: add a `leaf_type` and print booleans as JSON does), and `leaf_not_string` can carry `{note: "leaf is not encodable as UTF-8"}`, an English sentence inside the digest.
3. **`receipt` text.** 7.3 keeps the scientific string form because committed rows carry it and it cannot be turned back into the source token. The source token itself may be the better field.
4. **A number whose exponent is too large to hold** (`1e99999999999999999999`): today's code does not answer; it stops with an error. This text classes it under `leaf_not_numeric` by 7.3's bound. A vector must pin it.
5. **JSON nesting depth** in receipt bytes: today's limit is the host language's recursion limit, answered as `receipt_not_json`. No number is stated here.
6. **`document.name` is inside the digest.** The same bytes under two names give two digests. So is the `commit` label, which v0 never uses.
7. **`rounding` and `cap`, `minimum_bytes`** repeat constants of this text inside every row.
8. **Manifest tolerance.** Top-level keys beyond the seven are ignored and not covered by the manifest digest (5.4); unknown keys inside an entry are covered and ignored. An entry that is malformed but never cited is never looked at.
9. **Claims in a document with no canonical text** are not bound (rows have null offsets), even for a span that reads HELD there. One stray `</sworn>` anywhere in a document is enough: it removes the canonical text, so no span in that document has its claim bound. Such a document is `SWORN-FAILED` in any case, but its receipt no longer says what was written. Binding there needs rows that carry offsets into the document itself, or a projector that lexes; neither is in this draft.
10. **Disclosure.** D3 puts up to 300 code points of the document per span into the receipt. There is no hash-only option.
11. **One HELD span is enough** for `SWORN-HELD` whatever the UNRESOLVED count (D1 chose the narrower rule).
12. **`UNSWORN` against `SWORN-FAILED`.** A document whose only tag-shaped thing is a malformed tag is `SWORN-FAILED` with `sworn_total` 1.
13. **`path:` and `prereg:`** are parsed and never resolved (4.4), so that a v0 checker answers as today's does when it is handed no repository. A later profile resolves them.
14. **Stricter than today's code, by this text's choice:**
    - manifests at 0.1 are refused (one in-scope conformance vector uses one);
    - a manifest needs string `harness`, `turn`, `minted_at` (today a missing `minted_at` is filled from the clock, and a non-string `harness` is carried into provenance);
    - a manifest that holds a non-integer number, a key above U+007E or an integer outside 3.1 is not intact (`manifest_integrity`). Today it is digested with the lab's own canonical form, which admits finite doubles and larger integers, and it is read;
    - a manifest that holds an unpaired surrogate in any string, a `NaN` or an infinity, and carries no `digest` key, is not intact. Today it is read, its rows can be HELD, and the verdict core records `manifest_digest` null, because no digest is computed when none is declared. The projector refuses such a core (`manifest_not_canonical`) when one of its `rN` rows got past 6.1; it cannot see the other cases in this item;
    - a manifest with a duplicated key is refused (`manifest_form`). Today Python's reader keeps the last value;
    - an `authored_sha256` entry that is not 64 hex digits is refused (`manifest_authored_shape`). Today any string is accepted and lowercased by Python's full Unicode case mapping, so U+0130 becomes `i` followed by U+0307;
    - `WITHHELD` is not a verdict.

    The projector `to_v0.py` sees none of these from a verdict core except where the item says so, so a projected core and a v0 checker's core can differ on such inputs. Over the 1,718 in-scope conformance documents, the 0.1 manifest is the only input any rule in this item moves.
15. **Vectors.** No vector file exists for this text. The receipt check of 11.3 and the refusals `name_form`, `manifest_form`, `receipt_shape`, `receipt_digest`, `attestations_not_empty` and `profile_not_checkable` have no implementation in a checker; the projector checks the name the same way.
16. **Pad bits in base64.** 6.10 pins today's reading: pad bits that are not zero are ignored, so `QR==` reads as `A`. RFC 4648 section 3.5 lets a decoder reject such input, and some strict decoders do so by default; a checker built on one must accept it to agree. The other reading (`QR==` is `manifest_integrity`) would make the encoding of the bytes unique, at the cost of one more rule every emitter must follow. No in-scope conformance manifest has non-zero pad bits, so neither reading moves one.

## 14. What this text is not

It is a lab draft of a format, not a standard and not a specification anyone else has adopted. No second implementation of it is independent of the lab. No outside party emits or checks these objects. Nothing here is claimed to be new: binding statements to digests of evidence is older than this lab. Neighbours the lab has credited, with dates, in `papers/sworn/NOTE_prior_art_credit_2026_09_29.md`: Inline XBRL 1.1 (2013), where the displayed number in a filing is the tagged fact; Trusty URIs (2014); showyourwork! (2021); DeerFlow's zero-citation report verdict (2026); metacheck (2026); VeriFin (2026). The tool is a checker of bindings an author hands over; it does not find claims and it is not a detector of false statements in prose.
