# AMENDMENT — a diffgate record may name the instrument that wrote it, and capsule v0.2 layer 2 compares that identity apart from the content

Fathom Lab · 2026-10-11 · Amends `SPEC_oath_capsule_v02_2026_08_31.md`, in this directory, committed at
`f6d469c4` and not edited. Written by the lead, on the operator's decisions of 2026-10-11 (O1, O3 and O9 of
the design approved that day for item 3 of step zero of the 2026-10-06 study: every diffgate record names its
build and runtime). This is phase 1 of that design. Committed alone, ahead of the code that implements it
(`styxx/instrument.py`, new; the reader in `styxx/capsule.py`). The record-writing side, `styxx/diffgate.py`,
is phase 2 and is not in the change this file lands with: until it lands, every record still has eleven keys
and names nothing, and the reader described here answers that case as item 3 says.

## What the spec says today

- The embedded gate record is "a total pure function of (summary bytes, diff bytes)" (lines 21-24). Its
  only identity field is the constant `"diffgate": "v0"`.
- Layer 2 (lines 105-115) re-runs `gate_diff_text(summary, diff, run=None, strict=False)` at the installed
  instrument and compares the full record, with two named softenings: a `diffgate` version-key mismatch is
  INSTRUMENT SKEW (`ok: false`, with the pip pin to reproduce under), and `unparsed_claims` is advisory in
  both directions.
- R2 refuses a supplied `--gate` whose `diffgate` key is not `"v0"`; R5 refuses one that diverges from the
  live re-gate on any field but `base` and `head`.
- Layer 1 verifies `sha256(JCS(gate))` against `binding.gate` and re-runs nothing.
- The payload's `verifier` block (`styxx_version`, `pip`) is outside every binding.

## What was measured that the spec did not foresee

- **One version string has named two different readers.**
  `NOTE_handoff_capsule_reproduces_under_its_minting_build_2026_10_11.md` (phase 0): the HANDOFF capsule's
  record reproduces on every key under the `styxx/diffgate.py` that wrote it and under the `v7.47.0` tag's
  file, and stopped reproducing at `4c90d9cf`, where one reason string changed on a claim whose verdict did
  not move. Its pin, `styxx==7.47.0`, is satisfied by the tag's tree and was also the stamp of the log writer
  whose bytes do not reproduce it. From the record's own bytes the verifier can say only `gate.claims not
  reproduced`, which reads as an edit. The record names no build, so the verifier cannot tell an instrument
  change from an edit, and nothing written in afterwards could name the build honestly.
- **The runtime is part of the instrument.** Re-measured for this amendment on 2026-10-11, with
  `styxx/{diffgate,claimdetect,evidence,declare,_version}.py` at `f469f25b` staged under an empty
  `__init__.py` (this box's CPython 3.14 has no numpy, so `import styxx` does not run there): the summary
  `U+11BF3 files changed.` (a Sunuwar digit three, Unicode 16.0) over a one-file diff reads PASS with no claim
  under CPython 3.12.10 (Unicode 15.0.0) and FAIL under CPython 3.14.2 (Unicode 16.0.0), with
  `files_changed_count` CONTRADICTED, `diff changes 1 files, claim says 3`. The same bytes, the same module
  bytes, two verdicts. A verifier on the other interpreter reads the honest capsule as a sealed record that
  does not reproduce. The committed fixture that makes a verifier witness this is phase 2b of the design.

## The amendment

1. **The record may carry two identity keys, `build` and `runtime`, after its eleven content keys.**
   `"diffgate": "v0"` remains the format key of the content, and R2 is unchanged: it still accepts `"v0"`
   and nothing else. A content key is any key that is neither the format key nor an identity key.

   `build` names the bytes that wrote the record: `schema` (`styxx.diffgate/instrument/v1`), `reader`
   (`styxx.diffgate`), `styxx` (the version string of `styxx/_version.py`, or null), and `modules`, an
   ordered list of `[name, sha256]` pairs over `styxx.diffgate`, `styxx.instrument`, `styxx.declare`,
   `styxx.evidence` and `styxx.claimdetect`, each sha256 over the module file's bytes with `\r\n` read as
   `\n` (a lone `\r` is kept), the rule `web/gate/differential/py_side.py` and charon's `_content_sha256`
   already follow. The reader hashes its own file, never a module looked up by name, so a reader built from a
   string names a null sha and not another file's bytes.

   `runtime` names what ran it: `implementation` (`sys.implementation.name`), `version`
   (`platform.python_version()`), `unicode` (`unicodedata.unidata_version`; one Unicode database feeds
   `str`, `re` and `int()` in CPython) and `case_floor`, a label from a fixed table of case pairs (13.0:
   U+A7C7; 14.0: U+2C2F, U+10570; 16.0: U+1C89, U+10D50, U+A7CB; the highest version whose pairs and every
   lower version's pairs hold, `<13.0` otherwise), shared with the browser panel, which cannot name a Unicode
   version and can run the same pairs.

   What is compared: `build.reader`, `build.modules` by name, `runtime.implementation`, `runtime.version`,
   `runtime.unicode`. What is recorded and never compared: `build.schema` (checked for readability),
   `build.styxx` (one version string has named two readers), `runtime.case_floor` (a label). Every value is
   a string, a list of string pairs, or null; every key is ASCII; so JCS canonicalises the record unchanged.
   Anything the runtime cannot know is null, never a guess, and a null equals nothing: no "same" is claimed
   from an unnamed field, and no skew is claimed from one.

2. **Layer 2 compares every content key, and the identity apart from them.** This is a third named
   softening beside the two at spec lines 109-115. The per-key reproduction is unchanged in what it compares
   and in what it prints: a content key that does not reproduce is `gate.<key> not reproduced`, as today, and
   `unparsed_claims` stays advisory. The identity is then read from the embedded record and compared with the
   identity of the instrument that re-ran it, field by field. A record whose identity key is not an object,
   names an unknown schema, holds a compared value that is neither a string of at most 128 characters nor
   null, or lists modules otherwise than as at most 16 `[name, sha]` pairs, is read as unnamed, with an
   advisory that quotes the unreadable value cut short. Hostile bytes get a reading, never a traceback
   (#196's rule).

3. **INSTRUMENT SKEW is a reading of the identity, not of the format key alone.** A record that does not
   reproduce and names an instrument that differs on a compared field is INSTRUMENT SKEW: a problem that
   leads the problem list (`ok: false`, as the spec's "the verifier never lies" requires), names the
   differing fields and the recorded identity, and says how to separate the two readings: reproduce under
   the instrument the record names. The content lines follow it unchanged. A record that names no
   instrument, or not all of one, and does not reproduce cannot be classified from its own bytes, and the
   verifier says so in an advisory beside the unchanged problem lines: the failure cannot be told apart from
   instrument skew or an edit from the record alone, and the payload's pin is outside every binding and
   names a version, not a build. Every other identity outcome is an advisory: a record that reproduces under
   another instrument (its content did not depend on the difference for these bytes), a record that
   reproduces and names nothing, a record that names this verifier's own build and runtime and does not
   reproduce (what moved is not the instrument). The format-key mismatch of spec line 110 stays INSTRUMENT
   SKEW with no field comparison. Problem texts carry the recorded identity and the names of the differing
   fields, never a value of the live identity, which goes in the result and in advisories.

4. **R5 never compares identity keys.** A supplied `--gate` written by another build or runtime is accepted
   when its content reproduces from these bytes; the embedded record is always the live re-run, as today.
   When the content diverges and the supplied record names an instrument that differs, the refusal says so
   and names the fields.

5. **Layer 1 displays the recorded identity and proves only that it is the one sealed.** The page's meta
   line shows the recorded reader, its sha prefix, the interpreter and the Unicode database when the record
   carries them as strings; `binding.gate` covers them as it covers every key of the record. The page
   re-runs nothing and compares no identity.

6. **The pure-function sentence is restated.** The content of the record is a pure function of (summary
   bytes, diff bytes) under the instrument the record names, and the runtime is part of the instrument. The
   Unicode-16 measurement above is why the sentence needs the clause: the spec's sentence held for one
   interpreter at a time.

7. **`binding.gate` depends on the runtime for records that carry an identity.** Two honest mints of one
   summary and one diff on two runtimes no longer share a gate binding. Anyone who deduplicates capsules by
   `binding.gate` deduplicates by instrument as well; deduplicate by content (the summary and diff bindings
   and the content keys) instead.

8. **Disclosure.** A capsule minted after phase 2 carries, inside the sealed record: the interpreter's
   implementation name and version, the Unicode database version, a case-table floor label, the styxx version
   string, and five module names with the sha256 of each module's bytes. It carries no `sys.version` (the
   operator's decision O9: that string names the compiler, the build date and the OS family, and two builds
   of one release differ in it), no path, no host name, no user name, no locale and no operating system, in
   sealed records and in charon lines alike. A module sha names a file; it does not prove that file is the
   code that ran (a stale `.pyc`, a module patched at run time, a frozen application), and the identity says
   "names", never "proves".

## What does not change

- The spec text. R2. The format key. The content-key texts, so every committed charon line that quotes a
  problem re-derives as it is.
- `DOGFOOD_session_2026_08_31.capsule.html` verifies with no problem; `HANDOFF_capsule_v02_2026_08_31.capsule.html`
  fails with its one problem, `gate.claims not reproduced`, as today. Each gains one advisory: its record
  names no build or runtime. HANDOFF stays an advisory (O3); its classification is the phase-0 NOTE's.
- `papers/charon/charon.log.jsonl` lines seq 89 and 95 re-derive with no compared key moved: charon reads
  `problems` and `ok`, never `advisory`. The stranger's ferry-log step passes. No line is rewritten or
  appended.
- Sworn receipts and the conformance vectors: `styxx.sworn` imports nothing from the diff gate.
- Records minted before a record named its instrument will never name one. The build that wrote them is not
  in them, and writing one in afterwards would fabricate it.

## What this amendment does not say

- That a SKEW reading proves an instrument change. The binding is unkeyed; a re-sealer can claim another
  instrument. `ok` stays false, and the line names the test that separates the readings.
- That a reproduced record is right. Reproduction is a function of the bytes under an instrument, and a
  reader's defect reproduces as faithfully as its reading.
- That HANDOFF reads SKEW. It does not, and will not from its own bytes.
- That the browser's Unicode version is known. `case_floor` bounds the case tables and nothing else.

## ERRATUM — 2026-10-12: what the INSTRUMENT SKEW problem quotes, as the reader applies it

The text above is not edited. Two review rounds of phase 1 changed the reader after this file was
committed, and item 3 now differs from the code in two places: the SKEW problem withholds a recorded name
where item 3 says it names it, and a record that names part of its instrument and differs on a part it
names reads SKEW. Where item 3 and this section differ, this section states what `styxx/capsule.py` does
(`_identity_outcome` and the `_charon_quotable` helpers); `tests/test_instrument_identity.py` holds the
withholding, the unknown list and the four hint forms.

**Why a name is withheld.** `styxx.charon` reads a capsule problem holding `ambiguous`, `bytes !=`, or
`!=` beside `binding` as a broken binding, and derives the line UNRESOLVED. The SKEW problem quotes values
the minter chose, so without the rule a minter who names a module `styxx.ambiguous` turns a capsule that
does not reproduce into an unresolved line in the ferry log.

**The quoting rule.** It applies to the INSTRUMENT SKEW problem of item 3, the only problem text the
identity comparison writes.

1. The tokens of the record's choosing that the line can carry are four: the recorded identity, written as
   one line, `<reader> <sha10> (styxx <styxx>) on <implementation> <version>, Unicode <unicode>`; a module
   name inside a `build.modules[<name>]` field of the differing or unknown list; the recorded
   `runtime.implementation` and `runtime.version` in the hint; and the sealed `build.styxx` as the hint's
   pin. Before the test, every recorded value in the identity line and the hint is cut to 80 characters,
   with each character outside `A-Z`, `a-z`, `0-9`, `.`, `_`, `+` and `-` written as `?`, and a null
   written as `unknown`; a module name is already bounded to `^[a-z_.]{1,64}$`, matched whole.
2. A token is withheld when it holds `ambiguous` in any letter case, or `!=`. This test is wider than
   charon's own rule. After step 1 no token can hold `!` or `=`, so the word is what it refuses.
3. A withheld identity is written `an instrument whose recorded name this verifier does not quote (charon
   reads the name as a broken seal)`. A withheld module name, implementation or version is written
   `(a name this verifier does not quote)`, so such a field reads
   `build.modules[(a name this verifier does not quote)]`. A withheld `build.styxx` is not written at all,
   and the hint takes its next form below.
4. Neither replacement holds the word or `!=`, so the SKEW line never makes charon derive a capsule
   UNRESOLVED; charon reads the capsule as not reproduced.
5. Withholding is in the problem text only. The result's `gate_identity` keeps `recorded`, `differs` and
   `unknown` as read, and the field lists in advisories are not withheld (charon reads no advisory). No
   value of the live identity is in any problem, as item 3 says.
6. The rule does not cover the lines item 3 keeps unchanged. The content lines (`gate.<key> not
   reproduced: embedded ... vs live ...`) quote the content as before. The format-key SKEW line of spec
   line 110 quotes the payload's unsealed `verifier.pip` as before, so a pin holding the word there still
   makes charon derive UNRESOLVED, as it did before this phase. The R5 refusal at mint names the differing
   fields as read; it is a refusal message, never a capsule problem.

**A record that names part of its instrument.** Item 3 says that a record naming "no instrument, or not
all of one" that does not reproduce "cannot be classified from its own bytes". The reader reads `differs`
ahead of `incomplete` (`styxx.instrument.delta`): a record that names part of its instrument and differs
on a field it names (its runtime and no build, for one) is INSTRUMENT SKEW, and the advisory is for a
partial identity that differs on no field it names. The SKEW line lists every compared field null on
either side after the differing ones, `(differs: <fields>; unknown: <fields>)`, the second list only when
it is not empty, and its hint, `reproduce under that instrument (<hint>)`, points only at what the sealed
record names, in this order:

- `` `pip install styxx==<build.styxx>` on <implementation> <version> ``, when `build.styxx` is a
  non-empty string the rule lets through, with an advisory that a release installing another
  `styxx.diffgate` sha than the record names is not the minter's build;
- `the styxx whose styxx.diffgate the record names, on <implementation> <version>`, when the record names
  a `styxx.diffgate` sha;
- `a build the record names only by a version this verifier does not quote, on <implementation>
  <version>`, when its `build.styxx` was withheld and it names no `styxx.diffgate` sha;
- `the record names no build, only the runtime <implementation> <version>`, otherwise.

The payload's unsealed pin is never the SKEW line's hint.
