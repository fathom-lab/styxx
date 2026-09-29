# web/gate — the diff gate where Python cannot run

The instrument is `styxx/diffgate.py`. Two browser surfaces cannot import it: the paste-in
preview page and the bookmarklet. They run `diffgate.js`, a JavaScript transliteration of one
specific file — `styxx/diffgate.py` as it stands on `main` (BC-2 + COMPAT-1 + BIN-2 + COMPAT-2 +
PATH-1 + DECLARE-1, pull requests #113, #115, #120, #124, #127 and #129, plus the `fetch_pr` door),
re-cut for the PATH-2 repairs
(`papers/closed-model-frontier/PREREG_path2_resolution_2026_09_17.md`: a path claim resolves exact,
then suffix, then basename, #97; a dotfile keeps its dots in the path key, #121; a `def` the same
file's removed lines also define is changed, not added, #101; as amended by
`AMENDMENT_path2_resolution_2026_09_17.md`: definitions pair one to one per name, `only_touches`
lists only the paths outside by more than a dot, COMPAT's scaffold reading keeps `.storybook/` as
scaffolding; and by `NOTE_path2_third_pass_2026_09_25.md`, `NOTE_path2_fourth_pass_2026_09_25.md`,
`NOTE_path2_fifth_pass_2026_09_25.md`, `NOTE_path2_sixth_pass_2026_09_25.md`,
`NOTE_path2_seventh_pass_2026_09_25.md`, `NOTE_path2_eighth_pass_2026_09_27.md`,
`NOTE_path2_ninth_pass_2026_09_27.md`, `NOTE_path2_tenth_pass_2026_09_28.md`,
`NOTE_path2_eleventh_pass_2026_09_28.md`, `NOTE_path2_twelfth_pass_2026_09_29.md` and
`NOTE_path2_thirteenth_pass_2026_09_29.md`) on the file that carries
them, sha256 `c3eed72edfdf2ea8cf11667e42d47259964340b87d26b7fd56f5358a60012ac3` (LF line endings; a wheel
built on Windows carries CRLF and hashes differently, so `py_side.py` normalises before it
compares), reading names by the Unicode table `styxx/_xid.py` carries (15.0.0, table sha256
`8df68f21…`) and abstaining on a name that meets the skew set beside it (the code points the Pythons the
package supports, Unicode 13.0 to 16.0, read differently from the table; sha256 `0b7134fd…`; `diffgate.js`
carries the same bytes, `gen_xid.py` writes both, `xid_versions.json` holds the skew set's sources), comparing two
header paths' case by the fold `styxx/_fold.py` carries (Unicode 16.0.0, sha256 `a52cda82…`; `gen_fold.py` writes
it into both files), and making the file list unsure where a header path holds a code point outside the set of code
points Unicode 16.0.0 assigns, carried beside the fold (sha256 `56a413eb…`; the thirteenth pass, section D) — and this
directory is the
receipt for that port: the differential test that holds it to the Python's output, and the build that turns
it into the bookmarklet people drag into their bookmarks bar.

That gap is closed. Until #126 the port lacked COMPAT-2's sharpened compatibility reading (surface
vs scaffolding, signature changes, the candidate flag) and the differential counted 10 `compat_claim`
records as disagreements. The port carries that reading now, and the run below reads 0.

The port was first cut from the 7.47.0 wheel (`fb2d9b3e…`) and re-cut on 2026-09-16 for issue
#110: the 7.47.0 templates count Python `def` lines and accuse a TypeScript commit that says
"added 2 tests" of lying, and the bookmarklet is a public door, so it moved to the repaired file
ahead of the release rather than after it. What changed between the two files is listed in the
header of `diffgate.js` and measured below under *Drift*.

(The `/gate` page on the site is different: it runs the real released Python under CPython 3.13
via Pyodide, with the two module files hashed in the browser against the wheel. No port there.)

## What is here

`diffgate.js` — `gateDiffText(summary, diff, {strict})`, `parseUnifiedDiff(text)` and
`parseUnifiedDiffSides(text)`, the same closed template set (eleven templates, `compat_claim`
included), the same verdict strings, the same `why` text, the same `detail` — including the
`removed` list the compatibility reading attaches, and Python's `repr()` quoting where the
Python builds a reason with `!r`. Two deliberate gaps, both
stated in the file: the structural "unparsed claims" observer (`styxx.claimdetect`) is not
ported, and `--run` / `--evidence` do not exist, so "tests pass" is always UNCHECKABLE, exactly
as the CLI without `--run`.

`diffgate_ref.js` — `origin/main`'s `web/gate/diffgate.js` at `2a6ce0a3`, byte for byte (sha256 `06688702…`;
`.gitattributes` keeps its bytes, `tests/test_diffgate_guard.py` and `build_bookmarklet.py` refuse any other): the
guard's reference (`NOTE_path2_eleventh_pass_2026_09_28`). `gateDiffText` reads every claim with it too, and keeps a
verdict other than its verdict only where one of the three repairs, switched off alone, gives its verdict back and
that repair's own precondition holds on the claim; else the claim is UNCHECKABLE and names its verdict. The Python
carries the same reference, `styxx/_diffgate_ref.py`, `origin/main`'s `styxx/diffgate.py` (`9b620e00…`).

`bookmarklet_ui.js` — the panel: on a `github.com/OWNER/REPO/pull/N` page it reads the
description and the diff from `api.github.com` (two unauthenticated requests, nothing else,
nothing stored, nothing sent anywhere) and pins `[ok ]` / `[LIE]` / `[ ? ]` lines, the verdict
and the never-read count to the page.

`build_bookmarklet.py` — assembles the three (the reference in a function scope of its own, handed to the port) into
`bookmarklet_src.js`, minifies with
`terser -c -m --format ascii_only`, writes `bookmarklet.min.js` and `bookmarklet.href.txt`.
`--check` rebuilds all three in memory and compares them with the files on disk, byte for byte, and
writes nothing; every output is written with LF. `.gitattributes` marks `bookmarklet_src.js` `-text`,
so a checkout keeps those LF bytes. Before that line existed, `--check` failed on a stock Windows
checkout (`core.autocrlf=true`) with the committed blob correct: git rewrote the source to CRLF and the
byte comparison reported `bookmarklet_src.js … DIFFERS`, exit 1 (NOTE_path2_fourth_pass_2026_09_25,
B-1). Measured after the line, on this Windows checkout, with terser 5.46.0: all three `matches`,
exit 0. The shipped bookmarklet is

    bookmarklet.min.js    sha256 51f06338e13dd2fd24de7fd6074cdadd627f0e8b8375e3d8b8a56a45e4f189e0   83,768 chars
    bookmarklet.href.txt  sha256 1f803689069b21ec94b283ae9c4e6f89a56593613596e6db7f7896c20a495ad3   83,779 chars

(4,551 characters more than the twelfth-pass build, `830b4ba7…`, 79,217 characters: the set of code points Unicode
16.0.0 assigns, 3,225 of them, its decoder and the thirteenth pass's facts. The twelfth-pass build was 1,542 more than
the eleventh's, `ce8c5d99…`, 77,675 characters, which was 25,302 characters more than the tenth-pass build: `main`'s
port, carried whole as the guard's reference (`main`'s
own bookmarklet is 24,335 characters), and the guard; the tenth-pass build was 4,446 more than the ninth's: the
case fold, 1,996 of them, and the tenth pass's K-1 to K-5; the ninth-pass build was 8,577 more than the eighth's,
`main`'s own reading in both of its spellings and the ninth pass's abstentions; the eighth-pass build was 4,895 more
than the seventh's, 440 of them the skew set; the seventh-pass build was 5,417 more than the sixth's, 3,136 of them the
name table.)

(Earlier builds: `b04d14dc…`, 11,437 chars, from the 7.47.0 file — accuses outside Python;
`9ea8f572…`, 17,686 chars, the BC-2 + COMPAT-1 re-cut — cannot see a binary file; `4b2d34e1…`,
19,002 chars, the BIN-2 re-cut — resolves a path claim to an earlier basename match, keys a
dotfile without its dot, counts a changed `def` as added; `22c31746…`, 20,255 chars, an unmerged
PATH-2 cut — one changed test cancels every same-named new one, and `only_touches` abstains when the
dot is on the prefix; `af252434…`, 26,236 chars, the unmerged third-pass cut — reads "only touches
.gitignore" as not a path, counts an NBSP re-indent of a test as an added test, and reads a line
holding U+000B, U+000C, U+2028 or U+2029 differently from the Python; `5b6f3167…`, 26,691 chars, the
unmerged fourth-pass cut — reads a form-feed re-indent of an existing function as an added one, does not
count a new test led by a form feed, and reads a COMPAT line holding U+0085, a header path followed by
U+0085 and a printed path differently from the Python; `d8ce5111…`, 27,738 chars, the unmerged
fifth-pass cut — reads a line opening `--- ` or `+++ ` inside a hunk as a file header, ends a claimed
non-ASCII name where the Python does not, and reads a U+FEFF-led definition in the middle of a file;
`5d15861e…`, 29,038 chars, the unmerged sixth-pass cut — reads identifiers by its engine's Unicode (16.0 on
Node 24) where the Python read 15.0, and reads a GNU-style next-file header inside a hunk that declares more
lines than it carries as content; `03d1e1a1…`, 34,455 chars, the unmerged seventh-pass cut — verifies a test
count left after pairing changed tests away beside a `def test_` line in a markdown file or a string, counts a
U+FEFF-led test at line 1 where a count elsewhere had balanced it, reads a file list holding a header that may
be a SQL comment or a `++` line, and reads a name meeting a letter Unicode 15.1 or 16.0 assigned as its
prefix; `caf3682b…`, 39,350 chars, the unmerged eighth-pass cut — reads a test count, a symbol claim, a file
count or a path claim otherwise than `main` where `main`'s answer had been right by a second error both share
(a test behind a lone CR, a `def test_` in markdown, a `-U0` header triple, a monorepo's same-named file in
another directory); `f13f056d…`, 47,927 chars, the unmerged ninth-pass cut — licenses a file list that differs
from `main`'s by a file `main` read from a hunk's content, beside a changed file neither reading counts (a `Submodule`
line), compares two header paths' case by its engine's Unicode, raises on a `+++ /dev/null` with no `---` line
where `main` does not, and abstains on a path the Python extracts otherwise where the Python verifies;
`c9a23982…`, 52,373 chars, the unmerged tenth-pass cut — licenses a file count over #121's dotted key beside a
`---`/`+++` pair under a header neither reading can read (`git diff --no-prefix`) or a pair read as a header without a
header's shape, where `main` read the count right, and prints two reasons' keys unfolded; `ce8c5d99…`, 77,675 chars,
the unmerged eleventh-pass cut — licenses #121 in renderings with no `diff --git` header and #97 on a match only in
case; `830b4ba7…`, 79,217 chars, the unmerged twelfth-pass cut — licenses #97 and #121 on a typechanged path (git's
deletion then creation for one path) and #97 on a name ending in whitespace in `difflib`'s rendering. A bookmark
that hashes to any of them is an old port; drag the new one.)

Whatever a browser holds under that bookmark either hashes to the line above (drop the
`javascript:` prefix) or is not this build. terser 5.46.0 produced these bytes; the same terser
rebuilds the `4b2d34e1…` bookmarklet byte for byte from its sources, which terser 5.51.2 produced.
`build_bookmarklet.py`'s docstring names the same version.

`differential/` — the test. Read on.

## The differential test

A port is held to the original by output, not by trust. `differential/` builds a corpus of
(summary, diff) pairs, runs the released Python and the JavaScript over every pair, and compares
the records field by field: verdict, measured, why_unmeasured, sentences_total,
uncovered_sentences, the never-read list in order, and every claim as
(kind, verdict, why, text, detail) in order. A port that gets the verdict right for the wrong
reason, or reads one sentence more or less, is a disagreement.

    cd web/gate/differential             # on a checkout carrying #113 and #115 (or 7.48.0)
    python build_corpus.py               # 176 real pairs, pinned to shas (below)
    python fuzz_corpus.py                # 3,000 synthetic pairs, seeded
    python py_side.py                    # refuses to run unless styxx/diffgate.py hashes to c3eed72e…,
                                         # styxx/_diffgate_ref.py (main's reader) to 9b620e00…,
                                         # styxx/_xid.py's name table to 8df68f21… and its skew set to 0b7134fd…,
                                         # and styxx/_fold.py's case fold to a52cda82… and its assigned set
                                         # to 56a413eb…
    node js_side.js
    python differential.py
    node check_pairs.js                  # the 389 pinned pairs against their expect blocks

The pin moved twice between the last two runs of this differential, and one of those moves is a
finding rather than a routine bump. COMPAT-2 (#124) changed the compat reading and the port had to
follow it — that is ordinary. But #124 merged with only its Python half, and separately the
`fetch_pr` door landed on `main` after the previous pin was written, so `py_side.py` had been
**refusing to run on `main`** ever since: the check that guards the two-implementation claim was
itself disabled, which is exactly how the port was able to fall a whole cycle behind without
anything failing. `fetch_pr` fetches a pull request over the network and is no part of the reading
the port transliterates; it moves this whole-file hash without changing a single verdict. Both
facts are recorded here rather than quietly corrected.

`path1_pairs.json` carries eight pairs for PATH-1: a basename claim satisfied by files in
subfolders, a nested single file, a basename claim that still accuses when something else changed,
a dotted identifier (`Assert.NotNull`), a CSS selector (`.k-step-link`), a slashed prefix that
still anchors, and two pinning the failure modes PATH-1 deliberately does **not** repair. Those
last two assert the instrument is still wrong; they exist so a later change cannot claim an
unrepaired mode without its own preregistration.

The corpus needed them. Before PATH-1 added these, `corpus_real.json` carried exactly **four**
`only_touches` claims in 3,212 pairs and PATH-1 changed none of them — so the differential's
"0 disagreements" was true and almost meaningless for that change. A check that does not exercise
what changed is not evidence, and the honest py/js comparison for PATH-1 was run separately over
the 604-row BENCH corpus (297 `only_touches` readings, 0 disagreements).

Result, 2026-09-29, this branch after the twelfth pass (`NOTE_path2_twelfth_pass_2026_09_29`), merged with `main` at
`2a6ce0a3` (whose `styxx/diffgate.py` is the one at `98a5c368`):

    3529 pairs, 7519 claims (649 verified, 1607 contradicted, 5263 uncheckable) — 0 disagreement(s)
    353 pinned pairs, 0 disagreement(s)

The same 3,529 pairs with `main` on both sides read **48** disagreements, none of them on the 44 pairs this pass adds.
With `main`'s Python on one side and this port on the other, **454** records differ; this Python against `main`'s port,
**462**. Against the eleventh-pass head, 29 records move in the Python and 29 in the port, every one a pinned pair: the
five this pass re-pins and 24 of the 44 it adds. No generated or real record moves. The bookmarklet is rebuilt with
terser 5.46.0 (`bookmarklet.min.js` sha256 `830b4ba7…`, 79,217 characters); loaded in Node with the browser stubbed, it
reads all 3,529 records as the port does. A page that loads `diffgate.js` without `diffgate_ref.js` (or the bookmarklet's
bundle) now gets an error from `gateDiffText`; it used to read every decided claim as UNCHECKABLE with "main's reading
raises on this diff", which was false. Any page that embeds the port must load `diffgate_ref.js` with it.

The twelfth pass tightens two licences (the note, section A). Round 11's review, judged against truth, read 1,133
claim-door cells worse than `main` on the raw door, 1,133 in the port and 6 at the git door on one set of 4,000 cases,
and every one was a licensed difference. #121 was licensed in renderings with no `diff --git` header: there a
directory `a/` or `b/` reads as git's prefix, `strip()` merges a name ending in a space, and `difflib` and GNU
`diff -uN` leave an empty file out. #97 was licensed on a match that held only once lower-cased, or on an entry the
reader knew it had misread. So #121 now licenses only where the diff is git's own rendering, on every door:
- every file section sits under a readable `diff --git a/X b/Y` header;
- its `---`/`+++` pair and its rename lines name that header's paths;
- no line is one that no reading places.

On a path claim, #121 also needs the entry it resolved to match the claim as written, case kept, by the tier it
resolved by. #97 licenses only a case-kept exact or suffix match, and never in a reading that holds one of Z-3's
doubts. Tightening a licence can only turn a verdict into an abstention (the note, A.3).

Five pinned pairs written as plain `---`/`+++` diffs over dotfile twins re-pin to UNCHECKABLE. The same pairs written
as git writes them keep #121's reading, and are pinned beside them.

**What is measured, and how.** Three different things carry numbers in this README, and they are not interchangeable:
- **The regression differential is judged against truth.** It sets `main`'s file against this branch's on every door
  (the raw door, the git door on a real repository, the port). Each claim is judged by a model that reads no line of
  either instrument: CPython's parser over the base and head files, and git's `--name-status` for the file list. A
  claim reads **worse** where its verdict is false by that truth and `main`'s is not.
- **The guarantee harness is self-consistency.** It asks the instrument's own `_precondition` and switches whether each
  difference from `main` is licensed. It shows that the guard does what the module says. It cannot see a licence that
  the module's rules grant and truth refutes. The eleventh pass's "0 violations" were of this kind, on inputs where
  round 11 then read 1,133 cells worse.
- **The property test against the scorer** holds the instrument's final claims to `path2_gates.py`'s own guard:
  `main`'s verdict from the baseline bytes, the switched readings from the scorer's reverts of #97, #121 and #101, and
  the preconditions from the scorer's code. That is two implementations of one rule agreeing; it is not truth either.
  `tests/test_diffgate_guard.py` runs it over the pinned pairs, 500 randomised diffs and the reproductions on the raw
  door, and over 60 rebuilt records at the git door.

The regression differential of this pass used round 11's reviewer's harness and truth model, `origin/main` against
this branch. It ran 11 sets, 30,017 cases:
- 19,000 written this round at fresh seeds. 10,000 come from a generator for the shapes round 11 found: names ending in
  spaces beside their twins, directories `a/` and `b/` under `--no-prefix`, case-insensitive twins on git's
  case-sensitive paths, `difflib` and per-file GNU `diff -uN` leaving empty files out, and `strip()`-merged keys in
  plain renderings. 4,000 come from round 11's own generator and 5,000 from the builders' earlier generators. Of the
  9,482 git cases, 3,456 are rewritten for the raw door as `--no-prefix`, GNU, mnemonic-prefix, mailbox or CRLF text.
- Round 11's three sets (11,000 cases) and its 17 aimed reproductions.

Under Python 3.12 (535,957 claim-door cells) and 3.14 (535,510) it reads:
- **0 claims worse than on `main`, judged against truth**, on the raw door, the git door and the port;
- **0 new Python/port disagreements on the same input**, in verdict or in reason;
- no raise that `main` does not have.

The licences as committed at `e4586637` read 14 cells worse (4 raw-door, 4 port, 6 git-door): #121 licensed a path claim
whose entry matched only once lower-cased. The tier-kept case check (`98f74833`) fixes them. The six records are pinned
pairs and truth-judged tests, and the tests refuse `e4586637` on 4 of them. The same truth-judged check refuses the
eleventh-pass instrument on 12 of round 11's 17 reproductions.

The git door reads the repository and the port reads the text, so the two can differ on the same case. Where `main`'s
two doors agreed:
- 38 cells carry opposite verdicts, 32 on texts rewritten with mnemonic prefixes and 6 on texts rewritten as GNU diffs;
- in 19,244 cells the port abstains where the git door decides (#121 licensing nothing in those renderings, and the
  text's own doubts), and in 47 the reverse.

The cost is recall. On the differential corpus `main` decides 2,726 claims and this branch abstains on 486 of them. On
the 3,485 pairs the eleventh pass measured, it abstains on 472 of 2,660, five more than the eleventh pass (the five
re-pinned pairs). Over the regression sets' raw door under 3.12 (214,951 claims), this pass abstains on 8,993 claims
the eleventh pass decided:
- 3,060 of them false by truth where `main`'s were not: the regressions this pass removes;
- 5,705 right, and 228 undecided by the harness;
- by kind, `files_changed_count` 8,476, `file_deleted` 195, `file_created` 184, `only_touches` 100 and `file_touched` 38.

It decides no claim the eleventh pass left UNCHECKABLE, and changes no decided verdict. At the git door #121 now
licenses nothing under `diff.noprefix`, `diff.mnemonicPrefix` or `diff.submodule=log` either, although git's
`--name-status` was right there; the note takes that cost knowingly (A.1).

Result at the eleventh pass (`NOTE_path2_eleventh_pass_2026_09_28`), kept as it was measured, merged with `main` at
`2a6ce0a3` (whose `styxx/diffgate.py` is the one at `98a5c368`):

    3485 pairs, 7423 claims (623 verified, 1580 contradicted, 5220 uncheckable) — 0 disagreement(s)
    309 pinned pairs, 0 disagreement(s)

The same 3,485 pairs with `main` on both sides read **48** disagreements, with the tenth-pass head on both sides **2**
(R10-WHY1 and R10-WHY2, round 10's two reasons that printed a runtime's lower case, pinned by this pass). With
`main`'s Python on one side and this port on the other, **415** records differ — what `py_side.py --installed`
measures against 7.48.0, whose file is `main`'s; this Python against `main`'s port, **423**. Against the tenth-pass
head, 8 records move in the Python and 8 in the port, every one a pinned pair this pass added or re-pinned: no
generated or real record moves. The minified bookmarklet, loaded in Node with the browser stubbed, reads all 3,485
records as the port does. Round 10's K-5 reproductions (R10-K5A to R10-K5D), and a U+0085 and a U+FEFF separator
beside them, are tests, not pinned pairs: on `main` the two ports already extract different claims from those
sentences, and each port now reads them exactly as `main`'s same port did.

The eleventh pass moves the licensed-difference rule from the reading to the verdict (the operator's decision of
2026-09-28; the note, sections A and B). `main`'s reader is vendored unchanged — `diffgate_ref.js` here is
`origin/main`'s `web/gate/diffgate.js` byte for byte, and `styxx/_diffgate_ref.py` `origin/main`'s `styxx/diffgate.py`
— and `gateDiffText`, like the Python's two doors, holds every claim to that reader's verdict on the same input: an
equal verdict or an abstention is kept, and a different one only where one of the three repairs (#97's tiers, #121's
dotted key, #101's pairing), switched off alone by an explicit parameter, gives `main`'s verdict back and that
repair's own precondition holds on the claim; otherwise the claim is UNCHECKABLE and names `main`'s verdict. Where
`main` raises, or makes no such claim, a decided claim abstains; the verdict and `--strict` are recomputed from the
final claims. The three repairs are therefore the only surface where a verdict other than `main`'s can arise. That
bounds **where** a new false verdict can come from, not **whether** one does: a licensed difference can be false, and
round 11 measured 1,133 such cells on one set (the twelfth pass, above; its note, F.1). A claim read
from a sentence the two ports' templates may read apart — a non-ASCII word character (by the pinned table and skew
set), U+001C to U+001F, U+0085, U+FEFF, U+2028, U+2029 or a CR with a character after it; for a symbol claim only
outside every name — reads as `main`'s same port read it (K-5 at the sentence); the em dash, emoji and curly quotes
read alike in both templates, so `integrations/git/README.md — created.` keeps #97's repair. The guard does not make
the two ports agree; that stays the parity layers' work, and the differential above is its measure.

Round 10's review found four regression classes and four scorer gaps (the note, section C). The two verdict
regressions had passed through #121's licence, so the guard alone would not have caught them; each is fixed in the
reader. A `---`/`+++` pair under a `diff --git` header neither reading can read (`git diff --no-prefix`,
`diff.noprefix=true`), and a pair read as a header without a header's shape, are now doubts `main`'s reading also
held, and beside #121's dotted key the file-list claims abstain. The two Python/port disagreements are fixed too: K-5
reads the whole sentence, and the dot-miss and refused-file reasons print their keys through the fold. The raw-door
reproductions (R10-NP1, R10-NP2, R10-UNDER1, R10-UNDER2, R10-WHY1, R10-WHY2) are pinned pairs and read UNCHECKABLE in
both ports. At the git door, where the file list is git's `--name-status` and not the no-prefix text, the no-prefix
shapes keep #121's licensed verdict: over the 18 cases written for them in real repositories (`diff.noprefix=true`,
`diff.mnemonicPrefix=true` and git's default, each beside dotted twins and a whitespace twin or an `a/` or `b/`
directory), the git door reads 38 claims right that `main`'s git door read wrong and reads none worse on those 18
cases; round 11 then read 3 git-door cells worse on matches that held only in case (R11.3, fixed at the twelfth pass).
The port, which has only the text, abstains there, as does the Python's raw door.

The regression differential of this pass (the round's own harness and the round-10 reviewers' runners, `main`'s file
against this branch, CPython's parser and git's `--name-status` as the judge), over 37 sets and 75,047 cases — 14,018
written this round (11,000 randomised diffs at fresh seeds, 379 of them rendered as `git diff --no-prefix` prints them
and 375 with mnemonic prefixes, beside 1,500 aimed at K-5's sentences, 1,500 at header-shaped pairs after an exact
hunk and 18 cases in real repositories set to `diff.noprefix=true`, `diff.mnemonicPrefix=true` or git's default),
25,000 from the round-10 reviewers' generators and 36,029 from earlier rounds — reads, under Python 3.12 (2,054,053
claim-door cells) and 3.14 (2,052,816): 0 claims worse than on `main` on the raw door, the git door and the port,
judged against truth on those sets. None of their generators rendered a plain diff with an `a/` directory, an omitted
empty file, a name ending in a space or a case-only suffix match; round 11's generator for those shapes read 1,133
raw-door, 1,133 port and 6 git-door cells worse on 4,000 cases (the twelfth pass, above);
0 raises `main` does not have; **0 new Python/port disagreements**, in verdict or in reason, on the same input. Where
the git door (which reads the repository) is set against the port (which reads the text), 136 cells on 80 records
carry opposite verdicts, every one on a text rewritten with mnemonic prefixes (`c/`, `i/`, `w/`): the git door right
on 126 and the harness undecided on 10, and the tenth-pass head read 79 of the 80 records the same on the git door and
all 80 on the raw door.

The guarantee (the note, section B) was checked claim by claim over the regression sets' 75,047 cases and the
differential corpora's 3,485 pairs (78,532 inputs): every final claim reads `main`'s verdict, UNCHECKABLE, or the
branch's own verdict where a repair switched off alone gives `main`'s verdict back and that repair's precondition
holds. In the port, 871,428 claims; in the Python under 3.12, 869,220 raw-door claims and 339,918 git-door claims (the
git door against `main`'s `gate_diff` on the same repository and range), and under 3.14, 868,182 and 339,713: 0
violations, and nothing raises. That measured **self-consistency**, not truth: the harness decided each licence with the
instrument's own `_precondition` and switches, so a licence those rules grant and truth refutes passed it; round 11 read
1,133 cells worse on inputs where it reported 0 violations (the twelfth pass, above). `tests/test_diffgate_guard.py`
commits the same check over the pinned pairs, the differential corpora and 1,500 randomised diffs at a fixed seed (the
corpora test has two parameters, `corpus_fuzz` and `corpus_real`, both gitignored, so it skips in a clean checkout and
in CI), and plants eight defects in the reader outside the
three repairs (a hunk always read exact, `/dev/null` read with anything after it, async tests counted, every diff
holding Python, A-1's abstention dropped, a count off by one, a prefix held by string, F-2 breaking a line at a form
feed): each gives new verdicts without the guard and, with it, only abstentions or `main`'s own verdict.

The cost is recall, and this pass measured it rather than the note's estimate (section D expected most of the round-8
and round-9 generators' W-1, F-2 and Y-4 gains to become abstentions; they did not, because the ninth and tenth
passes' reading-level rule had already abstained on almost every difference no repair explains, and the guard found 12
more cells over 861,797 raw-door claims under 3.12, none where `main` was right). On the differential corpus `main`
decides 2,660 claims and this branch abstains on 468 of them (file_created 23, file_deleted 6, file_touched 70,
files_changed_count 73, only_touches 152, symbol_added 47, tests_added 97): 11 more than the tenth pass (R0.0's doubt
6, R0.2's 4, and 1 path claim K-5 now reads with this reading, which abstains). Over the regression sets' raw door
(3.12), `main` decides 612,047 claims; the tenth pass abstained on 308,091 of them and this pass abstains on 306,472:
3,040 new abstentions (R0.0 648, R0.2 2,334, main's paths folding apart 25, the guard 8, other 25; 1,993 of them where
`main` was right), and 4,669 claims the tenth pass left UNCHECKABLE decided again: 10 by a licensed repair (4 right, 6
undecided by the harness) and 4,659 by K-5, which takes `main`'s claim whole — 685 right, 2,096 undecided and **1,878
wrong**, each `main`'s own wrong verdict, where the tenth pass had abstained. That is the operator's K-5 rule as
written (a claim the two ports may read apart reads as `main` read it), not worse than `main`, and it is a verdict the
tenth pass withheld: it is the largest single move this pass makes against the tenth, and it is listed as a follow-up.

Per call, over the 3,485 records of the differential corpus (the fastest of three rounds each, on this Windows box
with other work running, the never-read observer on): the Python `gate_diff_text` takes 3.93 ms on average (median
0.59 ms, 95th percentile 1.61 ms, slowest 2.0 s) against `main`'s 0.60 ms (median 0.19 ms) and the tenth-pass head's
3.36 ms (median 0.36 ms) — 6.6 times `main`'s total and 1.17 times the tenth pass's. The port takes 0.86 ms on average
(median 0.16 ms) against `main`'s port's 0.17 ms and the tenth pass's 0.67 ms — 4.9 and 1.3 times. The guard reads
every input with `main`'s reader as well, and where a claim differs from `main`'s, once more with each repair switched
off; the git door runs git's two reads twice.

Result at the tenth pass (`NOTE_path2_tenth_pass_2026_09_28`), kept as it was measured, merged with
`main` at `2a6ce0a3` (whose `styxx/diffgate.py` is the one at `98a5c368`):

    3475 pairs, 7402 claims (622 verified, 1579 contradicted, 5201 uncheckable) — 0 disagreement(s)
    299 pinned pairs, 0 disagreement(s)

The same 3,475 pairs with `main` on both sides read **47** disagreements, with the ninth-pass head on both
sides **6**, all six on pairs this pass adds (round 9's two regression classes, which were the ninth pass's own
Python/port disagreements). With `main`'s Python on one side and this port on the other, **404** records differ —
what `py_side.py --installed` measures against 7.48.0, whose file is `main`'s; this Python against `main`'s port,
412. Against the ninth-pass head, 29 records move in the Python and 26 in the port, every one a pinned pair this
pass added or re-pinned: no generated or real record moves. The minified bookmarklet, loaded in Node with the
browser stubbed, reads the 299 pinned pairs as the Python does. Three pairs this pass wrote are tests, not pinned
pairs: where the two ports' path templates extract different paths from one sentence (a non-ASCII character in or
before the path, as on `main`), the details differ, and the pinned pairs are held at full width.

The tenth pass narrows the licences to #97's exact and suffix tiers, #121's dotted key and #101's pairing (the
note, section B). A file list `main` read from a hunk's content abstains where the ninth pass licensed its removal
(K-1: that file had balanced a changed file neither reading counts, git's `Submodule` line under
`diff.submodule=log`, svn's and hg's binary notices); two header paths are compared by one fixed case fold carried
in both files, not by the engine's lower-casing (K-2); a `+++ /dev/null` with no `---` line no longer raises (K-3);
beside #121's dotted key, a line no reading places abstains the file list (K-4); a path claim the two ports'
templates may extract differently reads as `main` read it (K-5); and a key a reason prints is the same text on every
engine. Measured on every door under Python 3.12 and 3.14 with CPython and git as the judge, over 22 sets and 44,129
cases, 16,000 of them randomised at fresh seeds (the note, section E): 0 claims read worse than on `main`, 0
Python/port disagreements `main` did not have, and nothing raises where `main` does not. The cost is recall: on the
differential corpus this branch abstains on 452 of the 2,642 claims `main` decides, 56 of them new at the tenth pass
(K-1 34, K-2 16, K-4 6), 44 of those on the pairs this pass adds.

Result at the ninth pass, kept as it was measured:

    3443 pairs, 7311 claims (623 verified, 1576 contradicted, 5112 uncheckable) — 0 disagreement(s)
    267 pinned pairs, 0 disagreement(s)

The same 3,443 pairs with `main` on both sides read **43** disagreements, with the eighth-pass head on both
sides **0**. With `main`'s Python on one side and this port on the other, **375** records differ — what
`py_side.py --installed` measures against 7.48.0, whose file is `main`'s; this Python against `main`'s port,
383. Against the eighth-pass head, 155 records move in the Python and 155 in the port: 59 pinned pairs this pass
added or re-pinned, and 96 fuzzed records, every one by Z-4 (the fuzzer names a file with a directory that
only a same-named file elsewhere matches). No real-corpus record moves. The minified bookmarklet, loaded in
Node with the browser stubbed, reads the 267 pinned pairs as the Python does.

The ninth pass adds no reading `main` lacked. For `tests_added`, `symbol_added`, `files_changed_count` and the
path claims it computes `main`'s own reading beside its own, in both of `main`'s spellings (its Python's line
split, whitespace and name table; its port's split, JavaScript's whitespace, ASCII word characters and line
starts), and where the two differ and no repair licenses the difference (#97's exact and suffix tiers, #121's
dotted key, #101's pairing, an exact hunk's counts) the claim abstains, identically here and in the Python,
with a reason that names the difference (the note, section B: Z-1 to Z-5). A path claim with a directory
component that only a same-named file in another directory matches abstains (Z-4); a definition line CPython
refuses abstains the claims that read its file (Z-5). Measured on every door under Python 3.12 and 3.14 with
CPython and git as the judge, over 23,110 cases (the note, section E): 0 claims read worse than on `main`,
and 0 Python/port disagreements `main` did not have. The cost is recall: on the differential corpus this
branch abstains on 369 of the 2,557 claims `main` decides, 152 of them new at the ninth pass (Z-4 81, Z-1 34,
Z-2 18, Z-3 15, Z-5 4). Those zeros did not hold: round 9's review measured 877 and 828 claim-door cells per door
reading worse at two seeds (a file `main` read from a hunk's content, licensed away beside a `Submodule` line) and
267 and 242 new Python/port disagreements under 3.12 (a case pair Unicode 16.0 assigned), and the tenth pass's own
differential found three more classes (NOTE_path2_tenth_pass, sections A and D); the tenth pass is the answer.

Result at the eighth pass, kept as it was measured:

    3416 pairs, 7259 claims (715 verified, 1595 contradicted, 4949 uncheckable) — 0 disagreement(s)
    240 pinned pairs, 0 disagreement(s)

The same 3,416 pairs with `main` on both sides read **41** disagreements, with the seventh-pass head on both
sides **0**. With `main`'s Python on one side and this port on the other, **248** records differ — what
`py_side.py --installed` measures against 7.48.0, whose file is `main`'s; this Python against `main`'s port,
259. Against the seventh-pass head, 52 records move in the Python and 52 in the port, every one a pinned pair
this pass added or re-pinned — no generated record moves. The minified bookmarklet, loaded in Node with the
browser stubbed, reads the 240 pinned pairs as the Python does.

The eighth pass adds no reading `main` lacked except one repair: where this branch cannot be sure it reads a
claim at least as well as `main`, it abstains, identically here and in the Python (the note, section B). A
file list read from a `---`/`+++` line that may be a SQL comment or a `++` line, holding two paths one key in
case, or missing a file GNU names outside any header pair abstains the count, the scope and the path claims
(Y-1). A U+FEFF-led definition the diff does not show is line 1 abstains the claim that could read it, and any
U+FEFF-led added test abstains the test count (Y-2). A name that meets a code point the supported Pythons read
differently from the table abstains (Y-3). A count equal to what the #101 pairing leaves abstains: the pairing
withdraws and never verifies (Y-5). And GNU's `+++ /dev/null<TAB>timestamp` is a deletion again (Y-4). Measured on
every door under Python 3.12 and 3.14 with CPython as the judge (the note, section E): 0 claims read worse than
on `main`, and 0 Python/port disagreements `main` did not have. That zero did not hold: round 8's review
measured 507 (3.12) and 525 (3.14) claim-door cells reading worse, on shapes the eighth pass's sets did not
hold (NOTE_path2_ninth_pass, D.3); the ninth pass is the answer.

Result at the seventh pass, kept as it was measured:

    3381 pairs, 7186 claims (722 verified, 1594 contradicted, 4870 uncheckable) — 0 disagreement(s)
    205 pinned pairs, 0 disagreement(s)

The same 3,381 pairs with `main` on both sides read **34** disagreements, with the sixth-pass head on both
sides **3** (all on this pass's pinned pairs for the characters the two engines' Unicode tables read
differently). With `main`'s Python on one side and this port on the other, **213** records differ — that is
what `py_side.py --installed` measures against 7.48.0, whose file is `main`'s (see *Drift*); this Python
against `main`'s port, 227. (The sixth pass's text tied `--installed` to its 219; it measured 205, the
same comparison as this 213.) The minified bookmarklet, loaded in Node with the browser stubbed, reads the
205 pinned pairs as the Python does.

The seventh pass closes the three regressions against `main` the sixth pass's review found, each in both
implementations. **One name table**: the two read identifiers from their engines' Unicode (Python 3.12's
15.0, Node 24's 16.0), so "Added function foo<U+30FB>bar." read VERIFIED here and CONTRADICTED there; both
now read the table `styxx/_xid.py` carries, Unicode 15.0.0, the same 3,136 bytes in both files, and this port
also builds Python's `\w` from it instead of `\p{L}\p{N}`. **The claim side read as the definition side**: a
claimed name that runs on past the identifier ("Added function fo²o.") names no identifier and abstains,
instead of being truncated to `fo` and verified on `def fo`. **A hunk's counts read only when it carries
them**: a hunk that declares more lines than it carries is read as `main` read it, so a GNU-style next-file
header after it is a header again, and a hunk git wrote is read by its counts with no exception. An added
`async def test_` now abstains the test count beside it. Measured on every door with CPython as the judge
(the note, section E): 0 claims read worse than on `main`, and 0 Python/port disagreements `main` did not
have.

The sixth pass makes two pairs of readings one reading each. **One parse**: the status map, the added
lines and both definition pairings come from one hunk-aware reading of the diff, so a removed SQL comment
(`-- users`, printed `--- users`) no longer hides a changed `def` from the pairing, and an added `++ x`
(printed `+++ x`) is no longer a phantom file on the raw door and here. **One reading of a name**: a name,
claimed or defined, is the Python identifier that starts there, in both implementations. Both were
enumerated on every door with CPython's parser as the judge (the note, section E): a 467-cell
definition-line grid and a 103-cell name grid, each a real repository read by `gate_diff`, by
`gate_diff_text` on git's bytes and by this port on the same bytes. This branch: **0** Python/port
disagreements and **0** verdicts wrong where `main` was right, on every door; `main` disagrees on 44 and 63
of them, the fifth-pass head on 0 and 63, and the fifth-pass head read 91 true claims over non-ASCII names
as CONTRADICTED here (`main`: 9). The name grid, the test shapes and the twelve name-end cells are committed
tests (`tests/test_diffgate_path2.py`, `test_w2_*`), beside the fifth pass's 300 raw-door cells.

The fifth pass's result, kept as it was measured: 3,337 pairs, 7,094 claims (671 verified, 1,571
contradicted, 4,852 uncheckable), 0 disagreements; 161 pinned pairs, 0.

**Zero is a count over this corpus, not a property of the two implementations.** The third pass reported
the same zero while the two returned opposite `tests_added` verdicts on any re-indent by U+000B, U+000C,
U+2028 or U+2029, and the fourth pass reported it while they disagreed on `compat_claim` lines holding
U+0085 and on a symbol name followed by a non-ASCII letter; no record in the corpus carried such a line.
The same 3,337 pairs with the fourth-pass head on BOTH sides read **17** disagreements, and with `origin/main`
on both sides **27**. The fifth pass enumerates the definition-line class instead of sampling it: every
character of Python's `\s` and U+FEFF leading an added or removed test or symbol definition, between `def`
and the name, and after the name, plus six name-end characters that are not spaces — 312 inputs, through
`gate_diff_text` and the port on one text, and through `gate_diff` on a real repository and the port on the
bytes git printed. `origin/main` disagrees on 40 of them on each door, the fourth-pass head on 4 (the `\b`
case), this branch on **0** on both; the 300 raw-door inputs are a committed test
(`tests/test_diffgate_path2.py::test_v1_the_port_reads_the_grid_as_the_python_does`), so that zero is a
receipt rather than a number typed once.

**What still disagrees, counted and not repaired.** The claim templates read the DESCRIPTION with
JavaScript's `\s`, `\w` and `\b`, not Python's. One whitespace character between the words of twelve claim
shapes (Python's 29 and U+FEFF, 360 inputs): **72** disagreements at the fifth pass, exactly the six
characters whose membership differs (U+001C–U+001F and U+0085 are whitespace to Python, U+FEFF to
JavaScript) in all twelve positions; on the sixth pass's own twelve shapes, 66 — the same on `main`, the
fifth-pass head and this branch. A non-ASCII letter or mark inside a claimed name or path (20 inputs):
**12** at the fifth pass, and that sentence said the count was the same on `origin/main`. The count was;
the verdicts were not (NOTE_path2_sixth_pass, D.2): at the fifth-pass head a claimed non-ASCII name read
CONTRADICTED on a true claim, here and for some names on the Python too. On the sixth pass's 88 claimed-name
inputs, whole records compared, `main` and the fifth-pass head disagree on 63 and this branch on **0**: the
verdict and the reason read the identifier in both implementations, and this port reads the template's
`name` group with Python's `\w` for the claim's detail. A declared `adds_symbol` with a non-ASCII
identifier is still MALFORMED here and read by the Python, as on `main`. On the diff side the fifth pass closed two classes: a
`---`/`+++` path followed by one of 28 such characters (6 disagreements on
`main`, 0 now) and a reason printing a path that holds one (25 on `main`, 20 at the fourth-pass head, 0
now). Below any test sat the engines' Unicode version: Node 24's `\p{L}\p{N}` read 5,004 code points
assigned after Unicode 15.0 as word characters where Python 3.12 does not. Since the seventh pass the port
reads names and Python's `\w` from the pinned 15.0.0 table, not its engine; what still reads the engine's
Unicode is the claim templates' own JavaScript classes (above) and 27 code points that lower-case
differently.

At the tenth pass, against `origin/main`, PATH-2 moves **404** of the 3,475 records in the Python and
**412** in the port; against the ninth-pass head, 29 and 26, every one a pinned pair this pass added or re-pinned
— no generated record moves.

At the ninth pass, against `origin/main`, PATH-2 moves **375** of the 3,443 records in the Python and
**383** in the port; against the eighth-pass head, 155 and 155: 59 pinned pairs this pass added or re-pinned
and 96 fuzzed records, every one by Z-4.

At the eighth pass, against `origin/main`, PATH-2 moves **248** of the 3,416 records in the Python and
**259** in the port; against the seventh-pass head, 52 and 52, every one a pinned pair this pass added or
re-pinned — no generated record moves.

At the seventh pass, against `origin/main`, PATH-2 moves **213** of the 3,381 records in the Python and
**227** in the port; against the sixth-pass head, 11 and 14, every one a pinned pair this pass added — no
generated record moves.

At the sixth pass, against `origin/main`, PATH-2 moved **205** of the 3,366 records in the Python and
**219** in the port (3 real, 108 fuzzed, the rest pinned pairs); against the fifth-pass head, 20 and 24, every
one a pinned pair that pass added or re-pinned.

At the fifth pass, against `origin/main`, PATH-2 moved **185** of the 3,337 records in the Python and **194** in the port (the
difference is pinned pairs on which the two disagreed on `main`); 111 of them lie outside the pinned-pair
files. The fifth pass itself moves, of the 3,301 records that predate it, the four pinned pairs it re-pins
(V-1: a form-feed re-indent now pairs as a changed test, a U+2028-led `def` defines nothing, and two
changed generic definitions pair and abstain) and **90 fuzzed records**, 92 claims CONTRADICTED →
UNCHECKABLE: the fuzzer writes `docs/.` followed by a sentence period, and V-4 reads a prefix written to
end in `..` as the parent it spells. No real-corpus record moves.

`path2_gates.py differential` at the twelfth pass ran from a clean tree at `188faeba`. It exits 0: every gate passes, no
violation, and the provenance records the guard's reference as the baseline, byte for byte.
- **Moves.** 451 records moved, three of them records the baseline raises on. Claims are attributed to #97 14, #121
  118, #101 51, F-2 28, V-1 11, V-4 94, W-1 2, Y-1 14, Y-4 6, Z-2 1, Z-3 4, Z-4 118 and Z-5 2, and to 33 sets (158
  joint), among them W-1+Z-3 39, Y-1+Z-3 32 and #97+Z-4 27. There are 10 new accusations (`files_changed_count` 7 and
  `only_touches` 3, #121's dotfile twins counted apart in git's own rendering), each admitted.
- **G-C9** (the guard, the scorer's own, with the tightened licences written out): on the raw door 10 claims licensed
  by #97, 28 by #121 and 34 abstained by the guard; at the git door 12 by #97, 29 by #121 and 13 abstained.
- **G-C7**: 0 oracle violations. This includes the licence facts the instrument reads (git's own rendering, a Z-3
  doubt, each key's paths as written), held to the scorer's own reading of them.
- **G-C8**: every one of the 3,529 records tried. 1,170 were rebuilt and scored through `gate_diff` (57 holding a
  dotted path), 2,359 were not rebuildable faithfully, and 367 were scored again with rename detection on (2 renames
  detected).
- **The canaries** (20 records: seven door canaries and thirteen raw-door canaries): 0 violations. The door canaries are
  paths holding U+0085, U+2028 or U+2029, K-5's sentence, a case-only #97 match, a dotted status mismatch and a lone
  dotfile. The canaries reach every guard outcome on both doors (K-5, an abstention, and on the raw door an abstention
  whose precondition held without its switch), and a run whose canaries reach none fails.
- Scorer `93f3c169…`, harness `75bfbc39…`, repaired `diffgate.py` `ede86d7b…`, reference and baseline `9b620e00…`
  (`98a5c368`).

**Planted defects.** Committed tests plant each of these in a copy of the instrument, and the scorer program itself
refuses it through the canaries that every run of either mode scores:
- the git door returning the reading unguarded, or calling `main`'s `gate_diff` and ignoring it (round 11's PG1 and
  PG1c, which both modes admitted at the eleventh pass);
- a licence without its precondition, or without its switch (PG3, PG4);
- a dotted entry satisfying any path claim, or a count off by one beside a dotfile (PD2, PD3);
- each tightened licence dropped.

Keeping a decided claim where `main` makes no such claim, or raises (PG7), is held by stub tests only. The two readers
extract the same claims, and where `main` raises Z-1 to Z-3 abstain every claim unless `--run` or `--evidence` is
given, which the scorer never passes.

In the port, these plants are refused by `check_pairs.js`: #121 licensing in any rendering, the tier-kept case, #97's
case and doubt, and a rename line naming anything. A line no reading places read as git's own rendering, a `+++` line
naming another file, and a missing reference read as `main` raising are refused by the committed Python tests.

`path2_gates.py differential` at the eleventh pass, from a clean tree at `cabaa6cc` (the scorer, the harness,
`styxx/diffgate.py`, `styxx/_diffgate_ref.py`, `styxx/_xid.py`, `styxx/_fold.py`, `styxx/declare.py` and
`path1_extensions.txt` unmodified; the commits after it touch tests, the port's comments, `py_side.py`, the bookmarklet,
this README and the CHANGELOG, none of them a file the scorer's provenance reads): exit 0, every gate passes, no
violation; the provenance records the guard's reference as the baseline, byte for byte. 412 records moved, three of
them records the baseline raises on. Claims attributed to #97 11, #121 68, #101 51, F-2 28, V-1 11, V-4 94, W-1 2,
Y-1 14, Y-4 6, Z-2 1, Z-3 4, Z-4 118, Z-5 2, and to 33 sets (196 joint), among them W-1+Z-3 39, Y-1+Z-3 32 and
#97+Z-4 19. New accusations 8 (`files_changed_count` 4 and `only_touches` 4, #121's dotfile twins counted apart),
each admitted. **G-C9** (the guard, this file's own): every final claim reads as the scorer's guard reads it, reason and
detail included, and each of the instrument's switches reads as the scorer's own revert of the same rule — on the raw
door 8 claims licensed by #97, 18 by #121 and 2 abstained by the guard; at the git door 10 by #97 and 11 by #121.
**G-C7**: 0 oracle violations, the oracles reading the instrument's reading before the guard. **G-C1**: the strict
gates' reports compared key for key, 0 violations. **G-C8**: every one of the 3,485 records tried, 1,142 rebuilt and
scored through `gate_diff` (38 holding a dotted path), 2,343 not rebuildable faithfully, 366 scored again with rename
detection on (2 renames detected), 229 moved, all attributed; the nine canaries (three door canaries, U+0085, U+2028 and
U+2029 paths, and six raw-door canaries: `main` raising twice, K-3, Y-4, an unreadable header beside dotted twins and
a header-shaped pair after an exact hunk; every run of either mode scores them), 0 violations. Scorer `9178b4fd…`,
harness `75bfbc39…`, repaired `diffgate.py` `73a03de6…`, reference and baseline `9b620e00…` (`98a5c368`).

`path2_gates.py differential` at the tenth pass, from a clean tree at `75bc4b06` (the scorer, the harness,
`styxx/diffgate.py`, `styxx/_xid.py`, `styxx/_fold.py`, `styxx/declare.py` and `path1_extensions.txt` unmodified; this
README and the CHANGELOG were not yet committed and are not files the scorer's provenance reads): exit 0, every gate
passes, no violation; 402 records moved, two of them records the baseline raises on (every claim abstains there, and
the gate, strict and summary-only fields are scored). Claims attributed to #97 11, #121 55, #101 51, F-2 28, V-1 10,
V-4 94, W-1 2, Y-1 46, Y-4 6, Z-2 1, Z-3 4, Z-4 117, Z-5 1, and to 32 sets (115 joint), among them W-1+Z-3 39 (with
Z-3 reverted the repair still reads W-1's map, so only the pair gives `main`'s claim back: K-1), #97+Z-4 16 and
W-1+Y-1+Z-3 6. New accusations 7: `only_touches` 4 through the amendment's dotted-prefix exception and
`files_changed_count` 3, #121's own (a dotfile twin counted apart), each admitted by the table; none explained only by
a post-amendment rule. `compat2_candidate` flips 8, each one rule alone (F-2 six False → True, W-1 one each way); the
gate-level fields moved on 3 records, given back by F-2 (2) and Y-4 (1); 4 F-4 withdrawals. **G-C7**: 0 oracle
violations over the 3,475 records, the tenth pass's oracles (K-1 to K-5, the fold pinned and checked sound against this
interpreter) included; the parse oracles hold every excluded PR in corpus mode. **G-C1**: the never-read sentences,
the strict pass and the report (`to_dict`, base and head) compared, 0 violations. **G-C8**: every one of the 3,475
records tried, 1,140 rebuilt through `git fast-import` and scored through `gate_diff` (38 of them holding a dotted
path), 2,335 not rebuildable faithfully; 366 scored again with rename detection on, 2 renames detected; 226 moved, all
attributed; the two door canaries (U+0085 and U+2028 paths, where `main`'s `--name-status` split cuts the path and
Z-3 abstains at the git door) scored, 0 violations. Scorer `f09b6eb1…`, harness `75bfbc39…`, repaired `diffgate.py`
`0fc470c5…`, baseline `98a5c368`.

`path2_gates.py differential` at the ninth pass, from a clean tree at `8e89333a` (the scorer, the harness,
`styxx/diffgate.py` and `styxx/_xid.py` unmodified; this README and the CHANGELOG were not yet committed and
are not files the scorer's provenance reads): exit 0, every gate passes, no violation; 374 records moved, one of
them a record the baseline raises on (no verdict is admitted there). Claims attributed to #97 11, #121 31, #101
51, F-2 28, V-1 10, V-4 94, W-1 14, Y-1 16, Y-4 5, Z-2 1, Z-3 3, Z-4 115, Z-5 1, and to 31 sets (91 joint), among
them #97+Z-4 16 (#97 reverted resolves by basename, where Z-4 then abstains, so only the pair gives `main`'s
claim back), #121+Y-1 14, F-3+Z-5 12 and the four- and five-rule sets a U+FEFF-led test needs
(#101+R-1+W-1+Y-2+Z-1+Z-5 4, R-1+W-1+Y-2+Z-1+Z-5 3, #101+R-1+Y-2+Z-1+Z-5 3). New accusations 4, all
`only_touches` through the amendment's dotted-prefix exception; none explained only by a post-amendment rule.
`compat2_candidate` flips 8, each one rule alone (F-2 six False → True, W-1 one each way); the gate-level fields
moved on 2 records, both given back by F-2; 4 F-4 withdrawals. **G-C7**: 0 oracle violations over the 3,443
records, the ninth pass's oracles (`main`'s reading in both spellings, Z-1 to Z-5, `_status_notes` on every
record and on its case-folded variant, the git door's Z-3 reading) included. **G-C1**: the never-read sentences
compared as text and every record scored again with `strict=True`, 0 violations. **G-C8**: every one of the 3,443
records tried, 1,129 rebuilt through `git fast-import` and scored through `gate_diff` (36 of them holding a dotted
path), 2,314 not rebuildable faithfully; 366 scored again with rename detection on, 2 renames detected; 220 moved,
all attributed; 0 violations. Scorer `5847677a…`, harness `75bfbc39…`, repaired `diffgate.py` `e975d098…`,
baseline `98a5c368`.

`path2_gates.py differential` at the eighth pass, from a clean tree at `56320b90` (the scorer, the harness,
`styxx/diffgate.py` and `styxx/_xid.py` unmodified; this README and the CHANGELOG were not yet committed and
are not files the scorer's provenance reads): exit 0, every gate passes, no violation; 248 records moved.
Claims attributed to #97 27, #121 28, #101 51, F-2 16, F-3 4, V-1 13, V-4 94, W-1 15, Y-1 16, Y-2 7, and to
sets #101+A-1 2, #101+F-2+V-1+W-2 2, #101+V-1 4, #101+V-1+W-2 4, #101+W-1 4, #101+Y-2 3, #121+V-4 2,
#121+Y-1 10, F-2+V-1 4, F-2+Y-1 2, F-3+V-1+W-2 6, F-3+Y-3 5, V-1+W-2 4, V-1+W-2+Y-3 1, W-1+Y-1 6, W-1+Y-2 8
(38 joint); no claim is attributed to Y-4 or Y-5, alone or in a set. New accusations 16: `only_touches` 4
through the amendment's dotted-prefix exception and 12 explained only by post-amendment rules, for which
G-C3 is not asked (F-2 1, F-2+V-1 1, F-3 2, F-3+V-1+W-2 3, V-1 `symbol_added` 5). `compat2_candidate` flips
8, each one rule alone (F-2 six False → True, W-1 one each way); the gate-level fields moved on 2 records,
both given back by F-2; 4 F-4 withdrawals. **G-C7**: 0 oracle violations over the 3,416 records, declared and
file-list claims included; the scorer refuses to start unless the name table equals Python 3.12's 15.0.0
database code point by code point and the skew set is the one its sources give, each hashing to the sha256
the scorer pins. **G-C8**: every one of the 3,416 records tried, 1,076 rebuilt as repositories and scored
through `gate_diff`, 2,340 not rebuildable faithfully; 122 moved, all attributed; 0 violations. Scorer
`9eec3d6c…`, harness `75bfbc39…`, repaired `diffgate.py` `d7c298d4…`, baseline `98a5c368`.

`path2_gates.py differential` at the seventh pass, from a clean tree at `a6a215be` (the scorer, the harness,
`styxx/diffgate.py` and `styxx/_xid.py` unmodified; this README and the tests were not yet committed and are
not files the scorer's provenance reads): exit 0, every gate passes, no violation; 213 records moved. Claims
attributed to #97 27, #121 35, #101 26, F-2 18, F-3 4, V-1 11, V-4 94, W-1 18, and to sets #101+A-1 2,
#101+F-2+V-1+W-2 2, #101+V-1 4, #101+V-1+W-2 4, #101+W-1 7, #121+V-4 2, F-2+V-1 4, F-3+V-1+W-2 9, V-1+W-2 4 (4
joint). New accusations 20: `only_touches` 4 through the amendment's dotted-prefix exception and 16 explained
only by post-amendment rules, for which G-C3 is not asked (F-2 1, F-2+V-1 1, F-3 2, F-3+V-1+W-2 5, V-1
`symbol_added` 7). `compat2_candidate` flips 8, each one rule alone; the gate-level fields moved on 2 records,
both given back by F-2 where the splits differ; 4 F-4 withdrawals. New this pass: **G-C7**, the scorer's own
reading of every rule it re-implements held against the instrument on all 3,381 records, 0 violations (the name
table checked against Python 3.12's 15.0.0 database); **G-C8**, 150 records rebuilt as repositories and scored
through `gate_diff`, 15 moved, all attributed, 0 violations (350 of the 500 tried could not be rebuilt
faithfully). Scorer `b69f3fe5…`, harness `75bfbc39…`, repaired `diffgate.py` `67fb1b75…`, baseline `98a5c368`.

`path2_gates.py differential` at the sixth pass, from a clean tree at `4694aa99` (the scorer, the
harness and `styxx/diffgate.py` unmodified; this README was not yet committed and is not a file the scorer's
provenance reads): exit 0, every gate passes, no violation; 205 records moved. Claims attributed to #97
27, #121 35, #101 24, F-2 16, F-3 4, V-1 7, V-4 94, W-1 14, and to sets #101+W-1 7, #101+V-1 4,
#101+V-1+W-2 4, #101+F-2+V-1+W-2 2, F-2+V-1 4, F-2+W-1 2, F-3+V-1+W-2 6, #121+V-4 2 (4 of them joint: no
single revert gave the baseline back). New accusations 14: `only_touches` 4 through the amendment's
dotted-prefix exception, and 10 explained only by post-amendment rules, for which **G-C3 is not asked** —
F-2 `tests_added` 1, F-2+V-1 1, F-3 2, F-3+V-1+W-2 3, V-1 `symbol_added` 3, printed as
`G-C3_no_accusation_added.waived_for_post_amendment_rules` (the fifth pass waived 6 the same way and did not
say so). `compat2_candidate` flips 8, each explained by one rule alone (F-2 six False → True, W-1 one each
way); 4 F-4 withdrawals; 30 records whose splits differ, 21 whose parses differ. F-2 and W-1 now admit a
move only on a diff where they can act. Scorer `b8c09a5e…`, harness `75bfbc39…`, repaired `diffgate.py`
`b837f7e4…`, baseline `98a5c368`.

At the fifth pass, `path2_gates.py differential` attributed every moved claim by counterfactual: its own copy of the
repaired instrument, with one rule reverted, must give the baseline claim back (NOTE_path2_fifth_pass,
V-3). From a clean tree at `7f6d9303` (the commit
before this README's; it touches no file the scorer's provenance reads): exit 0, every gate passes, no
violation; 185 records moved; claims attributed to #97 27, #121 35, #101 26, F-2 18, F-3 4, V-1 3, V-4 93,
jointly #101+R-1 4, #101+V-1 4, F-2+V-1 4, #121+V-4 2; new accusations 4 `only_touches` (the amendment's
dotted-prefix exception), 2 `symbol_added` (V-1: definitions CPython refuses) and 4 `tests_added` (F-2,
F-3, V-1), each counted in the payload; 6 `compat2_candidate` flips, all False → True and all admitted
because F-2 alone explains them; 4 F-4 withdrawals. Scorer `522b1e87…`, harness `75bfbc39…`, repaired
`diffgate.py` `0a5522eb…`, baseline `98a5c368`. The gate can fail: on a synthetic three-PR shelf, with a
scratch copy of this `diffgate.py` carrying the round-3 blocker swapped in, one record touching only
`.github/` with a stray U+2028 in an unrelated context line, `path2_gates.py corpus` exits 1 with
`G-C4_direction:only_touches`; the fourth-pass scorer (`60d678a5`) admitted that move under F-2 and
exited 0. With the instrument unmutated the same shelf exits 0.

The earlier result, kept as it was measured: at the fourth pass, 3301 pairs, 7056 claims (668 verified,
1652 contradicted, 4736 uncheckable), 0 disagreements; 125 pinned pairs, 0; with the third-pass head on both
sides 9, with `origin/main` 11; a 242-input whitespace grid read 1 disagreement (the `\b` case, closed by
the fifth pass). Its `path2_gates.py` run (exit 0, 69 records moved) used the record-wide F-2 attribution
the fifth pass replaced. At the third pass: 3282 pairs, 7028 claims (653 verified, 1640 contradicted, 4735
uncheckable), 0 disagreements.

Result, 2026-09-18, this branch at the COMPAT-2 reading plus PATH-1:

    3220 pairs, 6933 claims (606 verified, 1616 contradicted, 4711 uncheckable) — 0 disagreement(s)

The 3,220 are the 3,176 below plus 44 pinned pairs committed as JSON (the `.gitignore` here
ignores generated JSON and names these five as exceptions): `bc1_pairs.json`, the four pairs BC-2
owes the differential (a TypeScript commit saying "Added 2 tests", "adds a method to reload",
"only modifies the footer", two prefixes), also checked on the Python side by
`tests/test_diffgate_bc1.py`; and `compat_pairs.json`, nineteen more — the compatibility
reading in every branch (Python, JS/TS, Go, Rust, Java and Kotlin, a moved definition, no
covered language, more than five names, all nine phrases, an empty diff), the V14 bare-name and
containment cases, "added 3 test cases", second prefixes that are and are not path-shaped, a
changed path with an apostrophe (Python's `repr()` switches to double quotes; so does the port),
and the demo diff with CRLF line endings; and `bin1_pairs.json`, six for the #118 repair — a
binary beside a text file, three binaries added / modified / deleted, a pure rename and a mode
change, a file count that is true only once the binaries are seen, an `only_touches` lie hidden
behind a binary, a quoted path — also checked on the Python side by `tests/test_diffgate_bin1.py`;
and `compat2_pairs.json`, seven for the COMPAT-2 reading — a surface drop beside a scaffolding drop
and a signature change, scaffolding-only removals, a move with the same parameters, a Go signature
re-flowed over two lines, an exported arrow function's parameters, Java's name-then-paren regex,
seven surface drops beside one under `internal/`. Their `expect` blocks are the Python's output,
written down so a reader can see the intended readings without running anything. The 3,199
pre-BIN-2 records were byte-identical before and after that repair (G-BIN-2); of the 3,205
pre-COMPAT-2 records, 3,196 are byte-identical after it and 9 differ only in `compat_claim`
reasons and details (17 claims), which is COMPAT-2's G-C2-4.

`path2_pairs.json` carries the PATH-2 pairs — two README files in either order, a suffix match
beating an earlier basename, an exact match beating an earlier suffix, the basename fallback, a
dotfile and its undotted twin, "only touches github/" over `.github/` (abstains on the dot), two
prefixes and leading slashes, a dotfile beside a nested undotted name, a dotfile outside the
prefix, the #101 issue diff, a new test beside a changed one at three claimed counts, a changed
`def` beside a fresh one elsewhere, a rename, a test moved between files (counts as added, the
disclosed limit) and counted cases over a changed test; and, for the amendment, the same test name
in two classes, a changed test beside a same-named new one, the shelf's fold under an `A` and an `M`
header, a BOM strip on a test and on a function, non-ASCII test names, a non-ASCII suffix, a
same-named method added in another class and one changed in place, a changed `def` under an `A`
header, a generic `def`, a dot miss beside a real outside path (one and two prefixes), a dotted
prefix over an undotted file and directory, a `..` path, binary dotfile twins with no hunks, a pure
rename to a dotted name, a file named `.py`, and the `.storybook/` COMPAT-2 reading; and, for the fourth
pass, a slashless dotted prefix whose suffix the extension list does not hold in the three positions it
can take (VERIFIED, a real outside path, C-3's accusation), a test re-indented by U+000B, U+000C,
U+2028 and U+2029, a line separator inside an added line, inside a context line and before a
header-shaped fragment, a form-feed indent that does define a function and a line-separator indent the
symbol test still read as one (a disclosed limit, pinned so it could not move unseen; the fifth pass
re-pinned it, and the line now defines nothing), an NBSP and an
ideographic-space re-indent, and an off-tree prefix beside an on-tree one read three ways; and, for the
fifth pass, the definition-line class (a form-feed re-indent of a function, a form-feed-led new test, a
vertical-tab re-indent, NBSP-, U+0085-, U+001C- and U+FEFF-led definitions, a definition after a
mid-line U+2028, a form-feed separator, a changed generic class, a name followed by a non-ASCII letter
or a middle dot, an added `async def` alone and beside a changed `def`), the port's COMPAT reading in
every language with U+0085 in a `\s` position and the U+001C, U+001F, U+FEFF, `\w` and `\b` cases, a
header path's strip, binary headers holding U+2028, `repr()` of a printed path, a prefix written to end
in `..` or holding `..` after a named segment, and F-4's two boundaries — also checked on
the Python side by `tests/test_diffgate_path2.py`. Their `expect` blocks are the Python's output, written down so a
reader can see the intended readings without running anything. The 3,199 pre-BIN-1 records are
byte-identical before and after that repair (no binary in the corpus), which is its G-BIN-2.

The first result, the 7.47.0 port against the 7.47.0 file, was 3176 pairs, 8904 claims
(1024 verified, 2172 contradicted, 5708 uncheckable), 0 disagreements; that port is in this
directory's history.

The corpus is pinned so it is the same bytes everywhere: the 160 commits reachable from
`fa8bcde725252dd6264d89045fa58fce7f3d9e51` (each message against its own diff), three
pull-request descriptions written that day (`bodies/pr94.md`, `pr95.md`, `pr98.md`, each against
its branch head at the time — shas in `build_corpus.py`), the README demo, twelve edge cases from
the test suite, and 3,000 fuzzed pairs from `random.seed(20260916)`. The fuzzer aims at the
template set's soft spots on purpose (quoted and bare paths, Node.js-style non-file nouns,
"the same way x.py was", bullets, counts of zero, empty diffs, text that is not a diff), because a
port that agrees on easy input proves little. The figure posted the same day, 3,177 pairs, had one
more pair: an external project's PR description, left out of the committed corpus because it is
not ours to republish. It carried zero diff-shaped claims, so the claim count is the same 8,904.

## Drift: the port is ahead of the release

7.48.0 is on PyPI (released 2026-09-25 from `main`), and it ships `main`'s `styxx/diffgate.py`, sha256
`9b620e00…` (LF) — the wheel built at the cut carries that file — which this branch changes. So
`pip install styxx` gives a file the port does not match, and `python py_side.py --installed` against it
disagrees by the PATH-2 repairs: on the seventh pass's 3,381 pairs, 213 records with the port on one side
and 7.48.0's file on the other (the same file as `main`'s, measured in section *The differential test*); on
the eighth pass's 3,416, 248; on the ninth pass's 3,443, 375 (the sixth pass measured 205 on its 3,366).
When PATH-2 merges and a release carries it, `--installed` should read 0 again.

The measurement below is the older one, kept as it was taken: 7.47.0, the release before, whose file the
port had already moved past on 2026-09-16:

    3199 pairs, 8932 claims (1031 verified, 2189 contradicted, 5712 uncheckable) — 5448 disagreement(s)

What moved, on the 3,176 corpus pairs (the pinned pairs were written for the new file and are
left out of these counts; the #118 repair changes none of the 3,176 records, and COMPAT-2 changes
only `compat_claim` reasons, a kind the wheel never reads, so the figures below are unchanged by
either): the wheel makes 573 accusations the port does not — `symbol_added`
219, `only_touches` 205, `tests_added` 149 — and the port makes none the wheel does not. 286
pairs flip from FAIL to PASS and none flip the other way. The port reads 2,044 fewer
`file_touched` claims (the V14 repairs declining bare and ambiguous path mentions) and one more
sentence kind, `compat_claim`, which the wheel never reads (one sentence in this corpus carries
it). Every one of the 573 now ends in a reason citing #110 — 368 "no Python file in the diff;
this template counts `def` lines", 205 "prefix '…' is not a path" — because every one was an
accusation the issue showed to be unsupported by construction. On the EXTERNAL-1 corpus the same repair withdrew 569 of
665 (`papers/closed-model-frontier/RESULT_bc2_by_construction_lands_2026_09_16.md`).

That sentence used to end: when 7.48.0 is on PyPI, `--installed` should print 0. It could not: PATH-2
did not merge before 7.48.0 was cut, so 7.48.0 is `main`'s file, not the one the port claims to be. The
header of `diffgate.js` says which one the port is.

## What this is not

It is not a second instrument. The Python is the instrument; the JavaScript exists only where the
Python cannot run, and the day the two disagree on a pair in this corpus, the JavaScript is wrong
by definition. It is not a check of anything but transliteration fidelity — the template set's
own precision was measured elsewhere (EXTERNAL-1: path accusations at 0.23 against a 0.95 floor
on 71,016 agent-authored PRs, and withheld since), and no number here bears on that.
