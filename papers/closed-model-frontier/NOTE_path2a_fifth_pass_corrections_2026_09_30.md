# NOTE — PATH-2a, fifth pass: where the code departs from the pass-5 note

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`. `NOTE_path2a_fifth_pass_2026_09_30.md` (commit `0f791265`) was
written and committed before this pass's code, as the process requires. Carrying its designs into the code and its
tests turned up the departures below. The pass-5 note is not edited. This note is committed alone, before the code it
describes. Its figures were measured on the scratch copy that becomes the code commit (CPython 3.12.10 and 3.14.2,
Node 24.13.0).

## 1. C-1: the count seam is narrower than the note's

**What the note said.** Any of ten code points anywhere in the summary is a seam: the six white-space code points
only one port's `\s` reads (U+001C to U+001F, U+0085, U+FEFF), and the four CPython's IGNORECASE folds to an ASCII
letter (U+0130, U+0131, U+017F, U+212A).

**What that cost.** The builder's own truth world writes U+212A (KELVIN SIGN) in paths of its case-outside-ASCII
family. With the note's set, 37 right count verdicts were lost there, in both ports. They were caught by no rule of
#121 and are no count the two ports read apart. The note's prediction for the truth pins ("0 misses; the counts at the
head go into the README") held, but its cost was not stated.

**What the code does.** The seam is exactly where a count match can pass through a character read apart:
- **White space.** A count reads `\s+` in three places: between the number and `file`, between `file(s)` and
  `were`/`changed`, and between `were` and `changed`. On the left of each run of spaces stands an ASCII digit (a number
  outside ASCII is already withheld, `extract`), or `e` or `s` in either ASCII case, or U+017F. On the right stands `c`,
  `f` or `w`, in either ASCII case. So a white space of one port alone is a seam only inside a maximal run of either port's white space
  whose neighbours are those. The run is found in one pass (`_p2a_seam`; the port's `_p2aSeam`), not by a
  backtracking pattern.
- **Letters.** IGNORECASE reads a code point outside ASCII as a letter of the count only where the count's words hold
  that letter. The words `files`, `were` and `changed` hold `i` and `s` and no `k`. So U+0130 and U+0131 matter only as
  the `i` of `file`, U+017F only as its `s`, and U+212A never. The seam is `file` spelled so:
  `[Ff][U+0130 U+0131][Ll][Ee]` or `[Ff][Ii U+0130 U+0131][Ll][Ee]U+017F`.

Constants: `_P2A_ONE_SPACE` (the six), `_P2A_ANY_SPACE` (either port's white space), `_P2A_FILE_FOLD_RX`. They replace
the note's `_P2A_COUNT_SEAM`.

Tests pin, by enumeration on the running interpreter and engine:
- the six as the symmetric difference of the two ports' pinned `\s` tables;
- the four fold code points with the ASCII letters each matches (`i`/`I`, `i`/`I`, `s`/`S`, `k`/`K`);
- that the count template's words, read from `main`'s own source, hold `i` and `s` and not `k`;
- that the port's IGNORECASE matches none;
- nine example summaries, each with its answer.

**Measured.** The reviews' X2-13, G1, G2, G3 and the pinned X2 still give `seam` in both ports and one gate verdict.

On the inputs pass 4 pinned, only the X2 pair moves (`count` to `seam`), and the truth pins are those of `5ebe0b6b`
exactly:
- raw door: 245 right lost;
- port: 169 right lost;
- git door: 0 right lost.

The fuzz sets:
- The reviews' five generators (34,000 inputs) and a count-seam generator (`gen_seam`, 20,000 inputs over twin diffs)
  give 0 claim splits under every key of the committed `cross_port`, and 0 gate splits where every claim pairs.
- On the count-seam set, where `main`'s gates agree, the overlay's gates split on 0 inputs; at `5ebe0b6b`, on 247.

**Soundness, argued.** A count read by one port alone, with a clean number, needs a `\s+` of the count's own that
holds a white space of that port alone, or a letter of `file` that only CPython folds. The first is bounded by the
neighbours above. The second is spelled as above. Every other way the templates part (`\b` and `\d` beside a code
point from 0x80 up) leaves a number outside ASCII, or a wordish code point before the number, which pass 4's guards
already withhold in both ports.

## 2. The plants asked for pins the note did not name

Each rule this pass adds, dropped alone, must be caught (`test_plants_are_refused`). Three were not caught by the pins
the note named, and one port plant did not split the ports. They get these pins:
- **A counted name read through NFKC** (B-2, the reverse of t-nfkc: base `def test_a`, head `def test_` + FULLWIDTH
  LATIN SMALL LETTER A). A pinned pair `path2a:p5-t-nfkc-added`, and a case in the truth fixture.
- **The case doubt's base-name fact, and its suffix fact** (A-2). Each alone decides a claim only `main`'s Python reads:
  a base name outside ASCII, which the port's path template cannot read. Neither fits a pinned pair, which both ports
  must match. They are pinned in `XPORT_CASES` (`P5-case-base-name-only`: `y/dé.md` against `x/dÉ.md`;
  `P5-case-suffix-only`: `é/dé.md` against `z/É/dé.md`). `_pairs_catch` in the plants test now also reads every
  `XPORT_CASES` entry's Python decisions.
- **The count seam, dropped from the port alone.** The seam pairs read different counts in the two ports, so the
  plant split no claim that `main`'s two ports read alike. A pinned pair `path2a:p5-count-seam-without-a-second-count`
  gives both ports one count and a seam elsewhere ("files" U+001F "collapse").

So `path2a_pairs.json` gains 20 pairs, not 18. The truth fixture `tests/fixtures/path2a_pass5_repros.json` holds 11
cases: the review's ten and the reverse NFKC case, judged at three doors (33 claims).

## 3. Smaller departures

- **C-2, the port.** `Number`, `parseInt`, `parseFloat`, `isNaN` and `isFinite` are refused as whole words in code
  (`JS_BANNED_WORDS`), outside comments and strings, rather than as substrings of the block. So `_p2aNumbers` and a
  comment naming them are not refused, while an alias (`const p = parseInt`) is.
- **C-2, the widened count head.** The note said a committed case shows the error fallback withholding there. The case
  gives a count claim, by hand, a reason `main` never writes (a U+3000 before the count). In both ports the unplanted
  overlay reads it `unparsed`, and a copy with the widened head raises in the digit table and withholds `error`. The
  port side runs through a new `check_path2a.js --abstain` mode, which applies a port's overlay to a given record.
- **C-5 / I-4.** `check_path2a.js --bookmarklet` is the stub-page check. `--tables` also reports the port's
  one-port white space, its IGNORECASE folds (none) and the strings its digit table refuses.
- **B-2's cost, on the committed inputs.** 25 claims move through NFKC names. 24 of them lie in the seeded PATH-2a
  fuzz, and 3 of those 24 also hold joined lines. #161's y5 is the 25th. No committed input moves through the joined
  reading alone.

## 4. What does not change

Everything else in the pass-5 note stands: the dispositions, A-1, A-2, A-3, B-1, B-2, I-1, I-2 and the minors. Its
predictions stand too, with §1's figures in place of its C-1 figures.
