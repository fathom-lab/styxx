# NOTE — PATH-2a, tenth pass: where the pass's note and what landed differ, and what was measured, after its code

2026-10-05. `NOTE_path2a_tenth_pass_2026_10_05.md` was committed alone at `bb9f6131`, before the pass's code (DECIDE
and APPLY, the lints and their tests at `1261535b`; the hostile inputs strengthened at `094496ac`). The committed code
does what that note says. This note records where a sentence of it does not match what landed, and the figures it
left to be measured at the head; the earlier note is not edited.

## 1. Departures

1. **The hostile inputs (§2, Tests).** The note says the hostile DECIDE functions run "over #161's reproductions in
   both strict modes". As committed, every other reproduction carries one more sentence, one that `main` reads as a
   `tests_pass` claim. A mutation check of `1261535b` found why that is needed: an APPLY edited to look a decision up
   for any index, in reach or not, passed every hostile test in both ports; only the pin of APPLY's text failed.
   APPLY checks reach twice (when it takes a decision, and again when it writes, which is what makes a second decision
   for one claim a no-op), so one of the two checks alone still holds the relation, unless the record holds a claim of
   a kind the overlay has no tag for. With such a claim in the inputs, the edit raises in Python and falls back in the
   port, and the hostile tests fail on it.
2. **`int` in the Python lint (§3).** The note lists `int` among the names the lint refuses. APPLY reads a decision's
   index with `isinstance(i, int)`, so the lint as committed refuses `int` except as the type `isinstance` tests; two
   plants pin that `int(...)` is still refused beside it.
3. **The cross-port cases (§7, C-2 and I-3).** The note says the reviewer's inputs become cross-port cases. Eleven
   were added, ten for the reviewer's eleven one-port edits (one input tells two of them, one in each port) and one for
   the defect tag, so there are 84; and twelve one-port plants, each named with the case that tells it.
4. **The pins of bar C(iii) on a runtime not measured (§7, C-3).** As the note says, and exercised more widely than
   it promised: under emulation with an engine reporting Unicode 15.1 and one reporting 17, beside CPython 3.12.10
   and 3.14.2. On all four pairs the three seeded sets pass, and the patched-engine set's figures equal one of its
   two pinned rows exactly: the 16.0.0 row where the two runtimes fold U+A7DC alike, the 15.0.0 row where one does.
   No pin was added for CI's Node; that stays with the operator (I-2).

## 2. Measured at the head

**No decision moved.** The records of `1261535b` against `c69b161b`'s two files: 0 differing in 13,344 Python runs
and on 6,672 port inputs (both strict modes) over the committed inputs; 0 differing on the 160,000 adversarial inputs
of the ninth pass's sets, in 319,986 Python runs and on all 160,000 port inputs, and on the 19,000 read again under
the patched engine. Every counter of that adversarial run (the relation, bar C, the claims withheld) equals the ninth
pass's.

**Bar A at run time.** 26 hostile DECIDE functions in Python and 24 in the port, 958 and 962 runs each: 0 records
outside the relation. APPLY takes 2,633 of the 2,633 decisions the block's DECIDE returns on the committed inputs.
The new tests run against `c69b161b`'s two files: 49 of the 51 selected fail; the fourteen plants of the ninth
construction review fail there with a claim dropped, added or rewritten in `main`'s record, and pass at the head.

**The mutation check**, in a scratch archive of the head, one edit at a time. Eleven edits of the Python's APPLY or
DECIDE and eleven of the port's (a decision taken for any index, written twice, with any phrase, with any tag; the
record itself, or a claim's own detail, handed to DECIDE; no `try`; a non-list ignored; the gate verdict not
recomputed; a copy returned, in Python; a decision read twice, in the port; a tag APPLY refuses): each fails a
behaviour test as well as the pin, the "any index" pair after item 1 of §1. Three edits of the lints (escapes not
decoded, bitwise operators passed, `int` passed anywhere): each fails its plants. The twelve one-port plants, each
planted in the archive: `test_cross_port_reproductions` fails on every one. For the review's ten port edits it is
the only cross-port, bar C or truth test that does, which is the review's finding seen from the other side (its one
Python edit also trips a truth test whose own plant anchors on the edited line; the tag plant fails six tests). The
nearest-pin rule set back to `c69b161b`'s under the emulated Unicode 15.1 engine: the patched-engine set fails, as the
review found.

**The EXTERNAL-1 shelf** (§5), at the head, the database opened read-only and immutable. 71,104 pull request ids
have a body that is not blank; 2,003 of them have no file row with a name and 47 are skipped (a rebuilt diff over
3 MB); 69,054 are read, each once. 25,316 of those have a file in more than one row and 11,702 a file whose rows
disagree in status. `main`'s parser reads 69,053 of the rebuilt diffs back as the folded listing. 138,108 Python
runs: 0 outside the relation, no raise. 7,836 pull requests hold a decided claim in reach, 12,785 claims; the
overlay withholds 161 (`dir` 136, `tier` 13, `dot` 1, `extract` 4, `redefined` 6, `tests` 1); the port 161 of 12,768.
A committed variant of `main` reads 152 of the 12,785 otherwise, and all 152 are withheld. Bar C: `main`'s two lists
are the same on 69,035 (108 with a withheld claim) and differ on 19, all on the description side; the gate verdicts
differ on none without `--strict` and on 4 with it, under `main` and under the overlay alike. The same run on
`c69b161b`'s files gives the same figures. For the record, the three-matching reading against the folded listing,
which the README does not print (§5 of the note says why): 143 claims read otherwise than `main` by exact, suffix and
base-name matching alike, 118 of them withheld, the other 25 read by no variant otherwise; 7,355 read as `main` does,
2 of them withheld; 5,287 not judged.

**From the coverage review.** At the head, the reviewer's scripts give the reviewer's figures: on the decorated world
(seed 9103, 20,400 inputs) 12,559 of 12,559 attributable false verdicts withheld in Python and 11,521 of 11,521 in
the port; against the parent head `3bdc3bc4`, 903 more false CONTRADICTEDs withheld and 1,812 more right ones lost,
with 796 attributable verdicts kept at the parent and none at the head; on the committed truth world, 8 false tests
verdicts of the defined-again kind kept at three lines of context, 6 more at context 0 and 1, and 12 withheld as
`redefined`.

**Recall.** `main`'s committed corpora 80 of 2,231 under both path flavours; with #161's pairs 280 of 2,761; the
overlay's own pins 90 of 155 (Windows flavour) and 89 of 154 (POSIX).

**Cost.** The module body: 27.5 ms (least of 25 runs in turn) against `main`'s 5.1 on CPython 3.12.10, ×5.4, and
28.0 against 5.3 on 3.14.2, ×5.3. `import styxx.diffgate` in a fresh interpreter: 67 ms against 18 (×3.8) and 76
against 21 (×3.7). The copy APPLY makes: 5.5 to 5.8 ms for 10,000 claims in Python, 9 to 10 ms for 30,000 in the
port. The committed timing cases' slowest overlay 0.161 s (3.12.10), 0.132 s (3.14.2), 50 ms (Node); the rest is in
the README's *Cost*.

**Tests.** The three PATH-2a modules: 455 passed on CPython 3.12.10 and 455 on 3.14.2 (terser on the path on both;
3.14 through the earlier passes' bare-package plugin, since that interpreter has no numpy here).

## 3. Process

Three scratch scripts of this pass were written through shell heredocs, against the task's rule for files holding
backslashes. Two failed before writing anything (one at the shell, one at its own assertion); the third wrote a
scratch driver that did not parse. Each was rewritten as a file. A scratch fuzz run given a relative output directory
stopped after three sets and was run again for the rest. A test helper's variable carried a word the project's text
rules forbid and was renamed before the commit. Nothing outside the worktree and this pass's scratch directory was
written; nothing was pushed or fetched; the secrets directory was not read.
