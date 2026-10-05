# NOTE — PATH-2a, ninth pass: where the pass's note and what landed differ, after its code

2026-10-04. `NOTE_path2a_ninth_pass_2026_10_04.md` was committed alone at `2448ab98`, before the pass's code (the
JSON pins rewritten one record per line at `5a542223`, the removal and its tests at `1a01eb57`, more tests at
`cccee803`). The committed code does what that note says. Five of its sentences do not match what landed or what was
measured afterwards; the note is not edited, and this one records them.

1. **Lines removed (§1, item 3; §5, P-5).** The note gives the probe's count, 282 lines of the Python block and 228 of
   the port's. The removal as committed takes 284 and 229: the same definitions, with two more blank lines in the
   Python and one in the port; and the three names `apart`, `kinds` and `names` are out of the self-check's attribute
   list as well, which the probe's deletion left in.

2. **What was measured again at the head (§7, last paragraph).** The note says the adversarial fuzz sets and the
   EXTERNAL-1 shelf would be quoted as the probes'. Both were read again at the head, by the builder: 160,000 inputs
   from the earlier passes' and reviews' generators (not the probes' 354,675, whose generators the builder did not
   all have), in both ports and both strict modes; and the shelf, 69,058 bodies, in both ports. The README's figures
   for them are the head's. They agree with the probes' wherever the two read the same thing: on the shelf 152 of
   11,859 decided claims in reach withheld, 19 bodies whose `main` lists differ, all on the description side, 0 gate
   differences without `--strict` and 7 with it; and the same counts on each of the ten sets both generated the same
   way (the two hostile sets, the two seam sets, `hx`, `gen7`, `gen8`, the line-break set, the case-skew mix and the
   decorated world). The builder's 9,000 inputs from the seventh review's `fuzz7` generator are not the probe's 9,000.

3. **A gap the mutation check found (§4, I-2; §6).** The note says the U+2028 and U+2029 starts of `_p2a_counted` and
   `_p2a_anchored` "are checked by planting in this pass". They were, at `1a01eb57`, with thirteen one-edit mutants of
   the two instrument files in a scratch tree. Eleven were caught. Two were not: with those starts dropped from
   `_p2a_anchored`, or from the port's `_p2aAnchored`, every committed test passed, because no committed input held a
   removed or unchanged line with a definition after one of them. `cccee803` adds three cross-port cases that hold
   them (a removed `def` after U+2028, an unchanged one after U+2029, an added `def test_` after U+2028 that only the
   port's `main` counts) and four plants, two per port; the four mutants are caught at that commit.

4. **The cross-port cases (§6).** The note names the 26 that move and the pins for O-11 and for the eighth review's
   scope input. Eight were added in all, so there are 73: two for O-11, the scope input, two for the decorations of
   P-4 (a U+FEFF before the summary; styxx beside U+2028), and the three of item 3.

5. **The pins of bar C(iii) (§6).** The note says "pinned for each runtime tuple measured here ... within stated
   bounds elsewhere". As committed: exact on CPython 3.12.10 (Unicode 15.0) and 3.14.2 (16.0) against Node 24.13.0,
   under both path flavours for the committed inputs; on any other runtime each figure must lie within a tenth of
   the measured one, or within 10, and is printed. The abstention pins, which the note does not mention, were also
   read again under the earlier passes' emulation of Unicode 13.0 (the inputs' five Unicode 14 letters, U+2C2F, U+2C5F,
   U+A7D3, U+10570 and U+10597, mapped to code points no version assigns): no figure moved.

One process slip, in scratch: a helper script written through a shell heredoc lost half of its doubled backslashes
and wrote a broken `tests/test_diffgate_path2a_truth.py` into the working tree; it was rebuilt from a script written as
a file before anything was committed, and no committed file was affected. And one observation outside the PATH-2a
tests: on this Windows machine `tests/test_ledger.py` left `papers/LEDGER.md` modified in the working tree (CRLF line
endings) when it was run beside them; the file was restored with `git checkout` and is unchanged in every commit of
this pass.
