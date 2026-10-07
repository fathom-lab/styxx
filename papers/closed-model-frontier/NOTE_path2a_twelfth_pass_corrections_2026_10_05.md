# NOTE — PATH-2a, twelfth pass: where the pass's note and what landed differ, and what was measured, after its code

2026-10-05. `NOTE_path2a_twelfth_pass_2026_10_05.md` was committed alone at `b35f6a1c`, before the pass's code (the
port's APPLY and the comments at `602fc819`, the tests and the harness at `30aa1d4a`, the bookmarklet at `5598885d`,
the README and the CHANGELOG after them). The committed code does what that note says. This note records where a
sentence of it does not match what landed, and what was measured at the head; the earlier note is not edited.

## 1. Departures

1. **The cost figures (§4 g).** The reviewer's three shapes are given in the README as a paragraph below the
   line-heavy table, not as rows of it: the reviewer measured `main`'s call against the branch's whole call, and the
   table gives the overlay alone.
2. **The Python's pinned text moved too.** The sentence on `strict` (§4 e) is in APPLY's docstring, which the pin of
   APPLY's text covers, so the Python hash moved with the port's (76 and 78 lines, under the bound of 80).
3. **Test edits the note does not name.** One committed port plant was anchored on APPLY's `reach.push(` line and now
   anchors on the index store; the port's hostile test asks for at least seven one-realm cases, not five; the port
   checker's loader runs a page's script in the realm before the port loads (for the new case), and `--tables`
   reports the literal key list; the eleventh pass's single-prefix test is renamed
   `test_the_single_prefix_family_is_kept_and_counted` and reads each case in both strict modes.
4. **The pins pass on `ee82d2f3`'s code** (§2), as that note says they would: no rule changed. The new tests that fail
   there are the two one-realm cases (604 of 962 runs outside the relation each, against `ee82d2f3`'s port).

## 2. Measured at the head

**No decision moved.** Against `ee82d2f3`'s two files: 0 records differing on the committed inputs (13,344 Python runs;
6,672 port inputs, both strict modes) and on the six adversarial sets the eleventh pass read again (62,000 inputs, no
record differing in either port; 0 of 124,000 Python runs and 0 of 106,000 port runs outside the relation, the
patched-engine set's port relation not read).

**The mutation check**, in a scratch archive of the head, one edit at a time: an `__init__` put back on `_P2aSeen`; a
method put on `_P2aClaim`; the key list built by `for … in` again; the per-claim arrays filled by `push` again; the
literal list one key short; `.toSorted(` let through the scan; the general rule of O-10 planted in both ports. Each
fails a named test. The general rule moves the committed truth world by 98 right verdicts withheld in each port and
nothing else (right verdicts lost 253 → 351 in Python, 177 → 275 in the port), the eleventh coverage reviewer's figure.

**Tests.** The three PATH-2a modules: 489 passed on CPython 3.12.10 and on 3.14.2 (through the earlier passes'
bare-package plugin). `build_bookmarklet.py` gives `6bf6121a…`, 53,371 characters; the build equals the port on
14,308 runs. Recall: 80 of 2,231 under both path flavours, 280 of 2,761 with #161's pairs, 90 of 155 and 89 of 154 on
the overlay's own pins. A scratch pull-request body, checked with `gate_diff_text` against
`git diff origin/main...HEAD` under `main` and the branch, as written, with CRLF and stripped, in both strict modes:
PASS, 0 claims.

## 3. Process

Every file changed was searched for the words the house rules forbid before each commit. One scratch edit of the
README went through a Python script in a shell heredoc that held no backslash. No `sed -i` was used. Nothing outside
the worktree and this pass's scratch directory was written; nothing was pushed or fetched; the secrets directory was
not read.
