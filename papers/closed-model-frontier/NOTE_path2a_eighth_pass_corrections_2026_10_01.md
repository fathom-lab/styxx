# NOTE — PATH-2a, eighth pass: two corrections to the pass's note, after its code landed

2026-10-01. `NOTE_path2a_eighth_pass_2026_10_01.md` was committed alone at `8c3b5c6b`, before the code (`9eb93b04`), the
tests (`838f59cf`) and the README, CHANGELOG and bookmarklet (`9a1b40f2`). The committed code does not depart from it.
Two of its sentences do not match what landed; it is not edited, and this note records them.

1. **B-2's pins (§2).** The note says "Two pinned pairs." There are three:
   `path2a:p8-an-accent-outside-the-window` and `path2a:p8-a-count-sentence-decorated-after-changed` (withheld), and
   `path2a:p8-an-accent-inside-the-window-keeps-contradicted` (kept in both ports). `path2a_pairs.json` holds 136 pairs.

2. **The PR's size (§5, I-6).** The note gives "about 20,000 at this head". At `9a1b40f2`, `git diff 1cde8b82 HEAD |
   wc -l` reads 20,015 lines (35 files, 19,604 insertions, 33 deletions; this pass alone 13 files, 2,261 insertions,
   270 deletions). That is above the 20,000 lines GitHub accepts for a pull-request diff, so the repository's diffgate
   job would print DID NOT RUN on this pull request and exit 0 without `--strict`. The CHANGELOG's 20,013 was measured
   before its own last lines were written. Compacting the two JSON pin files (`path2a_pairs.json`,
   `tests/fixtures/path2a_repros.json`, about 5,000 lines between them) or splitting the pull request is the operator's
   call; this pass does neither.
