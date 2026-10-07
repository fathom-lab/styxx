# NOTE — PATH-2a, sixth pass: where the committed code departs from the pass-6 note

## 0. Status

2026-09-30. Branch `fix/diffgate-abstain-where-wrong`. `NOTE_path2a_sixth_pass_2026_09_30.md` (commit `aae04710`) was
written and committed alone before this pass's code, as the process requires. Carrying its designs into the code and
the tests turned up the departures below. The pass-6 note is not edited. This is the last pass of the workflow, so
there is no next pass's note to carry them: they are recorded here, committed alone, after the code they describe
(commits `174215e0`, `c964ebb2`, `d9db4af7`, `1de43460`, `a3cc622b`). Figures were measured at `a3cc622b` (CPython
3.12.10 and 3.14.2, Node 24.13.0).

## 1. A-1: the tokens, and the port's scan

- **The note:** `_p2a_abstain` names the tokens of the claims in reach; `_p2a_found` reads more than 32 of them through
  one automaton.
- **The code:** `tokens` keeps only the words with no character the two ports read apart (`_P2A_BAD_RX`). Those are the
  only words a claim looks up there: a path or a name reaches the lookup only in ASCII, a count only as ASCII digits,
  and a scope prefix only without such a character. So every named word is text of the Basic Multilingual Plane outside
  the surrogates, and the port's `_p2aFound` reads the text by UTF-16 unit, where the Python reads code points, and
  finds each word at the same places.
- **The port's speed:** the unit tables are packed arrays of booleans, and the scan skips units no word starts with
  while at the root.
- **The found set is the same either way,** and no decision moved.

## 2. A-1: the port's timing bound

- **The note:** the large cases are bounded within twice `main`'s call in the port.
- **The code:** the port reads them at three times the size (3.6 MB, 30,000 claims) and within three times `main`'s
  call. At twice, one run of the full suite on CPython 3.14.2 failed on a busy machine, where the overlay read 1.3 times
  `main`'s call alone. `495d2204` reads 8.6 to 9.6 times there and fails the bound either way.
- **Python** is as the note says: 1.2 MB and 10,000 claims, within `main`'s call.
- The pass-6 note's §5 cost prediction ("within twice it (port)") is replaced by this.

## 3. Pins the plants asked for

Each rule this pass adds or keeps, dropped alone, must be caught (`test_plants_are_refused`,
`test_port_plants_make_the_ports_disagree`). C-1 keeps CONTRADICTEDs where the two ports' views count apart, and that
emptied some earlier catches. The fixes:
- **`split`.** Its committed inputs had all been CONTRADICTEDs whose two views count apart. A pinned pair,
  `path2a:p6-split-where-the-views-count-alike`, keeps it reachable: both views count one site and name it apart.
- **A port reading a scope claim's text.** Its catches had been CONTRADICTED scopes. A pinned pair,
  `path2a:p6-a-long-scope-sentence`, gives it a VERIFIED scope whose text `main` cuts at 160.
- **The port's `_P2A_OWN`.** It now decides only claims one `main` alone decides so. It cannot split a claim both
  `main`s read alike: where the views count apart a CONTRADICTED stands, and a VERIFIED is read by one `main` only. The
  port plant test therefore also catches a plant that moves the port's own decision on `PORT_ONLY_CASES` (a created
  test after U+FEFF, "Added 2 tests.", VERIFIED in the port only), against the real port.
- **C-1's diff part.** A pinned pair, `path2a:p6-apart-a-test-one-port-counts`: a created test after U+FEFF, so
  main's two ports count 1 and 2.
- **C-1's fence part.** A cross-port case, `C1-a-fence-only-the-port-opens`, where only the port's `^` opens a
  DECLARE-1 fence after U+2028. The port plant test now reads the cross-port cases' inputs too.
- **Unrelated to a plant:** B-3's rename pair (`path2a:p6-a-claim-on-a-moved-files-old-path`) and B-1's symbol pair
  (`path2a:p6-name-defined-again`) are pinned as well.

So `path2a_pairs.json` gains 7 pairs (126 in all), not the 2 the note named.

The plant `symbol sites read anywhere in a removed line` names more context, since its old text now also occurs in
`_p2a_name_defs`.

## 4. Smaller departures

- **The inherent case.** It is pinned in its own form. `C1-inherent` is `Added 0 tests. 9<U+FEFF>files changed.` over a
  changed test: the port reads two claims and the Python one, each kept.
- **The committed inputs** are 6,662 (7 new pairs), not 6,655. The abstention pins move as the tests' comment says:
  `redefined` 17, `again` 6, `split` 1, `divergent` counts 70, scopes 3, `extract` scopes 4, counts 1, `seam` 1, `tests`
  345.
- **The node checker** gains `--opts` (A-2), `--found` (A-1), gate verdicts under `--strict` in `--decisions` (C-1,
  C-4), and `main`'s own call time in `--overlay-timing`.

## 5. What does not change

Everything else in the pass-6 note stands: the dispositions, the C-1 rule and its proof, B-1, B-2, C-2, I-1 and the
minors, and its predictions but the one in §2. The suites pass on CPython 3.12.10 (365 passed, 1 skipped) and 3.14.2.
