# ERRATUM — AMENDMENT_path2_resolution_2026_09_17: three statements corrected, one pre-code edit disclosed

Fathom Lab · 2026-09-17 · Beside `AMENDMENT_path2_resolution_2026_09_17.md`, which is not edited.
Written after the code of this round, from re-reading the amendment against the round-1 review
record. No rule, gate or limit in the amendment changes. Every correction below is to a sentence
that describes the review or the patterns.

## 1. The count of review findings

The amendment says "The review found one blocker, three majors and eight minors." As the three
reviewers labelled them, the record holds 1 blocker, 5 majors and 11 minors. Merging findings that
name the same defect:

- **1 blocker.** R-101 counted one removed name against every same-named added line. Two reviewers
  filed it as a major.
- **3 majors.** COMPAT-2's scaffold reading moved on a dot; running the #101 regexes on removed lines
  added Python/JS disagreements; the `only_touches` reason listed dot-only paths.
- **10 minors.** The pre-commit smoke run; per-PR attribution and issue #121's own gates; no committed
  differential receipt, and `py_side.py`'s promise of 0 disagreements; the ASCII name capture; the
  dot abstention firing in both directions, with its prefix and mixed cases untested; #97's order
  inside one tier; dotfile renames; `symbol_added` recall on same-named definitions; the untested
  BIN-1 registration; `build_bookmarklet.py`'s terser version and `--check`.

## 2. Generic definitions (limit 3)

The amendment says "PEP 695 `def f[T](` is not a definition line for either pattern". That holds for
the symbol pattern only: `NAME(?=[ \t(:]|$)` does not accept `[`. The test pattern captures a name up
to the next space, tab, `(` or `:`, so `def test_x[T](` reads as the name `test_x[T]`, and a changed
generic test pairs with its removed line like any other. The limit, restated: a changed generic
function or class still verifies "adds function f", exactly as it did on `87dded26`.

## 3. Where the reviewer inputs ran

The amendment says of the blocker's inputs: "each run on `87dded26` and on the branch, both doors and
the port". The record is narrower:

- The two-class inputs ran on `87dded26` and on the branch, over a real repository through both
  doors and through the port (correctness lens), and through `gate_diff_text` and the port (parity
  lens).
- The fold input ran on the branch through `external1_harness.reconstruct` and `gate_diff_text`
  (protocol lens), and through the port as a raw diff (parity lens).
- The non-ASCII input ran on both instruments and both doors, and agreed with the port.

## 4. An edit to the amendment before any code of this round

The amendment's commit was amended once before any code commit of this round, and it had not left
this machine (`af7df5b8` → `efe2878d`). The only change: three literal U+FEFF characters, which a
tool had decoded from the pattern text, were replaced with the escape text `\uFEFF` (`\ufeff` in the
quoted reviewer input). The rendered patterns had shown an invisible character where the escape
belongs. No word or rule changed. `styxx/diffgate.py` and `path2_gates.py` had the same decoding
before they were committed, and were repaired the same way. The committed code carries the escape
text, and the tests pin its behaviour (a BOM strip pairs; `\u00a0` indentation does not).
