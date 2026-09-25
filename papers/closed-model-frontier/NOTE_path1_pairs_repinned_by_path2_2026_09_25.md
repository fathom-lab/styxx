# NOTE — two PATH-1 pinned pairs were re-pinned by PATH-2

2026-09-25. Written beside `PREREG_path1_only_touches_repair_2026_09_17.md` and
`RESULT_path1_only_touches_repair_2026_09_17.md` (merged in #127), which this note does not edit.

`web/gate/differential/path1_pairs.json` belongs to PATH-1. Branch `fix/diffgate-path-resolution`
(pull request #161, PATH-2) rewrote three expected reason strings in two of its pairs, in commit
`a341d1d9` ("web/gate/differential: fourteen more PATH-2 pairs, the COMPAT-2 gap closed, two PATH-1
blocks re-pinned"). A reader arriving from PATH-1 had no way to learn that; this note is where they
find it.

## What changed

PATH-2 repairs issue #121: the path key keeps a dotfile's leading dots. Before it, the key was
`lstrip("./")`-ed, so a claimed `.k-step-link` printed as `k-step-link` and a changed
`.github/workflows/dependabot.yml` printed as `github/…`. The expect blocks were written against that
older key. **No verdict moves.** Only the printed strings do.

| pair id | claim kind | verdict (unchanged) | expected reason, as PATH-1 pinned it | as re-pinned by PATH-2 |
|---|---|---|---|---|
| `path1:css-selector` | `only_touches` | UNCHECKABLE | `prefix 'k-step-link' is not a path (#110)` | `prefix '.k-step-link' is not a path (#110)` |
| `path1:unrepaired-typo` | `file_touched` | VERIFIED | `diff status 'M' for 'github/workflows/dependabot.yml'` | `diff status 'M' for '.github/workflows/dependabot.yml'` |
| `path1:unrepaired-typo` | `only_touches` | CONTRADICTED | `paths outside 'githiub/workflows/dependabot.yml': ['github/workflows/dependabot.yml']` | `paths outside '.githiub/workflows/dependabot.yml': ['.github/workflows/dependabot.yml']` |

Each pair's `verdict` (PASS / FAIL) and `uncovered_sentences` are also unchanged. The new strings are
the claim and the path as written: the claim really says `.k-step-link` and `.githiub/…`, and the file
really is `.github/workflows/dependabot.yml`.

`path1:unrepaired-typo` still asserts what PATH-1 said it would: mode 6 (a typo in the stated path) is
not repaired, and the instrument still accuses `open-policy-agent/cert-controller#415` falsely. It now
does so in a reason that prints both dots.

## Where it is recorded on the PATH-2 side

`NOTE_path2_third_pass_2026_09_25.md`, section A, records the two `only_touches` strings; it leaves
out the `file_touched` string in the same pair, which `NOTE_path2_fourth_pass_2026_09_25.md` records.
Neither PATH-1 document quotes any of the three strings (the RESULT quotes one reason,
`prefix 'assert.notnull' is not a path (#110)`, which carries no leading dot and does not move), so
nothing in them is out of date.
