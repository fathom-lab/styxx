# Seat instructions v2 — the single permissible clarified re-run text (frozen at prereg freeze)

This version exists so that if a seat fails validity, the re-run cannot be steered: v2 was
written before any seat ran, and a re-run may use exactly this text and nothing else.

Everything in v1 applies unchanged. The labels are A (this commit changed files/code in a
specific, diff-checkable way), B (a result/measurement whose evidence lies outside any diff),
C (neither). The clarifications below are restatements with examples — they add no new rules.

## Rule 1 restated — subject lines

- "certify: collapse the third rung into one branch" → **A** (imperative naming a concrete
  code change this commit makes).
- "papers: notes from the week-two retro" → **C** (names no concrete change to files or code).

## Rule 2 restated — tense and agency

- "Rebuilt tests/test_seam.py after the split." → **A** (bare past-tense action verb, file
  object, no other actor: this commit did it).
- "tests/test_seam.py had been rebuilt in the prior cycle." → **C** (pluperfect + another
  cycle named: not this commit's act).
- "The certificate is present in the tree with drifted content." → **C** (stative: reports a
  state, asserts no act by this commit).
- "module.py: committed OATH-HELD" → **C** (reports a document's recorded status; asserts no
  change made by this commit to that file).

## Rule 3 restated — compound sentences

- "Fixed the band split; 2575 passed, 11 skipped." → **A** with `also_result_clause: true`
  (a change assertion AND a test-result assertion in one sentence).
- "2575 passed, 11 skipped." alone → **B**.

## Output — unchanged from v1

{"labels": [{"id": "...", "label": "A"|"B"|"C", "also_result_clause": true|false}, ...]},
every id present exactly once.
