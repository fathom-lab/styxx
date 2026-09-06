# SPEC — the path-segment class is identical in both implementations, and pinned (v0.1)

**Frozen 2026-09-06, before the code.** One rule, P1. Found by the eight-dimension adversarial
audit (`wf_9466dcba-f49`, dimension `parity`).

## The defect

`sworn.py` rejects bad path segments with `_PATH_SEG_BAD = [\\\s\x00-\x1f\x7f*?\[\]]`.
`sworn_verify.js` cannot write `\s` and mean the same thing, so it **hand-expands** the class
(l.471). The expansion omits one member of Python's `\s`: `U+0085 NEXT LINE`.

Measured over every Unicode scalar value on this build:

```
python rejects: 58 code points
node rejects  : 57 code points
rejected by PYTHON only: ['U+0085']
rejected by NODE only  : []
```

A `path:` receipt whose target contains `U+0085` therefore splits the two verifiers on the
**document verdict**:

```
PYTHON  document_verdict SWORN-FAILED   span MALFORMED/receipt_form
NODE    document_verdict SWORN-HELD     span UNRESOLVED/no_repository
```

One implementation refuses the document; the other says it held. This is exactly what the
conformance set exists to prevent, and it survived 1689 replayed vectors because no vector puts a
`U+0085` in a path.

## Why the one-character fix is not the repair

Adding `U+0085` to the JavaScript class closes today's gap and leaves the mechanism intact: a
hand-maintained enumeration of another language's character class, with nothing checking it.

The enumeration cannot be avoided. **Neither language's `\s` means the other's**: Python's includes
`U+0085` and excludes `U+FEFF`; JavaScript's includes `U+FEFF` and excludes `U+0085`. So a literal
list is required on the JS side, and a literal list drifts unless something compares it.

## P1 — the two classes are equal over every scalar value, and a test says so

1. `sworn_verify.js` adds `U+0085`, making the classes equal today.
2. A guard evaluates **both** classes over all 1,114,112 scalar values and asserts the rejected sets
   are identical — not that either contains some expected list, but that they agree with each other.

The guard reads the JavaScript's regex out of the shipped file rather than restating it, so it
cannot pass by testing a copy of the source it is meant to police.

This is the shape the other repairs in this audit argued for. Five of the six were identical defects
in *both* implementations, which a conformance set built from those same two implementations cannot
see. Here the two disagree, and the durable fix is a check that compares them directly rather than a
vector that happens to cover one code point.

## What moves

- **Nothing committed.** No committed document carries `U+0085` in a `path:` target.
- JavaScript only. `sworn.py` is unchanged, so the conformance set does not move at all — not even
  by the build pin, for the first time in this run of repairs.
- The JS bar stays at 1689.

## Guards, watched to fail before the code

| # | guard | before | after |
|---|---|---|---|
| P-G1 | the two rejected sets are identical over all scalar values | red: differ by `U+0085` | green |
| P-G2 | a `path:` target containing `U+0085` gets the same document verdict from both | red: SWORN-FAILED vs SWORN-HELD | green |
| P-G3 | an ordinary path target is unaffected in both | green throughout |

P-G1 is the guard that must be seen red, and it is the one that outlives this defect: any future
edit to either class that does not edit the other fails it.

## What this does not claim

That the two implementations agree generally. It pins one character class. The audit found this
divergence by enumerating a class; the same method applied to the other hand-mirrored classes in
`sworn_verify.js` — `TOKEN_RE`, `GRAM_RE`, `HEXRUN_RE`, `DIGIT_RE` — has not been run, and is the
obvious next measurement.
