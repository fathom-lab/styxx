# SPEC — the tree channel reads history, not a local rewrite of it (v0.1)

**Frozen 2026-09-06, before the code.** One rule, T1. Found by the eight-dimension adversarial
audit (`wf_9466dcba-f49`, dimension `tree`), survived its skeptic, and re-derived here before this
spec was written.

**This spec also corrects a merged finding of my own.** `FINDING_tree_channel_states_no_kind_2026_09_06.md`
(PR #76) tabulates the three tree handles and marks `GitTree` as the one that "consults git" and
therefore **cannot** say `committed` with no history behind it. That is wrong, and this is the
counter-example. The correction is written into that finding by this leg, not left standing.

## The defect

`GitTree._git` runs `git -C <repo> …` with neither `--no-replace-objects` nor
`GIT_NO_REPLACE_OBJECTS` set (sworn.py:918). Git's object lookup honours `refs/replace/*` by
default, in `cat-file` and `ls-tree` alike. So **one local ref makes the verifier serve a different
commit's bytes under the commit id the document names.**

Measured at `f6179c6e`, in a throwaway repository:

```
commit A (the document names this one) = 899d8e045f0c4de99a50c15271fbe1c5679efbdc
bytes at f.txt in A                    = the precision is 0.9900

no replace ref:     FAILED needle_missing  resolved_sha256=1071bbb4399e441c  doc=SWORN-FAILED
                    provenance: committed object at 899d8e04…; authorship unchecked
git replace A B
with refs/replace:  HELD   None            resolved_sha256=cc63391eb7b3d0cd  doc=SWORN-HELD
                    provenance: committed object at 899d8e04…; authorship unchecked

bytes at f.txt in A (still)            = the precision is 0.9900
```

**A false document becomes SWORN-HELD**, and the provenance note still names commit A. Nothing was
rewritten: no object was created, no history changed, and `git --no-replace-objects cat-file` still
prints the original bytes. One ref, and it is invisible in the receipt.

This is worse than the gap PR #76 describes. There the unbacked `committed` came from handles that
never consult git at all (`MemoryTree`, `SnapshotTree`), and the audit shipped with that leg says a
third party re-deriving with a real `GitTree` would catch it. **This defeats that defence**: the
re-deriver uses `GitTree`, and a replace ref in the repository they were handed answers for it.

Not a documented decision: `DECISIONS` has no entry for replace refs, grafts or alternates; no
docstring mentions them; no test covers them. `GitTree`'s own docstring says "A verdict must be a
function of bytes, not of somebody's checkout" — this makes it a function of somebody's refs.

## T1 — object lookup ignores replacement

Every `git` invocation in `GitTree._git` passes `--no-replace-objects`, before the subcommand, and
the process environment sets `GIT_NO_REPLACE_OBJECTS=1`. Both, because the flag is the documented
mechanism and the variable also covers any git subprocess that a future edit adds without
remembering the flag.

`git` is invoked as:

    git --no-replace-objects -C <repo> <subcommand> …

with `GIT_NO_REPLACE_OBJECTS=1` in the child's environment.

### Why not detect and refuse instead

Refusing when `refs/replace/*` exists would be a stronger statement and a worse one: it makes an
UNRESOLVED out of a repository that merely *has* a replace ref, including one that has nothing to do
with the commit the document names. The verdict must be a function of the bytes the document names,
so the right move is to read those bytes and ignore the replacement — not to decline because
something unrelated is present in the repository handed to us.

### Scope this does not cover, stated

`--no-replace-objects` addresses replacement refs. It does **not** address `info/grafts` (deprecated
and already ignored by modern git in most paths), `objects/info/alternates`, a `commit-graph` that
disagrees with the objects, or a `git` binary earlier on `PATH` than the real one. A verifier that
runs somebody else's `git` in somebody else's repository is trusting that environment, and the tree
channel does not attempt to escape it. This closes the one route that needs no privileges, writes
no object, and leaves the repository looking untouched.

## What moves

- **Nothing committed.** No replace ref exists in this repository; the fix is a no-op on every
  input that has none, which is every input the corpus has.
- The conformance set moves only by the verifier build pin, per the ERRATA in
  `SPEC_numeric_sign_is_not_dropped_v01_2026_09_06.md`: the bar is the same vector count with an
  identical multiset of expected outcomes.
- **No parity concern.** `sworn_verify.js` has no tree at all — `path:` and `prereg:` always resolve
  `UNRESOLVED no_repository` there — so this is a one-sided change by construction, and the JS bar
  should be unmoved at 1689.

## Guards, watched to fail before the code

| # | guard | before | after |
|---|---|---|---|
| T-G1 | with a replace ref pointing A at B, a span whose needle is absent from A's bytes is not HELD | red: HELD, document SWORN-HELD | green |
| T-G2 | with a replace ref present, the bytes resolved are A's, by sha256 | red: B's sha | green |
| T-G3 | with no replace ref, behaviour is unchanged | green throughout — catches over-reach |
| T-G4 | a replace ref unrelated to the named commit does not make the document UNRESOLVED | green throughout — the "why not refuse" clause |

T-G1 is the guard that must be seen red.

## What this does not claim

That the tree channel is now trustworthy. PR #76's finding stands in every other respect: the
channel still records no `kind_of_tree`, `MemoryTree` and `SnapshotTree` still say `committed` with
no history, and the format-level remedy is still unadopted and still the operator's call. This
removes one route by which even the git-backed handle could be made to lie.
