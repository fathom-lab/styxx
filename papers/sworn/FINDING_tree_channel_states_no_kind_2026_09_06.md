# FINDING — the tree channel states no kind, so `committed` is asserted on the caller's word

*2026-09-06. Scope: the Python API and the receipt as a travelling artifact. The `sworn` CLI is not
affected. No published number changes.*

## The asymmetry

The format has two channels for binding a sentence to bytes, and it treats them very differently.

**The manifest channel (`rN`) does not trust its input.** Every entry must declare
`kind_of_source`; the verifier validates it against an allowlist, refuses `kind_of_source_unknown`,
and refuses outright when the kind is one the author could have minted
(`receipt_author_minted`). The rung travels into `provenance` so a reader sees which it was.

**The tree channel (`path:`, `prereg:`) trusts its input completely.** It records:

```json
{"form": "path", "note": "committed object at <40-hex>; authorship unchecked"}
```

and nothing else. There is no `kind_of_tree`. The verifier prints *"committed object at X"* on the
word of whatever handle the caller passed, and these spans count in `rungs` under `committed` — the
strongest provenance the format states.

Three handles can answer, and only one consults history:

| handle | what it checks | can it say `committed` with no history? |
| --- | --- | --- |
| `GitTree` | `cat-file -t <commit>` is a commit; reads blobs from git | no |
| `SnapshotTree` | the handle's commit equals the commit its own entries were read at | **yes** — a fabricated snapshot satisfies this trivially |
| `MemoryTree` | nothing; answers from a dict the caller filled | **yes** |

`SnapshotTree` looks like the bound variant and is not. Its check is internal consistency, not
provenance: `SnapshotTree.from_memory(fabricated)` sets `snapshot_commit == commit` and passes.

## Reproduction

```python
FC = "0"*39 + "1"                      # well-formed object id that names nothing
tree = sworn.MemoryTree({"results/metrics.txt": b"run 7\nthe measured value was 0.9910\nrun 8\n"},
                        commit=FC)
core = sworn.verify(doc, name="claim.md", manifest=None, tree=tree, commit=FC)
```

```
document_verdict : SWORN-HELD
rungs            : {'committed': 1}
provenance note  : committed object at 0000000000000000000000000000000000000001; authorship unchecked
```

Identical output via `SnapshotTree.from_memory(tree)`. Nothing in the core records which handle
answered: no field of the core contains the string `Memory`, `Snapshot` or `Git`.

## It is not hypothetical: it is in this corpus

An audit of every tracked receipt (`receipt_provenance_audit.py`):

```
receipts appealing to the tree channel : 39
  naming a commit this repository has  : 34
  naming one it does not               : 5
```

The five are `papers/sworn/measurement/dryrun/SYN-0{1,2,3}` and `SYNX-0{1,2}`, together asserting
**`committed object at aaaaaaaa…aaaa` across 34 spans**. Those bytes were never committed and that
object id names nothing. They are legitimate — `synthetic.py` builds them over a `MemoryTree` at the
placeholder `C40 = "a"*40`, publishes the tree beside each receipt, and the population marks the
source `kind: "synthetic"` — but **the receipt cannot say any of that about itself.**

Establishing they were sound took three out-of-band lookups: the population file, the generator's
source, and the GitHub API. For a format whose premise is that the receipt is independently
re-derivable and can be checked without trusting its author, needing the author's directory layout
to interpret its strongest provenance claim is the defect.

Two groups were checked and cleared in the process, and neither is a problem:

- Eight `sworn_action_sample*` receipts name `f808c3c6…`, which exists in neither this repository nor
  the remote. They resolve **entirely through the `rN` channel** (`rungs: {L2: n}`) and appeal to the
  tree for nothing; the commit is a recorded field, not a claim. The id is a real commit in the
  deterministic fixture repository the committed generator builds.
- Eight receipts name no commit at all.

## What is not wrong

- **The CLI is unaffected.** `_load_tree` returns `GitTree` and nothing else, so
  `python -m styxx.sworn verify` cannot produce an unbacked `committed`.
- **Re-derivation still defends.** A third party re-deriving with a real `GitTree` at that commit
  gets `path_absent` or a byte mismatch. That is the format's actual guarantee and it holds. The
  gap is for a reader who *reads* a receipt rather than re-deriving it — which the format invites by
  printing a provenance note in prose.
- **No published number moves.** The five receipts are dry-run fixtures for the measurement harness
  and are cited as evidence for nothing.

## Remedies

**(a) State the kind — format-level, NOT taken here, operator-gated.** Add `kind_of_tree` to
tree-form provenance, mirroring `kind_of_source`, so a receipt says whether git answered. This is the
remedy that actually fixes the finding, and it **moves the digested core**: `provenance` is inside
the digest, so every v0.2 receipt in history would stop re-deriving. That is a spec change and a
version bump, and by the lab's own rule a moved core is not something to do quietly on the way past.
It is written down here and left for the operator.

**(b) Declare the exceptions — corpus-level, shipped.** `receipt_provenance_audit.py` computes,
over every tracked receipt, whether a `committed` claim names a commit this repository has. The
exceptions are declared once in `receipt_provenance_declarations.json` with a kind and a reason, and
the audit fails on any undeclared claim *and* on any declaration whose receipt no longer needs one.
Three out-of-band lookups become one committed, checked file.

(b) does not fix (a). It converts an unanswerable question about the corpus into an answered one,
and it does nothing for a receipt that leaves this repository. The declarations file says so about
itself: a declaration records that someone decided a claim is a fixture; it does not make it true.

## An absent commit is not an accusation

The audit's first version would have gone red on its first CI run, accusing all 34 backed receipts
at once. `actions/checkout` clones at depth 1 and `test.yml` sets no `fetch-depth`, so in CI nearly
every commit is missing and `cat-file` fails on all of them. Reading that absence as fabrication is
the same inference that shipped a path-claim accusation at 0.23 precision, and it was caught here
before pushing rather than by the corpus a second time.

So the audit establishes whether it can see history before it says anything:

| exit | meaning |
| --- | --- |
| 0 | every tree-claiming receipt is backed by history, or declared with a reason |
| 1 | at least one is neither -- an accusation, only ever made against complete history |
| 2 | the history here is incomplete, so the question cannot be answered |

Verified against real git rather than assumed: in a depth-1 clone,
`git rev-parse --is-shallow-repository` is `true`, `cat-file -t` on an older commit fails, and
`ls-files` still works -- which is exactly the state that would have produced the false accusations.

The cost is honest and worth stating: **in CI this guard reports INDETERMINATE and checks nothing.**
It has force in a full clone -- a developer's, or any run after `git fetch --unshallow`. Making it
bite in CI means setting `fetch-depth: 0`, which is a change to `.github/` and is left for the
operator.

## Watched to fail

| perturbation | result |
| --- | --- |
| no declarations file at all | 5 UNDECLARED, exit 1 |
| one declaration removed | that receipt UNDECLARED, exit 1 |
| a declaration with no `reason` | refused before any checking |
| a declaration for a receipt that *is* backed | STALE DECLARATION, exit 1 |
| a fabricated receipt staged into the tracked set | named as UNDECLARED, exit 1 |
| shallow history, no declarations | INDETERMINATE, exit 2, accuses nobody |

The last is the one that matters: the guard catches a *new* unbacked receipt entering the corpus,
not only the ones that were there when it was written.
