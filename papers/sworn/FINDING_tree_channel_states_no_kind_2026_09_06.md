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
| `GitTree` | `cat-file -t <commit>` is a commit; reads blobs from git | **yes** — see the correction below |
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

## The guard's own coverage, stated

The audit's first version read only the top level of each tracked JSON file. That is 51 of the
**2118** receipt-shaped objects in this repository; the other 2067 are nested, and it would have
reported "every receipt" while seeing 2.4% of them. None of the 2067 carries a tree claim, so
nothing was slipping through -- but coverage that holds by accident is not coverage, and the next
artifact to nest a tree-claiming receipt would have passed in silence.

It walks nested receipts now. The 2067 are all conformance vectors: expected verifier outputs over
synthetic documents, fixtures by construction and history by this lab's rules, so they are not
claims about the world and are not audited. That exclusion is scoped to `conformance/sworn/vectors/`
**by name**, and both numbers print on every run:

```
  conformance vectors, not audited     : 2067 receipt(s) under conformance/sworn/vectors/,
                                         of which 0 make a tree claim
```

The second number is the one that matters -- it is what the exclusion currently costs, and it is
zero. If a vector ever gains a tree claim, the line says so instead of hiding it.

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

---

## CORRECTION 2026-09-06 — `GitTree` was not the sound handle

**This finding said `GitTree` is the one handle that consults history and therefore cannot say
`committed` with no history behind it. That is wrong.** The table above is amended in place rather
than left standing, and the reason is recorded here.

`GitTree._git` ran `git -C <repo> …` with nothing suppressing object replacement, and git honours
`refs/replace/*` by default in `cat-file` and `ls-tree` alike. **One local ref makes the verifier
serve a different commit's bytes under the commit id the document names**, while the provenance note
still reads `committed object at <the named commit>`:

```
commit A (the document names this one) = 899d8e045f0c4de99a50c15271fbe1c5679efbdc
no replace ref:     FAILED needle_missing   doc=SWORN-FAILED
git replace A B
with refs/replace:  HELD                    doc=SWORN-HELD
bytes at f.txt in A (still)            = the precision is 0.9900
```

No object is written, no history is rewritten, and `git --no-replace-objects` still prints the
original bytes. Nothing in the receipt shows it.

**This also breaks a defence this document leans on.** The section "What is not wrong" says a third
party re-deriving with a real `GitTree` would catch an unbacked `committed` from `MemoryTree` or
`SnapshotTree`. They would not, if the repository they were handed carries a replace ref. That
sentence should be read as narrower than it was written: re-derivation defends against a fabricated
*handle*, not against a repository prepared for the re-deriver.

Repaired in `SPEC_tree_ignores_replace_refs_v01_2026_09_06.md` (T1): every `git` call now passes
`--no-replace-objects` with `GIT_NO_REPLACE_OBJECTS=1` in the environment. That closes the route
needing no privileges, writing no object, and leaving the repository looking untouched; it does not
address alternates, a commit-graph that disagrees with the objects, or a different `git` on `PATH`,
and the spec says so.

**How the error was made, since that is the more useful part.** The handles were classified by
reading what each one calls — `GitTree` calls git, so it was marked as consulting history. What was
never asked is the question this corpus asks of everything else: *what would have to be true of the
environment for that call to answer honestly?* An adversarial reviewer asked it and had the answer
in one command. The finding was published between those two moments.
