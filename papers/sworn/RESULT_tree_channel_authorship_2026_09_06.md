# RESULT — invariant 2 held on one channel of three, and now holds on all of them

Fathom Lab · 2026-09-06 · Spec: `SPEC_tree_channel_authorship_v01_2026_09_06.md`, frozen before
the repair. Receipt: `tree_channel_repair.json`, re-derived from git by
`tree_channel_repair_receipt.py`. **This document is itself sworn.**

## The invariant

**The agent cannot swear to bytes it minted.** A receipt whose `sha256` appears in the manifest's
`authored_sha256` — every byte-object the agent produced this turn — is refused as
`receipt_author_minted`. Without this, a verifier is something an author can feed its own output to
and call evidence.

## Where it was enforced

`_resolve` enforced it on the `rN` branch, at one line. The tree branch — `path:` and `prereg:` —
computed the same digest from the resolved bytes and never compared it to anything. Confirmed by
execution: the same bytes, which the manifest itself listed as agent-authored, sworn to three ways.

| form | `numeric` | `absent` |
|---|---|---|
| `rN` | MALFORMED `receipt_author_minted` | MALFORMED `receipt_author_minted` |
| `path:` | **HELD** | **HELD** |
| `prereg:` | **HELD** | — |

Naming the same bytes by path instead of by id was the whole attack. And `absent` — the verdict that
says *this never happened*, the strongest in the format — was reachable over the author's own
committed file that way.

It came from the sidecar battery's adversary, in its list of what nobody had attacked. The
conformance set regenerated after the repair with **no moved core**, which says the set never
exercised this case either — consistent with `RESULT_suite_power_2026_09_06.md` finding the tree
layer defended at 4 of 14.

## The repair, and how it was checked

One comparison on the tree branch, with the existing reason string. `complete = True` stays where
it was: a committed blob *is* complete, and the defect was never completeness — it was assuming
completeness before checking authorship.

The guard was written first and run against the verifier as shipped. Of
<sworn r="path:papers/sworn/tree_channel_repair.json#/guard_tests" k="numeric">10 tests,</sworn>
<sworn r="path:papers/sworn/tree_channel_repair.json#/before/failed" k="numeric">3 failed before the repair</sworn>
— `path:`, `prereg:`, and `absent`-by-path — and
<sworn r="path:papers/sworn/tree_channel_repair.json#/after/failed" k="numeric">0 failed after.</sworn>
The
<sworn r="path:papers/sworn/tree_channel_repair.json#/before/passed" k="numeric">7 that passed in both states</sworn>
are the ones that give the repair its shape: the `rN` refusal unchanged, an honest committed file
still resolving by path and by digest, `absent` still holding over it, and an empty
`authored_sha256` refusing nothing. A repair that refused every tree receipt would have failed them.

The receipt is re-derived from git, not remembered: `tree_channel_repair_receipt.py` reads the
verifier at the repair commit and at its parent, runs the guard against each in a scratch copy, and
writes both counts.

## What this does not say

**That `path:` and `prereg:` are now safe.** The tree channel has no `kind_of_source`, so the other
half of the `rN` refusal — `kos in SOURCE_KINDS_AUTHOR` — has no analogue there. A file the agent
committed under a name the harness never digested is still not caught. `authored_sha256` is the
manifest's record of what the agent produced; if the harness recorded no digest, nothing here can
refuse it. That is a limit of the manifest, stated rather than solved.

**That this was hard to find.** It was one line, visible by reading the two branches side by side,
and it was named by an agent asked to list what nobody had attacked. It survived because no test
and no vector ever swore author-minted bytes by path.

---

*The rule was written once and applied to one branch. Two others computed the same digest, and
neither looked at it.*
