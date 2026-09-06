# SPEC — invariant 2 on the tree channel: the same refusal on every form a receipt can take

Fathom Lab · 2026-09-06 · **A spec, not a result.** Frozen in its own commit before the repair is
made. It fixes what the repair must satisfy, so the repair cannot be scored by whether it made the
probe go green.

## The invariant, and where it was enforced

Invariant 2 of the sworn format: **the agent cannot swear to bytes it minted.** A receipt whose
`sha256` appears in the manifest's `authored_sha256` list — every byte-object the agent produced
this turn — is `MALFORMED / receipt_author_minted`. A verifier that cannot enforce this is a verifier
an author can feed its own output to and call evidence.

`_resolve` enforces it on the `rN` branch, at one line:

```python
if kos in SOURCE_KINDS_AUTHOR or sha in manifest.authored_sha256:
    return _Resolved(status="malformed", reason="receipt_author_minted")
```

The tree branch — `path:` and `prereg:` — computes `sha` from the resolved bytes and never compares
it to anything. Confirmed by execution, same bytes, same manifest, three forms:

| form | numeric | absent |
|---|---|---|
| `rN` | MALFORMED `receipt_author_minted` | MALFORMED `receipt_author_minted` |
| `path:` | **HELD** | **HELD** |
| `prereg:` | **HELD** | — |

The invariant holds on one channel of three. Naming the same bytes by path instead of by id is the
whole attack, and `absent` — the strongest verdict, the one that says *this never happened* — is
reachable that way.

## What the repair must satisfy

**T1 — one refusal, three forms.** A receipt whose resolved `sha256` is in `manifest.authored_sha256`
is `receipt_author_minted` on every form. The reason string is the existing one; no new vocabulary.
*Attack:* a repair on `path:` that forgets `prereg:`, or one keyed on the path rather than the
bytes. *Answer:* T4's test swears the same bytes all three ways and asserts the same refusal.

**T2 — `complete` on the tree branch stays `True`, and says why.** A committed blob is complete —
the verifier holds every byte of it. That was never the defect; the defect was that authorship went
unchecked before completeness was assumed. The hard-coded `True` gains the comment its two-word
form has been missing.
*Attack:* "fixing" completeness instead of authorship, so `absent` becomes unreachable on the tree
channel and the invariant is still unenforced. *Answer:* T4 asserts `absent` still HELDs over a
committed file that is *not* author-minted.

**T3 — the `rN` branch is untouched, and the vectors say so.** The 1689-vector bar frozen by
`SPEC_sworn_browser_verifier_v01` B3 holds; the JavaScript verifier has no tree channel and is not
changed. Any conformance vector whose core moves is examined and named: a moved `path:`/`prereg:`
vector over author-minted bytes would be the repair working, and the generator's refusal is read
rather than overridden.
*Attack:* regenerating the set to make the refusal go away. *Answer:* a moved core is reported in
the RESULT with the vector id and the reason its verdict changed.

**T4 — the guard is watched to fail.** The test exists before the repair, fails against the shipped
code on `path:` and `prereg:`, passes on `rN`, and passes on all three after. It also pins the
honest case — a committed file whose digest is *not* in `authored_sha256` — so a repair that refused
every tree receipt would fail it.

**T5 — the manifest without `authored_sha256` behaves as before.** An empty or absent list refuses
nothing on any branch. The repair adds a comparison, not a requirement.

## What this spec does not say

That `path:` and `prereg:` are now safe: the tree channel still has no `kind_of_source`, so the
other half of the `rN` refusal — `kos in SOURCE_KINDS_AUTHOR` — has no analogue there, and a file
committed by the agent under a name the manifest never saw is still not caught. `authored_sha256`
is the manifest's list of what the agent produced; if the harness did not record a digest, nothing
here can refuse it. That is a limit of the manifest, stated rather than solved.

---

*The rule was written once and applied to one branch. Two others compute the same digest, and
neither looks at it.*
