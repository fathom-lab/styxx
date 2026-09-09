# The sampled challenge — a second party's bytes at k/n of the cost

Fathom Lab · 2026-09-09 · **A design and its measured detection power.** No challenge has been
run by anyone, by any party, at any k. This document specifies a mechanism and reports what it
would establish if run; it reports no reproduction. Module: `styxx/v8/sampled_challenge.py`.
Tests: `tests/test_v8_sampled_challenge.py`, 34 tests, all passing, every certificate-bearing
test reading the published artifact rather than a fixture. Not sworn.

## Why

`papers/v8/THE_BOUNDARY_2026_09_09.md` states the invariant that survived five rounds of the lab
attacking its own design: a check on bytes an issuer wrote can only ask whether that party
contradicted itself. Its design consequence is that the challenge — a second party running the
battery and putting their own bytes in the log — is the only part of the system that introduces
a byte the issuer did not write. That document also says a design that cannot attract a second
party has failed at the only thing that would have made the engineering mean anything.

The challenge mechanism has never been run. One reason is cost: a challenge as specified in
§9 is a full battery re-run, and a challenge that costs a full re-run is a challenge nobody
performs. This design makes the second party's cost `k/n` of one, at a detection power that is
computed rather than assumed.

## The published artifact this is specified against

`papers/v8/first_verdict_2026_09_09/` — a verdict on `google/gemma-2-2b-it`, a 64-prompt
battery, greedy, 16 new tokens, five floor runs, a seven-entry log. Read off those bytes:

| quantity | value | where |
|---|---|---|
| battery items `n` | 64 | `fp_bf16/fingerprint-canonical-73a09ffa3f1e.json`, `body.items` |
| per-item digests present | 64 of 64, both `output_sha256` and `token_ids_sha256` | same file |
| canonical cert leaf index | 6 | `log/entries/000000/00000006.meta.json` |
| tree size of the published head | 7 | `log/sth/000000000007.json` |
| head root | `sha256:d61a6a78…1bcea4b` | same file |

Leaf index 6 is inside tree size 7, so the published head already commits to the certificate a
challenger would sample. That is the precondition the whole design rests on, and on this
artifact it holds.

## The selection function

A challenger draws k of the n item ids. The draw must not be choosable or predictable by the
issuer, or the issuer fabricates exactly the items nobody will sample. It is derived from a
**selector**, an object every member of which is a commitment:

```json
{
  "challenger_key": "ed25519:…",
  "k": 20,
  "log_id": "sha256:88300489…bae7c4f",
  "method": "styxx.v8/challenge/sample/1",
  "mode": "head",
  "n": 64,
  "nonce": "00112233445566778899aabbccddeeff",
  "root_hash": "sha256:d61a6a78…1bcea4b",
  "target": "sha256:73a09ffa…766a5c3",
  "tree_size": 7
}
```

`selector_id` is `"sha256:" + sha256(JCS(selector))`. The ranking seed is
`sha256(JCS(selector without "k"))`, and the n item ids are ordered by

```
sha256("styxx.v8/challenge/rank/1" || 0x00 || seed || 0x00 || item_id)
```

ties broken by the id, first k taken. Two tags are introduced,
`styxx.v8/challenge/sample/1` and `styxx.v8/challenge/rank/1`, framed the way
`styxx.v8.keys.tagged` frames a cert id and `fingerprint._ORDER_TAG` frames a plan's item
order, so neither preimage can collide with a cert id, a tree head, a seal, a sweep record or a
plan order.

Sampling is **without replacement**: one ranking, a prefix of length k. The power is therefore
hypergeometric, which is above the with-replacement figure at every k (table below).

### `k` is committed but is outside the seed, and that is a security decision

Putting `k` in the seed was the first draft, and a test caught it. With `k` in the seed each k
gives an **independent** permutation, so a colluding challenger holding the list of fabricated
items computes all 64 rankings and publishes whichever k has a clean prefix — 64 free draws at
n = 64, on top of nonce grinding. With `k` outside the seed there is one ranking per (head,
key, nonce, target), samples nest — k and k+1 share a prefix — and the challenger's only
remaining freedom over the draw is the prefix length, which is `k` itself and is the number a
reader already discounts by. `k` stays inside the selector and therefore inside `selector_id`,
so the claimed sample size is still a commitment; it just does not move the draw.
(`test_k_is_committed_but_does_not_move_the_ranking`.)

### A worked draw anyone can re-derive

Selector as printed above, with `challenger_key` set to the artifact's own issuer key for the
purpose of the example:

```
selector_id  sha256:b3507792f2906c40e9acbb63c61efede4657f001be5aeae7a2b99ab2a9ba189f
drawn (k=20) a13d1b0ebe3558d9 b4712a870051b4d6 d7efa933c8ad300d 205501a795c91a18
             283fef992ecff8d3 46be29730041af90 e56ece70332f0fe3 15aa43154e4e8a58
             b1a031d8c6eca320 c6402f34a998589f cf24f2f1a59981cc db3c4d9c98016c3e
             51b37322772567b3 4cc3bf784a888574 adcdbc5352081297 85354edf605aadc5
             ea86bf79e2e8283d 6bb1d390a81c9153 97154ba031f43747 c2c7eb9413d0ba18
```

The first eight of the k = 64 ranking are the first eight of this list, which is the nesting
property being visible rather than asserted.

## The challenge cert body

A `challenge` cert, `refs` = `[{role: target, id: <fingerprint>}, {role: own, id: <the
challenger's partial fingerprint>}]`, body:

```json
{
  "environment": {"hardware": "…", "runtime": "…"},
  "recipe_core": {"battery": "sha256:…", "decoding": {}, "…": "…"},
  "sample": {
    "disagreements": 0,
    "item_ids": ["a13d1b0ebe3558d9", "…19 more, in ranking order"],
    "results": [
      {"agree": true, "item_id": "a13d1b0ebe3558d9",
       "own_output_sha256": "…", "own_token_ids_sha256": "…",
       "target_output_sha256": "…", "target_token_ids_sha256": "…"}
    ],
    "sampled_exact_distance": 0.0,
    "selector": {"…": "the object above"},
    "selector_id": "sha256:b3507792…9ba189f"
  },
  "sample_of": "exact",
  "subject": {"…": "the challenger's own S_identity"},
  "synthetic": false
}
```

**`per_channel` and `coverage` are deliberately absent.** Both are floor-relative — a
`target_floor` per channel, and a within/beyond classification against the floor's `covers` —
and a one-run k-item reproduction has measured no floor. Emitting them with the target's own
floor copied in would let a reader read a floor comparison out of bytes that contain none,
which is the shape of the defect `THE_BOUNDARY` round 4 describes: the recomputation is exact,
of whichever bytes the issuer chose to name. The scope is carried explicitly in `sample_of`
instead.

A row agrees only when **both** digests match. The text digest alone would let a tokenizer
difference pass as agreement; the token digest alone would let a decoding difference pass
(`test_both_digests_must_match`).

## The verification predicate

`verify_sample(challenge, target_cert, *, sth, battery_item_ids, target_index)` returns the
list of reasons this is not a valid sampled challenge; empty means it is one. It is computed
from bytes a third party holds — the two certs, a tree head, the battery's item ids, the
target's leaf index — and it checks:

1. the cert is a `challenge` and `body.sample.selector.method` is `styxx.v8/challenge/sample/1`;
2. the selector's `log_id`, `tree_size` and `root_hash` are the given head's, member for member;
3. **the head already commits to the target**: `target_index < tree_size`. A caller who passes
   no index gets a reason beginning `UNCHECKED:` rather than a pass — without it the issuer may
   have minted the target after seeing the root, and the design's central property is gone;
4. the selector's `target` is the target cert's id, and the challenge names `target` exactly
   once (§9's C-DUPREF rule, applied here);
5. `selector.challenger_key` is the challenge cert's own `issuer.key` — no drawing with one key
   and signing with another;
6. `n` is the battery's item count and the target cert's items are exactly that battery;
7. **the drawn ids ARE `select_items(battery_item_ids, selector)`, in order** — this is the one
   that stops a challenger picking its own twenty;
8. every row quotes the target's own recorded digests for its item — a challenger cannot
   misquote the cert it is challenging;
9. every row's `agree` follows from its four digests;
10. `disagreements` and `sampled_exact_distance` follow from the rows;
11. `selector_id` is the selector's digest, `sample_of` is `exact`, and a synthetic challenge
    does not challenge a measured target (§9's S9-03 marker, both directions).

Signature checking is **not** in this list. `cert.check` and the log own that, and a caller who
skips them holds an unsigned claim whatever this predicate returns.

Each of 1–11 has a test that constructs the corresponding forgery against the published
certificate and asserts the refusal: a challenger-chosen sample, a misquoted target digest, a
flipped `agree` flag with its two derived counts, a substituted `selector_id`, a key mismatch,
a substituted head, a target outside the head, a duplicated `target` ref, a synthetic
mismatch, a wrong battery, and a redacted target that carries no per-item digests to sample.

## What a k-item challenge establishes, and what it does not

**Establishes.** A sampled challenge with k items, zero disagreements and a verified selection
establishes that a second key ran k of the n battery items under the target's recipe and
observed the output digests the target cert records for exactly those k items, where the k were
fixed by a tree head the target cert was already inside. It is evidence against the hypothesis
that a fraction f of the target's items were fabricated, at the strength tabulated below — and
at no strength at all against a challenger who colluded, since a colluding challenger can
report digests it never computed.

**Does not establish.**

- **Nothing about the noise floor.** A floor is a function of R runs (§5.1); one challenger run
  measures no floor. The module computes none, compares no sampled distance to `target_floor`,
  and emits no `coverage`. A reader who wants "is this difference inside the machine's own
  variability" needs the challenger's own multi-run floor, which is a `prereg noise-plan` plus R
  runs and is **not** what this design makes cheap. A `disagreements > 0` result is a
  disagreement, not drift.
- **Nothing about `seqlp`, `topk`, `resid` or `lens`.** Those are numeric channels whose
  Appendix B distances are means over per-item quantities. A sampled version is constructible
  the same way and is not built here, because each carries its own tolerance question and the
  digest comparison carries none.
- **Nothing about the n − k items not drawn**, except through the stated sampling inference,
  which is a statement about a procedure and not about any particular unsampled item.
- **Nothing about whether the target's subject is what its cert names.** That is the
  runner-reported-identity check repaired earlier (`runner.py`), and it is a separate predicate.

## Detection power

`n = 64`. `m = round(f·n)` is the integer count of fabricated items; `f_effective = m/n` is
reported because at n = 64 the two differ — f = 0.01 asks for 0.64 items and gets 1. All
figures from `sampled_challenge.power_table()`, pinned in `test_power_pinned_values` and
`test_power_table_row_shape`.

**Hypergeometric — what `select_items` actually does (without replacement):**

| k | f=0.01 (m=1) | f=0.05 (m=3) | f=0.1 (m=6) | f=0.25 (m=16) | f=0.5 (m=32) |
|---|---|---|---|---|---|
| 1 | 0.0156 | 0.0469 | 0.0938 | 0.2500 | 0.5000 |
| 2 | 0.0312 | 0.0923 | 0.1801 | 0.4405 | 0.7540 |
| 4 | 0.0625 | 0.1787 | 0.3323 | 0.6938 | 0.9434 |
| 8 | 0.1250 | 0.3347 | 0.5669 | 0.9147 | 0.9976 |
| 16 | 0.2500 | 0.5849 | 0.8363 | 0.9954 | 1.0000 |
| 32 | 0.5000 | 0.8810 | 0.9879 | 1.0000 | 1.0000 |
| 64 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**With replacement — `1 − (1−f)^k`, the weaker figure, kept because it is what a colluding
challenger pays per grinding attempt:**

| k | f=0.01 | f=0.05 | f=0.1 | f=0.25 | f=0.5 |
|---|---|---|---|---|---|
| 1 | 0.0100 | 0.0500 | 0.1000 | 0.2500 | 0.5000 |
| 2 | 0.0199 | 0.0975 | 0.1900 | 0.4375 | 0.7500 |
| 4 | 0.0394 | 0.1855 | 0.3439 | 0.6836 | 0.9375 |
| 8 | 0.0773 | 0.3366 | 0.5695 | 0.8999 | 0.9961 |
| 16 | 0.1485 | 0.5599 | 0.8147 | 0.9900 | 1.0000 |
| 32 | 0.2750 | 0.8063 | 0.9657 | 0.9999 | 1.0000 |
| 64 | 0.4744 | 0.9625 | 0.9988 | 1.0000 | 1.0000 |

**k needed for 95% detection at n = 64:**

| f | m = round(f·64) | k, without replacement | k, with replacement | cost as a fraction of the battery |
|---|---|---|---|---|
| 0.01 | 1 | **61** | 299 | 0.953 |
| 0.05 | 3 | **40** | 59 | 0.625 |
| 0.1 | 6 | **25** | 29 | 0.391 |
| 0.25 | 16 | **10** | 11 | 0.156 |
| 0.5 | 32 | **5** | 5 | 0.078 |

The with-replacement column at f = 0.01 exceeds n, which is the row where the two models come
apart: 299 draws with replacement, or 61 of 64 without. A single fabricated item in a 64-item
battery is not cheaply detectable by sampling, and the table says so rather than leaving it to
be discovered.

**The one row the prompt named**: at f = 0.1, k = 20 the with-replacement figure is
**0.8784** and the hypergeometric figure (m = 6) is **0.9058**, for 20/64 = 0.3125 of a battery
re-run.

### The power was measured, not only computed

`test_empirical_detection_matches_the_formula` replaces 6 of the published cert's 64 items with
fabricated digests, then runs 2000 challengers who each draw k = 20 with their own nonce and
report the original digests. The rate at which a challenge lands a disagreement is asserted
against `detection_hypergeometric(64, 6, 20)` = 0.9058 within four standard errors. This lab
holds that an agreement number without its detection power is not a number; the same rule
applies to a detection number that was only derived.

## What a colluding challenger can still do

They can do a great deal, and the mechanism is worth exactly as much as this section says.

**1. They can report digests they never computed.** A challenger who never loads a model and
copies the target's own digests produces a challenge that passes every one of the eleven
conditions. Nothing over these bytes reaches that, and no sampling design of any k changes it.
This is `THE_BOUNDARY`'s invariant arriving intact: a second party's bytes constrain a first
claim only if the second party is actually a second party. The quantity that survives is the
count of **distinct keys in a reader's own trust file** that have challenged a cert — §9's
Disputed threshold, a client-side judgment, not a log predicate.

**2. In mode `head`, they can grind the sample.** The nonce is free and the key is free — an
ed25519 key costs nothing to make — so a challenger colluding with the issuer redraws until the
sample misses every fabricated item. The cost is `1/(1−f)^k`:

| m fabricated (of 64) | expected redraws at k=20 | expected redraws at the k for 95% |
|---|---|---|
| 1 | 1.37 | 2.61 (k=61) |
| 3 | 2.61 | 6.82 (k=40) |
| 6 | 7.16 | 11.72 (k=25) |
| 16 | 315.34 | 17.76 (k=10) |
| 32 | 1048576 | 32.00 (k=5) |

Seven redraws is not a defence. `test_grinding_is_cheap_and_the_module_says_so` performs the
grind against the published item ids — 20 independent runs, each finding a clean nonce, mean
cost tracking the predicted 7.16 — so the weakness is a measurement in the test suite rather
than a caveat in prose. **Mode `head` removes the issuer's choice and does not remove the
challenger's.**

**3. In mode `commit-then-head`, they can abort.** The two-phase form takes the choice from
both: the challenger logs (target, k, nonce, key) at some index, and the sample is fixed by the
**first** head whose `tree_size` exceeds that index — a root the challenger did not know when
committing. Two residues remain, and neither is closed by any predicate over these bytes. The
log operator chooses what is appended and when, and therefore what the next root is. And a
challenger who dislikes the drawn sample can simply never publish the challenge, so the
disclosure that matters is the number of **commitments with no matching challenge** per key,
and a challenger's commitment-to-challenge ratio is a number a reader should want.

**4. They can pick a small k.** Nothing stops a challenge at k = 1. The defence is arithmetic
rather than a rule: a k-item challenge claims k-item power, the table above is the discount, and
a reader who treats every challenge as worth 1 has misread the mechanism.

Stated once: **sampling changes the cost of being a second party. It does not change the trust
model.** A design whose challenges are all cheap and all colluding is worse than no challenges,
because it publishes a reproduction count that means nothing.

## What is owed, and cannot be done in this module

- **A partial fingerprint is a new forgery surface.** The challenger's `own` cert carries k of
  n items. It carries `body.sample` so a reader holding it alone cannot mistake it for a
  fingerprint of the battery, but the schema and the log are the only places that can *refuse*
  one where a full measurement is required: as a floor `run`, as a `previous`, as a `--ref`
  baseline. Unrefused, a k-item cert is a partial measurement wearing a full measurement's
  type.
- **"The first head above the commitment index"** is a predicate over the log's own entry
  order. This module holds no log and cannot check it.
- **"The head commits to the target"** needs the target's leaf index, which comes from the log.
  The module reports `UNCHECKED:` when a caller does not supply one; the log should supply it
  and never accept the unchecked form.
- **`schema/challenge.json`** does not know about `sample` or `sample_of`, and §9 of the spec
  describes only the full-battery form. Both are edits to files this work does not own.
- **The reproduction count** — `THE_BOUNDARY`'s consequence 2, the number of independent keys
  that ran this battery while running it was possible — is still computed nowhere, and a
  sampled challenge makes it cheaper to raise without making it more meaningful. It should be
  reported as a k-weighted quantity or not at all.

## Limits

No challenge has been run, by this lab or anyone, at any k. Every number here is either a
combinatorial identity or a simulation over the published artifact's item ids; none is a
reproduction of the model. The design has been attacked by its author for one session, which
this lab's own record says is worth little: the boundary paper reports its author's
impossibility claims running 0 for 5 in one day. Two sentences in this document are of that
form — that a colluding challenger's fabricated digests are reachable by nothing over these
bytes, and that the log operator's control of the next root is not closed by any predicate over
them — and both should be read as conjectures with a poor base rate behind them rather than as
findings.
