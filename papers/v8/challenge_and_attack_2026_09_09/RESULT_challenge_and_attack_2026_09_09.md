# RESULT — an adversarial pass over two defenses, and the challenge mechanism run end to end

Fathom Lab · 2026-09-09 · **A record of attacks that succeeded, on one machine, run by the party who
holds every private key.** Three agents ran against the v8 worktree: one exercised §9's challenge
mechanism, two attacked the two defenses added since `../vacuous_floor_2026_09_09/` — the append-time
check that a noise floor honoured its plan, and the §5.5 baseline rule with the log around it.
Twenty-five constructions defeated a defense. Each is written below with the exact steps that
produced it.

This is **not** a security review, not a completed repair, and not evidence about anyone else's
machine. Nothing here was sworn, nothing was committed, and nothing was written into
`C:\Users\heyzo\clawd\wt\v8` — the worktree is byte-for-byte as it was found, and the published log
of `../first_verdict_2026_09_09/` still hashes to `e1fe297b96b1075f57611a1ea42acee8` with 7 entries,
1 tree head and 1 roster key. Every artifact lives under the session scratchpad. The attacker held
the issuer key and the log key, both of which the earlier run left on disk at
`scratchpad/firstlog/out4/{issuer.pem,log.pem}`; that is the threat model these results describe and
the only one they describe.

## What was exercised and what was attacked

**Exercised.** §9's challenge mechanism, end to end, on a copy of the `first_verdict` log: a second
key generated on this machine, its own reproduction of the canonical recipe, four challenge certs
minted, appended and cross-verified. 28 of 28 driver commands exited exactly as predicted, 368.1 s
wall, six GPU runs of 64 prompts at 50.8–93.9 s each, one RTX 4070 with 8 GB, `HF_HUB_OFFLINE=1`.
The challenged log grew from 7 to 13 entries and from 1 to 2 roster keys, with tree heads at
`tree_size` 7 and 13.

**Attacked, defense 1.** `log.py`'s `_check_floor_honours_its_plan` (lines 441–557), the repair
written after the zero floor of `../vacuous_floor_2026_09_09/`. Eight constructions were built; `log
append` accepted 8 of 8 with exit 0. Six of them carry a floor of exactly `0.0` on exact, seqlp and
topk with all ten pairwise distances 0.

**Attacked, defense 2.** §5.5's baseline rule and the log layer beneath it — `Log.append`,
`verify_entry`, `mirror()`, the issuer roster, the tree heads. Six named attacks; five defeated the
system outright and one (truncation) defeated a reader who holds no earlier head.

## The challenge mechanism: the four cases

§9 defines four things a challenge is supposed to do. Three of the four ran. One has no code.

**Case 1 — a reproduction that agrees.** A second key
(`ed25519:TYemeLwfqzE1jIu3nzdu-COgFGxTLBwtv-TX0m6ab1s`) re-ran the canonical recipe at bfloat16, one
run, 77.3 s, and filed the result as a challenge against the bf16 canonical:

```
exact  distance=0.000000000  floor=0.046875000    ratio=0.0
seqlp  distance=0.000000000  floor=0.036070694    ratio=0.0
topk   distance=0.000000000  floor=2.140233900    ratio=0.0
overall: same (sensitivity unmeasured)   (exit 2)
```

The challenger's fingerprint body was byte-identical to the canonical's on the exact channel. This
works, and it says one thing: this box repeats itself at `batch_size=1`.

**Case 2 — a challenge that disagrees.** Two ran. Both minted, both appended (indices 10 and 11),
both re-verified. Both are listed under *Defeated* below, because what they demonstrate is that the
mechanism will sign a disagreement about a subject it never checked.

**Case 3 — a challenge cert that survives verification and re-appending.** `log verify-cert` exit 0;
`log append` exit 0. Two negative controls were refused with exit 4: flipping one distance in the
challenge body gives `id: does not recompute`, and substituting the target issuer's key gives a
refusal, so a challenge cannot be re-attributed to the party it challenges. The challenge's target
ref resolved to entry 6 and its own ref to entry 7, both type-checked at append.

**Case 4 — the client-computed `Disputed` status, and the issuer's `response` that resolves a
challenge. Neither exists.** `grep -rn` over `styxx/v8/*.py` for `disputed`, `trust.json` and
`coverage-contested` returns zero hits. `rglob('trust.json')` across the worktree finds no file. The
CLI verbs are `battery`, `fingerprint`, `key`, `log`, `prereg`, `verify` — there is no `attest`, no
`view`, no viewer, and nothing that displays a challenge at all. `verify --result-out` signs a
`kind=verify` result; no path mints `kind=response` with a `challenge` + `disposition` + `detail`
body. The log now holds four unresolved challenges and there is no way to answer them. The
`Disputed` predicate was computed by hand over the resulting log — 1 distinct challenging key against
a threshold of 2, so not Disputed — and that computation lives in a driver script, not in the system.

A fifth thing was run because §9 requires it: a challenge declaring an environment outside the
floor's coverage. `--environment` naming an RTX 4090 and driver 999.99 produced coverage
`beyond-floor-coverage` with `coverage_diff = [hardware.driver, hardware.gpu]`, and every channel was
marked beyond coverage instead of compared. That is the correct classification, and it is the shape
every genuinely external challenger gets.

---

## Defeated — twenty-five constructions, with the steps that produced them

These are the deliverable. Twenty-five: eight against the floor check, eleven against the baseline
rule and the log, six against the challenge mechanism.

### Against the noise-floor check (8)

**F1. The vacuous floor, re-obtained by deleting the factor that gave it away.** Recipe: recipe
`batch_size=1`; plan declares one factor, `item_order=canonical|p1|p2|p3|p4`, R=5. The five runs
really do take five different item orders, so the predicate is satisfied on its own terms — but at
batch size 1 each item is its own forward pass, so permuting the order permutes independent
computations. `log append` exit 0. Floor `0.0` on all three channels, all ten distances exactly 0,
and the cert advertises `runs=5, pairs=10, alpha_single=1/11, alpha_overall=0` — statistics identical
to the honest `first_verdict` floor. `verify --diff` against fp16 then reads `seqlp
distance=0.001683480 floor=0.000000000 exceeds_floor`, `topk distance=0.317773350 floor=0.000000000
exceeds_floor`, exit 2. The repair catches a plan that lies about what it varied. It does not catch
a plan that declares only the harmless factor.

**F2. A plan whose only factor declares a single value.** `prereg noise-plan --runs 5 --nuisance
batch_size=1` is accepted by the CLI. The append check skips every factor with fewer than two values
("a factor declaring a single value commits to no variation and is skipped"). Five byte-identical
runs, append exit 0, floor `0.0/0.0/0.0`, and the verdict header prints `covers=batch_size` — a floor
that covers nothing while naming a factor. One CLI flag, no library call, no forgery.

**F3. A factor that is applied and cannot be seen.** `padding_side=left|right` with recipe
`batch_size=1`. The value is written into each run's recipe and `runner_hf` sets
`tok.padding_side` from it, so nothing is faked; at batch 1 the tokenizer pads a single sequence to
its own length and no pad token is ever emitted. Assignments left, right, left, right, left. Append
exit 0, floor `0.0/0.0/0.0`, `covers=padding_side`.

**F4. Two batch sizes that are one batch size.** `runner_hf.run` batches with `range(0, len(items),
batch_size)`, so on an 8-item battery every value ≥ 8 is one batch of 8. Recipe `batch_size=8`, plan
`batch_size=8|16|32`: the certs record 8, 16 and 32 — three distinct values, no coercion collapse —
and the three executions are identical. Append exit 0, floor `0.0/0.0/0.0`, verdict `seqlp
0.001589091`, `topk 0.264210369`, both exceeding a zero floor.

**F5. A real factor on a battery where it provably cannot matter.** Battery of one item, recipe
`batch_size=1`, plan `batch_size=1|8|32`. Nothing about the factor is fake; there is never more than
one item to batch. Append exit 0, floor `0.0/0.0/0.0`, verdict `seqlp 0.000222533`, `topk
0.233994863`.

**F6. `noise_floor.runs` naming certs the floor was not computed over.**
`attach_floor(canonical_body, run_bodies, plan_id, run_ids, covers, not_covered)` computes
`per_channel` from `run_bodies` and writes `run_ids` beside it with nothing tying the two together;
the append check reads the assignments of the certs named in `runs`. Recipe: mint five honest runs
under `batch_size=1|8|32`, append them, then sign a canonical whose floor came from five
item-order-only runs at batch 1 and whose `runs` list names the honest four. Append exit 0. Floor
`0.0/0.0/0.0` while the named runs really produced `0 / 0.000453822 / 0.194520062`.

**F7. The floor over a hand-picked subset of the preregistered runs — run against the lab's own
published verdict.** The check never asks whether the named runs are *all* the runs the plan
committed to. R is fixed before the runs; the floor may be chosen after them. On
`papers/v8/first_verdict_2026_09_09` — same five run bodies, same plan, same battery, same issuer
key, no new runs and no model loaded — twelve subsets satisfy the predicate, with topk floors from
0.813925214 to 2.140233900. Naming only `run2` (one pair, though the plan fixed R=5 and the CLI
enforces R≥5 at mint) appended with exit 0 and cut the floor: exact 0.046875 → 0.015625, seqlp
0.036070694 → 0.017551866, topk 2.140233900 → 0.813925214. The published sentence "One channel says
`same`. That is the whole point of this run." then becomes `topk distance=1.407087824
floor=0.813925214 ratio=1.728767950 exceeds_floor`. The verdict word flips without touching a run
cert. This is the worst of the eight.

**F8. The predicate satisfied with structural zeros.** Plan `batch_size=8|16|4` on 8 items gives
assignments 8, 16, 4, 8, 16: three declared values, three recorded values, and 6 of the 10 pairwise
distances exactly 0 because 8 and 16 are the same execution. The whole floor rests on one real
condition (exact 0, seqlp 0.000287411, topk 0.133138261) compared four times. Append exit 0; the
verdict then reads ratios of 5.53 and 1.98 against it.

### Against the baseline rule and the log (11)

**L1. Move your own baseline.** Recipe: take the canonical cert at index 6, multiply every
`body.noise_floor.per_channel[c].floor` by 100, leave refs untouched, change `created`, re-sign with
the issuer key, `Log.append`. Lands at index 7. Cause: `log.py:432-439` refuses only when `not
_same_noise_plan(cert, previous)`, and `_same_noise_plan` (`log.py:762-769`) is true whenever both
certs carry the same `noise_plan` ref — which the whole floor set does by construction. The escape
hatch written for the R−1 run certs admits an unlimited number of canonical fingerprints with
fabricated floors, with no previous ref and no diff. Repair: gate that escape on `noise_floor` being
absent from the appending cert, which is what §5.5's Scope paragraph says.

**L2. The same move, pointed somewhere convenient.** Same cert with floors ×1000 plus a refs entry
`{'role':'previous','id':<the run cert at index 2>}`. Appends at index 8. Two holes at once:
`cert.ROLE_TYPES['previous']` is `None` (`cert.py:108`), which `log.py:417-419` turns into "want the
appending cert's own type", so `previous` accepts any fingerprint — a non-canonical run cert, a
floorless cert, any subject in the log. And §5.5's "submitted together with a `verify --diff` result
against it" is enforced nowhere: `grep` for `accompan` across `styxx/v8` returns nothing.

**L3. The `baseline-gap` signal does not exist.** §5.5 gives up the append-time refusal in exchange
for a client-computed `baseline-gap` status printed by `verify` and shown in §12; §6 lists
`baseline_gap` in the verify result body. `grep -rn 'baseline_gap\|baseline-gap' styxx/v8/*.py`
returns zero lines. Nothing computes it, nothing prints it, the result cert does not carry it. The
moved baseline of L1 is not merely appendable; it is invisible. One side of §5.5's trade was paid.

**L4. The floor numerals are never re-derived.** `verify.py`'s `_floor_map` (lines 205–224) reads
`body.noise_floor.per_channel[c]['floor']` as a bare float. `_check_floor_honours_its_plan` compares
nuisance assignments only — `'distance' in its source` is False. Nothing anywhere recomputes
`floor_c` from the referenced run certs, though §5.3 requires exactly that of the sensitivity
receipt. Measured on the real `first_verdict` certs, editing two numerals:

```
real canonical    exact 0.062500000 / 0.046875000   ratio 1.333333333  exceeds_floor
                  seqlp 0.058564664 / 0.036070694   ratio 1.623607907  exceeds_floor
                  overall exceeds_floor
forged (x100)     exact 0.062500000 / 4.687500000   ratio 0.013333333  same
                  seqlp 0.058564664 / 3.607069400   ratio 0.016236079  same
                  topk  1.407087824 / 214.023390000 ratio 0.006574458  same
                  overall inconclusive (cross-subject)
```

**L5. The same hole on the error rate.** Set `per_channel[c].runs = 1000`, `pairs = 499500`,
`alpha_single = 2.002e-6`, `alpha_overall = 0.000002`, leave `noise_floor.runs` naming its four run
certs. Appends. `alpha_single` is defined as `1/(pairs+1)` and `pairs` as the pairwise count over
the named runs; both are derivable and neither is derived. A reader who trusts the printed size is
off by five orders of magnitude.

**L6. Backdate the evidence.** On a fresh log: (1) append the battery; (2) append the four run certs
with their `noise_plan` refs stripped, chained to each other by `previous` refs so the baseline rule
lets each in; (3) append the reference run the same way; (4) *now* mint a noise plan whose declared
nuisance values are chosen from the assignments the runs already used, and append it at index 6; (5)
append the canonical at index 7 naming the late plan and the early runs. Every step appended. The
log's index order reads: 0 battery, 1–5 fingerprints, 6 prereg, 7 canonical — the preregistration
sits after every run it governs, and nothing objected. Two defects: the check reads the run certs'
`body.nuisance` and never their refs, so §5.1 step 1 is unenforced; and the check requires only that
observed values be a subset of declared values, so an issuer who runs first can always mint a fitting
plan. §8.6 admits the log cannot date a prereg; §5.1 step 5 asserts an append order the code does
not enforce.

**L7. Replay across logs.** Generate a new log key, `Log.init` a second log with a roster naming the
same issuer, append gold entries 0–6 verbatim. All seven appended, bytes identical, Merkle root
identical (`sha256:d61a6a78…` in both), `log prove 6` exit 0, `log verify-inclusion` exit 0 under the
second log's own head, every `verify_entry` True. Gold `log_id sha256:88300489…`, replay `log_id
sha256:97b21ea3…`, and the gold `log_id` string appears nowhere inside the certs. Nothing binds a
cert to a log: "this cert is in the log" is not a checkable statement, only "this cert is in some
log".

**L8. Replay minus the inconvenient entry.** Into a fresh log append gold entries 0–5, then the L1
forgery instead of the real canonical. All seven appended. The resulting log holds exactly one
canonical fingerprint for that subject and it is the fabricated one, with a valid tree, valid proofs,
and a roster naming the real issuer. No `baseline-gap`, no `previous` ref, and nothing to compare
against unless the reader independently holds the gold head.

**L9. Swap a run, three ways.** One plan A declaring `batch_size=1|8|32` only, so a cert lacking
`item_order` can pose as a run. (a) Different subject: take the fp16 fingerprint, set
`body.run_index=9`, add a `noise_plan` ref to plan A, re-sign — appended at index 4. (b) Different
battery: mint a 32-item pool battery, take a bf16 run cert, repoint `recipe.battery` and the battery
ref — appended at index 5. (c) Different plan: three run certs whose `noise_plan` ref names an
unrelated plan B — appended at 6, 7, 8. Then a canonical with `noise_floor.plan = plan A` and
`noise_floor.runs = [fp16 run, 32-item run, three plan-B runs]`, `per_channel` claiming `runs=6
pairs=15` — appended at index 9. `verify_entry` True, `mirror()` `verified=True tamper=[]
misbehaviour=[]`, and `verify --diff` prints a full verdict. `_check_floor_honours_its_plan`
(`log.py:493-557`) resolves each run and reads only `body.nuisance[factor]`; it never compares the
run's subject identity, `recipe_core`, `recipe.battery` or `noise_plan` ref against the canonical's.

**L10. Forge the roster, and then delete it.** (a) Append one object `{'name':'independent
lab','key':<new key>,'from_index':0,'retired_at_index':null}` to `keys/issuers.json`. Before the
edit: `issuer: key ed25519:v-Toz… is not in the roster at index 7`. After it, the identical bytes
append at index 7; the Merkle root over the first 7 entries is unchanged, the shipped head still
verifies and reproduces, and `mirror()` reports `verified=True, tamper=[], misbehaviour=[]` — it
copies `keys/` verbatim (`log.py:978`) and never reads `issuers.json`. (b) Emptying the roster
changes nothing about entries already in, because `verify_entry` checks the signature under the key
inside the cert. (c) Deleting `keys/issuers.json` makes `Log.issuers()` return `None`
(`log.py:328-334`), and append step 2 is guarded by `if roster is not None` (`log.py:399`) — the
entire issuer check is skipped and the rogue cert appends. An unsigned file whose absence disables
the check is a fail-open default. §8.2 states there is no `keys/issuers.json` in the layout while
`log.py` writes one at init and enforces it; code and spec disagree about whether the file exists.

**L11. Truncate, against a reader holding no earlier head.** Copy the log, delete
`entries/000000/00000005.*` and `00000006.*` and every file in `sth/`, then `log sth --key log.pem`
— a valid head at `tree_size` 5, root `sha256:58a81632…`. A mirror run by a party with no older head
reports `verified=True, entries=5, tamper=[], misbehaviour=[], unpublished=[]`. The canonical
fingerprint and one floor run are gone with no trace inside the artifact; `log.inclusion(6, 5)` now
raises `index: 6 outside 0..4`. A party who pinned the earlier head catches it — see *What held*.
The honest statement is that an artifact directory alone proves nothing about what was removed from
it, which is what §8.0 says.

### Against the challenge mechanism (6)

**C1. Subject identity is never checked, at any layer.** §9 rule 1 as repaired by finding S9-01 is
in the spec text and in no code path. Recipe: (1) `log append
papers/v8/first_verdict_2026_09_09/fp_fp16/fingerprint-canonical-d92ebe2089f9.json --log <copy>` —
exit 0, index 9; (2) `verify --ref <bf16 canonical> --log <copy> --runner hf --dtype bfloat16
--challenge --own sha256:d92ebe2089f9de0cc3e634a9a89751b97e9033025c4280c2a1c9e1e1cedf0f18 --key
challenger.pem --challenge-out challenge_wrong_subject.json` — exit 2; (3) `log verify-cert` exit 0;
(4) `log append` exit 0, index 11. An fp16 cert now sits as the `own` half of a challenge against the
bf16 canonical. Run by hand, `cert.comparable(target, own)` gives `['cross-subject:precision']` and
the `S_identity` mismatch is `['precision']`. Nothing in the CLI, `cert.check`, `log.append` or the
JavaScript verifier runs that check.

**C2. The §6 subject guard in `verify --ref` is dead code with the real runner, which turns a
swapped-weights run into a signed `drift` claim against someone else's cert.** Recipe: `verify --ref
<bf16 canonical> --log <copy> --runner hf --snapshot <the same gemma-2-2b-it snapshot> --dtype
float16 --challenge --own <fp16 cert id> --key challenger.pem --challenge-out
challenge_fp16_weights.json`. Result: exit 1, verdict `drift`, `mismatched=[]`, coverage `within`,
distances 0.062500000 / 0.058564664 / 1.407087824, 93.9 s, with a confirmation run that agreed.
Cause: `runner_hf.TransformersRunner` has no `subject` member, so `verify._observed_subject`
(`verify.py` ~line 630) falls back to the cert's own subject and
`certmod.comparable(cert, {'subject': cert-subject-again})` is trivially empty. `--dtype` silently
changes what is loaded and the cert records the target's precision, not the loaded one. The
challenge appended at index 10 and the JavaScript verifier passed it.

**C3. A challenge cert asserts nothing about what was run.** All three challenges produced carry
`subject: {}` and `recipe: {}`; the body is `{per_channel, coverage, environment[, note]}` and
`environment` holds hardware and runtime only. A reader of `challenge_fp16_weights.json` cannot tell
from the cert that fp16 weights produced those distances. The only pointer is the `own` ref, which
nothing validates (C1).

**C4. §9's "a challenge landing below the target's floor is a reproduction and must be displayed as
one" is not implemented, because nothing displays challenges.** The Case 1 reproduction printed
`overall: same (sensitivity unmeasured)`; the strings `reproduction`, `reproduces` and `reproduced`
appear nowhere in the verdict, report, challenge body or result body. The coverage report likewise
is never labelled — it prints `beyond-floor-coverage`, which is at least the right classification.
There is no `attest` verb and no viewer, so §12's incident-and-dispute record has nothing to display
it in.

**C5. Admitting a second issuer is an unsigned, out-of-tree file edit with no receipt.** Recipe:
append `{"from_index": 7, "key": "<challenger pub>", "name": "…", "retired_at_index": null}` to
`<log>/keys/issuers.json` with a text editor. There is no `log add-issuer` verb; `log init --issuer`
is the only mint and it creates a log. The roster is outside the Merkle tree, so who was trusted
from which index leaves no evidence in any tree head, and the JavaScript verifier's 47 checks do not
look at it. (This is the same object as L10, reached from the challenge side; it is counted once
here and once there because the two paths reach it for different reasons and both need closing.)

**C6. A reproduction can only be filed by borrowing the target's own noise-plan ref.** The
challenger's honest reproduction, filed the obvious way, is refused: `baseline: a comparable
fingerprint is at index 6; a fingerprint that starts a new baseline needs a previous ref` (exit 4).
`fingerprint` has no `--previous` flag, so the only path past §5.5 was `--plan <the target's plan
id>` — a challenger must attach their run to the issuer's preregistration to be allowed to file it
at all. The refusal is the baseline rule working; the consequence is a challenge surface that
requires the challenger to adopt the target's prereg.

---

## What held, and why that is weaker evidence than it looks

Every item below is an attack that failed, or a control that was refused. An attack that fails may
mean the defense works, or may mean the attacker did not find the way in. These are not proofs.
Where a defense was never attacked at all, it is in *What is owed*, not here.

- **Duplicate id.** Re-appending a cert already present: `duplicate: sha256:73a09ffa… is already at
  index 6` (`log.py:428`).
- **Unresolvable refs.** A `previous` naming a cert absent from the log is refused (`log.py:411-416`);
  a canonical whose `noise_floor.plan` or `noise_floor.runs` name absent certs is refused
  (`log.py:507-523`) — this cost two failed attempts in L9 before the refs were fixed.
- **Embedded ids must appear in refs.** Stripping the `noise_plan` ref while
  `body.noise_floor.plan` still names it is refused by `cert.check`.
- **A declared factor pinned across the named runs is caught, with an exact message.** Control C1 of
  the floor pass (two batch-1 runs from `first_verdict`): `floor: the plan declares nuisance factor
  'batch_size' with values ['1','32','8'] and all 3 runs ran at '1'; a factor declared and never
  varied is measured by nothing…` — exit 4. The check does fire on the shape it was written for.
- **A floor over fewer than two runs is impossible.** `attach_floor` refuses with "a floor needs at
  least 2 runs, got 1" before any cert exists.
- **A plan declaring no factor is refused by the schema, not only the CLI.** Hand-signing
  `nuisance: []` fails `cert.check` with `schema[prereg]: body/nuisance: [] should be non-empty`.
  The docstring's "a plan carrying no nuisance list is skipped whole" is unreachable.
- **Mint-time refusals that work by name.** Two names for one item order
  (`item_order=canonical|reference`, and `canonical|CANONICAL` after strip+lower); an order factor
  unrealizable on the battery (`canonical|p1` on one item); a factor no runner can set (`gpu=4070|a100`
  → "this process can set batch_size, item_order, order, padding_side and nothing else"). Nine
  mint-time probes: 4 refused, 5 minted.
- **The roster check itself, when the file is present and honest.** `issuer: key ed25519:TYem… is
  not in the roster at index 7` (exit 4). L10 defeats the file, not the check.
- **Log-key custody.** `log sth` with a freshly generated key is refused (exit 4); only the log's own
  key seals a head.
- **Challenge cert integrity and attribution.** One distance changed → `log verify-cert` exit 4, `id:
  does not recompute`. The target issuer's key substituted in → exit 4. The reproduction verifies
  under the challenger's key and not the issuer's.
- **§9's coverage classification.** An environment outside the floor's `not_covered` set produced
  coverage `beyond-floor-coverage` with `coverage_diff` naming exactly `[hardware.driver,
  hardware.gpu]`, and no channel was compared. The rule that an out-of-coverage reproduction
  concludes nothing about drift is in code.
- **Truncation is detectable by a party holding an earlier head.** Pinned mirror on the truncated
  log: `verified=False, tamper=['sth <pinned>: covers 7 entries, the mirror holds 5']`. On the
  truncate-and-refill log: `sth: tree_size 7 has 2 different root_hash values` and `the entries do
  not reproduce the signed root at tree_size 7`. The old inclusion proof for entry 6 verifies against
  the old head and fails against the new one.
- **Merkle consistency and the second implementation.** `log verify-consistency 000000000007.json
  000000000013.json` exit 0: the `first_verdict` head is a prefix of the 13-entry challenged log.
  `node papers/v8/first_log_2026_09_09/cross_verify.js <challenged log>` exit 0, 47 of 47 checks
  agree over 13 entries and 2 heads, including an inclusion proof for leaf 12 and its wrong-leaf
  negative control. Baseline on the untouched 7-entry log was 25 of 25.
- **Entry integrity.** `verify_entry` re-derives id, signature, canonical bytes, CR/newline handling,
  meta agreement and leaf hash from the bytes. No modified entry got past it — and none was tried:
  every cert appended above was legitimately signed by the key the roster names. That defense is
  unexercised, not confirmed.
- **The reproduction number.** A second key re-running the canonical recipe got `0.000000000` on all
  three channels, and its fingerprint body was byte-identical to the canonical's on the exact
  channel. That is a statement about this machine repeating itself, not about reproduction.

The last two entries are the pattern for the whole section. `verify_entry`, `cert.check`, the JCS
canonicalization, the Merkle code and the Ed25519 layer were not attacked; the attacker signed
everything honestly and worked above them. `mirror()` reported `verified=True` on a forged roster, on
a swapped floor, on a truncated log and on a log with entries no head covers — so its True is not
evidence of anything the attacks above touched.

## What is owed

**No independent party, and no outside reproduction.** Every key, every run and every byte here came
from one machine and one operator, with both private keys on disk. A second key generated on the
same machine by the same person is not an independent party; the Case 1 challenge is this box
challenging itself, and the `0.000000000` distance is a statement about repetition, not
reproduction. A genuine external challenger would differ on `hardware.gpu` and `hardware.driver` and
would land in the `beyond-floor-coverage` branch, which concludes nothing about the model. Nothing in
this document substitutes for an outside party reproducing any of it on hardware this lab does not
own, and no result here should be cited as if one had.

**Repairs, in the order the attacks argue for them.**

1. Recompute `floor_c` at append from the R run certs by Appendix B and refuse a mismatch (L4); this
   is what makes L1, L2, L8 and F6 pay.
2. Require `pairs == C(len(runs)+1, 2)` and `alpha_single == 1/(pairs+1)` (L5).
3. Require every cert in `noise_floor.runs` to satisfy `cert.comparable(run, canonical) == []`, to
   carry a `noise_plan` ref equal to `noise_floor.plan`, and to sit at a higher index than the plan
   (L6, L9).
4. Require `noise_floor.runs` to be all R runs the plan committed to, not a subset (F7).
5. Gate the `_same_noise_plan` escape on `noise_floor` being absent from the appending cert (L1);
   type-check `previous` and require the accompanying `--diff` result (L2); implement `baseline_gap`
   (L3).
6. Make a missing roster refuse rather than admit (L10c); publish the roster as a document-kind
   result cert inside the tree, and have `mirror()` diff the file against it (L10, C5).
7. Give the runner a real `subject` and compare it against the cert (C2); check `S_identity` at
   `cert.check` and at `log.append` for challenges (C1); carry `subject` and `recipe` in the
   challenge body (C3).
8. Decide what a floor covering a factor that cannot vary means, and refuse it: single-valued factors
   (F2), factors the battery cannot realize (F4, F5), factors invisible at the recipe's batch size
   (F1, F3), and value sets that collapse to fewer distinct executions than declared (F8).

**Untested, therefore unclaimed.** `verify --ref`'s confirmation path against a vacuous floor —
every floor verdict above came from `verify --diff`, which cannot reach `drift`. Whether a forged
sensitivity receipt lets a hostile issuer reach a bare `same` with exit 0; that is the verdict most
worth forging and it was not attempted. The submission path of §8.3 (an issuer without the log key,
through CI) — no CI exists. `cert.check`, JCS, Merkle and Ed25519. `log verify` or an Appendix D walk
over any attacked log. A recipe or battery that empties every output, which would give an honest,
genuinely varied plan both a zero floor and a zero distance. Whether `previous` accepts a fingerprint
of a different subject that *is* in the log. The `--diff` path into a §9 challenge. Two distinct
challenging keys, so the `Disputed` threshold of 2 was never crossed even by hand. The `result` cert
of kind `response`, which does not exist.

**Loose ends.** `mirror()` overwrites the locally computed proof's `first_root` and `second_root`
with the STHs' claimed values before calling `verify_consistency` (`log.py:1118-1119`), making four
of that function's equality checks unreachable inside `mirror`; the Merkle check still uses the local
path, so this was not turned into an attack, but the dead checks should go. `mirror()` reports
`verified=True` on a log with entries no head covers. `--challenge` on an `unavailable` outcome
discards the verification report and loses `unavailable_reason`. Three machine events were not
root-caused: one `verify --ref` returning `unavailable` after 14.2 s, one fp16 model load exiting
`0xC0000005`, and two CUDA failures caused by another session holding ~5.6 GB of the 8 GB card. One
attack leg's transcript (`results.json`) was overwritten by a later run; its certs and logs are on
disk and `summary.py` re-derives the outcomes from the bytes.

## Limits

One machine, one RTX 4070 with 8 GB, one model (`google/gemma-2-2b-it`, bf16 and fp16), batteries of
64, 32, 8 and 1 items, one operator holding every private key. Twenty-five defeated constructions is
a count of what one adversarial pass found in one afternoon against two defenses and one mechanism;
it is a lower bound on what is there and says nothing about what is not. Eight of the twenty-five
were found by reading code and could have been found by anyone with the repository; the rest required
the issuer key, which this threat model grants. The floor attacks were judged only at `log append`
and at `verify --diff`. The challenge pass ran 28 commands, all as predicted, but three of §9's four
cases were exercised against a target the challenger already agreed with. Nothing here was sworn,
nothing was committed, and nothing was reproduced by anyone outside this lab.
