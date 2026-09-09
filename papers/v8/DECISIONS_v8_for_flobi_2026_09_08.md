# v8 — the decisions only you can make

Fathom Lab · 2026-09-08 · **A decision sheet, not a result.** Thirty-one items the adversarial review
could not decide, grouped into eight questions, each with its options, a recommendation, and the finding
ids that raise it. Full texts: `REVIEW_v8_spec_2026_09_07.md`. The amended draft (`SPEC_v8_v0.2_draft.md`)
already implements every recommendation marked **R** below and marks each spot `[OPERATOR-GATED …]`, so a
"yes to all recommendations" is one commit and no edits. Nothing is committed until you say so.

Two questions were added after that count and are not in it: **Q9** (2026-09-08, the three decisions that
expire) and **Q10** (2026-09-09, the four that the emptying of the class-two roster creates). Counting the
draft's marked blocks rather than this sheet's items, **twenty-six blocks stand operator-gated** as of
2026-09-09.

Six of these are blockers: entry #0's type (Q3), floor scope (Q4a), the floor quantum (Q4b), the identity
verdict (Q4c), the depth GPU (Q6c), and cross-hardware canaries (Q4a again). The rest can ship as written.

---

## Q1 — What is the v8 log called? · S0-01, S8-17 (major)

`styxx/charon.py` is shipped, sworn, red-teamed, and holds a 243-line log with its own vocabulary.
The v8 log is a different object: a Merkle tree with signed heads. Two objects, one name.

- **(a) R — rename the v8 log.** Charon v0.1 stays the ferry log, unrebuilt; its head is pinned and
  becomes an early entry of the new log. v8 prose says "the log" until you name it.
- (b) Number the v8 log charon v1.0 with a migration section: v0.1 frozen at its head, its seven statuses
  mapped to v8 verdicts, its lines appended under a role. More work, one name, and a vocabulary change
  that needs your signature anyway.

**Also**: v8's `drift` and charon v0.1's `DRIFT` mean different things (behaviour vs bytes). Whichever
option, one of them gets a qualifier.

## Q2 — Two other name collisions · S4-17 (minor), S11-01 (major)

- **"canary."** `SPEC_sworn_measurement_machinery_2026_09_05.md` already uses it for a planted false span.
  **R: rename the v8 object** (`sentinel-v1`, `edge-v1`, `probe-v1`) rather than re-issue a sworn spec.
  Do not commit entry #0 with the collision live.
- **CLI verbs.** `fingerprint`, `attest` and `log` are live 7.x subcommands with different meanings;
  the draft's removal table covered 4 of 47. **R: namespace v8** under `python -m styxx.v8` (already
  built that way), and ship Appendix E listing all 47 as kept / shim / removed.

## Q3 — The type set, and what entry #0 is · S0-02 (blocker), S3-08 (major)

Entry #0 is "this spec's hash", the hourly agent roll is a tree head, and the docket summary is prose.
None is a cert of any type in the draft, so all three fail the log's own append rules.

- **(1) R — least change:** entry #0 and the summary are `result` certs of kind `document`
  (`{path, commit, git_blob_sha256, eol:"lf"}` — the git blob, never the CRLF checkout); the hourly roll
  becomes a `sublog` cert. Type-set change, so it needs your signature.
- (2) Add a generic `attachment` type for all three.
- **Instrument readings** (docket item 5) also have no type. **R: `result` kind `reading`**, and state
  once that `lens` and `resid` are channels, never readings.

## Q4 — The statistics. The heart of it. · S2-01, S4-01, S4-02, S4-06, S5-01, S5-02, S5-03, S5-06, S5-09, S13-02

The draft could declare drift that is really nuisance, and could declare `same` with no power to detect
anything. Six sub-decisions:

**(a) Does a floor cross hardware?** *(blocker)* Floors are measured on one GPU; Appendix D invites
strangers on other hardware to challenge. Every honest cross-hardware challenge would land above a
single-machine floor and read as a dispute.
**R:** floors carry `covers` / `not_covered`; a run outside the coverage is `beyond-floor-coverage`,
never drift, never a dispute; a *public* drift claim additionally requires two distinct (gpu, driver)
pairs in the plan. Alternative: require a challenger-side floor too.

**(b) Can one flipped answer be drift?** *(blocker)* Canaries are selected for nuisance stability, so
their floor is 0 by construction, and then a single flip exceeds it.
**R:** floor runs must be *fresh* and disjoint from the selection sweep (recorded as `selection_runs`),
the cert carries `selection_biased: true`, and a discrete channel gets a quantum: `max(observed, 2/N)`,
so one flip is never drift. `ratio` is null when the floor is 0.

**(c) Is one changed anchor a swapped model?** *(blocker)* The draft says an anchor flip means `identity`,
which overrides every other verdict — an accusation with no null behind it.
**R:** `identity` fires only when a subject *hash* differs (weights, tokenizer, config), always, and is
plain fact; anchor flips become a counted `anchor_change` with their own floor and the same confirmation
run. This is already written into v0.2.

**(d) Who controls the floor?** The issuer chooses R, the nuisance set, and which runs count.
**R:** the nuisance plan is logged *before* the runs (a `noise-plan` prereg); every run is a logged
fingerprint referencing it; a floor whose runs are not all in the log is void.

**(e) What error rate does "drift" carry?** Max-of-pairwise at R=5 is roughly a 10% per-channel false
positive, times five channels, with "drift if any channel drifts".
**R:** print the nominal size (`alpha_single = 1/(pairs+1)`) beside every verdict. Then pick one:
**R+** R=8 minimum for anything public; or require two channels; or rename the single-run outcome
`exceeds-floor`. (v0.2 prints alpha and reserves `drift` for a confirmed run.)

**(f) May `same` ship without a power measurement?** An agreement number with no detection power is the
exact defect the lab caught twice before.
**R:** a `sensitivity` result cert must exist before any `same`, listing what the battery detects **and
its miss list**; `verify` prints both. You choose the positive-control set (minimum: the precision
variants and the predecessor revision).

**Plus:** `--diff` can't run a confirmation, so **R:** it gets `{same, exceeds_floor, identity,
inconclusive}`, uses the left cert's floor, prints `floor_owner`, and can never say `drift`.

## Q5 — Verdict word: `unmeasured` or `inconclusive`? · S5-07, S7-09 (major)

The shipped logprob gate already prints `unmeasured` for an absent measurement. The draft introduced
`inconclusive` for the same situation. **R: keep `unmeasured`** for an absent channel and reserve
`inconclusive` for "no floor on record". Renaming a shipped verdict is a separate, signed commit.

## Q6 — Scope and schedule · S14-03, S13-01, S7-03

- **(a) Relation to the plan of record** (`PLAN_the_next_level_2026_09_02.md`, sworn: four legs, the
  measurement waiting on your signature). The draft silently supersedes it. **R: v8 runs after legs 2
  and 4**, which are not edited; v8 inherits the claim ledger by reference. Nothing in v8 needs GPU
  before week 4.
- **(b) Docket models.** 9B/8B do not fit 8 GB VRAM; the draft's 48-hour docket was ~50× over budget.
  **R: restrict v8.0 dockets to what fits** (gemma-2-2b, Llama-3.2-3B, Qwen2.5-3B — all cached), state
  the ceiling, and fill the GPU-hour column from the first real docket instead of estimating it.
- **(c) The depth GPU** *(blocker)*. The draft schedules a "keystone v2 prereg" and a future public
  negative for an experiment already run and closed negative. **R: reassign that GPU time** to the
  canary sweeps and the plan-of-record measurement; the committed negative enters the log as a
  `document` cert. If you want depth reopened it is a *v3* prereg that names a different model family,
  task family, or endpoint, and cites the negative.

## Q7 — Who may issue, dispute, and be believed · S2-14, S8-11, S8-16, S9-02, S1-02

- **Issuer identity is the key**, never the name. **R:** a roster file the log's CI reads; rotation and
  revocation as signed log entries. Sub-choice: **R — forbid key rotation in v8.0** (one hardware-held
  log key) rather than build rotation history now.
- **Sybil.** Keys are free and "distinct hardware" is self-reported, so "≥2 independent challengers"
  is not a fact the log can establish. **R:** drop "distinct hardware"; compute Disputed only over keys
  in the *client's* trust file; label everything else `unvetted`.
- **Open submission (v1).** **R: not in 8.0.** No hosted service, no rate limiting, nobody asking.
- **"Append-only."** A consistency proof only means something between two heads obtained through
  channels the operator does not control. Until a mirror or an anchor exists, v0 is a signed list, and
  **R: the verify commands print that sentence.** Anchoring stays deferred until a mirror exists.

## Q8 — What stays secret · S2-05, S4-18, S2-11, S7-05

- **Published prompts are defeatable.** A provider that recognises them can serve a pinned model.
  **R: say so in the spec, in those words**, and label alias `same` verdicts accordingly. There is no
  defence; claiming one would be the lie.
- **Sealed batteries?** A public battery is a distillation target; a secret one is an unverifiable
  number. **R for 8.0: keep them public** and state the limit plainly. Optional later: seal half a
  battery by commitment and reveal it on the drift claim that uses it.
- **Sealed preregs must seal everything** (endpoints, tests, exclusions, kill gates), not just
  hypotheses, and a result may only cite a sealed prereg that was revealed at a lower log index. **R.**
- **Alias floors expire.** **R:** mandatory window, re-measure on verify, `observed_model_id` change is
  `identity` — the one case where identity has direct evidence.

---

## Q9 — the three decisions that expire · added 2026-09-08 from `DURABILITY_v8_2026_09_08.md`

Everything above can be decided later at the cost of an edit. These three cannot. Each is cheap
today, each becomes impossible after an event we do not control, and two of them have to happen
**before entry #0 is signed**, because entry #0 is what they secure.

- **(a) Anchor the log from its first tree head.** The only property that survives a broken
  signature scheme, a dead lab, and a vanished model is *ordering and content under a dated root the
  operator does not control*. Today the log's dates are assertions by the party under audit. An
  anchor is one scheduled job on the machine already signing tree heads, and either nothing or a
  small fee. **A pin cannot be created retroactively** — a timestamp taken after the break proves
  nothing about what came before it. **R: anchor from the tree head that covers entry #0**, and
  delete "skip until a mirror exists" from §8.5 and §15.5. This reverses the recommendation the
  earlier draft carried, and the reason is that deferral here is not reversible.
- **(b) Pin the log key where you cannot reach it.** The key is currently published in the same
  repository the operator controls, so a reader has no independent way to know they are looking at
  the right log. **R: at least three records outside our control**, named by permanent identifiers
  in entry #0, plus one sentence: a log whose tree heads do not verify under this key identifier is
  a different log, not this one. Must predate the compromise or the lapse it defends against.
- **(c) Leave the hash-algorithm door open in the grammar.** The current rule accepts only
  `sha256:<64 hex>` and rejects everything else, and the version rule makes a successor algorithm a
  disjoint log with no proof crossing between them. Roughly six lines now buys a migration path
  later; the grammar cannot be relaxed for certs already signed. **R: adopt the `<alg>:<hex>`
  grammar with one registered member today**, and an optional identifier field excluded from the
  signature so adding an algorithm never invalidates one.

Signature schemes have a policy horizon, and current guidance points at the first half of the
2030s for the family this design uses. The Merkle structure is unaffected and survives intact. The
question these three answer is whether, on that day, the log still proves anything — and the answer
is decided now or not at all.

---

## Q10 — the four questions the empty class-two roster creates · B-SCHEDULE, B-XLOG, B-FLOOR3, B-CENSUS · added 2026-09-09

`THE_BOUNDARY_2026_09_09.md` split the surviving defects into what a check over logged bytes can reach
and what nothing can. **The second class is now empty** — all four members were reclassified, the last
two against this lab's own published log (`papers/v8/class_two_empty_2026_09_09/`, three scripts that
read stored entry bytes and import no styxx). Every one of the four errors was the same: the argument
examined one certificate instead of the log the certificate sits in. The repairs are written into the
spec as A-61 to A-65. Four decisions inside them are yours, and each is marked in place.
**With these, twenty-six blocks in the draft stand operator-gated.**

- **(a) Plans already logged without a schedule.** *B-SCHEDULE, §5.1.* A noise plan now carries the
  assignment of every run, so the reference run — the one whose values become the subject's
  fingerprint — is fixed before the data exists. The published plan does not: two factors at three
  values and five runs admit **57,600 schedules that satisfy it, and it names none of them**, so the
  reference was chosen from nine cells after the plan was signed, by the party the fingerprint is
  about. **R:** the requirement binds from the version that adopts it; a floor whose plan predates the
  field appends and prints `schedule: absent (plan predates the field)`. The alternative — require it
  everywhere — makes the lab's own only real floor unappendable to a fresh log and costs a
  re-measurement for a repair that fixes nothing already logged. Second half, same decision: a
  schedule may **not** leave a position free, because one `any` per factor hands the choice back.
- **(b) The half of the snapshot check that has an honest failure mode.** *B-XLOG, §2.2.* One
  (`hf_repo`, `revision`) naming two sets of content hashes is a contradiction in the log's own bytes.
  One set of hashes under two revisions is not always: a commit touching only a README gives two
  revisions identical A.2 hashes, and both certs are honest. **R:** refuse the first half, disclose the
  second as `revision-alias`. Today the implementation refuses both, so the interim is a refusal whose
  message names the innocent case. This is the EXTERNAL-1 lesson (an accusation predicate at 0.23
  precision) applied before rather than after.
- **(c) What a contradicted floor does.** *B-FLOOR3, §5.8.* A floor claiming that a factor level pair
  separates nothing, against a logged floor that measured it separating something, is now detectable:
  on the published log the forgery is refused on **9 of 9** comparable cells across three channels, the
  honest floor agrees on all **12**, and a fresh subject returns **`unconstrained`** — which is not a
  pass and must never be printed as one. **R:** refuse the zero-against-a-logged-positive class at
  append; disclose anything weaker; leave the stricter determinism check (same-batch runs in the
  published floor were byte-identical) as a disclosure for 8.0, because one subject on one box on one
  day is not enough evidence to turn an honest driver update into a refusal.
- **(d) What an empty census does to the exit code.** *B-CENSUS, §6.* Every verdict now prints how much
  prior logged material constrained it, and a verdict nothing constrained prints **`first claim: no
  prior logged cert constrains this subject on this recipe`** on the verdict line rather than in a
  limits section. **R:** print-only for 8.0, exit codes unchanged, and the stricter form (`same
  (unconstrained)` exits 2) at the first docket. The arithmetic is why: the published log has **7 cert
  entries, 1 issuer key and no challenges**, so the reproduction count is 0 for every cert in it and
  every verdict the tool can currently produce is a first claim — an exit code that is constant carries
  no information. Second half: the census lives in the printed report and not yet in the signed result
  body, because those bytes are pinned by the committed conformance vectors; moving it into the body is
  a schema version and fresh vectors beside the old ones, never over them.

**What none of the four buys, stated because it is the point of the document that forced them.** Every
predicate above asks whether one party contradicted itself. The residue is no longer a list of fields
it cannot reach; it is one act — **the first claim about anything** — and the only thing that
constrains a first claim is a second party running the same battery and putting their own bytes in the
log. The reproduction count is currently zero. That is a business fact before it is an engineering one.

## Q11 — the question a second implementation found, and it costs us a published number · B-TOPKABSENT · added 2026-09-09

**Background.** A second implementation of the floor arithmetic was written from the spec by an
author forbidden to read `styxx/v8/distances.py`, `floor.py` or `fingerprint.py`. It reproduced all
thirty published numbers exactly — ten pairwise distances on each of three channels, to the last
digit. That is the good news and it is the first cross-implementation agreement this project has
about a *measurement* rather than about cryptography.

It then found that §3.2 does not say what an **absent** `topk_forced_on` means, and the field is
absent on both sides of every certificate we have published.

- Read absent as *a value, equal on both sides*: the comparison is permitted, everything published
  stands.
- Read absent as *the run scored its own output*: the comparison is topk-inconclusive, §6 says print
  no topk number, and seven of ten floor pairs drop out, leaving **a topk floor of 0.0** against the
  published 2.1402339.

**What it costs.** Under the honest reading, the most quotable line in
`RESULT_first_verdict_2026_09_09.md` — *changing the batch size perturbs this model's top-five
distribution more than changing precision from bf16 to fp16 does* — is not printable. That RESULT is
preserved byte-identical and now carries `ERRATUM_topk_comparability_2026_09_09.md`; it is not
edited, because a receipt is history.

**The options.**

- **(a) Absent means no comparison.** Print `topk: inconclusive (no forcing regime stated)`. Treats
  an unstated regime as unstated. Costs us the topk claim above until the field is written.
- **(b) Absent means not-forced and comparable.** Preserves every published number. Reads silence as
  a claim, and it is the reading that happens to favour whoever is publishing.
- **(c) Refuse an absent field at append.** The question cannot arise again, and every certificate we
  have is unappendable to a fresh 8.0 log.

**Recommendation: (a).** It is the only one of the three that does not infer a fact from silence, and
(b) fails the test this lab applies everywhere else — the reading that benefits the party writing the
bytes is the one to distrust. The cost is real and is the reason this is your call and not ours: it
retracts a public claim we made this morning.

**This decision cannot be unmade** for certificates appended under it. Once a log has seated topk
numbers under one reading, changing the reading changes what those entries mean.

**Also in A-66 and not separately gated**, because neither changes a published verdict: Appendix B's
topk averaging is fixed as flat over (item, position) pairs, which is what both implementations
computed — the alternative gives 1.875141894 and leaves the verdict `same` either way — and §5.7's
`alpha_overall` is printed with its margin, because ours is published as 0 and the nearest run clears
its threshold by **0.0004**.

---

## Q12 — the attack that broke today's repair, and what a floor may declare · B-NEWLEVEL · added 2026-09-09

**Background.** We closed `THE_BOUNDARY`'s class-two member 1 with a cross-certificate floor
predicate (A-63) and published that as a result. A sixth adversarial pass broke it the same day,
using only our own published bytes.

The attack declares a batch level the log has never seen. Four of five runs in the forged floor are
our published bytes verbatim; the fifth is spliced from genuine runs at the other precision. Every
check passes — the floor predicate, determinism, snapshot agreement, and every structural property
we cited as the signature of a real measurement. Our own published comparison is re-judged from
`exceeds_floor` to `same` on all three channels.

**The structural point, which is the reason this is a decision and not a bug fix.** A comparison
against the log constrains a claim only at factor levels the log already holds. The issuer declares
the levels. The space of levels is unbounded. So the record growing does not close in on a liar —
the empty space in it grows too, and the issuer picks where to stand.

**The options.**

- **(a) Refuse nothing; disclose.** Print `levels_without_prior` beside the verdict so a reader sees
  which part of a floor rests on no comparison. Costs nothing and refuses no honest first
  measurement of a genuinely new setting.
- **(b) A floor whose widest distance comes from a level with no prior may not be the reference for
  a drift claim.** Refuses this exact attack. Also refuses an honest lab that adds a batch size.
- **(c) A factor level must be preregistered in a plan logged before any run at that level.** Folds
  into Q10's B-SCHEDULE and makes a level a commitment instead of a declaration.

**Recommendation: (a) with (c).** The disclosure is free. (c) puts the choice of levels under the
same before-the-data rule the schedule is already going under. (b) on its own buys a refusal an
attacker routes around by declaring two new levels instead of one.

**Cost of (c):** every floor we have already logged rests on a plan that fixed levels but not their
assignment, so those are grandfathered as stale-not-contradicted, and a fresh 8.0 log holds new ones
to the rule.

**Also recorded and not separately gated:** `precision` is a field with no corroborating byte
anywhere in the design. The published bf16 and fp16 subjects differ in that one string and nothing
else, and excluding it from snapshot agreement is correct — a rule that hashed it would refuse an
honest pair. It is the pivot of every forgery above and it appeared on no roster in the source
document. There is no repair on offer here; it is on the record so the next roster is not written
without it.

**This decision cannot be unmade** for certificates appended under it.

---

---

## What "yes to all R" gets you

The v0.2 draft as written, minus the gate markers, plus Appendix E and the two renames. Weeks 1–4 are
CPU-only and already half-built: the canonical-bytes, Merkle, and signature layers pass 573 tests, an
independent second implementation is being written from the spec alone, and the batch-invariance probe
(`probe_batch_invariance_2026_09_08/`) is the first v8 sentence with a receipt.

**Two things block entry #0 regardless of the answers above:** the log key must exist (Q7 — generated
offline, held on a token), and the canary rename (Q2) must land, because entry #0 freezes the text.
