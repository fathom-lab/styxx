# v8 — the decisions only you can make

Fathom Lab · 2026-09-08 · **A decision sheet, not a result.** Thirty-one items the adversarial review
could not decide, grouped into eight questions, each with its options, a recommendation, and the finding
ids that raise it. Full texts: `REVIEW_v8_spec_2026_09_07.md`. The amended draft (`SPEC_v8_v0.2_draft.md`)
already implements every recommendation marked **R** below and marks each spot `[OPERATOR-GATED …]`, so a
"yes to all recommendations" is one commit and no edits. Nothing is committed until you say so.

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

## What "yes to all R" gets you

The v0.2 draft as written, minus the gate markers, plus Appendix E and the two renames. Weeks 1–4 are
CPU-only and already half-built: the canonical-bytes, Merkle, and signature layers pass 573 tests, an
independent second implementation is being written from the spec alone, and the batch-invariance probe
(`probe_batch_invariance_2026_09_08/`) is the first v8 sentence with a receipt.

**Two things block entry #0 regardless of the answers above:** the log key must exist (Q7 — generated
offline, held on a token), and the canary rename (Q2) must land, because entry #0 freezes the text.
