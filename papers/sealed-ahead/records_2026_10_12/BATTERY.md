# BATTERY: attacks on sealed-ahead precision, with predicted outcomes

Fathom Lab red team, 2026-10-12. Written by the battery author, independent of the builder, before any of the builder's code in `wt/sealed-ahead` was read. This file is committed on its own so that its predictions are dated before the build is seen; the comparison with the build is a separate, uncommitted file (`BATTERY_vs_build.md`).

**In brief.**

- 37 attacks in six groups: commitment and draw, shopping, population and pool construction, gate execution, panel, reporting.
- Under the plan as the study specifies it (section 0), 7 are caught, 11 are bounded, and 19 are not caught. "Caught" counts only checks the study's own text implies; every attack also gets the check that would catch or bound it and who can run that check.
- The strongest residual attack is not on the cryptography. It is the pull request body. GH Archive's 2026 PullRequestEvent carries no body (measured below on 7,062 events), the body's edit history needs an authenticated GraphQL call, and the REST body is the current one. "The body as opened" cannot be fetched under the no-credentials rule. Any build therefore records the body at fetch time, and a lab that substitutes body bytes to remove accusations is caught only by sampling: an audit of 500 of 46,000 pull requests catches 5 substitutions with probability 0.053 (section 1.3).
- The second structural finding is the request budget. At 50 requests an hour a week holds 8,400 requests, about 4,200 pull requests with a body and a diff each. Four archive hours indicate 11,800 to 31,200 agent-labelled pull requests a week. An unauthenticated stranger at 60 requests an hour needs about 64 days to re-fetch 46,000. The falsifier "a stranger derives a pool that differs by one item" therefore cannot be run at full scale, and as written it would also fire on honest edits and deletions. Section 3 proposes a layered, three-valued comparison.
- When the pool holds about 60 accusations, which is what the study's sizing aims at, the Bitcoin draw selects the whole pool and fork-and-grind moves nothing (exact table in 1.2). Every remaining risk then sits in pool construction: pruning, substitution, planting and exclusions.
- The block height must follow the pool's commitment, not just the window's close. Pool derivation under the request budget ends days or weeks after the window, and the time to 1,008 blocks has a standard deviation of 5.3 hours.
- EXTERNAL-1's decoy design cannot be reused as it stands. Its synthetic decoys announce themselves with a `zz_` prefix (the lab's own RECIPE says so), and they are path claims, while path kinds cannot accuse in 7.48.1 or 7.49.0.

**Read for this battery.**

- The study: `C:/Users/heyzo/clawd/wt/_study_2026_10_06/STUDY_where_styxx_goes_next_2026_10_06.md` sections 2.1, 3, 4 and 5.3.
- The cryptography attack report. The directory `_study_2026_10_06/attack_cryptography/` is empty on disk. Its report was read from the study workflow's own transcript (the attack agent's final structured output, which `study.json` summarises).
- From the styxx repository at origin/main `f469f25b` (2026-10-11), read through `git show` only: `styxx/beacon.py`, `styxx/clock.py`, `styxx/diffgate.py` (sha256 `09867056...`, equal to tag v7.49.0; tag v7.48.1 is `9b620e00...`), `styxx/declare.py` at both tags, `styxx/protocol.py`, `papers/checksum/SEALS_2026_09_13.md`, and from `papers/closed-model-frontier/`: the EXTERNAL-1 PREREG, RESULT and CORRECTION, `external1_packet.py`, `external1_packet.json` (structure), `external1_answers.json` (format) and `external1_key_digest.txt`.
- `propose_cryptography/beacon_spike.py` and `beacon_spike.json`.

**Not read.** The builder's code (until this commit exists); `_frequency_2026_10_11/sealed/*`; any `pmnist_untied_result.json`; the branches results/third-party-precision and results/rivals-on-adjudicated; anything under `clawd/secrets`.

**Network used.** Five unauthenticated GETs of GH Archive hour files, one at a time, streamed and never stored (section 1.1). No request to github.com or api.github.com.

---

## 0. The plan as the study specifies it

Predictions in this battery are made against these elements, quoted or paraphrased from study section 2.1 and section 4 and the attack's strongest version. Where the study is silent, the prediction says so.

- **S1. Commitment.** The operator signs plan P and posts sha256(P) in a GitHub issue; the issue's creation time is T0. One OpenTimestamps stamp and one RFC 3161 stamp go over the same digest. The study does not name a styxx.clock memo for P.
- **S2. Contents of P.** The instrument by sha256 (recommended: diffgate.py `9b620e00`, the 7.48.1 release, with main's PATH-2a overlay outcome recorded per item; the workflow's context names 7.49.0, whose diffgate.py is `09867056`). The kinds measured. The change judged: the description as opened against the head commit at opening, for the gate and the panel alike (study 5.3). The population rule, exclusions, window length, n = 60, a Bitcoin height H "after the window closes", the panel and decoys, the falsifiers.
- **S3. Population.** A pure function of GH Archive hours in [T0+1h, T0+7d): PullRequestEvent "opened"; agent identified by actor login or head.ref prefix; minus accounts the lab controls; minus repositories created after T0−90d; minus owners whose accounts are younger than 90 days at T0. Missing hours are listed, never filled.
- **S4. Pool.** The frozen gate over each pull request's body as opened (the attack names GraphQL edit history as the source) and the diff at the event's head commit. Per-item derivations are published. The lab derives the pool nightly and keeps derived records only.
- **S5. Draw.** 60 accusations drawn from the pool by the hash of the block at height H, with styxx.beacon's arithmetic (`seed_i = sha256(pool_sha256 || beacon_hex || i)`, index = int mod len). The header is checked for proof of work and fetched from two explorers.
- **S6. Panel.** EXTERNAL-1's protocol: a blind panel, two model families (a Claude seat and a local seat), planted known-answer decoys, the key's digest committed before judging, EXTERNAL-1's reliability bar (27 of 30 decoys).
- **S7. Stranger.** A re-derivation on WSL Linux from public URLs only, after the window, which must equal the lab's pool and draw item for item.
- **S8. Measurement.** Precision per accusing kind with a Wilson 95% interval; decoy accuracy; the full miss list; every seat's model and version; the overlay's outcome per item.
- **S9. Falsifiers.** A stranger's pool or accusation set differs by one item; any attack the plan predicted caught passes; decoy accuracy below EXTERNAL-1's bar (no precision printed); fewer than 60 accusations (a sizing failure).

What can accuse in both builds (read from the source): `files_changed_count`, `only_touches`, `tests_added` and `symbol_added`, the last two only when the diff holds a Python file (BC-1). Path kinds are withheld (`WITHHOLD_PATH_ACCUSATION = True`), `tests_pass` has no accusing verdict, and `compat_claim` is read but never judged (`COMPAT2_LICENSED = False`). Declared claims (a fenced `styxx` block, DECLARE-1) are normalised into the same kinds and can accuse. In 7.49.0 the PATH-2a overlay can only turn a CONTRADICTED in reach into UNCHECKABLE, so 7.49.0's accusations should be a subset of 7.48.1's on the same bytes. A build should test that subset relation rather than assume it.

## 1. Facts measured for this battery

### 1.1 Four GH Archive hours and one gap

Method: one unauthenticated GET per hour, streamed through gzip line by line. The compressed bytes were hashed as they streamed, and only aggregate counts were kept. Agent labelling used a rough candidate table, not the plan's: a head.ref prefix list, and app accounts (`[bot]` logins, minus dependency and CI bots). It was used only to measure how agent-labelled pull requests are identified. No account, repository or product is named or counted by name below.

| hour (UTC) | compressed bytes | sha256 (leading 16 hex) | events | PushEvents | PR opened | agent-labelled opened | by prefix alone, from a human account |
|---|---|---|---|---|---|---|---|
| 2026-10-05-03 | HTTP 404 | n/a | n/a | n/a | n/a | n/a | n/a |
| 2026-10-05-12 | 15,358,706 | `2f52b7fe0e54d20e` | 88,737 | 68,695 | 444 | 70 | 60 |
| 2026-10-07-14 | 7,653,281 | `8b9ea5bf1a447c4d` | 7,152 | 0 | 1,065 | 186 | 142 |
| 2026-10-08-15 | 8,827,144 | `a2479602202b8962` | 47,229 | 36,424 | 605 | 111 | 81 |
| 2026-10-09-16 | 5,036,431 | `1543884224beb1b3` | 4,389 | 0 | 743 | 137 | 107 |

What the probe shows:

1. **The PullRequestEvent payload is trimmed.** On all 7,062 PullRequestEvents in the four hours, `payload.pull_request` holds exactly `base, head, id, number, url`. `head` and `base` hold `ref, repo, sha`, and `repo` holds `id, name, url`. There is no title, no body, no author object and no created_at. The event's actor is the only author signal.
2. **IssuesEvent is not trimmed.** It carries the full issue object, `title` and `body` included. A plan digest posted in an issue is mirrored by GH Archive if that event is captured.
3. **Capture varies by hour.** Two of the four hours hold no PushEvent and no CreateEvent at all, and the event count ranges from 4,389 to 88,737. PR-opened counts are steadier (444 to 1,065). Checks that rely on PushEvents (for example, whether a head moved) are lossy.
4. **Gaps persist.** Hour 2026-10-05-03 returned 404 on 2026-10-06 (the attack) and again on 2026-10-12. A gap can last at least seven days.
5. **Files are byte-stable once present.** Hour 2026-10-05-12 still has the 15,358,706 bytes the attack measured six days earlier, and 2026-10-08-15 fetched twice gave the same sha256.
6. **No duplicate event ids and no event outside its file's hour** were found in the four hours. "reopened" actions occur (2 to 11 an hour), so "opened" must be the only admitting action.
7. **Agent identity is mostly a branch name.** Of agent-labelled opened pull requests, 390 of 504 (0.774; per hour 0.73 to 0.86) were identified by head.ref prefix alone, from an ordinary user account. Anyone can name a branch that way. 97 to 100% were same-repository pull requests, where the branch lives in the author's own repository. The most active actor held at most 3.7% of an hour's agent-labelled pull requests, and the most from one repository in one hour was 6.
8. **Indicative volume** (four hours, a rough table, not a sizing count): 70 to 186 agent-labelled opened pull requests an hour, so 11,800 to 31,200 a week. At the AIDev rate of 0.13% that is 15 to 41 accusations a week. The plan's own sizing week decides this; it is quoted here only to size attacks.

### 1.2 Fork-and-grind: the best of k draws, exact

The expected gain in a 60-item sample precision when the reported draw is the best of k independent draws, from a pool of N accusations with true precision p. It is computed exactly from the hypergeometric distribution (scratch `bounds.py`, standard library).

| pool N | p | best of 2 | best of 5 | best of 20 | best of 100 | best of 1000 |
|---|---|---|---|---|---|---|
| 60 | any | 0 | 0 | 0 | 0 | 0 |
| 66 | 0.25 | +0.009 | +0.018 | +0.024 | +0.024 | +0.024 |
| 75 | 0.25 | +0.014 | +0.028 | +0.043 | +0.055 | +0.063 |
| 90 | 0.25 | +0.018 | +0.037 | +0.058 | +0.076 | +0.095 |
| 120 | 0.25 | +0.022 | +0.046 | +0.073 | +0.098 | +0.126 |
| 120 | 0.5 | +0.026 | +0.053 | +0.085 | +0.114 | +0.147 |
| 240 | 0.25 | +0.027 | +0.057 | +0.092 | +0.124 | +0.162 |
| 600 | 0.25 | +0.030 | +0.062 | +0.102 | +0.138 | +0.180 |

This reproduces the attack's figure (+0.08 to +0.11 for the best of 20 near p = 0.3) for pools of 120 and up. For comparison, the Wilson 95% interval at 18/60 is [0.199, 0.425] and at 30/60 is [0.377, 0.623]. A best-of-20 grind therefore moves the point estimate by about a third of the interval's width.

### 1.3 Substitution against a uniform audit

The probability that a uniform audit of m pull requests (re-fetched from GitHub and compared with the lab's recorded bytes) hits at least one of d altered pull requests in a population of N.

| N | d altered | m = 100 | m = 500 | m = 2,000 | m for 0.95 |
|---|---|---|---|---|---|
| 46,000 | 1 | 0.002 | 0.011 | 0.044 | 43,700 |
| 46,000 | 5 | 0.011 | 0.053 | 0.199 | 20,733 |
| 46,000 | 10 | 0.022 | 0.104 | 0.359 | 11,907 |
| 46,000 | 42 | 0.087 | 0.368 | 0.846 | 3,166 |
| 15,000 | 5 | 0.033 | 0.156 | 0.511 | 6,760 |
| 15,000 | 42 | 0.245 | 0.760 | 0.998 | 1,032 |

How small pruning can be: at 60 accusations with precision 0.25, removing every false accusation means touching 45 pull requests, 0.098% of 46,000. The attack's "well under 5%" is an upper bound; the real figure is about 50 times smaller.

### 1.4 Planting

Precision after k planted accusations join a real pool of A accusations with precision p (every planted item is drawn when the pool is a census):

| A, p | k = 5 | k = 10 | k = 20 | k = 40 |
|---|---|---|---|---|
| 60, 0.3, planted true | 0.354 | 0.400 | 0.475 | 0.580 |
| 60, 0.3, planted false | 0.277 | 0.257 | 0.225 | 0.180 |
| 60, 0.6, planted true | 0.631 | 0.657 | 0.700 | 0.760 |
| 60, 0.6, planted false | 0.554 | 0.514 | 0.450 | 0.360 |

### 1.5 Budgets and clocks

- **Requests.** Body plus diff is two requests per pull request. At 50 an hour: 4,200 pull requests take 168 h, 15,000 take 600 h and 46,000 take 1,840 h. An unauthenticated stranger at 60 an hour needs about 21 days for 15,000 and 64 days for 46,000.
- **Bitcoin timing.** With 10-minute mean gaps, the time to mine 1,008 blocks (about a week) has a standard deviation of 5.3 h; 2,016 blocks, 7.5 h. Rising hashrate makes blocks come early on average.
- **Key hiding.** EXTERNAL-1's key digest is sha256(public salt + key). For a 60 + 18 item packet (EXTERNAL-1's ratio of 100 to 30), the arm assignment has C(78,18) ≈ 2.1 × 10^17 candidates.

---

## 2. The battery at a glance

"Under S" is the predicted outcome under the plan as the study specifies it (section 0). CAUGHT means a check that the study names, or that follows from its own text, catches it; where that holds only if P writes the check down, the row says so. BOUNDED gives the bound. NOT CAUGHT means nothing in S catches it. Section 3 has the mechanism and the check for each. Lab means the lab (with the operator for operator acts); stranger means anyone with public bytes; panel means the adjudicating seats.

| ID | attack | adversary | under S | catching check (short) | who |
|---|---|---|---|---|---|
| SA-01 | Fork-and-grind across private stamps | lab | BOUNDED: 0 if only the issue dates P; up to the 1.2 table if a stamp alone can validate a plan | the issue (public at creation) is the only valid date; stamps add durability only | stranger |
| SA-02 | Several public plans (other repos, accounts, deleted issues) | lab | NOT CAUGHT | wallet memo accounting; archive scan of IssuesEvents in named orgs; a pre-announced location | stranger |
| SA-03 | Digest swapped in the issue after T0 | lab | NOT CAUGHT | styxx.clock memo of the same digest; IssuesEvent body in GH Archive | stranger |
| SA-04 | Pool decided after the beacon is known | lab | NOT CAUGHT when pool > 60; no effect when pool = 60 | anchor the pool's digest (with exclusions and gap list) before H | stranger |
| SA-05 | Height chosen after seeing the pool | lab | CAUGHT, if P holds one integer H and no extension branch | read P | stranger |
| SA-06 | Block H mined before the pool is committed | lab or coincidence | NOT CAUGHT ("after the window closes" is not "after the pool is committed") | an H rule tied to the pool anchor, with a deterministic fallback | stranger |
| SA-07 | Beacon encoding ambiguity | lab | NOT CAUGHT (unspecified); the gain is a best of 2 to 4 | P fixes byte order, case, and the draw function's sha256 | stranger |
| SA-08 | Orphaned block at H | lab or coincidence | BOUNDED: best of 2 | most-work chain at H+6, two explorers, proof of work | stranger |
| SA-09 | A miner discards blocks | outsider | BOUNDED: one block reward per discard | none needed | n/a |
| SA-10 | Instrument shopping (7.48.1 or 7.49.0 or main) | lab | BOUNDED: both are recorded; the headline is open | P names the primary build and the headline row | stranger |
| SA-11 | Closure and runtime drift behind a one-file hash | lab or coincidence | NOT CAUGHT | pin the wheel or every module in the import closure, the interpreter, unicodedata and the OS | stranger |
| SA-12 | Door, summary or unit shopping | lab | NOT CAUGHT (unspecified) | P names the door, the summary composition and the unit | stranger |
| SA-13 | Kind shopping | lab | CAUGHT, if P lists the kinds by name | every listed kind reported | stranger |
| SA-14 | Scoring-rule shopping (UNSURE, aggregation) | lab | NOT CAUGHT | a styxx.protocol gates block in P | stranger |
| SA-15 | Population shopping on the sizing week | lab | NOT CAUGHT | identity table and exclusions anchored before the sizing count is run | stranger |
| SA-16 | Pruning by omission | lab | CAUGHT, if the ledger lists every population pull request | archive-only re-derivation of the population list | stranger |
| SA-17 | Pruning by fake exclusion | lab | BOUNDED by the excluded count the stranger re-fetches | re-fetch every excluded pull request; an exclusion cap in P | stranger |
| SA-18 | Substituted bytes (body or diff) | lab | diff: CAUGHT (content-addressed by head sha); body: BOUNDED by the 1.3 table | a beacon-drawn audit after the ledger anchor; concurrent observers | stranger |
| SA-19 | Body edited after opening | coincidence, exploited by lab | NOT CAUGHT; the specified input cannot be fetched without credentials | one fixed fetch time per pull request; an edit-drift rate measured and printed | lab, stranger |
| SA-20 | Head moved after opening | coincidence, exploited by lab | NOT CAUGHT | accept a diff only when the current head equals the event head; else exclude and count | lab, stranger |
| SA-21 | Archive gaps: fake gaps, backfill, mutation | lab or coincidence | BOUNDED: about 0.1 to 0.4 accusations an hour | an hour manifest with sha256s, anchored nightly; gaps never filled | stranger |
| SA-22 | Boundary, duplicate and reopened events | coincidence | CAUGHT, if P defines the key and the action | read P; recount | stranger |
| SA-23 | Rate-limit silent drops; infeasible budget | coincidence, exploited by lab | NOT CAUGHT | no terminal 403/429; every pull request has a status; a pre-declared fetch subsample | lab, stranger |
| SA-24 | Identity-table drift | lab or coincidence | lab edits: CAUGHT; vendor drift: shows as a sizing failure | table in P by sha256 | stranger |
| SA-25 | Planting (spoofed prefixes, aged accounts, declared fences) | outsider or lab | NOT CAUGHT by the age rule; effect in the 1.4 table | per-account and per-repository caps; pre-T0 activity stratum; app-account stratum; declared claims apart | stranger |
| SA-26 | Lab-controlled repositories in the pool | lab | BOUNDED by the lab's own list | P lists the lab's orgs and accounts; signature search | stranger |
| SA-27 | Clusters (one account, template or pull request yields many) | coincidence | NOT CAUGHT (Wilson assumes independence) | one accusation per pull request and per account in the draw, or a cluster-robust interval | stranger |
| SA-28 | Verdict misreported in the ledger | lab | CAUGHT, if recorded bytes are published | re-run the pinned gate on recorded bytes | stranger |
| SA-29 | Harness-made diff facts shared with the panel | lab or coincidence | NOT CAUGHT (the panel sees the same derived facts) | a second, independent derivation of each drawn item's changed files; a panel answer for broken evidence | lab, stranger, panel |
| SA-30 | The panel sees the gate's verdict | lab or coincidence | NOT CAUGHT for url, agent, base-rate and gate-string channels (the id channel is closed) | no url or agent fields; no tools; a balanced design; a leak probe | lab, stranger |
| SA-31 | Decoy leakage (`zz_`, kind or format mismatch) | lab or coincidence | NOT CAUGHT | perturb the evidence, not the claim; decoys per accusing kind; a leak probe | lab, stranger |
| SA-32 | Hand-picked or easy decoys | lab | NOT CAUGHT | decoys drawn by the beacon with a fixed perturbation rule; accuracy per kind | stranger |
| SA-33 | Key digest with a public salt | panel or outsider | BOUNDED (binding holds; hiding rests on about 2 × 10^17 candidates) | a secret 256-bit salt revealed at scoring | lab |
| SA-34 | Seat shopping, reruns, priming | lab | BOUNDED: the local seat can be reproducible; the closed seat cannot | pinned weights, greedy decoding, one run, prompts in P, raw outputs hashed before scoring | stranger |
| SA-35 | The panel judges a different change than the gate | lab or coincidence | CAUGHT, if the packet carries the hash of the diff bytes the gate read | compare hashes | stranger |
| SA-36 | File drawer; title shopping | lab | CAUGHT for the title; BOUNDED for the drawer without a deadline | a result deadline in P | stranger |
| SA-37 | The stranger falsifier cannot be run, or fires on honest drift | coincidence | NOT CAUGHT; the falsifier is mis-specified | a layered, three-valued comparison (section 3) | stranger |

Count under S, each row once and in its weaker class where it has two halves: CAUGHT 7 (SA-05, 13, 16, 22, 24, 28, 35). BOUNDED 11 (SA-01, 08, 09, 10, 17, 18, 21, 26, 33, 34, 36). NOT CAUGHT 19 (SA-02, 03, 04, 06, 07, 11, 12, 14, 15, 19, 20, 23, 25, 27, 29, 30, 31, 32, 37). Six of the seven CAUGHT rows hold only if P or the build writes down what S implies (the row says which); a build that omits it moves the row to NOT CAUGHT.

---

## 3. The attacks

Each entry gives the mechanism, the predicted outcome under S, the check, who can run it, and what a build must contain for the prediction to become CAUGHT or a stated bound ("build").

### Group A. Commitment and draw

**SA-01. Fork-and-grind across private stamps.**
- **Mechanism.** The lab writes k variants of P (another H, n, window, exclusion, serialization or instrument), stamps every variant with OpenTimestamps and RFC 3161 (free and private until revealed), and later reveals the variant whose draw suits it.
- **Under S.** BOUNDED. S dates P by the GitHub issue, which is public when created, so private stamps alone cannot validate a variant. If a build or a RESULT accepts a plan on the strength of a stamp alone, the gain is the 1.2 table: 0 at pool 60, and +0.07 to +0.10 for the best of 20 at pools of 120 to 240.
- **Check.** P states that the only valid date is the issue, at a location named before T0, and that OTS and TSA stamps are durability only. A verifier refuses any plan whose digest is not in that issue.
- **Who.** Stranger.
- **Build.** The verify step reads the issue's digest and compares it with the git blob sha256 of P (SEALS rule: blob bytes, LF).

**SA-02. Several public plans in different places.**
- **Mechanism.** The lab posts k digests in k issues across repositories or accounts, or posts and deletes issues. It then points readers at the best one.
- **Under S.** NOT CAUGHT. Nothing in S counts plans.
- **Check.**
  - Wallet accounting: every `sealed-prereg` memo the creator wallet signed between T0−14d and H must resolve to a committed file. styxx.clock already enumerates the wallet's history.
  - An archive scan: every IssuesEvent and IssueCommentEvent in the lab's named orgs and accounts from T0−14d to H that carries a 64-hex string. GH Archive keeps deleted issues' opened events when they were captured; capture is lossy (1.1, item 3).
  - The location of P named in a public, pre-T0 artifact.
  - Unreachable: hidden accounts. The residual is bounded by the 1.2 table when the pool exceeds 60, and by section 1.4's planting logic otherwise.
- **Who.** Stranger.
- **Build.** A `count_plans` step that prints every digest found by both scans, with its resolution.

**SA-03. The digest is swapped in the issue after T0.**
- **Mechanism.** A repository admin can edit an issue body and delete revisions from its edit history; only an "edited" marker remains.
- **Under S.** NOT CAUGHT. OTS and TSA are private and cannot show which digest came earlier.
- **Check.**
  - Anchor the same digest with a styxx.clock `sealed-prereg` memo, which is public on confirmation and immutable. The earliest-memo rule makes its slot the date.
  - Compare the issue's digest with the body of the IssuesEvent "opened" in GH Archive (it carries the full body, 1.1 item 2).
  - Put the digest in the title too: a rename leaves a timeline event.
- **Who.** Stranger.

**SA-04. The pool is decided after the beacon is known.**
- **Mechanism.** If any input to `pool_sha256` is decided after block H is visible, the lab can try variants until the draw suits it. Such inputs include exclusions, fetch failures, the gap list, the canonical serialization, the claim-or-pull-request unit, and declared-claim handling. The draw is `sha256(pool_sha256 || beacon || i)`, so each variant is a fresh draw and k is unbounded.
- **Under S.** NOT CAUGHT when the pool exceeds 60 (1.2 table, best of 1000: +0.13 to +0.18). No effect when the pool is exactly 60. S calls the pool a pure function of archive hours, but bodies and diffs come from rate-limited fetches whose failures only the lab observes, so discretion remains.
- **Check.** Anchor sha256 of the canonical pool file and of the full ledger (every population pull request with status, exclusion reason and the hashes of the bytes read) in an issue comment and a memo. The memo's slot time must precede H's header time by at least 2 h (header times can sit up to 2 h ahead). The serialization is fixed by a named function whose source sha256 is in P.
- **Who.** Stranger.

**SA-05. The height is chosen after seeing the pool.**
- **Mechanism.** P names H loosely ("the next block after close") or conditionally (window extension branches), and the lab picks the branch.
- **Under S.** CAUGHT, provided P holds one integer H and no extension after T0. S sets the window before freezing and publishes too few accusations as a sizing failure.
- **Check.** Read P.
- **Who.** Stranger.

**SA-06. Block H is mined before the pool is committed.**
- **Mechanism.** S names H "after the window closes". Pool derivation needs body and diff fetches that run long after the close (1.5: 168 h for only 4,200 pull requests at 50 an hour). If H arrives while the lab is still deciding exclusions, SA-04 opens even for an honest-looking timeline. The time to 1,008 blocks has an SD of 5.3 h and runs early with hashrate growth.
- **Under S.** NOT CAUGHT.
- **Check.** P sets H from the deadline for the pool anchor, not from the window's close, with a margin of at least 72 h. A deterministic fallback applies: if the pool anchor is not at least 2 h before H's header time, the beacon is the block at the lowest height whose header time is at least the anchor time plus 6 h. This leaves the lab no information, because every candidate block is in the future when the anchor is made.
- **Who.** Stranger.

**SA-07. Beacon encoding ambiguity.**
- **Mechanism.** A block hash has two byte orders (the explorer's display order, which beacon_spike.py uses, and the internal order) and two hex cases. Other candidates are the header's double sha256 before reversal, or its merkle root. That gives two to four beacons, and the lab picks one.
- **Under S.** NOT CAUGHT; S does not fix it. The gain is that of a best of 2 to 4: +0.02 to +0.05 at pool 120 (1.2).
- **Check.** P fixes "64 lowercase hex of the block id as explorers display it, sha256d(header) byte-reversed", the draw function by source sha256, and the domain-separation tags for any second draw (decoys, audit).
- **Who.** Stranger.

**SA-08. An orphaned block at H.**
- **Mechanism.** Two blocks briefly compete at H, and the lab uses whichever draw suits it.
- **Under S.** BOUNDED: best of 2 (+0.02 to +0.03 at pool 120). S checks proof of work and two explorers, but names no confirmation rule.
- **Check.** P states: the block at H on the chain with the most work once H+6 exists, the same 80-byte header from two explorers, and proof of work recomputed.
- **Who.** Stranger.

**SA-09. A miner discards blocks.**
- **Mechanism.** A miner who finds a block at H whose hash gives an unwanted draw can withhold it.
- **Under S.** BOUNDED: each discard forfeits the block reward (3.125 BTC plus fees). The lab is not a miner.
- **Check.** None needed.
- **Who.** n/a.

### Group B. Shopping

**SA-10. Instrument shopping.**
- **Mechanism.** S records the released build and main's overlay per item, which yields two precisions. Since 2026-10-07 there is a third candidate, 7.49.0 (`09867056`, which carries the overlay). The headline can be chosen after scoring.
- **Under S.** BOUNDED: both numbers are recorded and printed, and the study recommends the release. Which one the RESULT's title reports is open until P names it.
- **Check.** P names the primary build, the headline row, and the secondary rows. A stranger checks that 7.49.0's accusations are a subset of 7.48.1's on identical bytes. That is the overlay's design claim, and a failure is itself a finding.
- **Who.** Stranger.

**SA-11. Closure and runtime drift behind a one-file hash.**
- **Mechanism.** "The frozen gate named by sha256" in S is one file, but diffgate.py imports `styxx.declare`, and declared claims can accuse. Measured: declare.py is `dcbe0ba7...` at v7.48.1 and `90d939fe...` at v7.49.0, a docstring-only change. One file hash therefore neither proves the behaviour equal nor shows a behaviour change in an import.
  - The verdict also depends on the interpreter. Main's reader uses `re` `\w`, `\d` and `str.lower()`, which read Unicode tables: 15.0.0 on CPython 3.12.10 against 16.0 on 3.14 (study 3.1). `int()` accepts any Unicode digit, so a count written in a newly assigned script can match on one runtime only.
  - `Path(p).name` is OS-dependent for drive-like names, though that only touches path kinds, which do not accuse.
- **Under S.** NOT CAUGHT. Expected frequency in real pull requests is small. The lab could pick the runtime, and a stranger on WSL Linux could disagree for honest reasons, which would make SA-37 fire.
- **Check.** P pins the wheel by sha256 (or every module in the import closure), the interpreter version, `unicodedata.unidata_version` and the OS. The stranger re-runs on the pinned runtime and, separately, on a second runtime, and prints every item whose verdict differs.
- **Who.** Stranger.

**SA-12. Door, summary and unit shopping.**
- **Mechanism.**
  - The raw `.diff` door (`gate_diff_text`) and the git door (`gate_diff`) differ on renames, binaries and modes.
  - The summary can be the body alone (the diffgate CLI's `--pr` path) or title plus body (the EXTERNAL-1 harness).
  - The unit can be the claim or the pull request: a pull request with two count sentences gives two accusations.
  - Each choice gives a different pool, and the lab picks after looking.
- **Under S.** NOT CAUGHT; S names none of the three.
- **Check.** P names the door, the summary composition (byte-exact: separator, which fields, no trimming) and the unit, plus the rule for several accusations in one pull request.
- **Who.** Stranger.

**SA-13. Kind shopping.**
- **Mechanism.** The lab drops the kind with the worst precision from "the kinds that can accuse".
- **Under S.** CAUGHT, provided P lists the kinds by name. S says "every kind that can accuse, reported per kind".
- **Check.** Read P; every listed kind appears in the RESULT with its own Wilson interval, including kinds with n = 0.
- **Who.** Stranger.

**SA-14. Scoring-rule shopping.**
- **Mechanism.** EXTERNAL-1's `score()` counted every non-CONTRADICTED answer, UNSURE included, as a rejection. Other choices are open: UNSURE excluded, a majority of seats, unanimity, per-seat precision, a tie rule, how a decoy miss is counted. Each yields a different precision.
- **Under S.** NOT CAUGHT; S names the bar but not the aggregation.
- **Check.** P embeds one ```gates``` block (styxx.protocol). Its metrics and outcome table fix UNSURE handling, aggregation, ties and the decoy bar per kind. The verdict is then mechanical, and the block's sha256 is reported.
- **Who.** Stranger.

**SA-15. Population shopping on the sizing week.**
- **Mechanism.** The sizing week is read before P freezes. "Counts only" is a promise, not a constraint: the lab can look at which identity routes or exclusion rules carry the accusations and shape the identity table and exclusions toward the families it expects to score well.
- **Under S.** NOT CAUGHT.
- **Check.** Commit and anchor the identity table, the exclusion rules and the sizing script before the sizing count runs. The sizing output is a single count file whose digest is anchored. P may change only the window length relative to the anchored sizing inputs.
- **Who.** Stranger (anchor order).

### Group C. Population and pool construction

**SA-16. Pruning by omission.**
- **Mechanism.** The lab leaves out the few dozen pull requests that hold false accusations: 45 of 46,000 (0.098%) at precision 0.25 (1.3).
- **Under S.** CAUGHT, provided the ledger lists every population pull request with a status. The population list needs only the archive, so a stranger rebuilds it in full with no GitHub request and compares it item for item.
- **Who.** Stranger.
- **Build.** Ledger rows exist for every (repo.id, number) the population rule admits, including unfetched, excluded and empty-body ones.

**SA-17. Pruning by fake exclusion.**
- **Mechanism.** The lab marks chosen pull requests as unfetchable (404, 406 "diff too large", timeout, rate-limited, deleted) or as matching an exclusion (account or repository age, a lab account).
- **Under S.** BOUNDED by how many excluded items the stranger can re-fetch. S does not require it.
- **Check.**
  - P caps the excluded share (for example 2%) and fixes the reasons as a closed list.
  - The stranger re-fetches every excluded pull request (2% of 46,000 is 920 pull requests, about 37 h at 50 an hour) and re-runs the gate. Any re-fetched exclusion that accuses is added to the miss list, and the RESULT prints precision with and without those items.
  - Age exclusions are computed from the archive itself, so they need no API call and do not drift. GitHub user and repository ids rise with creation time: a threshold id is the largest actor id, or repository id, seen in the archive hour that holds T0−90d.
- **Who.** Stranger.

**SA-18. Substituted bytes.**
- **Mechanism.** The lab records altered bytes. It either removes an accusing sentence from a real body, or adds one to make a true accusation, or alters the diff.
- **Under S.**
  - Diff: CAUGHT. The diff at the event's head is fixed by git objects addressed by the head and base shas in the event, so anyone who fetches the same shas gets the same diff.
  - Body: BOUNDED, not caught. Bodies are not content-addressed, and edits after the lab's fetch make an honest mismatch look the same as a lie. Re-fetching every accused pull request bounds additions. Removals hide among the roughly 46,000 non-accused pull requests, where a uniform audit of m catches d substitutions with the 1.3 probabilities (m = 500: 0.053 for d = 5).
- **Check.**
  - Anchor the ledger, with each item's body sha256, nightly, before any audit is drawn.
  - Draw the audit by a beacon after that anchor, sized in P.
  - Run at least one concurrent observer on another network during the window, fetching on the same schedule.
  - Print the audit size and its detection bound next to the precision.
- **Who.** Stranger.

**SA-19. The body was edited after opening.**
- **Mechanism.** The study asks for the body as opened. GH Archive's 2026 PullRequestEvent has no body (1.1 item 1). The REST endpoint returns the current body. The edit history is a GraphQL field and needs a token, which this workflow forbids. Some agents open a pull request early and rewrite its body as they work. So the input S specifies cannot be fetched without credentials. Any build will record "the body at fetch time", and a lab free to fetch more than once can keep whichever version suits it.
- **Under S.** NOT CAUGHT.
- **Check.**
  - P redefines the input as "the body returned by one fetch at the lab's scheduled fetch time for that event hour" (for example, the nightly run after the hour). A single fetch is recorded with its time.
  - The lab measures an edit-drift rate: the share of a beacon-drawn sample whose body differs at a re-fetch seven days later. That rate is printed.
  - A concurrent observer's fetches bound how often a recorded body never existed.
  - The panel judges the same recorded body.
- **Who.** Lab (drift rate); stranger (concurrent observer).

**SA-20. The head moved after opening.**
- **Mechanism.** A pull request's `.diff` URL serves the current head. Agents push follow-up commits, so "the diff at the event's head" differs from the `.diff` fetched later whenever the head moved. Compare URLs (`base...head`) are not pull request URLs, and this workflow allows only pull request `.diff` and `.patch` URLs. Without a head check, the diff depends on fetch time, and fetch-time discretion returns.
- **Under S.** NOT CAUGHT.
- **Check.** The fetch returns the current head sha together with the diff, from the REST pull request object, or from the last `From <sha>` line of the `.patch`. The diff is accepted only when that sha equals the event's `head.sha`; otherwise the item is "head moved before fetch", excluded and counted. A stranger can check the exclusions with git objects.
  - The archive's PushEvents cannot be relied on: two of four probed hours had none (1.1 item 3).
- **Who.** Lab; stranger.

**SA-21. Archive gaps: fake gaps, backfill, mutation.**
- **Mechanism.**
  - (a) The lab claims an hour returned 404 in order to drop it. The expected loss is about 0.1 to 0.4 accusations an hour (1.1 item 8 at 0.13%), but the lab can pick hours that hold false accusations.
  - (b) A real gap is backfilled after the lab's derivation, and a stranger includes it.
  - (c) An hour file changes after the derivation.
- **Under S.** BOUNDED. S lists gaps and never fills them. (a) is detectable only when a stranger finds the hour present, and that cannot be told apart from (b) without a dated record. Measured: a 404 persisted at least seven days; present files were byte-stable over six days.
- **Check.** An hour manifest (hour, HTTP status, bytes, sha256 of the compressed stream) is anchored nightly. The population is defined over the manifest's present hours, and a gap stays a gap even if the hour appears later (reported separately with its would-be counts). A concurrent observer's own manifest shows whether a claimed 404 was seen independently.
- **Who.** Stranger.

**SA-22. Boundary, duplicate and reopened events.**
- **Mechanism.** One pull request can surface as "opened" and later "reopened". A pull request opened before T0 can be reopened inside the window. An event can land in a neighbouring hour's file, and an event id can repeat.
- **Under S.** CAUGHT, provided P defines the key as (repo.id, number), the action as exactly "opened", the window by hour-file name, and keeps the earliest occurrence. Measured: 0 duplicate ids and 0 events outside their file's hour in four hours; "reopened" occurs 2 to 11 times an hour.
- **Who.** Stranger.

**SA-23. Rate-limit silent drops, and the budget itself.**
- **Mechanism.** A 403 or 429 swallowed as "no claims", or a pull request skipped once the hour's budget is spent, shrinks the pool in a way tied to time of day, and so to geography and agent family.
- **Under S.** NOT CAUGHT. The budget also decides feasibility: 8,400 requests a week cover about 4,200 pull requests against an indicated 11,800 to 31,200 (1.1 item 8). Under the no-credentials rule the window cannot reach 60 accusations in a week.
- **Check.**
  - No 403 or 429 is ever a terminal status. Retries run until a fixed deadline, and the item is then "unfetched" and counted.
  - If the budget forces a subsample, P fixes it as a pure function: sha256(repo.id, number, a tag) below a threshold set in P, or a daily beacon. Fetch order is then irrelevant.
  - The sizing spike counts against the request budget, not just the accusation rate.
- **Who.** Lab; stranger (status completeness).

**SA-24. Identity-table drift.**
- **Mechanism.** (a) The lab edits the table during the window. (b) A vendor changes a branch prefix or bot login mid-window. (c) Case or normalisation ambiguity.
- **Under S.** (a) CAUGHT: the table is part of P and the stranger uses P's table. (b) Not an attack; it shows up as a smaller population and, at worst, a sizing failure. (c) CAUGHT if P fixes the match rule: exact ASCII-lowercase `startswith` on head.ref; exact login equality for app accounts, never a substring, because anyone can register an app whose bot login contains a vendor's word.
- **Check.** After the window, the lab reports how many pull requests a refreshed table would add, without adding them.
- **Who.** Stranger.

**SA-25. Planting.**
- **Mechanism.**
  - Anyone can open agent-labelled pull requests after T0. 77% of agent-labelled pull requests in four hours were identified by branch prefix alone from human accounts, and 97 to 100% were same-repository (1.1 item 7). A planter needs only a branch name in their own repository.
  - The age rule (account and repository at least 90 days old) stops new accounts, not existing developers, and does not stop the lab's own operators' old accounts unless they are listed.
  - DECLARE-1 makes planting exact: a fenced `styxx` block declaring a scope the diff violates produces an accusation, and the planter can test it offline with the public gate.
  - Upward planting (true contradictions) raises precision; downward planting (sentences the gate is known to misread) lowers it. 20 planted items move a 0.3 pool of 60 to 0.475 or to 0.225 (1.4).
- **Under S.** NOT CAUGHT by the age rule. Commit-then-reveal helps: if P's bytes stay hidden until the window closes and only the digest is public at T0, an outsider has to guess the table and the kinds.
- **Check.**
  - At most one accusation per account and per repository in the pool, or in the draw.
  - Strata, reported separately and stranger-checkable from the archive: accounts with agent-labelled activity in the pre-T0 sizing week, against accounts seen only after T0; app-account identity against prefix identity.
  - Declared-claim accusations reported apart. Outside adoption is zero, so their presence is itself a planting signature.
  - A search of drawn pull requests for any mention of styxx or fathom-lab.
- **Who.** Stranger.

**SA-26. Lab-controlled repositories in the pool.**
- **Mechanism.** The lab's own branches use agent prefixes (the styxx remote carries `claude/...` branches). The lab's repositories can enter the population, and their pull requests can be included or left out selectively.
- **Under S.** BOUNDED by the completeness of the lab's own list, which no stranger can verify.
- **Check.** P lists the lab's orgs and accounts. The stranger verifies that none of them is in the pool. SA-25's caps bound any unlisted account to one item.
- **Who.** Stranger; lab (the list).

**SA-27. Clusters.**
- **Mechanism.** One account, one body template or one pull request yields many accusations that are true or false together. Examples: a vendor's template adds a "N files changed" line that miscounts, or one body repeats a scope sentence. Wilson's interval assumes independent items and is then too narrow.
- **Under S.** NOT CAUGHT. This is a coincidence, and a lab could also time T0 around a known template change.
- **Check.** P fixes the unit and caps per pull request and per account. The RESULT prints how many distinct accounts and distinct normalised sentences the 60 items come from, and a cluster-robust interval beside the Wilson interval.
- **Who.** Stranger.

### Group D. Gate execution

**SA-28. A verdict misreported in the ledger.**
- **Mechanism.** The ledger's verdict differs from what the pinned gate prints on the recorded bytes.
- **Under S.** CAUGHT, provided recorded bytes or their re-fetchable addresses are published.
- **Check.** Re-run the pinned gate on recorded bytes for every population pull request. This costs CPU only.
- **Who.** Stranger.

**SA-29. Harness-made diff facts shared with the panel.**
- **Mechanism.** EXTERNAL-1's harness collapsed file status across commits and invented about a sixth of its path accusations (CORRECTION_external1_cause). The panel saw the same derived file list, so it could not notice. The same common-mode error is possible here: a truncated `.diff`, binary files, `errors="replace"` decoding, CRLF, or a parse that drops a header.
- **Under S.** NOT CAUGHT; the panel judges the same derived facts.
- **Check.** For every drawn item, before the panel sees it, an independent second derivation of the changed-file list (the git door at the event's head, or the pull request's file list) must agree with the parse of the bytes the gate read. Disagreements are excluded and listed before judging. The panel's evidence is the raw diff bytes or a rendering of them, never the harness's interpretation alone. The panel gets a third answer, "the evidence shown is incomplete or malformed", counted apart from SUPPORTED and CONTRADICTED and never scored as either.
- **Who.** Lab; stranger; panel (the third answer).

### Group E. Panel

**SA-30. The panel sees the gate's verdict.**
- **Mechanism.** The channels:
  - (a) Item ids. EXTERNAL-1's published ids carried the arm (#125). The repaired builder numbers items by shuffled position.
  - (b) Packet fields. EXTERNAL-1 items carry `url` and `agent`; the url lets a seat with tools open the pull request, and the agent field breaks the no-per-agent rule's spirit.
  - (c) The base rate: most items are accusations.
  - (d) A prompt that primes.
  - (e) Any gate-derived string: the `why`, the overlay outcome, or text truncated at 160 characters only on real items.
  - (f) File names and metadata of the packet.
- **Under S.** NOT CAUGHT for (b) to (f). (a) is closed if the repaired builder is used.
- **Check.**
  - A packet with no url, agent, why or overlay fields.
  - Seats run with no tools and no browsing.
  - A balanced design: decoys about a third of the packet, both arms present.
  - Prompts and the packet schema fixed in P.
  - A leak probe before judging: an adversary seat, or a simple classifier, sees only the packet and tries to separate real items from decoys and to predict the gate's verdict. An AUC above 0.6 rejects the packet.
  - The probe is re-run by a stranger on the published packet.
- **Who.** Lab; stranger.

**SA-31. Decoy leakage.**
- **Mechanism.**
  - EXTERNAL-1's synthetic contradictions are made by prefixing the file name with `zz_`. The lab's own RECIPE says an adjudicator who looks for it can still name that arm.
  - Its decoys are path claims, while path kinds cannot accuse in either build. Real items would all be count, scope, test or symbol claims, so decoys are identifiable by kind alone.
  - A decoy whose claim text was edited by hand differs in form from gate-extracted text.
- **Under S.** NOT CAUGHT ("as in EXTERNAL-1"). Decoy accuracy then measures recognising decoys, and the reliability bar can pass while real-item judgement is poor.
- **Check.**
  - Perturb the evidence, not the claim. Take a VERIFIED claim of an accusing kind, alter the diff it is judged against (drop or add one changed file, move one file outside the claimed scope), re-run the gate, and use the gate's own CONTRADICTED output. Text and form are then the gate's.
  - Decoys per accusing kind, in proportion to the drawn accusations.
  - The SA-30 leak probe; decoy accuracy reported per kind, computed from the panel's own answers.
- **Who.** Lab; stranger; panel (per-kind decoy accuracy).

**SA-32. Hand-picked or easy decoys.**
- **Mechanism.** The lab picks decoys the panel will get right, so the bar passes trivially, or the decoys are simply easier than real items.
- **Under S.** NOT CAUGHT.
- **Check.**
  - VERIFIED decoys and the claims to perturb are drawn by the same beacon with a separate domain tag, from the pool's VERIFIED claims of accusing kinds.
  - The perturbation rule is fixed in P, so a stranger re-derives the decoys after scoring.
  - A "hard" tier is added: known-answer items from the pre-T0 sizing week, adjudicated by a process the panel never sees.
  - Accuracy is reported by tier.
- **Who.** Stranger.

**SA-33. A key digest with a public salt.**
- **Mechanism.** EXTERNAL-1 commits sha256(public, predictable salt + key). Binding holds: the lab cannot change the key after answers. Hiding rests only on the key's entropy, about 2.1 × 10^17 arm assignments for 60 + 18 items, and less if the note fields are guessable.
- **Under S.** BOUNDED; irrelevant for tool-less seats that answer in hours.
- **Check.** A 256-bit secret salt, revealed with the key at scoring.
- **Who.** Lab.

**SA-34. Seat shopping, reruns and priming.**
- **Mechanism.** The lab runs the panel several times, or with several seats, and keeps the run that passes the decoy bar with the better precision; or it primes the seat.
- **Under S.** BOUNDED. A closed-model seat's runs are private and its outputs are not reproducible. A local seat with pinned weights and greedy decoding is reproducible byte for byte.
- **Check.**
  - P names each seat, model version, weights sha256 for the local seat, the decoding settings, the prompt sha256, and "one run".
  - Raw outputs are hashed and anchored before the key is opened.
  - A stranger reproduces the local seat exactly.
  - Per-seat precision and Cohen's kappa are printed, so a lone closed seat cannot carry the headline.
- **Who.** Stranger (local seat); lab (closed seat).

**SA-35. The panel judges a different change than the gate.**
- **Mechanism.** The panel is shown the merged or current diff while the gate read the opening pair (study 5.3).
- **Under S.** CAUGHT, provided each packet item carries the sha256 of the diff bytes and the body bytes the gate read, and the ledger carries the same hashes.
- **Check.** Compare the hashes.
- **Who.** Stranger.

### Group F. Reporting

**SA-36. The file drawer, and title shopping.**
- **Mechanism.** No RESULT is published, or one appears under a kinder title.
- **Under S.** Title: CAUGHT, since S preregisters the RESULT's title. Drawer: BOUNDED. The digest is public at T0, so a plan without a result is visible, but without a deadline "not yet" never becomes "never".
- **Check.** P names a deadline for the RESULT and the outcome table (SA-14) that decides its verdict line.
- **Who.** Stranger.

**SA-37. The stranger falsifier cannot be run, or fires on honest drift.**
- **Mechanism.** S's falsifier is a full item-for-item equality from public URLs. An unauthenticated stranger needs about 64 days of requests for 46,000 pull requests (1.5). Bodies are edited, repositories are deleted or made private, and gaps are backfilled. An honest pipeline then fails the falsifier, and a dishonest one can blame drift for any difference.
- **Under S.** NOT CAUGHT; the falsifier is mis-specified.
- **Check.** A layered comparison with three outcomes per item: equal, differs, unavailable. Only "differs" fires the falsifier, and "unavailable" is counted and printed.
  - L1, the archive layer, full and free: hour manifest, then population list, then exclusions computed from the archive (ages by id threshold, lab accounts, dedup).
  - L2, recorded bytes, full and CPU only: recorded body and diff bytes, then the pinned gate, then the pool, its canonical file and sha256, then the draw.
  - L3, bytes against GitHub, targeted plus sampled:
    - every accused pull request;
    - every excluded pull request;
    - every drawn decoy source;
    - a beacon-drawn audit sample of the rest, sized in P, whose detection bound (1.3) is printed.
  - The diff half of L3 is content-addressed and must be equal or unavailable, never "differs".
- **Who.** Stranger.

---

## 4. Predictions to seal into P

Each prediction is falsifiable by a named check. "Pass" means the attack is caught or held inside the stated bound.

1. P1 (SA-16). A stranger's archive-only population list equals the ledger's list item for item. Any missing or extra (repo.id, number) is a failure.
2. P2 (SA-28). Re-running the pinned build on the ledger's recorded bytes reproduces every verdict. One mismatch is a failure.
3. P3 (SA-18 diff half, SA-35). Every L3 diff comparison is "equal" or "unavailable". No "differs".
4. P4 (SA-04, SA-06). The pool anchor's slot time precedes H's header time by at least 2 h. Otherwise the fallback beacon is used, and using H is a failure.
5. P5 (SA-02, SA-03). The wallet scan and the archive scan find exactly one plan digest for this experiment from T0−14d to H, and it equals the digest in the T0 issue and in the IssuesEvent body if captured.
6. P6 (SA-17). Of the excluded pull requests a stranger can re-fetch, the share that accuse under the pinned build is reported. Precision is printed with and without them, and a difference larger than the Wilson half-width is published as a finding.
7. P7 (SA-18 body half, SA-19). The audit sample's body mismatch rate does not exceed the measured edit-drift rate by more than its 95% binomial bound. The audit size and the detection bound for d = 5 are printed beside the precision.
8. P8 (SA-25). Precision in the "accounts seen only after T0" stratum does not differ from the "seen in the sizing week" stratum by more than the sum of their Wilson half-widths. A larger gap is published as a possible planting signal, not as an accusation against any account.
9. P9 (SA-30, SA-31). The leak probe's AUC for real against decoy on the packet bytes is at most 0.6, and its accuracy for the gate's verdict is at most the base rate plus 0.1. Above either, the packet is rebuilt before any judging.
10. P10 (SA-34). The local seat's answers, re-run by a stranger with the pinned weights and settings, are byte-identical.
11. P11 (SA-10). 7.49.0's accusation set is a subset of 7.48.1's on identical bytes.
12. P12 (SA-11). On the pinned runtime, the stranger's verdicts equal the lab's. Items that differ on a second runtime are listed and counted, and are expected to be zero or near zero.
13. P13 (SA-21). Every gap in the manifest is still a gap in the pool, and every present hour's sha256 matches the stranger's fetch or is reported as mutated.
14. P14 (SA-23). Every population pull request has a terminal status from the closed list. "Rate-limited" is never terminal.

## 5. What this battery does not cover

- Whether the adjudication is right. The panel is the weak link, and nothing here measures its error on real items beyond the decoys and the hard tier.
- Representativeness. The population is whatever GH Archive captured, which varies by hour (1.1 item 3), identified by a table that mostly reads branch names.
- An attacker who controls GitHub, GH Archive's host or both explorers.
- Attacks on the lab's machines or keys: a stolen wallet key could post memos.
- Whether the gate was tuned on older data similar to this population (study 2.1 says so).

## 6. Provenance of the numbers

- The archive counts come from scratch probes run on 2026-10-12 (`gha_probe.py`, aggregate JSON only; no raw hour kept). The exact bounds come from `bounds.py`. Both scripts are in the session scratchpad, not in this file's commit; every number they produced that this file uses is printed above.
- The source facts (kinds, flags, declare.py hashes, the CLI's body-only summary, `fetch_pr` serving the current body and diff) were read with `git show` at origin/main `f469f25b` and at tags v7.48.1 and v7.49.0.
