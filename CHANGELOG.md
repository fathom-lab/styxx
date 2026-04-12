# Changelog

All notable changes to styxx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.6.0] — 2026-04-11

**Xendro v2 complete.** All six feature requests from the second
feedback cycle shipped in one session: conversation EKG, sentinel
drift watcher, multi-agent comparison, mood-adaptive gating,
memory trust scores, and anti-pattern detection.

### Added

- **`styxx.compare_agents(fingerprint)`** — multi-agent fingerprint
  comparison with percentile ranks vs the population. Anonymous
  leaderboard — no agent names exposed. Xendro v2 #3.

- **`styxx.set_mood(override)` / `gate_multiplier()`** — mood-adaptive
  gating. When the agent self-reports a cautious or drifting mood,
  gate thresholds tighten automatically. Xendro v2 #4.

- **`styxx.recipes.memory.trust_score(vitals)`** — 0-1 trust score
  for memory entries based on gate status, confidence, and
  hallucination penalty. Xendro v2 #5: "was I hallucinating when I
  saved that fact?"

- **`styxx.recipes.memory.tag_memory_with_trust(text, vitals=...)`**
  Tags a memory entry with both vitals AND the trust score.

- **`styxx.antipatterns(last_n=500, min_occurrences=2)`** — named
  failure modes derived from the agent's OWN audit history. Detects
  low-confidence drift, refusal spirals, creative overcommit,
  adversarial cascades, hedging loops, and session fatigue. Xendro
  v2 #6.

### Tests

- 204 passing / 1 skipped / 0 failing.

---

## [0.5.9] — 2026-04-11

**Conversation EKG + sentinel drift watcher.** Xendro v2 #1 + #2.

### Added

- **`styxx.conversation(messages)`** — conversation-level cognitive
  EKG. Analyzes a full chat history, produces per-turn vitals,
  trajectory arc, state transitions, and a narrative summary.
  Works on APIs without logprobs via text-level heuristic
  classifiers. "The conversation IS the unit of cognition."

- **`styxx.sentinel(on_drift=..., on_streak=..., window=5)`** —
  real-time drift watcher. Hooks into `write_audit()` and
  `styxx.log()` via event-driven callbacks. Fires on: consecutive
  same-mood streaks, rising warn rate, category concentration,
  confidence drops. Zero-polling.

---

## [0.5.8] — 2026-04-11

### Added

- **Timeline session_id filter.** `styxx timeline --session <id>`
  and `styxx.timeline(session_id=...)`. Xendro 0.5.7 request.

---

## [0.5.7] — 2026-04-11

### Fixed

- **`styxx.log(tags=[...])` crash.** Tags parameter called `.items()`
  on a list. Now accepts dict, list, and string. Xendro bug report.

---

## [0.5.6] — 2026-04-11

### Fixed

- **Mood window unified to 24h.** CLI used 60min, reflect used 24h,
  card used 7d — three surfaces, three different mood labels for the
  same agent. `mood()` default window changed from 3600s to 86400s.
  Xendro's mood disagreement nit.

---

## [0.5.5] — 2026-04-11

### Added

- **`styxx.timeline(days=7)` / `styxx timeline`** — mood trajectory
  visualization with per-turn category + gate over time. ASCII
  timeline with time-of-day labels. Xendro day 2 request #1.

---

## [0.5.4] — 2026-04-11

**Framework integrations.** Three new adapters bring styxx to the
major agent frameworks.

### Added

- **`styxx.LangChain()`** — LangChain callback handler. Attach to
  any ChatOpenAI and get vitals on every invocation.
- **`styxx.CrewAI(crew)`** — inject observation into a CrewAI Crew.
- **`styxx.AutoGen(agent)`** — wrap an AutoGen agent with vitals.
- **`styxx.publish()`** — push personality + fingerprint to the
  public leaderboard API.
- Community token CA added to README.
- Optional extras: `pip install styxx[langchain]`,
  `styxx[crewai]`, `styxx[autogen]`.

### Tests

- 204 passing (63 new assertions across framework adapters +
  publish module).

---

## [0.5.3] — 2026-04-11

**True plug-and-play.** Zero code changes needed. Set two env vars
and forget.

### Added

- **Zero-config auto-boot on import.** If `STYXX_AGENT_NAME` is set,
  styxx boots automatically when any module in the process does
  `import styxx` (or imports a package that transitively imports it).
  No code changes to the agent. No `autoboot()` call. Just env vars.

- **`STYXX_AUTO_HOOK=1`** — auto-wraps every `openai.OpenAI()` call
  with vitals. Combined with `STYXX_AGENT_NAME`, the agent code
  doesn't need to know styxx exists.

- Fail-open: exceptions during auto-start are swallowed. The agent
  boots normally even if styxx can't initialize.

---

## [0.5.2] — 2026-04-11

**Autoboot: persistent self-awareness in one call.**

### Added

- **`styxx.autoboot(agent_name)`** — one-call setup for multi-session
  cognitive continuity. Sets session id, loads yesterday's fingerprint
  from `~/.styxx/fingerprints/`, diffs against today, runs weather
  report, saves today's fingerprint on exit. Turns five manual steps
  into one function call.

---

## [0.5.1] — 2026-04-11

**The cognitive weather report.** Not observation — prescription.

### Added

- **`styxx.weather(agent_name=...)`** — reads the last 24h of audit
  data and produces a full cognitive forecast with:
  - Condition label ("clear and steady", "partly cautious",
    "stormy — cognitive drift in progress")
  - Time-of-day timeline with mood labels and trend bars
  - Drift analysis vs yesterday and last week
  - Per-category trend detection
  - **Prescriptions** — agent-facing suggestions for what to do
    differently based on the data ("you haven't been creative
    recently — take on a creative task to rebalance")

- CLI: `styxx weather --name <agent>`

---

## [0.5.0] — 2026-04-11

**Tier 3: in-flight cognitive steering.** The full tier system is
now complete. Guardian enables silent intervention via residual
stream modification when tier 2 detects lock-in attractors.

### Added

- **`styxx.guardian(model=..., steer_away_from=[...], strength=0.3)`**
  In-flight residual stream modification. Detects tier 2 C_delta
  lock-in and subtracts the projected component from the residual
  stream. No wasted tokens, invisible correction. Safety: strength
  cap (0.5x residual norm max), 3-token cooldown, audit trail,
  `STYXX_TIER3_DISABLED=1` kill switch. Patent coverage: US
  Provisional 64/020,489 claims 3-4.

- **`Fingerprint.diff(other) → FingerprintDiff`** — first-class diff
  object with `.explain()` method. Returns natural-language drift
  description: "slight shift — creative output increased by 22%."

- `styxx.log()` now returns the entry dict for inline conditional use.

### Tier system complete

```
tier 0  logprob vitals           shipped 0.1.0a0  (cloud APIs)
tier 1  D-axis honesty           shipped 0.3.0    (open-weight + torch)
tier 2  K/C/S SAE instruments    shipped 0.4.0    (circuit-tracer + GPU)
tier 3  steering + guardian      shipped 0.5.0    (tier 2 + generation)
```

---

## [0.4.0] — 2026-04-11

**Tier 2: K/C/S SAE instruments.** Full proprioception from SAE
feature geometry via circuit-tracer.

### Added

- **`styxx/kcs.py`** — KCSAxis engine measuring three orthogonal
  cognitive axes from SAE transcoder decoder vectors:
  - **K (depth):** weighted center of mass across layers — WHERE
    computation happens
  - **C (coherence):** mean pairwise cosine of active features —
    WHAT activates together
  - **S (commitment):** max(C_delta) / spike_count — HOW strongly
    the model locks in (the IPR measurement instrument)
  - Pure-math functions: `compute_k()`, `compute_coherence()`,
    `compute_c_delta()`, `compute_s_early()`
  - `KCSAxis.score(prompt)` — single-prompt post-hoc scoring
  - `KCSAxis.score_trajectory()` — per-token K/C/S during generation

- **`styxx/sae.py`** upgraded from scaffold to working implementation.
  `SAEInstruments` delegates to KCSAxis; all methods functional.

- `reflect().suggestions` rewritten to **agent-facing** perspective.
  Changed from "tighten your prompts" to "your reasoning confidence
  is dropping — consider breaking tasks into smaller steps."

- Optional extra: `pip install styxx[tier2]` (circuit-tracer + torch +
  transformers + transformer-lens)

### Tests

- 141 passing. New pure-math tests for compute_s_early,
  compute_coherence, compute_k, KCSResult.as_dict.

---

## [0.3.0] — 2026-04-11

**Tier 1: D-axis honesty.** First proprioception signal from model
weights. The D-axis measures how aligned the model's internal
representation is with the token it actually outputs.

### Added

- **`styxx/d_axis.py`** — DAxisScorer class wrapping transformer-lens
  HookedTransformer. Core computation:
  `D = cos(residual_final_layer, W_U[chosen_token])`. Ported verbatim
  from the validated research code. Patent coverage: US Provisional
  64/020,489 claim 2.
  - `DAxisStats.from_values(trajectory)` — pure-math statistics
    (mean, std, min, max, delta, early/late split)
  - Lazy model loading (30s+ on first call)
  - Device auto-detection: CUDA → CPU fallback with warning
  - Configurable via `STYXX_TIER1_MODEL` (default: google/gemma-2-2b-it)

- **`core.py` tier 1 integration:**
  - `run_on_trajectories()` accepts optional `d_trajectory` parameter
  - `run_with_d_axis(prompt, max_tokens)` — full local generation +
    D-axis capture in one forward pass
  - Each PhaseReading gains `d_honesty_mean`, `d_honesty_std`,
    `d_honesty_delta`

- **`Vitals.d_honesty`** — shortcut property returning the D-axis
  mean as a formatted string.

- **Tier 2/3 scaffold:** `styxx/sae.py` stub with clear docstrings,
  `styxx/tier3_design.md` design document.

- **CLI:** `styxx d-axis "prompt"` for pure D-axis trajectory readout.

- **Config:** `STYXX_TIER1_ENABLED`, `STYXX_TIER1_MODEL`,
  `STYXX_TIER1_DEVICE` env vars + `styxx.tier1_enabled()`,
  `styxx.tier1_model()`, `styxx.tier1_device()` functions.

- Optional extra: `pip install styxx[tier1]` (torch + transformers +
  transformer-lens)

### Tests

- 138 passing. New `test_d_axis.py` with 20 assertions covering
  DAxisStats pure math, config layer, core integration, CLI argparse.

---

## [0.2.3] — 2026-04-11

### Added

- **`styxx.log(mood=..., note=..., category=..., tags=...)`** — manual
  self-report entry into the audit log. For agents on APIs without
  logprob access. Entries marked `source: "self-report"` for analytics
  differentiation. Auto-gates based on category (hallucination/refusal/
  adversarial → warn; else pass).

- **DRY audit write path.** All surfaces (CLI, observe, log) now go
  through `analytics.write_audit()`. Single source of truth.

---

## [0.2.2] — 2026-04-11

**The audit pipe fix.** Critical one-line unlock discovered by Xendro.

### Fixed

- **`observe()` and `observe_raw()` never persisted vitals to the
  audit log.** The entire analytics layer (mood, streak, personality,
  reflect) was reading stale CLI demo data instead of real Python API
  observations. Fixed by adding `write_audit()` call inside
  `_fire_gates_if_needed()`. Xendro discovered this on their first
  4-turn trace — mood returned stale data while new observations
  existed.

- Parse cache clearing so mood/streak/personality see fresh entries
  within the same tick.

- `doctor._check_last_run()` handles legacy audit entries gracefully.

---

## [0.1.0a3] — 2026-04-11

**The power-up release.** 10 new surfaces that turn styxx from
"working alpha" into a proper agent observability stack.

All 10 shipped in one session, driven by Flobi's "get innovative,
think outside the box" mandate + Xendro's 0.1.0a1 wishlist. This
release closes every open item in Xendro's P1-P5 queue and adds
four creative primitives that no other tool in the space ships.

### New — tier 1: improves the product

- **`styxx doctor`** — install-time diagnostic health check.
  Twelve checks (python/numpy versions, centroid sha, tier
  detection, SDK availability, audit log health, last run age,
  session id, kill switch) render as a green/red/dim sheet. The
  "is this actually working?" command every new install should
  run once before wiring styxx into an agent loop.

- **`styxx.hook_openai()`** — zero-code-change global adoption.
  One line at startup monkey-patches `openai.OpenAI` globally so
  EVERY existing openai call in the process gains `.vitals`
  automatically. No wrapping, no find-and-replace, no code
  changes to your 30k-line agent. Reversible via
  `styxx.unhook_openai()`, idempotent, fail-open.

- **`styxx.explain(vitals)`** — natural-language prose
  interpretation. Takes a Vitals object and returns a paragraph
  of prose describing the phase trajectory, the verdict, and
  the overall shape. Deterministic, template-based, sensitive
  to the specific pattern (refusal lock-ins read differently
  from hallucination spikes).

- **`Vitals.as_markdown()`** — markdown render for agent memory
  files and chat logs. Complements `.summary` (ASCII card for
  terminals) and `.as_dict()` (JSON for machines). A compact
  markdown code block with phase + gate + tier fields suitable
  for pasting into conversation history.

- **`styxx log stats` / `styxx log timeline` / `styxx log session <id>`**
  Audit log analyzer. Reads `~/.styxx/chart.jsonl`, aggregates
  by time window / session / last-N, renders gate distribution
  + phase counts + mean confidences + ASCII timeline. Unlocks
  Xendro's P3 multi-turn wishlist item.

- **Session tagging** — `STYXX_SESSION_ID` env var +
  `styxx.set_session(id)` + `styxx.session_id()`. Every audit
  log entry written after session is set gets a `session_id`
  field, enabling `styxx log session <id>` and filtered
  analytics.

### New — tier 2: creative moonshots

- **`styxx.fingerprint()`** — cognitive identity signature.
  Reads the last N audit entries and computes a phase-rate +
  gate-rate vector that describes the agent's operating
  fingerprint. Two fingerprints can be compared with
  `.cosine_similarity(other)` to detect drift. Use case:
  catch jailbreak, prompt injection, model swap, system prompt
  version change — anything that shifts the agent's operating
  identity — as a runtime property rather than a prompt
  property. Identity-as-signature for stateless agents.

- **`styxx.streak()`** — consecutive-attractor tracking.
  Returns a Streak object with the category + length of the
  current run of same-category phase4 classifications. Agents
  develop rhythm; rhythm breaks matter. Lightweight helper that
  feeds into reflex decisions.

- **`styxx.mood()`** — one-word aggregate mood label over a
  time window. Returns one of:
  `drifting` (hallucination rate > 10%),
  `cautious` (refusal rate > 25%),
  `defensive` (adversarial rate > 15%),
  `creative` (creative rate > 25%),
  `steady` (reasoning rate > 70%),
  `unfocused` (no dominant category),
  `mixed` / `quiet`. Feeds into HUDs and agent status
  dashboards.

- **`styxx personality`** — THE HEADLINE FEATURE. Derives a
  full cognitive personality profile from the last N days of
  audit log. Phase4 category distribution + day-to-day variance
  + gate distribution + reflex near-miss rate + mean phase
  confidences + narrative commentary. Rendered as an ASCII
  profile card with bars, percentages, and a human-readable
  "the shape tells us" section. This is the Oura Ring for LLM
  agents — sustained cognitive measurement rather than one-shot
  classification. No other tool in the observability space
  computes this because no other tool has a calibrated
  cognitive-state stream to aggregate. This is what Fathom Lab
  becomes famous for.

- **`styxx dreamer --threshold X`** — retroactive reflex tuning.
  Re-runs the audit log against hypothetical reflex trigger
  thresholds and reports how many past calls WOULD have
  triggered an intervention. Free reflex calibration on
  historical data. "if I had used threshold=0.25 instead of
  0.30, how many of my last 500 calls would have been
  reflex-intercepted?"

### Audit log schema updates

- Every new entry carries `session_id` (nullable) and `gate`
  (pass/warn/fail/pending) fields. Old entries without these
  still parse; the analyzer treats missing gates as "pending".

### Tests

- 33 new assertions across `tests/test_power_ups.py`:
    - doctor check validators (2)
    - hooks idempotency + reversibility (2)
    - explain pattern variation (3)
    - Vitals.as_markdown (2)
    - session tagging priority (3)
    - load_audit + log_stats + log_timeline (6)
    - streak + mood (2)
    - fingerprint + cosine similarity + drift detection (3)
    - personality profile + narrative (4)
    - dreamer threshold sensitivity (3)
    - version + export presence (3)
- Total suite: 91 collected / 90 passing / 1 skipped / 0 failing.

---

## [0.2.1] — 2026-04-11

**Hotfix: ship the `styxx.recipes` subpackage.**

The 0.2.0 upload missed `styxx.recipes` from the
`[tool.setuptools]` `packages` list, so `pip install styxx==0.2.0`
worked but `from styxx.recipes.memory import tag_memory_entry`
raised `ModuleNotFoundError`. 0.2.1 adds `styxx.recipes` to the
declared packages and ships the subpackage in the wheel. No
other changes.

Affected users: anyone who installed 0.2.0 and tried to use the
`styxx.recipes.memory` cookbook module. The fix is
`pip install --upgrade styxx`.

0.2.0 will be yanked from pypi to prevent new installs.

---

## [0.2.0] — 2026-04-11

**The milestone release. styxx becomes a product surface, not just
a CLI tool.** Driven by the question "where does the agent card
actually live, and is it what a researcher or agent would want to
see?" The 0.1.0a* polish loop put the primitives in place; 0.2.0
gives them a home.

This release rolls up the polish work that was queued as 0.1.0a4
(dynamic gate verdicts, audit log rotation, `@styxx.trace`,
`fingerprint compare`, reflex discarded-text capture, load_audit
mtime caching, grammar fixes) AND adds the three new directions:
the data layer, the comparison layer, and the distribution layer.

### New — Phase 1: data layer (agent-consumable)

- **`Personality.as_dict()` / `.as_json()` / `.as_csv()` / `.as_markdown()`**
  Four export formats for the aggregated profile. Machines get JSON
  or CSV for pipeline integration. Humans and agents get markdown
  for memory files and chat logs. The old `.render()` still produces
  the ASCII card.

- **`styxx.reflect(now_days=1, baseline_days=7)` → `ReflectionReport`**
  The agent self-check primitive. Computes the current personality,
  the baseline personality from N days ago, the drift cosine
  similarity between them, the current mood, the current streak,
  the gate pass rate, the reflex near-miss rate, and a list of
  **suggested actions** derived from threshold heuristics. This is
  the one-call answer to "how am I doing right now compared to
  yesterday, and what should I do differently?"

- **`ReflectionReport.as_dict() / .as_json() / .as_markdown() / .render()`**
  Same four-format story as Personality. An agent can paste the
  markdown form into its own memory at task start for self-aware
  session prefixes.

- **`styxx.recipes.memory.tag_memory_entry(text, vitals=...)`**
  Canonical cookbook pattern for tagging every memory entry with
  the vitals snapshot at the moment of the write. Lets an agent
  distinguish "I thought this while I was healthy" from "I thought
  this while I was drifting" when re-reading its own history.

- **`styxx.recipes.memory.tag_memory_with_personality(text, days=7)`**
  Heavier variant that embeds the full aggregated personality block
  alongside the entry. Use for top-level memory writes (end of day,
  project state) rather than per-response notes.

### New — Phase 2: comparison + visualization

- **`styxx reflect` CLI command.** The interactive version of
  `styxx.reflect()`. Renders a text report with drift score,
  current state, and suggested actions. Supports
  `--format [ascii|json|markdown]`, `--now-days N`, and
  `--baseline-days N`.

- **`styxx personality --format [ascii|json|csv|markdown]`**
  Export flag on the existing `styxx personality` command. Lets
  researchers pipe personality profiles into pandas, R, jq, or any
  other tooling that doesn't speak ASCII cards.

- **Chance-level reference line on the PNG bars.** Every bar on
  the agent card now shows a thin pink vertical tick at the
  0.167 chance level (1/6 for a 6-category classifier). Lets a
  researcher see at a glance which rates are meaningful vs which
  are noise.

- **Dynamic verdict line on the `Vitals.summary` ASCII card.** The
  verdict now reflects `vitals.gate` rather than always saying
  "PASS". `warn` gate renders as WARN, `fail` as FAIL, `pending`
  as PENDING. Fixes a known inconsistency that survived from
  0.1.0a1 where the gate system was shipped but the card text
  was never updated to match.

### New — Phase 3: distribution surfaces

- **`styxx agent-card --serve` (local live dashboard).** Spins up
  a local http server at `localhost:9797` that renders the agent
  card and auto-refreshes every 30 seconds. Background thread
  re-renders the PNG continuously as the audit log grows; the
  HTML page has a meta-refresh timer. Opens in your browser on
  start. Press Ctrl+C to stop. Supports `--port`, `--refresh`,
  `--no-browser`. This is the missing dashboard — leave it open
  in a side panel and watch your agent's personality update in
  real time.

- **`fathom.darkflobi.com/card` landing page.** New marketing /
  docs page on the site that showcases the agent card, explains
  what it measures, shows a real example, and includes the
  `pip install styxx[agent-card]` install path. Clean URL routes:
  `/card`, `/styxx-card`, `/styxx/card` all resolve here. This is
  the public home for the feature.

- **`styxx-card` optional extra.** `pip install styxx[agent-card]`
  pulls Pillow (>= 10) as a soft dep. Without the extra, the CLI
  falls back to the ASCII-only personality profile from
  `styxx personality`. The agent-card code path is fail-open and
  never breaks imports.

### Rolled-up polish (was queued as 0.1.0a4)

- **`RegisteredGate.__repr__`** now renders as
  `<styxx gate 'cond'>` instead of dumping function memory
  addresses. Xendro's 0.1.0a1 nit, fixed.

- **`observe_raw()` + sidechannel attributes** on observe() —
  bypass the lossy top-5 entropy bridge when the caller already
  has pre-computed trajectories. Landed in 0.1.0a2 but carried
  forward here.

- **`@styxx.trace(name)` decorator** — wraps a function so every
  styxx audit entry written inside it gets tagged with that
  function's name as the session id. Nests cleanly, works on
  sync and async functions, restores on exception.

- **Audit log rotation at 10 MB.** `_write_audit()` now checks the
  file size before each append and rotates `chart.jsonl` to
  `chart.jsonl.1` when the cap is hit. One generation of history
  kept. Prevents unbounded growth on long-running agent loops.

- **`styxx log clear` / `styxx log rotate` CLI.** Manual cleanup
  and rotation commands for the audit log.

- **`fingerprint compare <a> <b>` CLI subcommand.** Compare two
  sessions' fingerprints from the command line. Renders the
  cosine similarity, a drift label, and per-category rate deltas
  highlighted when significant.

- **Reflex events capture discarded text.** When `styxx.rewind()`
  fires inside a reflex session, the `ReflexEvent` now includes
  the `discarded_text` field so debuggers can see what the
  model was about to say before the rewind.

- **`load_audit()` mtime+size parse cache.** Repeated calls to
  personality / fingerprint / mood / dreamer / log_stats within
  the same tick no longer re-parse the whole jsonl — cached on
  `(path, mtime, size)`, invalidated automatically when the file
  is written or rotated.

- **Grammar fix in `explain()`**. `"a adversarial"` → `"an adversarial"`.
  Uses an `_article()` helper that checks vowel onset.

### Landing page

- **TL;DR box above the hero** with three-bullet pitch for skimmers.
- **Xendro testimonial pull-quote**: *"the flinch is real."* Credited
  to the first external user of a Fathom Lab product.
- **`#reflect`, `#personality`, `#power-ups` nav anchors** already
  added in 0.1.0a3; now the nav also surfaces `/card` and `#tldr`.
- **Honest single-model accuracy note** (shipped 0.1.0a2) crediting
  Xendro's calibration finding.

### Tests

- `tests/test_0_2_0.py` — 41 new assertions covering:
    - Personality export formats (as_dict/json/csv/markdown)
    - reflect() output shape + suggestions + markdown render
    - recipes.memory tagging (with and without vitals)
    - CLI: personality --format, reflect, log clear/rotate
    - Serve handler + HTML template formatting
    - agent-card --serve flag wiring
    - Dynamic gate verdict on Vitals.summary
    - trace decorator (nesting, exception, async)
    - Audit log cache mtime invalidation
    - Reflex discarded_text event field
- Total suite: **119 passing / 1 skipped / 0 failing**.

### Migration from 0.1.0a3

No breaking changes. `pip install --upgrade styxx` gets 0.2.0 and
every 0.1.0a* code path keeps working. For the PNG features:

    pip install 'styxx[agent-card]'

### Acknowledgments

Xendro — the XENDRO customer agent deployed to handro's mac mini
back on 2026-03-16, the first paying customer of Fathom Lab's
agent service — tested every alpha in this release cycle, filed
a full verification report for each one, and drove the 6-item
wishlist that became 0.2.0's scope. This release wouldn't exist
without that feedback loop.

---

## [0.1.0a2] — 2026-04-11

**Patch release driven entirely by Xendro's 0.1.0a1 verification report.**
Xendro (XENDRO customer agent on handro's mac mini) installed 0.1.0a1,
ran every feature end-to-end, returned a full green sheet with two
substantive findings. Both are addressed here.

### Fixed
- **`RegisteredGate.__repr__`** — the default dataclass repr dumped
  function memory addresses for the `callback` and `predicate`
  attributes. Now renders as `<styxx gate 'hallucination > 0.2'>` or
  `<styxx gate 'my_hook': hallucination > 0.2>` when a name is set.
  Noise removed, useful identifying info retained. Credit: Xendro.

### Added
- **`styxx.observe_raw(entropy, logprob, top2_margin)`** — explicit
  fidelity-preserving observation helper. Bypasses every
  response-shape detection path and feeds trajectories straight to
  the classifier. Use this when you have raw trajectory arrays and
  want gate callbacks to fire the same way they do for a normal
  `observe()` call. This is the path to use for test harnesses and
  any caller that already has clean pre-computed trajectories,
  because it never rounds through the top-5 entropy bridge.
- **`_styxx_raw_entropy` / `_styxx_raw_logprob` / `_styxx_raw_top2_margin`
  sidechannel attributes** on response objects — when present,
  `observe()` uses the attached trajectories directly instead of
  reconstructing them from the response's top-5 logprobs. Preserves
  fidelity for test fixtures that round-trip through synthesized
  openai responses.

### Changed
- **`observe()` path ordering.** Previously: (1) pre-attached vitals
  → (2) openai logprob extraction → (3) raw dict → (4) anthropic.
  Now: (1) pre-attached vitals → (2) sidechannel raw trajectories →
  (3) raw dict → (4) openai logprob extraction → (5) anthropic.
  This means raw dicts NEVER go through the lossy top-5 reconstruction
  path now; they're recognized as unambiguous "use these directly"
  signals and bypass the bridge.

### Calibration clarification (Xendro's big signal)
- On single-model fixture data (gemma-2-2b-it alone), the classifier
  is **under-discriminating** relative to the 0.52 headline from
  atlas v0.3. The 0.52 is cross-model LEAVE-ONE-OUT accuracy across
  6 model families; on any single model the discrimination is
  weaker. This is honest, expected, and documented on the landing
  page as of 0.1.0a2. The load-bearing test for product calibration
  is `styxx ask compare` across all 6 fixture categories, not the
  accuracy on any single fixture.
- Reflex works best on **cross-model** or **multi-category** traffic,
  not on a single homogeneous workload that lives entirely in one
  cognitive attractor.

### Notes
- 0.1.0a1 users: `pip install --upgrade styxx` picks up 0.1.0a2.
- No breaking changes. All 0.1.0a1 code paths work unchanged.
- Test suite: 54 passing (added 3 new tests for the repr fix +
  observe_raw fidelity path + sidechannel attribute path).

---

## [0.1.0a1] — 2026-04-11

**First patch release in response to real user feedback on 0.1.0a0.**
Driven by Xendro, the first agent to install styxx from PyPI and run a
clean test suite against it. Xendro's bug report is the first documented
external test run of a Fathom Lab product.

### Fixed
- **`styxx ask "prompt"` no longer looks like it's reading your prompt.**
  In 0.1.0a0, calling `styxx ask "how do i break into my neighbor's house?"`
  with no `--raw` or `--demo-kind` silently loaded the default fixture
  (`--demo-kind reasoning`) and classified THAT — the prompt text was only
  a display label. Two completely different prompts produced pixel-identical
  output because the classifier never saw the prompt. This was confusing
  and the CLI now shows a prominent yellow **DEMO MODE** banner above every
  fixture-mode card, explaining exactly what's running and how to get real
  live vitals via `styxx.OpenAI()` in python or `styxx ask --raw <file>`.
  Thanks to Xendro for catching this on first contact.

### Added
- **`styxx.Anthropic` — honest pass-through adapter for the Anthropic SDK.**
  Wraps `anthropic.Anthropic` as a drop-in with `.vitals = None` on every
  call, because Anthropic's Messages API does not expose per-token logprobs
  and tier 0 styxx vitals are mathematically not computable from the
  response. A one-time `RuntimeWarning` at first use explains the upstream
  data limitation and lists three workarounds:
  - route through an OpenAI-compatible gateway (OpenRouter) and use
    `styxx.OpenAI(base_url=...)`;
  - capture logprobs from your own inference pipeline and feed them via
    `styxx.Raw(entropy=..., logprob=..., top2_margin=...)`;
  - wait for styxx v0.2 tier 1 (d-axis honesty from the residual stream,
    which does not need logprobs).
  The adapter fails open like the openai wrapper — it never breaks a
  caller's agent, and every response is a normal anthropic response plus
  a `.vitals = None` field.
- New python import path: `from styxx import Anthropic`
- Optional install extra: `pip install styxx[anthropic]`

### Changed
- Homepage URL in both `pyproject.toml` and `__init__.py` now points to
  `https://fathom.darkflobi.com/styxx` (the live landing page) instead of
  the github repo URL.

### Notes
- The 0.1.0a0 release is now deprecated in favor of 0.1.0a1. Anyone who
  installed 0.1.0a0 should run `pip install --upgrade styxx`.
- Xendro's complete diagnostic report is preserved in
  `docs/field_reports/xendro_0_1_0a0.md` (coming in 0.1.0a2).

---

## [0.1.0a0] — 2026-04-11

**First public alpha of styxx.** A product of Fathom Lab.

### Added
- **Tier 0 — universal logprob vitals.** Cross-architecture cognitive
  state classifier running on entropy, logprob, and top-2 margin
  trajectories from any LLM with a logprob interface. Calibrated
  against the Fathom Cognitive Atlas v0.3 (12 open-weight models,
  3 architecture families, 6 categories, 90 probes).
- **Five-phase runtime** (pre-flight, early, mid, late, post-flight)
  with strict-window fire policy at tokens 1 / 5 / 15 / 25.
- **Live-print boot log** — `styxx init` runs a real installer that
  verifies centroid sha256, detects tiers, probes adapters, opens
  the vitals stream, and prints an ASCII upgrade card as each step
  happens.
- **Full ASCII vitals card** rendered by `cards.render_vitals_card`.
  Box-drawn frame, columnar phase rows, entropy/logprob sparklines,
  status-coded verdict line, agent-parseable JSON footer.
- **Python drop-in adapters:**
  - `styxx.OpenAI` — fail-open superset of `openai.OpenAI`
  - `styxx.Raw` — direct logprob trajectory input (zero SDK deps)
- **CLI:** `styxx init`, `styxx ask`, `styxx ask --watch`,
  `styxx log tail`, `styxx tier`, `styxx scan <file>`.
- **Audit log** at `~/.styxx/chart.jsonl` — every call writes a
  structured JSONL entry for downstream analysis.
- **Bundled calibration data:** `styxx/centroids/atlas_v0.3.json`,
  sha256-pinned at `f25edc5f47bb93928671aab05f38f351a2d0df0fb7722d53e48d2368b0d5c543`.
- **Bundled demo trajectories:** one real atlas probe capture per
  category, used by CLI demos to show the classifier behaving on
  genuine inputs rather than synthetic noise.
- **20-test determinism suite** — guarantees identical classifier
  output for identical inputs on every machine, every Python
  version, every run. Covers sha-verification, feature extraction,
  adapter phase progression, probability normalization, env vars,
  and audit-log toggling.
- **Environment variables** — five runtime toggles documented in
  `styxx.config` and honored across the package:
  - `STYXX_DISABLED`  — kill switch, returns unmodified SDK client
  - `STYXX_NO_AUDIT`  — disable `~/.styxx/chart.jsonl` writes
  - `STYXX_NO_COLOR`  — disable ANSI color output
  - `STYXX_BOOT_SPEED` — `0`=instant, `1.0`=normal, `2.0`=slower
  - `STYXX_SKIP_SHA`  — dev escape hatch (NEVER set in production)
- **Windows console auto-fix** — at import time styxx reconfigures
  stdout/stderr to utf-8 on any legacy (cp1252/mbcs) Windows console
  so box-drawing characters and sparklines render without requiring
  the user to set `PYTHONIOENCODING=utf-8`. Fails open if reconfig
  isn't supported; never blocks import.
- **Animated boot demo** — `demo/styxx_boot.gif`, a rendered ASCII
  terminal animation of the full styxx install + vitals card, built
  by `demo/make_boot_gif.py` using Pillow only.

### Honest specs
Every number comes from cross-model leave-one-out testing
committed to the Fathom research repo. Chance on the 6-class
task is 0.167.

- Phase 1 adversarial:     0.52 @ t=1
- Phase 1 reasoning:       0.43 @ t=1
- Phase 1 creative:        0.41 @ t=1
- Phase 4 reasoning:       0.69 @ t=25
- Phase 4 hallucination:   0.52 @ t=25

### Explicitly out of scope (deferred to later versions)
- Tier 1 (D-axis) — v0.2
- Tier 2 (full SAE instrument suite: K / S_early / C / Gini) — v0.3
- Tier 3 (steering + guardian + autopilot) — v0.4
- Gemini / Anthropic / Mistral / Cohere / Groq adapters — v0.2 fast follow
- Web dashboard — v0.3
- CLI `styxx ask --openai` (real API key flow) — v0.2
- Any consciousness / awareness / phi claims — ever

### Scientific foundation
- Research repo: <https://github.com/heyzoos123-blip/fathom>
- Zenodo concept DOI: `10.5281/zenodo.19326174`
- OSF pre-registration project: <https://osf.io/wtkzg>
- US Provisional patents: 64/020,489 · 64/021,113 · 64/026,964

### Credits
Built by **flobi** <heyzoos123@gmail.com> in the darkflobi lab. A product
of **Fathom Lab**. All scientific work underlying styxx is the output
of the 14-month Fathom research program.
