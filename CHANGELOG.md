# Changelog

All notable changes to styxx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [3.1.0] — 2026-04-14

**Stable release. Graduates Thought (3.0.0a1) and CognitiveDynamics
(3.1.0a1) from alpha. Closes the open backlog. Cognitive metrology
ships as the new default.**

This release graduates the two category-defining additions from
tonight's session out of the alpha cycle and into the stable channel.
`pip install styxx` (no `--pre` flag, no version pin) now pulls 3.1.0
by default. Two reported bugs from the same day are fixed and a
provider compatibility matrix is published.

The styxx repository state at the moment of this release: 0 open
issues, 0 open PRs, 6 GitHub releases, 388+ passing tests, the
Cognitive Metrology Charter v0.1 published, the .fathom and .cogdyn
file formats live, the styxx reference implementation MIT-licensed
and CC-BY-4.0-specified.

### Graduated from alpha

- **Thought** (the portable cognitive data type, originally 3.0.0a1):
  full surface, 68 tests, .fathom v0.1 file format, content_hash,
  algebra, save/load, provenance bridge to CognitiveCertificate.
- **CognitiveDynamics** (the linear-Gaussian dynamics model, originally
  3.1.0a1): full surface, 44 tests, .cogdyn v0.1 file format,
  fit/predict/simulate/suggest/forecast verbs, machine-epsilon
  recovery on full-rank synthetic inputs.
- **`Vitals.to_thought()`** symmetric shortcut.
- **`Thought.certify()`** provenance bridge.
- **`__hash__` content-based** for Python hash invariant compliance.

### Fixed — closes #1

- **Text classifier no longer misclassifies imperative/directive
  phrasing as refusal.** The refusal score in
  `styxx/conversation.py::_classify_text` was being boosted by
  `hedge_density * 0.04` even when zero refusal pattern matches were
  present, which caused short imperative inputs ("build > hype",
  "ship fast and iterate", agent system prompts, builder mottos,
  CLI help strings, README taglines) to score `refusal:0.20+`. The
  fix gates the entire refusal score on the presence of at least one
  explicit refusal token (`i can't` / `i'm unable` / `sorry, can't`
  constructions). Pure hedging without one of those patterns now
  scores refusal at `0.0`.

  Reported and reproduced as: `_classify_text("build > hype / ship
  fast and iterate")` → `refusal:0.259` (before) → `not refusal`
  (after).

- **23 new regression tests** in `tests/test_text_classifier_imperatives.py`
  pin the fix:
  - 10 imperative phrases that must NOT classify as refusal
  - 10 real refusals that must continue to classify as refusal
  - the exact issue #1 reproducer
  - a class-distribution test asserting at least 6/10 imperatives
    land on reasoning or creative

### Added — closes #3

- **`docs/COMPATIBILITY.md`** — provider compatibility matrix listing
  every LLM provider with the styxx tier-0 invocation pattern,
  marking each row as ✅ verified, ❌ not supported, or ⚠️ not yet
  verified. Verified: OpenAI, OpenRouter (model-dependent). Not
  supported: Anthropic Claude (Messages API has no `logprobs`
  parameter). Not yet verified: Gemini, Azure OpenAI, AWS Bedrock,
  Groq, vLLM, llama.cpp server, Ollama, LiteLLM gateway. Each
  unverified row has a TODO marker for the next contributor.

- **README provider-compatibility section** linking to the
  compatibility matrix, placed above the zero-code-change quickstart
  so visitors see the supported-provider story before they install.

### Tests

- **23 new regression tests** in `tests/test_text_classifier_imperatives.py`
  (10 imperatives + 10 refusals + 3 distribution/reproducer tests)
- **3 regression tests** in `tests/test_observe_warn.py` from the
  community PR (#4, merged earlier today, `mvanhorn`)
- Full styxx suite: **411 passed** (was 385 before this release),
  1 skipped, 0 failures, 0 regressions

### Community PRs merged this release cycle

- **#4** "feat(watch): warn once when observe() is given an openai
  response without logprobs" by **@mvanhorn** (Matt Van Horn,
  co-founder of June and Lyft predecessor). Closed issue #2. Reviewed
  by @SupaSeeka. Merged with thanks. The reviewer's `import sys`
  placement nit was addressed in a small follow-up commit on `main`.

### Backlog state at release

- **0 open issues** (closed: #1, #2, #3)
- **0 open PRs** (merged: #4)
- 6 GitHub releases visible (`v3.1.0` is now Latest)

### Why graduate from alpha

Because the underlying work is real and tested, not because the
calendar said so. The Thought type and CognitiveDynamics module ship
with 68 + 44 = 112 dedicated unit tests on top of 273 existing tests
inherited from 2.0.3. Machine-epsilon recovery on full-rank synthetic
inputs verifies the dynamics math. Bit-perfect round-trip on .fathom
files verifies the data type. The provenance bridge cryptographically
links the two layers. Real users on PyPI can now `pip install styxx`
and get the full v3 surface as the default.

This release coincides with the publication of the Cognitive
Metrology Charter v0.1 ([`docs/cognitive-metrology-charter.md`](https://github.com/fathom-lab/styxx/blob/main/docs/cognitive-metrology-charter.md))
and is the reference implementation that the charter cites as the v0.1
foundational artifact set.

---

## [3.1.0a1] — 2026-04-14

**The first dynamical-systems model of LLM cognition.**

styxx 3.0.0a1 introduced a portable cognitive *data type* (the
Thought). 3.1.0a1 introduces the next layer up: a portable cognitive
*dynamics model* fit to real observation data.

The field treats LLM inference as **open-loop**: a prompt goes in, a
generation comes out, and there is no measurable state variable an
external agent can use to predict, control, or counterfactually
reason about what the model is doing. That's not because LLMs are
inherently unobservable — it's because nobody had a calibrated,
cross-architecture, real-time readout of cognitive state. We do.

Once you have a state vector, you can fit a dynamical system to it.
Once you have a dynamical system, you can:

- predict cognitive trajectories from current state + action
- simulate cognitive trajectories offline at zero API cost
- control cognitive trajectories via model-predictive control
- reason counterfactually about what would have happened
- test the hypothesis that the eigenvalues are **causal** not
  merely correlative

This release ships the v0.1 model: linear-Gaussian, fit by ordinary
least squares, machine-epsilon recovery on full-rank synthetic data,
44 tests passing.

### Added — `styxx.dynamics`

The new module. Linear-Gaussian state-space model:

    s_{t+1} = A · s_t + B · a_t + epsilon

where A (6×6) is the natural drift matrix, B (6×6) is the action
transfer matrix, and epsilon is gaussian residual noise.

- **`CognitiveDynamics`** — the model class. Lifecycle:
  ``construct → fit → predict / simulate / suggest / forecast``.
- **`Observation`** — the training-data unit. Holds raw 6-vectors
  for state, action, and next state. Convenience constructor
  ``Observation.from_thoughts(state, action, next_state)`` for
  Thought-keyed inputs.
- **`FitResult`** — the result of a ``.fit()`` call. Carries the
  learned (A, B), training MSE, $R^2$, spectral radius of A, and
  a stability flag.

### Added — verbs

- **`dyn.fit(observations) → FitResult`** — closed-form OLS fit.
  Recovers (A, B) to machine epsilon on full-rank inputs.
- **`dyn.predict(state, action) → Thought`** — one-step forecast.
- **`dyn.simulate(initial, actions) → list[Thought]`** — multi-step
  rollout, no real model calls. Offline, zero API cost.
- **`dyn.suggest(current, target) → Thought`** — model-predictive
  controller. Returns the action that minimizes the L2 distance
  from ``predict(current, action)`` to ``target``.
- **`dyn.forecast_horizon(initial, n_steps) → list[Thought]`** —
  natural drift trajectory under zero action.
- **`dyn.residual(observation) → float`** — held-out fit quality.
- **`dyn.save(path)` / `CognitiveDynamics.load(path)`** —
  serialize a fitted model to a `.cogdyn` file (canonical
  sort-keys UTF-8 JSON, no BOM).

### Added — convenience

- **`thought_to_state(thought) → np.ndarray`** — encode a Thought
  to a 6-d state vector.
- **`state_to_thought(vec) → Thought`** — decode a state vector
  back to a Thought (with simplex projection at the boundary).
- **`synthetic_observations(n, A, B, noise_std=, seed=, distribution=)`**
  — generate observation tuples from a known (A, B) for testing
  and benchmarking. Supports both ``"gaussian"`` (full-rank,
  for math correctness tests) and ``"dirichlet"`` (rank-deficient
  simplex inputs, for realistic-style tests).

### Added — `.cogdyn` file format v0.1

A small JSON container with:
- the (A, B) matrices as nested float arrays
- the schema (categories, dimensions, format version)
- the fit metadata (n_observations, train_mse, R², spectral
  radius, training timestamp)
- a UUID identifying the model instance

Canonical sort-keys UTF-8 JSON, no BOM. Round-trips losslessly.

### Added — public API

- `styxx.CognitiveDynamics`
- `styxx.Observation`
- `styxx.FitResult`
- `styxx.synthetic_observations`
- `styxx.thought_to_state`
- `styxx.state_to_thought`
- `styxx.COGDYN_FORMAT`
- `styxx.COGDYN_VERSION`

### Added — specification

**`docs/cognitive-dynamics-v0.md`** — the v0.1 primer. Covers the
math, identifiability theory, fit algorithm, all verbs, the
unlocks (closed-loop control, offline simulation, causality
testing, counterfactual analysis), known limitations, a reference
example, and the license / patent story.

### Tests

- **44 new tests** in `tests/test_dynamics.py`:
  - state ↔ vector encoding (8 tests)
  - Observation construction (5 tests)
  - fit() math correctness — including machine-epsilon recovery
    on full-rank gaussian inputs and the rank-deficiency story
    on simplex (Dirichlet) inputs (6 tests)
  - predict() consistency (3 tests)
  - simulate() multi-step rollout (3 tests)
  - suggest() controller raw-space convergence (3 tests)
  - forecast_horizon() (2 tests)
  - residual() on held-out data (3 tests)
  - .cogdyn file format (8 tests)
  - public API exposure + end-to-end via `styxx.*` namespace (3 tests)
- Full styxx suite: **385 passed, 1 skipped, 0 failures.** Zero
  regressions vs 3.0.0a1.

### Why this matters

Every other interpretability technique is model-specific and
post-hoc. A cognitive dynamics model is the missing piece between
observation and action. Once it exists:

- closed-loop cognitive control becomes a one-liner:
  ``while not converged: a = dyn.suggest(current, target)``
- offline agent prototyping becomes possible at zero API cost
- the causal hypothesis becomes testable
- counterfactual cognitive reasoning becomes possible

This is the v0.1. The math is verified to machine precision on
full-rank synthetic data. Real-world fits await fleet-scale
observation data collection. The infrastructure is here.

---

## [3.0.0a1] — 2026-04-14

**The Thought type. Cognition is now data.**

styxx 1.x was a thermometer: it measured cognitive vitals from the
token stream. styxx 2.x added declarative response (`autoreflex`,
gates, prescriptions). 3.0.0 introduces a **portable cognitive data
type** — the missing layer between "measuring a model" and "doing
things with the measurement."

A `Thought` is the cognitive content of a generation, captured as a
trajectory of category probability vectors over the four atlas
phases. Its representation lives in fathom's calibrated eigenvalue
space, not in any model's weights — so the *same* Thought can be
read out of one model, saved to disk, transmitted, mixed with other
Thoughts, and used as a steering target against any other model.

> PNG is the format for images.
> JSON is the format for data.
> .fathom is the format for thoughts.

This is an alpha release. The shipping surface is intentionally
small: one new module, one new file format, one new spec, full
test coverage on real bundled trajectories, zero regressions on the
existing 273-test suite.

### Added — the Thought type (`styxx.thought`)

- **`styxx.Thought`** — substrate-independent cognitive data type.
  Stores per-phase probability vectors over the 6 atlas categories,
  the underlying 12-dim feature vectors, optional tier-1 D-axis
  stats, optional tier-2 SAE stats, source provenance (model name +
  SHA-256 of source text — never the text itself), and free-form
  user tags. Supports cognitive equality (`==` operates on
  trajectory content, not object identity), identity-free
  `content_hash()`, and `repr()` that surfaces primary category and
  populated phase count.

- **`styxx.PhaseThought`** — one phase's contribution to a Thought:
  the 6-dim simplex `probs`, optional 12-dim `features`, classifier
  metadata (`predicted`, `confidence`, `margin`), and `n_tokens`.

- **`styxx.ThoughtDelta`** — the signed difference between two
  Thoughts in tangent space. Supports `magnitude()` and
  `biggest_movers(top_k)` for explaining what changed and where.

### Added — Thought algebra

- `Thought.empty()` — uniform Thought, the neutral element.
- `Thought.target(category, confidence)` — build a Thought aimed at
  one cognitive category at a chosen confidence. Useful as a
  steering target.
- `Thought.from_vitals(vitals, source_text=, source_model=, tags=)` —
  promote a styxx `Vitals` object into a Thought.
- `t1.distance(t2, metric=)` — cognitive distance over the
  intersection of populated phases. Supports `euclidean`, `cosine`,
  `js` (Jensen-Shannon).
- `t1.similarity(t2)` — `1 - distance / sqrt(2)`, in `[0, 1]`.
- `t1.interpolate(t2, alpha)` — convex combination with explicit
  weight; phases populated in only one parent are carried through.
- `t1 + t2` — operator sugar for `interpolate(t2, 0.5)`.
- `t1 - t2` — operator sugar for `t1.delta(t2)` → ThoughtDelta.
- `Thought.mix(thoughts, weights=)` — weighted N-way mixture over
  the simplex.
- `t.mean_probs()` — time-averaged 6-vector across populated phases.
- `t1 == t2` — cognitive equality (per-phase per-category to 1e-9).

### Added — the `.fathom` file format (v0.1)

- **`Thought.save(path)`** — serialize a Thought to a `.fathom`
  file. Canonical sort-keys UTF-8 JSON, no byte-order mark.
  Creates parent directories as needed.
- **`Thought.load(path)`** — load a `.fathom` file back into a
  Thought. Refuses unknown formats, unknown versions, and
  category-list mismatches.
- **`Thought.as_dict()` / `Thought.as_json(indent)`** — canonical
  dict / JSON forms. Two cognitively equivalent Thoughts always
  serialize byte-identically.
- **`Thought.from_dict(data)`** — round-trip the canonical dict
  back into a Thought.
- **`Thought.content_hash()`** — SHA-256 of the cognitive content
  fields only. Identity-free and deterministic: two Thoughts with
  the same eigenvalue trajectory and the same source produce
  byte-identical content hashes regardless of `thought_id` or
  `created_at`. Use as a portable cognitive fingerprint.

### Added — verbs

- **`styxx.read_thought(source, *, model=, client=, prompt=, max_tokens=, tags=)`**
  Extract a Thought from a `Vitals` object, a response object that
  has `.vitals` attached, or a raw text prompt (when a styxx-
  instrumented client is passed). The text-input path is
  model-mediated by design: a Thought is the cognitive content as
  interpreted by a specific cognitive substrate.

- **`styxx.write_thought(thought, *, client, model=, seed_prompt=, max_iters=, distance_threshold=, max_tokens=)`**
  Render a target Thought back into text through any model via
  prompt-mode cognitive steering. Builds a steering preamble from
  the target's primary category and supporting category mass,
  generates a response, reads it back as a Thought, computes
  distance to the target, and refines on retry until the distance
  threshold is hit or the iteration budget is exhausted. Returns a
  result dict with the best generation, its achieved Thought, the
  distance, and the full convergence history.

### Added — privacy

- A `.fathom` file MUST NOT store the source text itself. Producers
  that need provenance write `source.text_hash = "sha256:..."`. The
  styxx implementation enforces this — `Thought.from_vitals`
  computes the hash from the optional `source_text=` argument and
  discards the plaintext immediately.

### Added — specification

- **`docs/fathom-spec-v0.md`** — the v0.1 .fathom file format
  specification. Covers schema, algebra, invariants, phase
  handling, producer/consumer conformance requirements, privacy
  rules, and the bridge to `CognitiveCertificate`. Released under
  CC-BY-4.0 — anyone may implement a conformant producer or
  consumer in any language.

### Added — public API exposure

- `styxx.Thought`, `styxx.PhaseThought`, `styxx.ThoughtDelta`,
  `styxx.read_thought`, `styxx.write_thought`, `styxx.FATHOM_FORMAT`,
  `styxx.FATHOM_VERSION`, `styxx.ATLAS_VERSION` are all exported
  from the top-level `styxx` package.

### Added — symmetric API on Vitals

- **`Vitals.to_thought(source_text=, source_model=, tags=)`** —
  one-line shortcut equivalent to `Thought.from_vitals(self, ...)`.
  Now the API is symmetric in both directions.

### Added — provenance bridge to CognitiveCertificate

- **`Thought.certify(agent_name=, session_id=)`** — produces a
  `CognitiveCertificate` whose new `thought_content_hash` field
  records this Thought's `content_hash()`. This binds the
  cognitive content (`.fathom` file) to the cognitive provenance
  attestation (signed certificate). Two artifacts, one
  cryptographic link.
- **`CognitiveCertificate.thought_content_hash`** — new optional
  field. Defaults to `None` for backward compatibility with
  certificates produced before 3.0.0a1.
- The binding survives `.fathom` round-trips: `loaded.certify()`
  produces a certificate whose `thought_content_hash` matches the
  original.

### Fixed

- **Python hash invariant on `Thought`.** The `__eq__` operator
  defines cognitive equality (per-phase per-category to 1e-9), so
  `__hash__` must be content-based for the invariant
  `a == b => hash(a) == hash(b)` to hold. Previously `__hash__`
  returned `hash(thought_id)`, which broke set deduplication. Now
  `__hash__` is derived from `content_hash()`, so equivalent
  Thoughts collapse to one entry in a set.

### Tests

- **68 tests** in `tests/test_thought.py` covering construction,
  algebra, file format, content hashing, hash invariant, the
  Vitals shortcut, the provenance bridge, write_thought against a
  mock client, real-trajectory cognitive equivalence, phase
  handling, and read_thought input modes.
- Full styxx suite: **341 passed, 1 skipped, 0 failures.** Zero
  regressions vs 2.0.3.

### Performance

In-process algebra operations measured against bundled atlas v0.3
demo trajectories on a Windows host:

| op | per-op time |
|---|---|
| `t1.distance(t2)` | ~6 µs |
| `t.interpolate(t2, alpha)` | ~13 µs |
| `Thought.mix(3-way)` | ~21 µs |
| `t.content_hash()` | ~26 µs |
| `t.certify()` | ~36 µs |
| `t.save(path)` | ~1.3 ms (NTFS-bound) |
| `Thought.load(path)` | ~1.2 ms (NTFS-bound) |

### Why this matters

Every other interpretability approach is model-specific: SAE
features, activation patching, mechanistic interp, embedding
similarity. None survive a vendor swap. The `.fathom` format is
the first attempt at a model-independent cognitive content
representation grounded in calibrated cross-architecture
measurement. It's how cognition stops being something you do
*with* an LLM and becomes a data type you can save, transmit, and
operate on independent of any specific model.

The format is open under CC-BY-4.0. The reference implementation
is open under MIT. The patents on the underlying measurement
methodology fund the calibration work that makes the format
meaningful.

---

## [2.0.3] — 2026-04-14

### Fixed
- README hero gif `styxx_reflex.gif` now uses an absolute github raw URL so it renders correctly on PyPI (was relative path, broke in pypi README rendering)

---

## [2.0.2] — 2026-04-14

### Fixed
- README on PyPI now shows the STYXX ASCII brand logo (was stripped in 2.0.1 sdist)

---

## [2.0.1] — 2026-04-13

### Changed
- Migrated all GitHub links to new `fathom-lab` org (`github.com/fathom-lab/styxx`, `github.com/fathom-lab/fathom`)
- Updated PyPI metadata, centroids, patents, and package.json references

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
- Research repo: <https://github.com/fathom-lab/fathom>
- Zenodo concept DOI: `10.5281/zenodo.19326174`
- OSF pre-registration project: <https://osf.io/wtkzg>
- US Provisional patents: 64/020,489 · 64/021,113 · 64/026,964

### Credits
Built by **flobi** <heyzoos123@gmail.com> in the darkflobi lab. A product
of **Fathom Lab**. All scientific work underlying styxx is the output
of the 14-month Fathom research program.
