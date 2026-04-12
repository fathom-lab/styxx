# Changelog

All notable changes to styxx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
