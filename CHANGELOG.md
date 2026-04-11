# Changelog

All notable changes to styxx will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

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
