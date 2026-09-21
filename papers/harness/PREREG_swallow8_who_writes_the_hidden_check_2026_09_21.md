# PREREG — SWALLOW-8: who writes the hidden check — the gate's firings, by who made the commit

Fathom Lab · 2026-09-21 · Frozen before any commit of the population is classified.
Follows `RESULT_swallow7_the_differential_audit_2026_09_21.md` (INVALID on its frozen join; 6/7
reported, not claimed).

## Where this came from, stated before anything is measured

SWALLOW-7 ran the gate at 21,569 mainline commits of 96 repositories and found 104 that bring a
check hiding its own failure — 147 checks. Eight of those commits brought three or more at once,
and the two largest were a compiler's asset update and a bot's batch. This repository's own
thesis is that agents send pull requests. So: of the commits that touch a workflow, which are an
agent's, which are a bot's, which are a person's — by a stated rule on the commit's author,
subject and body — and does the gate fire on them differently?

## 1. The instrument

`benchmarks/harness_mutation/authorship.py`, frozen at `c3fb6e42…`, on the SWALLOW-7 receipt
(`swallow7_receipt.json.gz`, file sha256 `c6b12d09…`; differential `91e4a4a7…`) and the
SWALLOW-6 clones. It reads nothing but each mainline commit's author name and address, subject
and body, and writes no name and no address: a commit in the receipt carries its sha, its time,
its class, the label of the signal that classed it, and the gate's counts for it.

**The rule** (`classify`), in order:
- **agent** — the first of: the author name is an agent's own form (`Copilot`, `copilot-swe-agent`,
  `claude[bot]`, `Claude Code`, `devin-ai-…`, `openhands`, `sweep-ai`, `coderabbit`,
  `google-labs-jules`, `cursor-agent`, `codex[bot]`, `codegen-sh`, `gemini-code-assist`,
  `amazon-q`, `qodo`, `blackbox`, `gh-aw[bot]`); the author address is one (`noreply@anthropic.com`,
  `copilot@`, `devin-ai-integration`, …); a `Co-authored-by` trailer names one; the body or
  subject carries "Generated with/by Claude Code | Claude | Copilot | Codex | Cursor | Devin |
  Aider | Gemini | Jules", "Made with …", or "[CI] Agentic workflows"; the subject merges a branch
  under `codex/`, `claude/`, `copilot/`, `cursor/`, `devin/`, `aider/`, `jules/`, `gemini/`,
  `sweep/`, `openhands/`. **A bare first name is not a signal**: "Claude" and "Devin" are people's
  names.
- **automation** — otherwise, the author name or address matches a bot that is not a coding agent
  (`[bot]`, dependabot, renovate, github-actions, pre-commit-ci, release bots, …).
- **human** — otherwise. This class holds every agent-assisted commit that left no signature, so
  **the agent class is a floor**, and every agent share below is a lower bound.

**Population.** The non-root mainline commits of the SWALLOW-7 receipt (21,569), each joined by
sha to its class; the gate's firings and their checks come from that receipt unchanged.

## 2. What was known at the freeze

The rule was exercised on shaped records (the test file) and on the scripted history. **While
writing the rule, the authorship of six of SWALLOW-7's eight batch commits was read**, to see
what an agent's signature looks like in this population: `dotnet/maui`'s ten came in a commit
authored by Copilot; `nodetool`'s first six in a merge of a `claude/` branch and its second six
in a commit with no signal; `MontrealAI`'s two in merges of `codex/` branches; `vscode`'s and
`sentry`'s four each from people. So P1 is calibrated on those, and the rest are blind: no other
commit's author has been read, and no class count exists yet.

## 3. Predictions, committed now

Non-root commits; "brought by" a class means the firing commit is in that class.

**P1 — a fifth, at least** (calibrated). At least **20%** of the 147 newly hidden checks were
brought by agent-class commits.

**P2 — the agent's commit fires more** (blind). The share of agent-class commits that fire is
at least **2×** the share of human-class commits that fire.

**P3 — the bot's commit does not** (blind). At most **0.1%** of automation-class commits fire.

**P4 — no less repairable** (blind). Among newly hidden checks, the share with a verified repair
is at least as high for agent-brought as for human-brought.

**P5 — more at once** (blind). The mean number of newly hidden checks per firing commit is at
least **1.5×** higher for agent-class commits than for human-class commits.

**P6 — the trend** (blind). Agent-class commits are at least **5%** of the workflow-touching
commits dated 2026 and at least **3×** their share of those dated 2024.

**P7 — born, not turned** (blind). At least **90%** of agent-brought newly hidden checks are
`born hidden` (a new step), not an existing check made hidden.

## 4. Gates

| gate | what it holds | bar |
|---|---|---|
| G-S8-1 (instrument) | the rule on shaped records; the join on the scripted history; deterministic; no name written | `tests/test_harness_authorship.py` green |
| G-S8-2 (population) | every commit of the receipt is classed | 0 unclassified; ≥ 90 repositories |
| G-S8-3 (frozen underneath) | the instrument and its source | `c3fb6e42…`; the SWALLOW-7 receipt file at `c6b12d09…`, whose differential is `91e4a4a7…` |
| G-S8-4 (ledger) | every prediction scored | HIT/MISS for P1–P7 |

G-S8-1 to G-S8-3 are blocking.

## 5. What would abandon this

G-S8-2: a commit the receipt has that the clone's log does not is a join that is not on the
same history.

## 6. Honest statement of what a passing SWALLOW-8 means

That, by a list of names and phrases, the commits that carry an agent's signature bring hidden
checks at a higher rate than the commits that do not, and that dependency and release bots bring
none. It does not say an agent wrote the check: a merge of a `codex/` branch was reviewed by a
person, and a person's commit may have been an agent's work with no signature. It does not
identify anyone: the receipt holds classes, not names. The human class is the remainder of a
rule, not a finding about people.

## 7. Running it

```
python -m benchmarks.harness_mutation.authorship --receipt papers/harness/swallow7_receipt.json.gz --work <clones> --out papers/harness/swallow8_receipt.json
python papers/harness/swallow8_score.py
```
