# RESULT — SWALLOW-8: who writes the hidden check — a fifth of the population's workflow commits carry an agent's signature, two fifths of 2026's; those commits bring a third of the hidden checks, at 1.6× the rate of a person's commit, and repair no worse; dependency and release bots bring none

Fathom Lab · 2026-09-21 · Scores the receipt `swallow8_receipt.json.gz` against the
preregistration frozen at sha256 `316db0509151e4697c8f5c475812e212754e0d775f35017e38ec096bb535ecb3`.
Not amended. One run.

Receipt: `papers/harness/swallow8_receipt.json.gz` (sha256 of the JSON `a3dd0272…`, recorded by the
scorer; every non-root mainline commit of the SWALLOW-7 receipt with its class, the label of the
signal that classed it, and the gate's counts — no name, no address) · instrument
`benchmarks/harness_mutation/authorship.py`, sha256 `c3fb6e42…` · source: the SWALLOW-7 receipt
(`c6b12d09…`, differential `91e4a4a7…`) and the SWALLOW-6 clones · 21,569 commits, 96
repositories, 0 unclassified · 26 s · scored by `swallow8_score.py`.

**VALID. 4 of 7 predictions HIT** (P1, P3, P4, P6); P2, P5, P7 MISS. **4,098 of the 21,569
workflow-touching commits — 19% — carry a coding agent's signature** (a `Co-authored-by` naming
one, 1,858; an agent as the author, 1,286; a merge of an agent's branch, 905), in 80 of 96
repositories; they are 0.09% of the commits dated 2024, 21% of 2025's and **38% of 2026's** (P6).
They bring **49 of the 147 newly hidden checks** (33%; P1) from 31 firing commits — **0.76% of
agent commits fire, 0.46% of a person's**, a ratio of 1.64 (P2 predicted 2: MISS); within the two
years agents exist, 0.76% against 0.60%, and a person's commits before 2025 fired at 0.32%.
**Not one of 1,690 dependency, release and action-bot commits fires** (P3). The agent's hidden
check is repaired no worse: **36 of 49 have a verified repair (73%)** against 65 of 98 (66%) for a
person's (P4). It does not arrive in larger batches (1.58 per firing commit against 1.34; P5
predicted 1.5×: MISS), and it is not only born: **36 of 49 are new steps born hidden, 10 are
non-checks rewritten into hidden checks** — all ten in one Copilot commit to `dotnet/maui` — and
3 are existing checks made hidden (P7 predicted 90% born: MISS at 73%).

## 0. What was done

Every first-parent commit touching `.github/workflows` in the 96 clones was read for its author
name and address, subject and body, and classed by the frozen rule — `agent` on any of a stated
list of signatures (an agent's own author name or address, a `Co-authored-by` trailer naming one,
"Generated with …", "[CI] Agentic workflows", a merge of a branch under `codex/`, `claude/`,
`copilot/`, …), `automation` on a bot author that carries no such signature, `human` otherwise —
then joined by sha to the SWALLOW-7 receipt's commits, their firings, and the checks each firing
brought with its verified repair. A bare first name is not a signal, so a person named Claude or
Devin is a person. The receipt carries classes and signal labels, never a name.

## 1. Gates

| gate | bar | result |
|---|---|---|
| G-S8-1 instrument | the rule on shaped records; the join on the scripted history; deterministic; no name written | pass — `tests/test_harness_authorship.py`, 4 passed |
| G-S8-2 population | 0 unclassified; ≥ 90 repositories | pass — 21,569 commits, 0 unclassified, 96 repositories |
| G-S8-3 frozen underneath | `c3fb6e42…`; the SWALLOW-7 receipt at `c6b12d09…`, differential `91e4a4a7…` | pass |
| G-S8-4 ledger | P1–P7 scored | pass |

No deviation. Stated at the freeze and repeated: the authorship of six of SWALLOW-7's eight batch
commits was read while the rule was written, so P1 is calibrated; P2–P7 were blind.

## 2. Predictions, scored

| | predicted | observed | |
|---|---|---|---|
| P1 a fifth (calibrated) | ≥ 20% of newly hidden checks brought by agent commits | **49 of 147, 33%** | HIT |
| P2 fires more (blind) | agent fire rate ≥ 2× human | **0.76% vs 0.46%, 1.64×** | MISS |
| P3 bots do not (blind) | ≤ 0.1% of automation commits fire | **0 of 1,690** | HIT |
| P4 no less repairable (blind) | agent-brought repair share ≥ human-brought | **73% vs 66%** | HIT |
| P5 more at once (blind) | checks per firing commit, agent ≥ 1.5× human | **1.58 vs 1.34, 1.18×** | MISS |
| P6 the trend (blind) | agent share ≥ 5% in 2026 and ≥ 3× 2024 | **38% in 2026; 0.09% in 2024** | HIT |
| P7 born, not turned (blind) | ≥ 90% of agent-brought checks born hidden | **36 of 49, 73%** (10 rewritten into checks, 3 acquired) | MISS |

## 3. Reading

**The population has changed hands.** In 2024, three of 3,340 workflow commits carried an agent's
signature; in 2025, 1,358 of 6,396; in 2026 to the pinned HEADs, 2,737 of 7,168. Eighty of 96
repositories have at least one; `MontrealAI` (588 of 693), `gh-aw` (566 of 709), `mlflow` (585 of
1,323), `airbyte`, `aspire`, `CodenameOne`, `cmux`, `nodetool` and `py3plex` (71 of 73) are
largely written that way now. Every number here is a floor: a commit whose agent left no
signature is a person's under the rule.

**The agent's commit fires more, and the difference is smaller than the era.** 0.76% of agent
commits bring a hidden check, 0.46% of a person's — 1.64×, short of the 2× predicted. But a
person's commits before 2025 fired at 0.32% and in 2025–2026 at 0.60%: much of the gap is the
period, not the author. Within 2025–2026 the ratio is 1.27. The hidden check is more common now,
whoever writes it, and the agent writes a little more of it than that.

**Who, by signature.** Of the 31 agent firing commits, 13 carry a `Co-authored-by` trailer
(`aspire` ×4, `sentry-docs` ×2, `CodenameOne` ×2, `nodetool` ×3, `selfxyz`, `langfuse`), 11 have
an agent as the author (`py3plex` ×5, `gh-aw` ×4, `sdk`, `maui`), 7 merge an agent's branch
(`nodetool` ×4, `MontrealAI` ×3). `nodetool`'s two batches of six read differently under the
rule: the first is a merge of a `claude/` branch (agent); the second, "add ten scheduled
maintenance routines", carries no signature and is a person's — the floor, in one repository.

**The bots that never fire.** 1,690 commits by dependabot, renovate, github-actions and their
kin — version bumps, releases, generated updates — and not one brings a hidden check. The
`[CI] Agentic workflows` asset update that brought `maui`'s ten is authored by Copilot, and is
an agent's under the rule, not a bot's.

**Repaired no worse.** 36 of the 49 agent-brought checks have a verified repair on the commit's
own text (24 `no-continue-on-error`, 5 `no-default`, 3 guards, 3 `both`, 1 strict shell) against
65 of 98 for a person's — the agent writes the textbook line, `continue-on-error: true`, and the
gate's one-line fix answers it. It also writes the `|| echo` default (5 of the 7 `no-default`
repairs in the whole population are on agent-brought checks).

**Not only born.** The prediction that an agent writes new hidden steps but does not hide existing
ones missed on one commit: Copilot's update of `maui`'s generated assets rewrote ten
`Validate COPILOT_PAT_N` steps that were not checks into checks that swallow — the largest single
firing in the population — and `gh-aw`'s "Make error-message lint advisory" and two `MontrealAI`
`codex/` merges each hid an existing check.

## 4. What this does not say

The classes are a list of names and phrases. An agent that left no signature is a person under
the rule; a person who merged an agent's branch is an agent's commit under the rule; a
`Co-authored-by` trailer says the tool was in the loop, not who decided the `|| true`. The receipt
holds no name and this RESULT names no one. The rates are the frozen model's readings of hidden
checks, at commits on first-parent mainlines, in 96 repositories chosen for the agent pull
requests they receive — a population in which agents are common by construction.

## 5. What ships

Nothing in `styxx` changes: this cycle reads the receipts of the last two. `authorship.py` is
frozen at the sha256 its receipt names and pinned by its own test file.

## 6. Next

The floor is the finding's limit: a second signature list, wider, frozen and scored on the same
receipt, would say how much of the 19% is 30%. And the gate on the compiler's output: the
`maui` commit is one of a class — a generated asset update that rewrites steps — that
`styxx ci-audit --base` would have stopped, and that the compiler could run on itself.
