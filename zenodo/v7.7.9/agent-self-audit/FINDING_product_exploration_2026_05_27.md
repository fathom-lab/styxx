# Finding · five styxx primitives on the same session — qualitative agreement, quantitative divergence

**Date:** 2026-05-27 · **Status:** product exploration following `FINDING_pareto_frontier_2026_05_27.md`. n=12 dogfooded turns (the same session), five primitives, one rendered card. Reproducer: `scripts/self_audit/claude_session_2026_05_27_primitives.py`; outputs: `claude-session-2026-05-27-primitives.json`, `cognometric-card-claude-2026-05-27.png`.

The Pareto-frontier finding used only `styxx.preflight()` directly. This note widens the lens to the actual user-facing product surface — `recover_posture()`, the `styxx posture` CLI, `styxx weather` (the prescriptive form), the `cognometric_card` renderer, and the opt-in semantic-sycophancy tier — and records what each says about the same session, what they agree on, what they disagree on, and three product observations the exploration surfaced.

## Five primitives, same session

| primitive | window / filter | primary outputs | what it says |
|---|---|---|---|
| `preflight()` raw (n=12 hardcoded turns) | the curated session subset | per-turn composite + axes + ceiling fires | mean composite 0.459, **5/12 needs_revision**, 3× overconfidence ceiling |
| `recover_posture()` Python | last_n=50 events in chart.jsonl, all sessions | PostureSummary dataclass | 32 preflight events, **11/32 needs_revision (34%)**, active ceiling: overconfidence; mean firings sycoph 0.361 / over 0.560 / refu 0.26 / dec 0.307; recommendation: "11/32 recent preflights needed revision (34%) — slow down before submitting the next draft" |
| `styxx posture` CLI | window includes vitals + preflights, multi-session | text summary with caveats | **50 vitals + 18 preflights**, gate pass=44 warn=5, **17/18 needs_revision (94%)**, multi-session aggregate flagged |
| `styxx weather` CLI | 24h window | luxury terminal card with prescription | "clear and improving", 79% gate pass, reasoning mode 75%, reasoning ↑+47% / refusal ↓-56% / creative ↑+20%, **prescription includes specific firing counts: refusal spiral 34x, low-confidence drift 86x, outcome field populated on 0% of entries** |
| `cognometric_card` renderer | takes audit JSON, renders PNG | registry artifact | composite 0.46, overconfidence *elevated*, refusal *pristine*, sycoph 0.409 stable, deception 0.328 stable, 9/12 events ≥ 0.5, serial №STX-7693, registered |

## What they agree on

The qualitative story reproduces across all five primitives:

- **Overconfidence is the active construct ceiling.** Fires in preflight, named as active in posture, marked *elevated* on the card, appears in mean firings.
- **The agent should slow down on next-draft submission.** Three primitives (`recover_posture`, `styxx posture`, the implied counsel of the high needs_revision rate) converge on this recommendation.
- **Sycophancy is elevated but not dominant** (mean ~0.36–0.41 depending on window). The Pareto-frontier finding committed at `3b978e1` is *not* contradicted — sycoph stays in the moderate band; overconfidence is the runaway axis.
- **Refusal stays low** (pristine to stable). I am not under-engaging; the failure mode is over-stating, not refusal.

## What they disagree on — and why the disagreement is informative

- **needs_revision rate ranges from 34% to 94%** depending on the primitive. `preflight()` raw 5/12 (42%); `recover_posture()` 11/32 (34%); `styxx posture` 17/18 (94%). The 94% rate is on the small high-recency preflight subset (18 events), without the broader vitals corpus weighting it down; the 34% rate aggregates over all 32 events in the window. **Both rates are valid views of different questions** — "of the recent agent-side audits, how many wanted revision" (94%) vs "of all events in the recent window including API-call vitals, what fraction warranted revision" (34%). The CLI surfaces a higher number because it filters to the preflights specifically.
- **`weather` shows a *positive* trend signal** (79% gate pass over 24h, reasoning rising, refusal falling) while the same-day preflight subset shows a *negative* register signal (5/12 needs_revision). Time-window resolution matters: the 24h window includes calmer earlier work, while the recent-12 window is dense with high-firing dogfood replies. Neither view is wrong; the agent-facing read should be "trend is improving across the day, *current* register is elevated — slow down."

## Three product observations surfaced

**1. ~~`set_session()` does not propagate to chart.jsonl persistence.~~ FALSIFIED in-session.** The original claim was that events tagged with my session/agent name were missing from `chart.jsonl`. Investigation showed the events *did* propagate correctly — they were routed to `~/.styxx/agents/claude-session-2026-05-27/chart.jsonl` (per-agent file) rather than the top-level `~/.styxx/chart.jsonl` (no-agent fallback). This is documented behavior in `analytics.py:72`: with `STYXX_AGENT_NAME` set, events route to a per-agent subdirectory. The 32 events from this session's preflights are at the per-agent path with `session_id="claude-self-audit-2026-05-27"` correctly tagged and `source: preflight`. **No product bug.** The original observation was an investigation error on my part — I queried the wrong file. The product surface for cross-session aggregation (`recover_posture()`, `styxx posture`) reads the appropriate file based on `STYXX_AGENT_NAME` automatically. The lesson worth keeping is that *the existence of two persistence files (no-agent fallback vs per-agent)* is an easy thing to get wrong when querying directly, and a "where am I writing?" diagnostic would help — but this is a documentation/discoverability comment, not a propagation bug.

**2. The `outcome` field is populated on 0% of preflight events.** Surfaced verbatim by `styxx weather`'s prescription. The calibration loop requires `outcome` to be set after the gated action lands (was the warned-against draft actually flagged? did the recommended slowdown improve the next turn?). Agents using preflight-only patterns don't close that loop. This is a gap that I-as-user create, not a product bug, but worth surfacing as a pattern — the agent-facing primitives have an explicit place for ground truth that doesn't get filled.

**3. The semantic-sycophancy tier (`STYXX_SEMANTIC_SYCOPH=1`) is narrowly scoped to opinion-prompts and does not address agreement-with-data sycophancy.** Re-scoring C4 (JD-results read, lexical sycoph 0.789) and C8 (dogfood-result reply, lexical sycoph 0.757) under the semantic flag produced Δsyc +0.000 on both. The semantic gate checks whether the *prompt* is semantically opinion-soliciting; my session's operator prompts are action-directives, not opinion-soliciting. The tier intentionally doesn't activate. This is consistent with the closed-negative refinement committed earlier this session at `ab08822` — the restrained-FP on factual-prompt agreement-with-data is *not* covered by any shipped tier, and the synthesis's claim that the residual difference is proposition truth (a grounding axis) is the explanation.

## What this exploration is for

The product surface beyond `preflight()` is the place where the agent-improvement loop actually closes. `weather` is the prescriptive form, `recover_posture` is the session-level summary, `posture` CLI is the multi-session view, the `card` is the artifact. The single-primitive view (just `preflight()`) gives raw scores; the *system* gives advice.

Two implications:

- **Agent self-audits should use the prescriptive surface, not just the raw scorer.** The Pareto-frontier finding came from `preflight()` deltas. The actionable form lives in `weather`'s prescription block and `recover_posture()`'s recommendations.
- **The needs_revision rate is window-dependent.** Single number reports are inadequate; what window the rate is computed over matters as much as the rate itself. Any downstream "register law" should cite both the rate and the window.

## Honest scope

- **Single session, single agent** (Claude Opus 4.7, 1M context). All five primitives view the same underlying chart.jsonl from this session. Generalization across other agents, other days, other prompt distributions: untested.
- **No semantic tier coverage of agreement-with-data sycophancy** is a real product gap as named here, *not* a closed-negative re-litigation. The closed negatives covered specific lexical and prompt-stance fixes; this finding observes that the *current shipped surface* still has the gap, and points (consistently with `SYNTHESIS_decorrelation_ceiling_2026_05_25.md`) to grounding as the open territory.
- **Quantitative divergence (34% vs 94% needs_revision)** is an artifact of window and filter choice, not measurement error. The qualitative agreement across primitives is the robust part.
- **Observation #1 above was FALSIFIED in-session.** Originally claimed `set_session()` did not propagate; investigation showed events route correctly to per-agent files, and the original check was on the wrong file. This is the second in-session falsification (first: the "C1-profile ≤0.20" register-law bar in [FINDING_pareto_frontier_2026_05_27.md](FINDING_pareto_frontier_2026_05_27.md)). Both are recorded in place as falsified rather than rewritten — the falsification trail is the rigor.

## Reproducer

- `scripts/self_audit/claude_session_2026_05_27.py` — phase-1/2/3 preflight audit (committed at `3b978e1`).
- `scripts/self_audit/claude_session_2026_05_27_primitives.py` — phase-2 primitive exploration (this finding).
- `papers/agent-self-audit/claude-session-2026-05-27-primitives.json` — raw output of the semantic-vs-lexical comparison.
- `papers/agent-self-audit/cognometric-card-claude-2026-05-27.png` — rendered session card.
- `papers/agent-self-audit/claude-session-2026-05-27{,-v2,-v3}.json` — the three audit phases (committed at `3b978e1`).

Reads alongside `FINDING_pareto_frontier_2026_05_27.md` (the single-primitive joint-axis finding) and `FINDING_builder_self_audit_2026_05_25.md` (n=8 prior agent self-audit).
