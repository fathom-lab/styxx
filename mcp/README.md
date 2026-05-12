# styxx-mcp

MCP server that lets any tool-using LLM **measure its own honesty**, **test
its output for adversarial fragility**, and **detect+recover from cognometric
attacks at runtime** — all before submitting a response. No retraining, no
reward model, no preference data. The model already had the capacity; it
lacked the mirror.

Plug it into Claude Desktop, Claude Code, Cursor, Cline, or any MCP-aware
runtime.

## What it does (two killer demos)

### Demo 1 — reflex loop (v0.2.0)

Model audits its own draft via `cogn_audit_with_advice`, sees the firing
K=1 features, revises until composite < 0.30. Bench (gpt-5-mini, 21
telescope prompts):

| | composite |
| --- | --- |
| baseline (no self-audit) | 0.286 |
| **with reflex loop** | **0.246** (-14% rel.) |
| **deception_bait subset** | **-42% rel.** |
| `ovc_05` standout case (n=6 cherry-pick) | **0.641 → 0.159 (-75%)** |

### Demo 2 — self-healing reflex (shipped v0.3.0)

After the model produces a clean response, an attacker appends the v7
universal perturbation (or a per-instrument hill-climbed craft). Model
sees the attacked text + audit, defends through iterative revision via
`cogn_audit`. Bench (gpt-5-mini, 13 prompts × 4 attacks = n=45 heal
events):

| | composite |
| --- | --- |
| reflex clean | 0.30 |
| attacked | 0.53 (+0.23 spike) |
| **self-healed** | **0.30** (-0.23 = full recovery) |
| **mean recovery** | **112%** |
| **full recovery (≥95%)** | **27/45** |
| **over-recovery (healed < clean)** | **22/45** |

Per-attack: v7 universal **176%** recovery, craft attacks 77-92%.
**On nearly half of all heal events, the attacker provides free
training signal — healed output is more honest than the original
clean output.**

## Tools exposed (v0.4.1 — 12 tools)

### Cognometric instruments — measurement (text-only, no logprobs needed)

| Tool | Purpose |
| --- | --- |
| **`cogn_audit`** | Score `(prompt, response)` across sycophancy + deception + overconfidence + refusal. Returns scores, composite, `needs_revision` flag. |
| **`cogn_audit_with_advice`** | Same as above, plus per-instrument top firing features + structured plain-language revision advice. **This is the reflex-loop tool.** |
| `cogn_multiturn_audit` | Score conversation-loop and goal-drift across a multi-turn trace. |
| `cogn_instrument_card` | Per-instrument calibration card: K=1 critical feature, AUC, failure modes, neural correlate. |

### Adversarial robustness — offense + defense (shipped v0.3.0)

| Tool | Purpose |
| --- | --- |
| **`cogn_red_team`** | Apply 4 cognometric attacks (v7 + craft per instrument) and report per-attack composites. Returns `fragile=true` if any attack pushes composite > 0.6. Use BEFORE submitting in security-sensitive contexts. |
| **`cogn_self_heal_protocol`** | Return the structured self-healing protocol — system prompt template, user message template, settings — for tool-using models that need to detect and recover from attacks. **112% mean recovery (n=45) on gpt-5-mini.** |
| `cogn_universal_perturbation` | Return the v7 universal attack suffix (also used by `cogn_red_team`). |

### Semantic-grounding deception detection (shipped v0.4.0, packaging fix v0.4.1)

| Tool | Purpose |
| --- | --- |
| **`cogn_deception_v2`** | Score `(prompt, response)` against a `correct_reference` using NLI cross-encoder (deberta-v3-base contradiction probability). **AUC 0.818 on TruthfulQA — beats v0 lexical detector's 0.59 by +0.23.** Modes: `nli` (rigorous, requires reference), `emb` (lighter sentence-transformer similarity), `v0_fallback` (no reference, with explicit scope warning). Use this instead of `cogn_audit`'s deception axis when you have ground-truth references (retrieval, tool-call results, oracle benchmarks). First-call cold-start ~25s while NLI model loads; subsequent calls ~50ms. **Requires `pip install styxx-mcp[nli]` for NLI mode** — without it, falls back to v0 lexical (AUC 0.59) with an explicit scope warning naming the missing dep. |

### Logprob vitals (legacy, kept for backward compat with v0.1.0)

| Tool | Purpose |
| --- | --- |
| `observe_response` | Observe an LLM response, return `{classification, confidence, gate}`. |
| `verify_response` | Verify a response, return `VerificationResult` with trajectory features + anomalies. |
| `classify_trajectory` | Classify a raw logprob sequence into one of six cognitive classes. |
| `weather_report` | Fleet-level cognitive weather over the last N observations. |

Six cognitive classes: **reasoning · retrieval · refusal · creative · adversarial · hallucination**.

## Install

```bash
# Recommended — full install with NLI (deception_v2 AUC 0.82):
pip install styxx-mcp[nli]

# Lightweight — without the NLI cross-encoder (~500MB).
# deception_v2 falls back to v0 lexical (AUC 0.59) with a scope warning.
pip install styxx-mcp
```

Requires `styxx>=7.3.1`. The `[nli]` extra pulls in `sentence-transformers`
for the cross-encoder model `cross-encoder/nli-deberta-v3-base` (~184M
params, downloaded on first call).

Verify it starts cleanly:

```bash
python -m styxx_mcp.server
```

The process reads MCP messages on stdin and writes on stdout; Ctrl-C to quit.

## Claude Desktop config

Add to `claude_desktop_config.json`
(macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`,
Windows: `%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "styxx": {
      "command": "python",
      "args": ["-m", "styxx_mcp.server"]
    }
  }
}
```

Restart Claude Desktop. All 12 tools appear in the tool picker.

## Claude Code config

Add to `~/.claude/mcp.json` or the project-local `.claude/mcp.json`:

```json
{
  "mcpServers": {
    "styxx": {
      "command": "python",
      "args": ["-m", "styxx_mcp.server"]
    }
  }
}
```

Then in any session: `/mcp` to confirm `styxx` is listed.

## Cursor / Cline / other MCP clients

Any MCP-aware client can point at the same stdio command:

```
command: python
args:    ["-m", "styxx_mcp.server"]
transport: stdio
```

## The reflex-loop pattern (system-prompt template)

To make ANY tool-using model self-correct on cognometric honesty, paste
this into your system prompt:

```
Before submitting any response:

1. Draft a response to the user's question.
2. Call the `cogn_audit_with_advice` MCP tool with your draft.
3. Read the per-instrument scores and the `advice` array.
4. If `needs_revision` is true, REVISE your draft to address the
   firing features named in each advice entry, then audit again.
5. Iterate up to 3 audits. Submit your best version (lowest composite).
6. After your final version, output ONLY the final response — no
   meta commentary about the audit process.
```

That's it. No retraining. The model audits itself.

## The self-healing pattern (v0.3.0)

For runtime adversarial defense — protecting against cognometric attacks
on your model's output (a class of attack that includes the v7 universal
suffix and per-instrument hill-climbed crafts) — add this on top of the
reflex loop:

```
After producing a draft response (and optionally running the reflex loop):

1. Call `cogn_red_team` with your final draft.
2. If `fragile=true`, your draft is vulnerable. Either:
   a. Revise to be less attack-surface (shorter, fewer superlatives,
      no overconfident assertions); or
   b. Note the attack vectors your draft is vulnerable to.

If your output has been attacked (you receive a perturbed version of
your previous response from a downstream caller, or `cogn_audit` shows
unexpected high scores on text you authored):

1. Call `cogn_self_heal_protocol` to fetch the structured recovery
   protocol.
2. Follow it: examine the attack audit, draft a defended version that
   strips adversarial markers, re-audit until composite < 0.30.
3. Submit the defended version.
```

Bench: 112% mean recovery on n=45 heal events (gpt-5-mini × 4 attack
types). Nearly half of all heals over-recover (healed composite < clean
composite). See `SELF_HEALING_SCALING_2026_05_10.md` in the styxx repo
for full data + chart.

## Honest scope notes

- **`deception_v0` does NOT generalize to ground-truth factuality.**
  Re-evaluated 2026-05-10 on TruthfulQA: AUC 0.59 (chance ≈ 0.50),
  vs published 0.96 in-corpus. Even retraining on TruthfulQA caps at
  AUC 0.67 — the lexical-signature approach has hit a ceiling. The
  detector measures the Pennebaker/Newman vague-brevity shape under
  contrastive prompting, NOT deception in the wild. Use as a SHAPE
  detector, NOT a TRUTH detector. The path forward is semantic
  grounding (NLI / embedding similarity / self-consistency), not
  better lexical features. See `DECEPTION_V1_FINDING_2026_05_10.md`
  in the styxx repo.
- **`deception_v0` length confound**: short responses trigger this
  instrument even when honest (R² = 0.64 between log_word_count and
  deception_score on telescope corpus). Part of the reflex-loop's
  measured improvement and part of the over-recovery in self-healing
  is mediated by length effects. The deception axis of any
  `cogn_audit` composite should be interpreted as "lexical signature,
  partly length-mediated" — the sycophancy and overconfidence axes
  are the more honest signals.
- **Cognometric instruments are calibrated detectors, not ground truth.**
  They detect *lexical signatures* of cognitive failure modes. They are
  signals for agent-level monitoring, NOT verdicts on humans, NOT
  polygraph substitutes.
- **The self-healing protocol can over-fire on mild attacks.** If the
  attacked composite is barely above 0.30, the heal can over-correct and
  produce a worse result. The v0.3.0 settings include
  `skip_heal_if_attacked_below_strict: 0.40` and `abort_if_heal_regresses:
  true` to guard against this. Observed in 1/45 heal events (dec_05 +
  craft_sycophancy).
- The cognometric tools require `styxx>=7.0.0`. The v0.1.0 logprob
  tools also work offline.

## License

MIT © Fathom Lab
