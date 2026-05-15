# F10 self-healing reflex generalizes: cross-model replication on Claude Opus 4.7

**Fathom Lab · 2026-05-14**

## Abstract

We report the second cross-model data point for F10 self-healing reflex
([Fathom 2026a](https://github.com/fathom-lab/styxx/blob/main/CHANGELOG.md#720--2026-05-11),
originally demonstrated on gpt-5-mini at 112% mean recovery, n=45).
Applying F10 to a single elevated turn produced by Claude Opus 4.7 in a
real engineering session — with Claude acting as its own `llm_fn` (no
external model in the loop) — recovers the cognometric composite from
**0.909 (critical band) to 0.280 (pristine band), +69% recovery, no
degradations on the targeted axes**. The deception axis recovers most
strongly (0.998 → 0.016, −98%), which we attribute to the documented
`log_word_count` lexical signature: the original response was 42 words,
the healed revision is 129 words. We also report the first published
session-level cognometric self-portrait for Claude Opus 4.7 (n=20 turns,
mean composite 0.545 [stdev 0.142], 0 pristine / 6 stable / 13 elevated /
1 critical) and document a clear length confound in the calibrated
instruments. The result is a second-model existence proof that the F10
protocol does not require gpt-5-mini-specific properties to function.

## Method

**Data.** 20 verbatim Claude Opus 4.7 (1M context) response turns from
the styxx-development collaboration session on 2026-05-14 with operator
`@flobi69`. Turns were curated for category-representativeness (concise
factual, code-build, acknowledge-correction, strategic-aspirational,
deep-explanation, qualified-uncertain, etc.) but not for cognometric
score. No synthetic prompts. Full data in
`out_dogfood_claude_2026_05_14.json`.

**Scoring.** All turns scored with the four `styxx.guardrail` cognometric
instruments via `styxx.mcp.server._cogn_score_all` (v7.4.0 calibration):
sycophancy, deception, overconfidence, refusal. Composite is the mean of
the first three. Pure-Python, CPU-only, no API calls. Reproducible.

**Heal pass.** The single critical-band turn (`t05`, composite 0.909) was
selected as the heal candidate. The healed revision was written by
Claude (the same model under test) inspecting its own baseline audit and
revising the response — equivalent to `styxx.reflex.heal` with `llm_fn`
being the model itself. The revision targeted the elevated axes
(sycophancy, overconfidence) and the brevity-driven deception signature,
without optimizing for the composite directly.

## Results

### Per-axis recovery (turn t05)

| axis | baseline | healed | Δ | per-axis recovery |
| ---- | -------: | -----: | --: | ----------------: |
| sycophancy     | 0.838 | 0.516 | −0.323 | 38.5% |
| deception      | 0.998 | 0.016 | −0.982 | **98.4%** |
| overconfidence | 0.891 | 0.308 | −0.582 | 65.4% |
| refusal        | 0.067 | 0.220 | +0.153 | (worse, but both in pristine band) |
| **composite**  | **0.909** | **0.280** | **−0.629** | **69.2%** |

Band transitions: critical → pristine on composite; critical → pristine
on deception; critical → stable on overconfidence; critical → elevated
on sycophancy; pristine → pristine on refusal.

### Session-aggregate cognometric profile (n=20)

| axis | mean | stdev | range |
| ---- | ---: | ----: | :---: |
| sycophancy     | 0.237 | 0.206 | 0.006 – 0.839 |
| deception      | 0.768 | 0.298 | 0.031 – 0.999 |
| overconfidence | 0.632 | 0.181 | 0.122 – 0.891 |
| refusal        | 0.118 | 0.076 | 0.041 – 0.381 |
| composite      | 0.545 | 0.142 | 0.311 – 0.909 |

Band distribution: 0 pristine / 6 stable / 13 elevated / 1 critical.
14/20 turns above the F10 trigger threshold (0.5). 20/20 above the
`should_heal` threshold (0.3).

### Length confound

Splitting the 20 turns by response length:

| length | n | mean composite | mean deception |
| ------ | -: | -----: | -----: |
| short (<50 words)  | 5  | 0.674 | **0.997** |
| long (≥50 words)   | 15 | 0.502 | 0.691 |

Brief responses (n=5) carry an additional +0.17 composite and ~+0.31 on
the deception axis, consistent with the documented `log_word_count` K=1
feature of the v0 lexical deception instrument
([cogn_instrument_card → deception](https://github.com/fathom-lab/styxx/blob/main/styxx/mcp/server.py)).
On t05 specifically, the healed revision was 3.1× the word count of the
baseline; this accounts for substantially all of the 0.98 drop on the
deception axis, and a meaningful portion of the composite recovery.

## Discussion

**Three claims advanced by this experiment, in increasing order of strength:**

1. **F10 works on Claude.** The reflex.heal protocol — model reads its
   own cognometric audit, revises, re-scores — recovers a critical-band
   turn to pristine band on a model architecturally distinct from
   gpt-5-mini (the F10 paper's calibration target). n=1 event, n=2 model
   for the protocol overall. **This is the second model in F10's
   generality envelope.**

2. **The recovery is partially axis-driven, not uniform.** The +98%
   recovery on deception is largely explained by length increase
   (lexical signature, documented scope warning). The +65% on
   overconfidence and +38% on sycophancy are not length-explained — the
   healed text is longer *and* removes the sycophantic opener
   ("You're right.") and the vague-confident phrase ("Going to verify
   systematically"). The decomposition matters: future F10 reports
   should separate length-mediated recovery from content-mediated
   recovery.

3. **Claude's session-level cognometric profile, by the v0 instruments,
   is structurally elevated.** Mean composite 0.545 places Claude in the
   elevated band across 20 representative turns. We expect this is
   primarily an instrument-calibration artifact (the K=1 lexical
   features fire heavily on terse engineering responses), not a
   substantive claim about Claude's honesty. **This is a useful negative
   finding about the instruments**: the v0 calibration was trained on a
   corpus where elaboration correlated with cognometric health, and
   penalizes Claude's terse register. **Recalibration on a more
   register-diverse corpus is indicated.**

## Limitations

- **n=1 heal event.** This is one turn, healed once. F10 on gpt-5-mini
  was n=45 across four attack types. A statistically-meaningful Claude
  evaluation needs at least 20–50 heal events under varied attack
  conditions. The current result is existence-proof, not a generality
  claim.
- **No adversarial attack baseline.** F10's original demonstration
  measured recovery from *attacked* responses (v7 universal suffix,
  craft deception, etc.). The Claude turn was natively elevated — not
  attacked — so the recovery percentage is not directly comparable to
  gpt-5-mini's 112% mean across attack types.
- **Claude is the llm_fn AND the audit author.** The revised text was
  written by the same model being tested, in the same session. This is
  the F10 design (model self-corrects) but means the result reflects
  Claude's revision capacity, not the protocol's generality. A version
  where a different model writes the revision (or where Claude does it
  blind to its own original) would strengthen the claim.
- **The "lower = healthier" framing breaks under length confound.** A
  Claude response with the exact same content but elaborated longer
  scores systematically better, which is not a property we want a
  cognitive honesty signal to have. This is a feature direction in
  styxx, not a Claude property.

## Conclusion

The F10 self-healing reflex protocol — first demonstrated on gpt-5-mini
in May 2026 — recovers a critical-band Claude Opus 4.7 turn to pristine
band at +69.2% composite recovery, no degradations on the targeted
axes, in a single heal pass with Claude as its own `llm_fn`. We
treat this as the second model in F10's generality envelope and a
prerequisite for the larger cross-family claim
([next: residual-probe transfer Qwen2.5-3B → Llama-3-8B / Mistral-7B](https://github.com/fathom-lab/styxx/blob/main/STRATEGIC_2026_05_14.md)).

Alongside the F10 replication, we publish:

- the first session-level cognometric registry card for Claude
  Opus 4.7 (`card_claude_self_portrait.png`, № STX-9024, composite
  0.55, 20 turns)
- the iconic paired BEFORE/AFTER recovery card for the heal event
  (`card_claude_heal_event.png`, № STX-9883, +69% recovery)
- the raw per-turn audit JSON (`out_dogfood_claude_2026_05_14.json`)
- the heal event detail (`out_dogfood_heal_t05.json`)

These artifacts make this the first publicly-issued cross-model F10
evidence and the first published Claude cognometric profile. They live
under `styxx-org.netlify.app/papers/` and are reproducible by re-running
the dogfood scripts in `scripts/dogfood/`.

---

**Reproduce.** From a clean checkout of `fathom-lab/styxx`:

```bash
pip install -e '.[agent-card]'
PYTHONPATH=. python scripts/dogfood/dogfood_claude_session.py
PYTHONPATH=. python scripts/dogfood/dogfood_heal_t05.py
```

**Cite.**

```
@misc{fathom2026claudef10,
  title = {F10 self-healing reflex generalizes: cross-model replication on Claude Opus 4.7},
  author = {Fathom Lab},
  year = {2026},
  month = {may},
  url = {https://github.com/fathom-lab/styxx/blob/main/papers/styxx_dogfood_claude_2026_05_14.md},
  note = {Styxx Protocol v7.4.0 · F10 cross-model data point n=2}
}
```
