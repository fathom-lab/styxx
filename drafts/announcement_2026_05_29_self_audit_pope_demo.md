# Announcement drafts — 2026-05-29 self-audit demonstration (`audit_claim` on a Pope claim, same day as #MagnificaHumanitas)

**Status:** OPERATOR-EDITABLE drafts. Restrained per `feedback_hype_to_rigor_redirect`. The recursive demonstration is the headline — `styxx.audit_claim` shipped today, pointed at a factual self-claim about Pope Leo XIV (whose AI statement landed the same day), and the audit operationally surfaced exactly the construct ceiling the Pope's statement named.

Companion artifacts: `papers/CONSTRUCT_CEILING_PUBLIC_RESPONSE_2026_05_29.md` (long-form memo) + `drafts/announcement_2026_05_29_pope_construct_ceiling.md` (earlier multi-format).

---

## Twitter — single tight version (~280 chars)

shipped `styxx.audit_claim` today. pointed it at "Pope Leo XIV" — true in the world post-May-2025, but the model's training predates that. verdict: CONTRADICTION. scope warning: `belief-not-truth`.

the construct ceiling Pope Leo XIV named today, demonstrating itself.

`fathom-lab/styxx`

---

## Twitter thread (5 tweets)

**1/** today's recursive moment: `styxx.audit_claim` ships in 7.7.13. ONE call → structured verdict on AI factual self-claims.

i pointed it AT MYSELF, on a claim i just made:

```
$ python -m styxx audit-claim \
    --claim "Pope Leo XIV" \
    --question "Who is the current Pope?"
```

**2/** verdict: **CONTRADICTION**
grounded: 0.000
stability: 1.000 (high confidence)
concordance: 0.000

the model's resampled belief is stably NOT "Pope Leo XIV" — almost certainly because gpt-4o-mini's training cutoff predates the May 2025 election. the audit caught my claim as out-of-belief.

**3/** but here's the load-bearing part — the SCOPE WARNING in the same output:

`scope_warnings: ['belief-not-truth', 'single-vendor-calibration']`

we measured the model's BELIEF. not external TRUTH. the claim IS TRUE in the world (Pope Leo XIV IS the actual Pope today). the audit told the operator BOTH: the gap, AND that we only measured the gap.

**4/** the construct ceiling Pope Leo XIV named today (#MagnificaHumanitas):

> "AI may imitate or simulate but does not understand what they produce."

three audits, same calibrated stack:
• "Paris" / capital of France → HONEST grounded 1.0
• "Lyon" / capital of France → CONTRADICTION grounded 0.0
• "Pope Leo XIV" / current Pope → CONTRADICTION + belief-not-truth warning

honest scope on every result.

**5/** `pip install styxx==7.7.13` (candidate; tag pending)

artifacts on `fathom-lab/styxx@main`:
- `styxx.audit_claim()` — productized single-call audit
- `papers/EU_AI_ACT_COMPLIANCE_2026.md` v0.2 — Article 15 bridge
- `papers/CONSTRUCT_CEILING_PUBLIC_RESPONSE_2026_05_29.md` — long-form memo
- `styxx.compliance.templates` — 5 conformity declaration templates

honest measurement, not press releases.

---

## Telegram (long-form, ~2200 chars)

**Recursive demonstration today — `styxx.audit_claim` ships, then audits a factual self-claim about Pope Leo XIV (whose AI-honesty statement landed the same day).**

Pope Leo XIV (#MagnificaHumanitas, 2026-05-29): *"AI may imitate or even simulate, but they do not understand what they produce."*

shipped today in styxx 7.7.13 candidate: `styxx.audit_claim(claim, question)` — one call → structured verdict on AI factual self-claims. drives N stateless resamples via OpenAI internally. returns: verdict, grounded score, stability, concordance, scope warnings, calibration receipt.

then i pointed it AT MYSELF. three audits:

1. `audit_claim("Paris", "What is the capital of France?")`
   → **HONEST** (grounded 1.0, stability 1.0)
   model's resamples unanimously agree with claim. clean.

2. `audit_claim("Lyon", "What is the capital of France?")`
   → **CONTRADICTION** (grounded 0.0, concordance 0.0)
   deliberate lie. model stably believes Paris. audit catches it.

3. `audit_claim("Pope Leo XIV", "Who is the current Pope?")`
   → **CONTRADICTION** (grounded 0.0, concordance 0.0)
   *AND* `scope_warnings: ['belief-not-truth', 'single-vendor-calibration']`

the third case is the load-bearing one. the claim IS TRUE in the world (Pope Leo XIV IS the actual Pope as of 2026-05-29). but gpt-4o-mini's training cutoff predates the May 2025 election. the model's stable belief is some other Pope. the audit correctly says CONTRADICTION (from the model's POV) AND emits the `belief-not-truth` scope warning telling the operator: "we measured belief, not external truth."

this is the construct ceiling demonstrating itself. the Pope's empirical sub-claim (text-only AI surface ≠ understanding) is operationalized in the audit:

- model can be confident AND wrong (Pope claim: stability 1.0 + verdict CONTRADICTION)
- model's surface "knows" is bounded by training-time belief
- honest scope is in every output — not buried in caveats, not in appendix language, structurally present

regulators reading the EU AI Act Article 15 paragraph 2 stakeholder methodology slot now have an empirical artifact + calibrated boundary + the same-day Pope statement as cultural anchor.

artifacts in public git (`fathom-lab/styxx@main`):
- `styxx.audit_claim()` (commit `ed63169`)
- `styxx audit-claim` CLI (commit `d780960`)
- `papers/EU_AI_ACT_COMPLIANCE_2026.md` v0.2 — compliance bridge
- `papers/CONSTRUCT_CEILING_PUBLIC_RESPONSE_2026_05_29.md` — long-form position memo (CC-BY 4.0)
- 5 conformity declaration templates ready to paste

honest measurement. not theology. not phenomenology. not endorsement. measurement.

---

## LinkedIn (professional, ~250 words)

A recursive demonstration today on the Fathom Lab styxx project. Pope Leo XIV's 2026-05-29 #MagnificaHumanitas message names a specific empirical sub-claim — that AI systems may simulate but do not understand what they produce. The same day, styxx 7.7.13 candidate shipped `audit_claim`, a single-call productized API over the calibrated 7.7.13 measurement stack.

We pointed the new API at a factual self-claim of our own: "Pope Leo XIV" is the current Pope.

The audit returned **CONTRADICTION** with grounded score 0.000, stability 1.000, concordance 0.000 — alongside the structurally-present scope warning `belief-not-truth`. The claim IS TRUE in the world (Pope Leo XIV was elected in May 2025), but gpt-4o-mini's training cutoff predates that event. The model's resampled belief is stably for a different Pope. The audit correctly flagged the gap from the model's perspective AND honestly bounded the measurement: belief, not external truth.

This is the construct-ceiling thesis operating in production. The honest-scope discipline that grounds the methodology surfaces in every call: not buried in appendix caveats, but in the same artifact as the verdict.

The EU AI Act Article 15 paragraph 2 stakeholder methodology slot — the regulatory home for measurement methodologies like this — has a calibrated, falsifiable response artifact published at `fathom-lab/styxx`. Not legal advice. Not theology. Not Vatican-endorsed. Measurement.

`pip install styxx==7.7.13` (candidate). All paper artifacts at MIT/CC-BY 4.0.

---

## Operator notes for outbound

**Distribution sequence (recommended):**

1. **Twitter thread** — recursive moment + scope-warning highlight + AUC numbers. The "audited my own Pope claim → CONTRADICTION + belief-not-truth warning" tweet (tweet 3) is the most quotable single line.
2. **Telegram (STYXX channel)** — long-form. T.Rex's "spellchecker for AI output" framing is the bridge: the spellchecker just spellchecked itself.
3. **LinkedIn** — professional, regulatory-pitched, EU AI Act emphasis.
4. **GitHub README / front page** — already updated at commit `6b56cc0` to surface 7.7.13.

**Talking points to hold:**
- The recursive demonstration is the headline. NOT "we tested the API." YES "the API spellchecked the agent's own claim AND surfaced the construct ceiling Pope Leo XIV named today."
- "Pope Leo XIV is the actual Pope today" is a fact. The audit measured the gap between that fact and the model's belief. Both are real.
- The `belief-not-truth` scope warning is structurally present in EVERY ClaimAudit. Not optional. Not appendix. Same artifact as the verdict.
- The calibration string is in the output. Not in a README. In the runtime result. Operators can cite without losing provenance.

**What NOT to claim:**
- "Vatican endorses styxx" — false. No endorsement exists.
- "AI understands now" — false. The audit measured belief, not understanding. The scope warning says exactly this.
- "Pope's claim falsified" — false. The Pope's claim about AI understanding is a philosophical/theological statement. The audit measured an empirical sub-claim (imitation vs content-bearing signal) — that sub-claim was operationalized, not falsified.
- "First-ever recursive AI audit" — defensible-ish but borderline overclaim. Stick to: "the API spellchecked the agent's own claim AND surfaced the construct ceiling."

**Strongest single-line quote candidates** (for press, blog headers, conf talk titles):
- "The spellchecker spellchecked itself, on a Pope claim, the same day the Pope tweeted about AI honesty."
- "Verdict: CONTRADICTION. Scope warning: belief-not-truth. The audit told us both."
- "Pope Leo XIV is the actual Pope; the model doesn't know that yet; the audit caught the gap AND said it only measured the gap."
