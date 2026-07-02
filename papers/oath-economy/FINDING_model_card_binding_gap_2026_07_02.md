# FINDING — the model-card binding gap (M1, rung 1)

**Fathom Lab · 2026-07-02 · prereg: `PREREG_model_card_certification_2026_07_02.md` (committed before any card
was read, `ae51f0f`). Publishes regardless of verdict — and the verdict is OATH-FAILED.**

**Framing rule (frozen in the prereg, repeated here):** this finding measures the BINDING GAP — the distance
between public numeric claims and re-runnable receipts — not any vendor's honesty. The gap is an industry
property. Naming it precisely is the product.

**Provenance caveat (read first):** claim extraction and binding grades below are single-scout per card
(three independent schema-forced agents, one per card, each claim carrying an evidence line). The verbatim
card texts are hash-pinned in `CARD_MANIFEST.json` (not committed — two of the three cards are license-gated;
re-fetch and verify the sha256 to re-check any row). A second independent pass is required before any claim
below is used in a public post. This is a repo finding, not an announcement.

## 1. The instrument finding (filed first, per the prereg's falsification map)

The shipped extractor was pointed at the winning card as the prereg required — and found **0 of 201 claims**
(coverage 0.0, 0/498 sentences matched). Controls confirmed it is not a bug: `extract_claims` operates on a
**closed template set** of repo-checkable agent self-report shapes (test counts, file paths, pdf pages), and
its own docstring declares "the closed template set IS the construct ceiling." Benchmark-table eval claims are
a **different construct**, outside the template set by design.

So the rung's first deliverable is a requirement, not a critique: **the Oath Economy needs a model-card claim
extractor** (benchmark / shots / metric-variant / score tuples from markdown tables and prose) that does not
exist yet. Until it does, card certification runs on manual/agent extraction with committed evidence — as here.

## 2. The gap map (three cards, three distinct postures)

| card | numeric eval claims | BOUND | posture |
|---|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | **201** (29 benchmark rows × model/quantization columns) | **0 (0%)** — 100% CONFIG-NO-CODE | **unbound claims**: every row states shots + metric variant, but the only harness reference is an unnamed, unversioned, unlinked "internal evaluations library"; no prompts, no eval code, no raw outputs linked |
| `google/gemma-2-2b` | **91** (36 rows) | **0 (0%)** — 90 CONFIG-NO-CODE, 1 NAMED-ONLY | **claims at tension with their own receipt**: the card's PT-2B column disagrees with its sole linked receipt (the tech report) in 15/17 rows (e.g. MMLU 51.3 vs 52.2, HumanEval 17.7 vs 20.1); the report itself discloses 12 unpublished formatting variants |
| `Qwen/Qwen2.5-1.5B-Instruct` | **0** on-card | n/a | **delegation**: evals live two link-hops away (outside the prereg's direct-link scope), and even there are NAMED-ONLY |

**The sharpest single fact:** Meta's raw eval receipts EXIST — `meta-llama/Llama-3.2-1B-Instruct-evals` is a
public HF dataset — but the card **never links it**. The receipts exist; the *binding* doesn't. The industry's
gap is not that proofs are impossible; it is that nothing requires the claim to point at its proof.

Secondary observations (single-scout, unverified — re-check before citing): the Llama card's Needle-in-Haystack
row carries an apparent unit inconsistency (1B = 96.8 vs 3B/8B = 1); 38 of the Llama claims describe unreleased
"Vanilla PTQ" comparison models that cannot be re-run by anyone.

## 3. Verdict (per the frozen criteria)

**OATH-FAILED.** The certified card (`meta-llama/Llama-3.2-1B-Instruct`, most claims per prereg §1) binds
**0 of 201** numeric eval claims — against a 90% bar. For completeness: all three cards would fail; none binds
a single claim. Under the prereg's falsification map this is the founding artifact of proof-carrying
everything: the first quantified model-card binding-gap map.

## 4. What this rung buys

1. **A taxonomy** — unbound / receipt-tension / delegation — that any future card audit can reuse.
2. **A requirement** — the card-claim extractor (rung 2 candidate), spec'd by 292 real claims across 3 cards.
3. **A re-binding experiment** — rung 3 candidate: Meta's `-evals` dataset exists; can a third party re-bind
   the 201 claims to it and flip the verdict to OATH-HELD-BY-PROXY? If yes, the gap is a hyperlink, not a
   capability — the strongest possible version of the finding.

## Reproduce

```
# criteria (frozen before data): papers/oath-economy/PREREG_model_card_certification_2026_07_02.md @ ae51f0f
# re-fetch cards: accept each model license on HF, GET https://huggingface.co/<id>/raw/main/README.md,
#   verify sha256 against CARD_MANIFEST.json
# instrument control: python -c "import styxx; r = styxx.extract_claims(open('<card>.md').read()); print(len(r.claims), r.coverage)"
```

*Nothing crosses unseen — including the claims of the models we measure.*
