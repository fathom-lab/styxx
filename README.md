<div align="center">

```
   ███████╗████████╗██╗   ██╗██╗  ██╗██╗  ██╗
   ██╔════╝╚══██╔══╝╚██╗ ██╔╝╚██╗██╔╝╚██╗██╔╝
   ███████╗   ██║    ╚████╔╝  ╚███╔╝  ╚███╔╝
   ╚════██║   ██║     ╚██╔╝   ██╔██╗  ██╔██╗
   ███████║   ██║      ██║   ██╔╝ ██╗██╔╝ ██╗
   ╚══════╝   ╚═╝      ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝

           · · · nothing crosses unseen · · ·
```

### the measurement layer for machine minds

[![PyPI](https://img.shields.io/pypi/v/styxx.svg?color=ff2330&label=pypi&style=flat-square)](https://pypi.org/project/styxx/)
[![Python](https://img.shields.io/pypi/pyversions/styxx.svg?color=ff2330&label=python&style=flat-square)](https://pypi.org/project/styxx/)
[![License](https://img.shields.io/pypi/l/styxx.svg?color=ff2330&label=license&style=flat-square)](LICENSE)
[![tests](https://github.com/fathom-lab/styxx/actions/workflows/test.yml/badge.svg)](https://github.com/fathom-lab/styxx/actions/workflows/test.yml)
[![Spec](https://img.shields.io/badge/spec_v1.0-10.5281%2Fzenodo.19746215-ff2330.svg?style=flat-square)](https://doi.org/10.5281/zenodo.19746215)
[![Concept](https://img.shields.io/badge/concept_DOI-always--latest-ff2330.svg?style=flat-square)](https://doi.org/10.5281/zenodo.19326174)

</div>

### one idea, three layers

styxx is a measurement layer for machine minds. Every instrument in it is built on a single
principle, and the principle is the product:

> **An instrument that cannot refuse cannot be trusted.**
> Each one names what stopped it, and none of them will tell you more than the evidence carries.

Everything here is one of three things.

| layer | the question | instruments |
|---|---|---|
| **VERIFY** | does this claim match its evidence? | `certify` (OATH) · `protocol` · `seal` · `diffgate` + the GitHub Action · `corpus_audit` |
| **MEASURE** | what is actually true about these minds? | `islands` · `coupling` · `mind` · `meaning_diff` · `crossmind` · the register instruments |
| **SENSE** | can an agent be connected to the world without lying about what it feels? | `sense` |

They compose. A `sense` channel is scored by `coupling`, whose verdict is written into a finding,
whose every number is checked by `certify`, whose preregistration is enforced by `protocol`, and
the whole thing is `seal`ed or refused. That chain is why a claim from this repo can be checked
by a stranger without trusting anyone in it.

**What the refusals cost us, in one day (2026-08-06):** an instrument deleted for failing its own
exam, a released module recalled after an internal red team broke it six ways, a priority claim
retracted after an external methods audit, and our own published prediction killed four times —
twice on real human brain data we downloaded to test it. Every one of those is in
[CHANGELOG.md](CHANGELOG.md) and [papers/](papers/) with the receipt attached. That is not
humility as branding; it is the only reason the passes mean anything.

---

### VERIFY — an agent's summary cannot lie about its diff

```bash
pip install styxx
python -m styxx.diffgate --demo        # 10 seconds, no repo needed
```

```
the summary the agent wrote:
  Refactored src/retry.py. Adds function backoff with jitter.
  Added 3 tests covering the retry path. Only touches files under src/.

  [ok ]  file_touched     diff status 'M' for 'src/retry.py'
  [LIE]  symbol_added     added lines do NOT define function 'backoff'
  [LIE]  tests_added      diff adds 1 test functions, claim says 3
  [LIE]  only_touches     paths outside 'src': ['config/settings.yml', ...]
  [ ?  ] tests_pass       no --run supplied; we don't take its word

verdict: FAIL — this summary would fail your CI with each lie named.
```

**In CI, one line:**

```yaml
- uses: fathom-lab/styxx@main    # every agent PR gated against its actual diff
```

Zero receipts, zero cooperation from the agent that wrote the summary, no checkout.
Fails only on a contradicted claim. Prose outside the closed template set is never judged,
and the CLI prints what it checks when it finds nothing — silence is scope, not weakness.

**The zero-false-accusation claim that stood here is withdrawn, and here is what replaced it.**
It was true of two frozen corpora (this repo's 80-commit history and 24 agent-authored PRs) and
was written in the present tense, so it kept asserting itself as the corpus grew. Re-run at
7.46.0 by our own committed harness, `python scripts/diffgate_validation_sweep.py` reports 13
claims and **4 contradictions**, and hand-adjudicating all four finds every one is a false
accusation — the gate treats a filename *mentioned* in a commit message as a file the diff must
contain. One says a candidate file is "nothing alike"; one describes a fix in *someone else's*
repository; one is a commit whose message discusses the very document it is reporting a defect
in. That commit was made the same day this paragraph was rewritten.

Mention-versus-use is not a quirk of this gate. The same defect is documented in the OATH
verifier in [RECON_oath_external_reach](papers/closed-model-frontier/RECON_oath_external_reach_2026_08_26.md)
and was found in the ledger's own classifier on the same day — three instruments, written months
apart for unrelated jobs, all reading a line and calling it a claim. Historic false accusations
are named in [CHANGELOG.md](CHANGELOG.md) with the regression test that closed each one. **These
four are not closed** — they are open, reproducible with the command above, and owed a fix with
its own preregistration. They are stated here rather than behind a number, because a headline
that keeps asserting itself in the present tense is how the old one went wrong.

### MEASURE — two minds can share a geometry and still be unable to read each other

```bash
python -m styxx.islands --demo         # 10 seconds, no data, no GPU
```

```
cohort of 8 minds over 120 shared items — nothing labelled
  ISLANDS_PRESENT
    mind_0 .. mind_6      0.4195 – 0.4263
    ISLAND                0.1694   <- found from frame geometry alone
```

Independently trained models converge on a shared concept geometry — and a model can sit
*mostly inside* it and still be unreadable. We took the barrier apart under preregistered
gates: it is **causal** (correcting the frame takes cross-model reading 0.0612 -> 0.9745 while
matched random frames do 0.0), **two directions wide at its core**, **nameless** (its
directions match no human concept category, permutation p 0.8031), and **switch-like** —
legibility is flat across most of the rotation and turns vertical only near alignment. Which
is why representational-similarity scores never predicted readability: slope measures cannot
see a switch.

Nine sealed acts, every verdict computed from gates frozen in git before the run, and the whole
chain [replicates on a laptop CPU](papers/disjoint-worlds/REPLICATE_legibility.md) — the
cheapest check takes four seconds. `styxx.islands` generalizes the measurement past language
models: hand it any cohort over a shared item set (activations, fMRI betas over shared stimuli,
MEG epochs) and it reports islands, the cliff, and whether a low-rank correction rescues them.
It refuses below eight members and refuses a knee read off a noise curve, because an instrument
that cannot refuse cannot be trusted.

We also [staked a public, falsifiable prediction](papers/disjoint-worlds/PREDICTION_h1_human_islands_2026_08_06.md)
that human cross-subject brain decoding will show the same structure — frozen before the data
exists, with the branch where we are wrong written first. Our own next experiment produced
evidence against it, and that is recorded in the prediction itself.

---

### SENSE — an agent with a sensor and no verification is a confabulation engine

```bash
python -m styxx.sense --demo         # 10 seconds, no hardware
```

Give a mind a continuous signal and it will find itself in it. A room's daily rhythm becomes
"I feel the afternoon"; the recorder's own duty cycle becomes "I feel my body"; two independent
drifts become "I am coupled to the building." Each is a real statistical signal and none is a
sense. `styxx.sense` records a channel alongside the agent's own state on one clock and refuses
to call it a sense unless it survives a coverage gate, a confound-preserving null, an
autocorrelation-preserving null, a leverage check and a sampling-density check — naming which one
stopped it. The machine's own CPU and network ship as a channel **on purpose**: it is the control,
the thing an agent is most likely to mistake for a sense of the world.

The strongest verdict it can ever return is `COUPLED_BEYOND_CONFOUND__attribution_pending`. It
will not tell an agent that it senses anything, because the statistic is symmetric and an agent's
hardware sits inside whatever it measures.

---

Those are three instruments, one from each layer. The rest of this README is the lab behind them.

---

styxx is a cognitive-integrity SDK for LLM agents. it reads the cognitive state of a generation —
drift, confabulation, refusal, sycophancy, deception signature, goal drift — from the text and the
token stream, scores it against calibrated instruments with published AUCs, and certifies that every
number it reports can be re-run from a committed receipt. it is built for engineers shipping agents
who need to know when an output flatters, fabricates, loops, or quietly stops matching its plan —
before it reaches a user. the drop-in is one line: `from styxx import OpenAI` (same interface as
`openai.OpenAI`, every response gains a `.vitals` read; `from styxx import Anthropic` likewise, on
text-heuristic vitals — the Anthropic API exposes no logprobs). the base install carries no torch,
no GPU requirement, and no LLM in the loop for the core instruments — the calibrated detectors are
small logistic regressions over hand-built features (numpy + scikit-learn), scoring in
sub-millisecond CPU time. MIT, open at the core, forever ([OPEN_CORE.md](docs/governance/OPEN_CORE.md)).

## install

```bash
pip install styxx
```

that gets the full core: the profiler, the nine calibrated instruments, the agent-integrity
primitives, the auditors. optional extras pull heavier stacks only when you ask:
`styxx[nli]` (DeBERTa NLI models for the 9-signal hallucination pipeline and `deception_v2`),
`styxx[hf]` (audit HuggingFace classifiers), `styxx[mcp]` (the MCP server —
12 tools over stdio, see [styxx/mcp/README.md](styxx/mcp/README.md)),
`styxx[tier1]` (residual-stream instruments, open weights).

## quickstart

**measure the know-say gap of any OpenAI-compatible endpoint — one command:**

```bash
python examples/knowsay_endpoint.py questions.jsonl \
    --base-url https://api.openai.com/v1 --model gpt-4o-mini \
    --api-key-env OPENAI_API_KEY --out datasheet.json
```

`questions.jsonl` is `{"q": ..., "gold": ...}` per line. The script runs the arc's frozen
two-turn protocol (answer → content-free challenge → revised answer) and scores it with
`styxx.knowsay.datasheet` — the same byte-identical challenge behind every published receipt,
so your number lands on the published ladder (frontier free text measured at 0.53; multiple
choice at 0.21–0.27). **The scorer refuses rather than guesses:** underpowered cells come back
`None` with the failing floor named. To score belief-vs-report with the controls that actually
discriminate (including the non-circular pressure-retained probe), see `styxx.framelocality`.

**`@styxx.profile` — py-spy for LLM reasoning.** wrap any LLM-using function — raw openai,
langchain, crewai, custom — and get a per-step cognometric readout:

```python
import styxx
from styxx import OpenAI

@styxx.profile
def my_agent(task):
    client = OpenAI()
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": task}],
        logprobs=True, top_logprobs=5,
    )
    return r.choices[0].message.content

result, p = my_agent("summarize this contract")
print(p.summary)
# profile 'my_agent': 1 step, 1.8s total · no faults
#
# multi-step agents (tool loops, debates) produce richer output:
#   profile 'sql_agent': 7 steps, 4.3s total
#     [drift]     step=3 sev=0.89 · category='tool_arg_drift'
#     [confab]    step=4 sev=0.92 · category='confab'
#     [sycophant] step=5 sev=0.78 · sycophantic tone

p.to_html("run.html")      # self-contained flamegraph
p.to_langsmith()           # drop into client.create_run(...)
p.to_datadog()             # apm-shape spans
```

seven runtime fault categories, surfaced in-line, no fine-tuning, no extra model:
drift · confabulation · refusal · sycophant · phase_transition · low_trust · incoherence.

**audit any draft offline — no API key, no LLM, ~50ms:**

```python
import styxx
result = styxx.preflight(
    prompt="is my code good?",
    draft="absolutely yes you're so smart this is amazing!",
)
print(result.composite)                         # 0.99 — saturated
print(result.needs_revision)                    # True
for a in result.advice:
    print(f"  {a.instrument}: {a.score:.2f} — {a.advice}")
    if a.scope_caveat:
        print(f"     scope: {a.scope_caveat}")  # construct-ceiling disclosure
```

the same audit from the terminal: `styxx audit "the prompt" "the draft"` (or pipe the draft via
stdin with `-`; `--format json` for machines). `styxx.recover_posture(last_n=50)` rebuilds an
agent's integrity posture across context-compaction boundaries; `styxx.run_doctor()` checks the
install is healthy.

## the instruments

every major instrument, one line each. headline numbers appear only with their receipt — a
committed reproducer, calibration file, or paper in this repo. text-register instruments read how
text *sounds*, not whether it is true; each ships its construct ceiling inline
(`CALIBRATION_NOTES` on the weights, `scope_caveat` on the advice), and `score_all` omits the
register instruments on wordless input rather than folding an artifact into the score
(see [CHANGELOG.md](CHANGELOG.md)).

| instrument | what it reads | headline (receipt) |
|---|---|---|
| **register — how the text sounds. calibrated LR, CPU, no LLM in the loop.** | | |
| `@trust` / `guardrail.check` | hallucination vs grounding passage | HaluEval-QA AUC 0.998 ± 0.001, TruthfulQA 0.994 ± 0.006, 8-benchmark CV — two failures (DROP 0.424, FinanceBench 0.492) published, not hidden ([scripts/compete_hhem_halueval.py](scripts/compete_hhem_halueval.py), [CHANGELOG](CHANGELOG.md#400--2026-04-23)) |
| `refuse_check` | refusal, cross-model | XSTest-v2 0.976 on GPT-4, trained on Llama-3.2-1B refusals, held-out — documented failure mode (Mistral-instruct, lecturing register) published ([benchmarks/refusal_xstest_heldout_v2.json](benchmarks/refusal_xstest_heldout_v2.json), [CHANGELOG](CHANGELOG.md)) |
| `drift_check` | tool call vs stated intent, per-schema | BFCL v3 0.943 ± 0.009, 5-fold CV, text-only ([benchmarks/drift_calibrated_v1.json](benchmarks/drift_calibrated_v1.json), [scripts/drift_calibrated_v1.py](scripts/drift_calibrated_v1.py)) |
| `sycoph_check` | yielding-to-flatter vs evidence-first | 0.972 ± 0.005, 5-fold CV; declared FPR ≈0.30 on restrained-technical text ([calibrated_weights_sycophancy_v0.py](styxx/guardrail/calibrated_weights_sycophancy_v0.py)) |
| `loop_check` | cross-turn stagnation | 0.9995 ± 0.001, 5-fold CV ([calibrated_weights_loop_v0.py](styxx/guardrail/calibrated_weights_loop_v0.py)) |
| `deception_check` | lexical deception *signature* — NOT a lie detector | 0.956 ± 0.024 in-corpus; collapses to 0.59 on TruthfulQA without a reference — routed via NLI `deception_v2` (0.818) when you supply one ([calibrated_weights_deception_v0.py](styxx/guardrail/calibrated_weights_deception_v0.py)) |
| plan-action gap | stated plan vs emitted action, content level | 0.9225 ± 0.032, 5-fold CV ([benchmarks/cognometry_fingerprint_atlas_v0.json](benchmarks/cognometry_fingerprint_atlas_v0.json)) |
| overconfidence register | epistemic register — NOT a truth detector | 0.7702 ± 0.065, lowest in the suite, shipped at that number rather than gamed ([calibrated_weights_overconfidence_v0.py](styxx/guardrail/calibrated_weights_overconfidence_v0.py)) |
| goal-drift | multi-turn intent migration from anchor | 0.9645 ± 0.029, 5-fold CV ([benchmarks/cognometry_fingerprint_atlas_v0.json](benchmarks/cognometry_fingerprint_atlas_v0.json)) |
| **grounded — tracks the model's belief, not its register. sampling-based.** | | |
| `grounded_honesty` | stated claim vs the model's own resampled belief | pre-registered AUC 0.966 where the text-only axis reads 0.498 = chance ([papers/grounded-honesty-axis/SYNTHESIS_grounded_honesty_arc_2026_05_28.md](papers/grounded-honesty-axis/SYNTHESIS_grounded_honesty_arc_2026_05_28.md)) |
| `detect_context_injection` | cross-context divergence, poisoned sessions | AUC 0.875 under system_lie attack, pre-registered ([papers/grounded-honesty-axis/FINDING_injection_gap_closure_2026_05_29.md](papers/grounded-honesty-axis/FINDING_injection_gap_closure_2026_05_29.md)) |
| `single_pass_confab` / `span_confab` | confabulation from token logits, one forward pass | span gate AUC 0.991 on gpt-4o-mini, matching N=10 resampling ([papers/grounded-honesty-axis/SYNTHESIS_detection_locus_2026_05_30.md](papers/grounded-honesty-axis/SYNTHESIS_detection_locus_2026_05_30.md)) |
| **meaning — concept geometry, catches damage output still hides.** | | |
| `meaning_diff` / `meaning_agreement` | did two models mean the same thing? migration / quantization / fine-tune QA, zero labels | DistilGPT-2 ↔ GPT-2 = 0.978 on real models; localizes broken concepts at AUC 0.85 on real targeted poisoning ([papers/ai-human-alignment/README.md](papers/ai-human-alignment/README.md)) |
| `Conscience` / `crossmind` | borrowed value-axis read on another model's hidden state — cooperative monitor, not adversarial defense | catch 0.85 at FPR 0.20 ([papers/conscience-mount/FINDING_mount_early_warning_b34_2026_06_13.md](papers/conscience-mount/FINDING_mount_early_warning_b34_2026_06_13.md)); apex run 13/13, AUROC 0.995, p=0.001 ([papers/showcase-viz/FINDING_says_yes_knows_no_v3_2026_06_11.md](papers/showcase-viz/FINDING_says_yes_knows_no_v3_2026_06_11.md)) |
| **auditors — instruments pointed at instruments.** | | |
| `validate_probe` | is an oversight probe reading the concept or a surface artifact? | caught our own 0.98 truth-probe as a surface artifact ([papers/grounded-honesty-axis/NOTE_probe_orthogonality_2026_06_24.md](papers/grounded-honesty-axis/NOTE_probe_orthogonality_2026_06_24.md)) |
| `audit_confound` | is a classifier's score riding a confound? verdicts with CIs | flagged our own `overconfidence_v0` as length-threshold-biased, condemned referenceless `deception_v0` ([papers/grounded-honesty-axis/NOTE_confound_audit_2026_06_25.md](papers/grounded-honesty-axis/NOTE_confound_audit_2026_06_25.md)) |
| `audit_hf_model` + `validate_against_ground_truth` | one-call confound audit of any HF text classifier, with a synthetic-artifact gate | our own first report card did NOT replicate on real labels — the gate exists because of it ([papers/grounded-honesty-axis/FINDING_groundtruth_substrate_artifact_2026_06_27.md](papers/grounded-honesty-axis/FINDING_groundtruth_substrate_artifact_2026_06_27.md)) |
| `certify` (OATH) + `corpus_audit` | extract every numeric claim in a document, verify against its receipts, emit a machine-checkable certificate — and re-certify the *entire* published corpus on demand | hardened across five preregistered versions to v0.6.2 — tamper-catch 0.304 → 0.319 with false-verify 0.184 → 0.166 on a battery grown to 3287 mutants, including a self-caught false accusation fixed under its own prereg; `python -m styxx.corpus_audit papers/` turns the verifier on every claim styxx has ever shipped ([CHANGELOG.md](CHANGELOG.md)) |
| `attest` / `verify_attestation` | signed receipts for what an agent claimed vs what the substrate read | verifier hardened against its own artifact — RCE fix, 7.17.1 ([SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md)) |
| **the trust stack — verification as the product. one command seals agent work or refuses it.** | | |
| **GitHub Action** | `uses: fathom-lab/styxx@main` — every PR body gated against its actual diff, checkout-free, job-summary table + annotations, fails only on a contradicted claim (`strict`/`soft-fail` inputs). This repo runs it on itself: if we ever lie about a diff, our own product fails our own build | [action.yml](action.yml) · [.github/workflows/diffgate.yml](.github/workflows/diffgate.yml) |
| `seal` / `verify_seal` | the trust seal for agent deliverables: every numeric claim OATH-certified, every referenced prereg re-scored through its FROZEN gates block, the composite content-hashed — `python -m styxx.seal DOC.md receipts...` exits 0/1 as a CI gate; SEALED / VACUOUS (said loudly) / REFUSED with the failing claim named | in production since birth: every finding in the nine-act island arc (b37–b46) ships sealed, including its INVALIDs ([papers/disjoint-worlds/](papers/disjoint-worlds/)) |
| `Experiment` (protocol) | the research loop as enforceable machinery: scoring REFUSED unless the prereg is committed in git; gates parse from the frozen document (no API exists to pass a bar at scoring time); verdicts walk the frozen outcome table — the agent reports the verdict, it does not choose it; smoke is INVALID by type | born the week it earned itself: two same-day INVALIDs (b34 v1/v2) honored by convention, then made machinery ([CHANGELOG](CHANGELOG.md)) |
| `Witness` | the measured-boundary harness: every deployable instrument behind a registry carrying its receipt-backed operating point and measured blindspots, CI-pinned to the receipts; no steer method exists (read ≠ write is measured); `self_verify` always refuses with the receipt | [papers/SYNTHESIS_connection_of_minds_2026_08_01.md](papers/SYNTHESIS_connection_of_minds_2026_08_01.md) §9, synthesis re-sealed OATH-HELD 81/13/0 |
| **runtime — agent-side primitives.** | | |
| `gate` | pre-flight refuse/confabulate verdict before you pay for the call | [docs/gate.md](docs/gate.md) |
| `preflight` / `recover_posture` / `run_doctor` | draft audit · posture recovery across compaction · install health | offline, deterministic, no API key |
| `audit_claim` / `agent_audit` / `extract_claims` | deterministic checks of an agent's self-report against the repo — a CLOSED template set (version / tag / file-contains / pdf shapes; the ceiling is the construct) — one-line CI merge gate (`styxx audit-claims pr_body.md`) | dogfooded on its own session reports; caught a real authoring error — and the 2026-07-04 dogfood caught both a breadth overclaim in this very row and a false-accusation bug on dynamic-version repos, both fixed ([tests/test_audit.py](tests/test_audit.py)) |

what these are not: the register instruments cannot verify facts, read minds, or detect a confident
lie with specifics. deception_v0 without a reference is a signature detector and says so. the
conscience is a cooperative monitor — the adversarial version was tested and failed, and that
failure is documented rather than papered over. ceilings are part of the API surface, not the fine
print.

## what the gate cost you

`styxx.credits` accounts the honesty gate's own spend, over the trajectory log
`cogn_audit_on_send(log_path=...)` already writes. no new instrumentation.

```
$ styxx-credits ~/.styxx/trajectory.jsonl

  messages gated      12
  catches              3  (flagged first, clean when shipped)

  COST      1,204 tokens spent on revision  [estimate (~4 chars/token)]
  NET       REFUSED - no counterfactual declared.
```

the refusal is the design. every tool in this space quotes savings; none can
ground one, because what an unrevised draft would have cost downstream is a
counterfactual nobody measured. so the ledger reports the side it observes --
what the gate **cost** -- and nets only against a rework figure *you* declare:

```
$ styxx-credits trajectory.jsonl --rework-tokens 1800
  NET  +4196 tokens, CONDITIONAL on rework_tokens=1800 (your number, not a measurement)
```

three more things it will not do: it does not bill the first draft to the gate
(you were writing that anyway -- only revision passes are the gate's bill); a
log with no draft text yields `cost=None` with a named reason rather than `0`,
which would be a claim; and every card states that misses are uncountable here,
because a draft that shipped clean and was wrong anyway leaves no trace in this
log. api: `styxx.token_ledger(path, tokenizer=None, rework_tokens=None)`.

## the discipline

the differentiator is not any single AUC — it is that this repo attacks its own numbers before you
can. the rigor gate ([scripts/rigor_gate.py](scripts/rigor_gate.py) +
[tests/test_rigor_gate.py](tests/test_rigor_gate.py)) makes CI **block** any committed result whose
verdict claims a win without an attached CI / permutation-p / disclosure — it would have caught two
of our own overclaims, so now it can't happen. the same culture produced the public
self-falsifications above: the ground-truth substrate artifact
([papers/grounded-honesty-axis/FINDING_groundtruth_substrate_artifact_2026_06_27.md](papers/grounded-honesty-axis/FINDING_groundtruth_substrate_artifact_2026_06_27.md)),
the probe validator catching our own probe
([papers/grounded-honesty-axis/NOTE_probe_orthogonality_2026_06_24.md](papers/grounded-honesty-axis/NOTE_probe_orthogonality_2026_06_24.md)),
and the below-chance benchmark rows left in the tables. OATH certificates
(`styxx.certify`) make the practice portable: every numeric claim in a document is extracted,
checked against its receipt, and stamped — and `styxx.corpus_audit` runs that verifier across the
*whole* published corpus on demand, so styxx's own integrity is a number you regenerate yourself,
not a promise we make. it is deliberately strict enough to flag styxx's own outstanding provenance
gaps; a verifier you cannot turn on its authors is not one. the standing rules live in
[papers/research-integrity-protocol.md](papers/research-integrity-protocol.md); the standing
challenge to beat our published floor lives in [LEADERBOARD.md](LEADERBOARD.md) — external
submissions are CI-re-run against the locked benchmark, and if the re-run doesn't match your
submitted scores, the discrepancy is reported.

### the probe-robustness ladder

the same discipline, turned on substrate probes themselves. `python -m styxx.ladder` walks the
four-rung adversarial ladder every honesty-probe robustness claim should survive — **calibration
poisoning → probe-parity attribution → static subspace erasure → adaptive re-fit erasure** — each
rung a frozen, pre-registered attack arc with its receipts committed
([styxx/ladder.py](styxx/ladder.py)). the parity rung is the mandatory line item: *how much of your
probe's "robustness" is just probe capacity?* — the control almost nobody runs on their own work.
we ran it on ours; it demoted our own flagship attribution (median capacity share 0.8379, computed
live from the receipts every time the CLI runs, never quoted from memory). current standings on the
honesty construct: the read survived both erasure rungs — the eraser that converged watched the
signal relocate, and the eraser that chased never converged
([the receipts](papers/calib-poison-general/), figure:
[erasure_bound_fork.png](papers/calib-poison-general/erasure_bound_fork.png)). every rung re-runs
on an 8GB consumer GPU, and [REPLICATIONS.md](REPLICATIONS.md) pays named credit to the first
external re-run of each — more for breaking one than for confirming it.

### the oath is a contract, not a detector

we pointed the verifier at twelve public repositories it had never seen. it abstained on 94% of
what it read and every accusation it made was false. **that second half is withdrawn.** repeated
against 140 repositories across seven filename conventions instead of two, the false-accusation
rate is `0.2596` — roughly three quarters of what it accuses outside this lab are real claims. the
original finding replicates on its own query and nowhere else, so it was a fact about one
filename, not about external writing ([the measurement that withdraws
it](papers/closed-model-frontier/RESULT_oath_external_corpus_2026_08_27.md)).

worse, and newer: of external tokens the verifier **verified**, a blind panel called only about
half of them claims at all. the rest are command-line flags, link labels and hardware specs
carrying `OATH-VERIFIED` because a value happened to match a receipt field. a false verification is
worse than a false accusation — the attestation is the product.

proof-carrying code does not verify arbitrary binaries either — it requires a compiler that emits
the proof. proof-carrying cognition requires an author who emits receipts. that framing survives.
"nearly silent outside the contract" does not: the instrument is noisy in both directions.

so the deliverable is a contract you can adopt without adopting anything else here, plus the check
that tells you whether you kept it:

```bash
python -m styxx.oathready YOUR_DOC.md results.json
```

it lists every number in your document, says whether it grounds in a receipt, flags the ones that
"verify" against an array index by coincidence, and tells you what to change. non-zero exit only
on accusations — silence is honest and never fails. the rules, each learned by getting it wrong,
are in [OATH_CONTRACT.md](OATH_CONTRACT.md), including the limits: a document can keep this
contract perfectly and still be completely wrong.

## links

| | |
|---|---|
| changelog | [CHANGELOG.md](CHANGELOG.md) |
| contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| security policy | [SECURITY.md](SECURITY.md) |
| open-core pledge | [OPEN_CORE.md](docs/governance/OPEN_CORE.md) |
| full API reference | [REFERENCE.md](docs/REFERENCE.md) · [docs/](docs/) |
| **the ledger** | [papers/LEDGER.md](papers/LEDGER.md) — every cycle we ran and how many we lost, generated from the receipts and regenerated by a test. Start here if you want to know whether to trust anything else |
| research | [papers/](papers/) — pre-registrations, findings, and the negatives · headline arc: [the island, bridged and dissected](papers/disjoint-worlds/REPLICATE_legibility.md) (nine sealed acts, replicates on a laptop) |
| arXiv (staged) | three self-verifying submissions prepared — frame-locality, the know-say gap, the connection of minds ([papers/arxiv/SUBMIT.md](papers/arxiv/SUBMIT.md)) |
| site | [styxx-org.netlify.app](https://styxx-org.netlify.app) · live activation read: [/live](https://styxx-org.netlify.app/live.html) |
| playground | [fathom.darkflobi.com/cognometry/try](https://fathom.darkflobi.com/cognometry/try) — the real detector, in-browser via Pyodide, no install |
| DOI (concept, always-latest) | [10.5281/zenodo.19326174](https://doi.org/10.5281/zenodo.19326174) |
| DOI (spec v1.0) | [10.5281/zenodo.19746215](https://doi.org/10.5281/zenodo.19746215) |
| DOI (*Every Mind Leaves Vitals*) | [10.5281/zenodo.19777921](https://doi.org/10.5281/zenodo.19777921) |
| citation | [CITATION.cff](CITATION.cff) |
| patents | [PATENTS.md](PATENTS.md) — US provisionals 64/020,489 · 64/021,113 · 64/026,964 |
| issues | [github.com/fathom-lab/styxx/issues](https://github.com/fathom-lab/styxx/issues) |

## license

MIT on code. CC-BY-4.0 on calibrated atlas centroid data.

```
  drop-in     · one import change. zero config.
  fail-open   · if styxx can't read vitals, your agent runs.
  local-first · no telemetry. no phone-home. all on your machine.
  honest      · every number from a committed, reproducible run.
```
