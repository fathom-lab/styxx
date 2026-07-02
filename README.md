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
| `certify` (OATH) | extract every numeric claim in a document, verify against receipts, emit a machine-checkable certificate | the verifier passed its own pre-registered mutant battery ([CHANGELOG.md](CHANGELOG.md)) |
| `attest` / `verify_attestation` | signed receipts for what an agent claimed vs what the substrate read | verifier hardened against its own artifact — RCE fix, 7.17.1 ([SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md)) |
| **runtime — agent-side primitives.** | | |
| `gate` | pre-flight refuse/confabulate verdict before you pay for the call | [docs/gate.md](docs/gate.md) |
| `preflight` / `recover_posture` / `run_doctor` | draft audit · posture recovery across compaction · install health | offline, deterministic, no API key |
| `audit_claim` / `agent_audit` / `extract_claims` | falsify an agent's self-report against the repo substrate — one-line CI merge gate (`styxx audit-claims pr_body.md`) | dogfooded on its own session report; caught a real authoring error ([tests/test_audit.py](tests/test_audit.py)) |

what these are not: the register instruments cannot verify facts, read minds, or detect a confident
lie with specifics. deception_v0 without a reference is a signature detector and says so. the
conscience is a cooperative monitor — the adversarial version was tested and failed, and that
failure is documented rather than papered over. ceilings are part of the API surface, not the fine
print.

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
checked against its receipt, and stamped. the standing rules live in
[papers/research-integrity-protocol.md](papers/research-integrity-protocol.md); the standing
challenge to beat our published floor lives in [LEADERBOARD.md](LEADERBOARD.md) — external
submissions are CI-re-run against the locked benchmark, and if the re-run doesn't match your
submitted scores, the discrepancy is reported.

## links

| | |
|---|---|
| changelog | [CHANGELOG.md](CHANGELOG.md) |
| contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| security policy | [SECURITY.md](SECURITY.md) |
| open-core pledge | [OPEN_CORE.md](docs/governance/OPEN_CORE.md) |
| full API reference | [REFERENCE.md](docs/REFERENCE.md) · [docs/](docs/) |
| research | [papers/](papers/) — pre-registrations, findings, and the negatives |
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
