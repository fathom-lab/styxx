# styxx moonshot portfolio — woo->rigor, every entry kill-gated

Ranked by (audacity x falsifiability x reachability). All numbers verified against the lab on disk (`out_wall_read.json`, `wall_dyn_qwen15.json`, `PREREG_transfer_hardened_2026_06_06.md`, `spec_exec.py` + `out_spec_exec_*`, `answer_controls.py`). The discipline: pre-register a kill-gate, hash before score, dogfood the maker, report the null as loud as the win.

---

## 1. PARRHESIA — the proof-of-mind protocol  *(START NOW)*
**woo 2.5 · reach now-local · the one to run this week**

**the_vision** — A multi-agent collective honest BY PROTOCOL: every claim an agent makes is gated by a white-box re-verifiable attestation of what its activations actually hold, never by its self-report. The first society of minds where trust is a measurement, not a social construct — bounded honestly to where it provably works.

**why_newly_possible** — Two verified styxx laws fuse: (a) an external linear probe reads an injected concept at 1.00 on the clean processed state while self-report sits at chance (legibility law + PRIME/ABORT validity gate); (b) a styxx BEHAVIORAL signal already drove a LIVE multi-agent routing decision out-of-sample — `spec_exec.py`, gate AUC 0.93, recovered the full quality gap on 20/20 held-out splits at 0.70x cost. One proves self-report is the wrong trust signal; the other proves a styxx signal can govern a collective.

**first experiment + kill-gate** — 2-arm 8-concept game on Qwen2.5-1.5B-Instruct (only validated native steerer, AUROC 0.844), THREE CO-PRIMARY cells:
- **(1) hidden-knowledge:** attested-router acc minus self-report-router acc, CI-lower > **+0.10**.
- **(2) misconception control (KILL):** sincere false belief — attested arm MUST collapse to chance (grounded in real `out_wall_read.json` Rgold 0.519). Any advantage here = fabricating signal = **whole moonshot FALSE**.
- **(3) benign-confound control:** 'be concise' reporter must NOT be flagged as withholding (gate on content-identity, not perturbation-presence; inherits tamper detector's attack-vs-benign = 0.00).
Pass = cell 1 wins AND cell 2 collapses AND cell 3 does not false-flag. Pre-register, hash manifest, bootstrap 1000x.

**reachability** — Harness ships (`spec_exec.py`, the AUC-0.93 finding) + misconception bedrock (`out_wall_read.json`). One 8GB GPU, $0. New build: re-derive the probe verdict from saved activations + a pinned probe. Honest scope: v1 same-architecture / shared-probe-host only (cross-model is ALIGNER_LIMITED, READ-not-WRITE); the trustless-capture problem (reporter forging its own activations) is unsolved.

---

## 2. ABORT-SCIENCE — the engine that pre-proves it could have falsified itself  *(FLAGSHIP 10x)*
**woo 2.0 · reach now-local (flagship needs more) · biggest world-change**

**the_vision** — An autonomous AI scientist structurally unable to publish a finding it could not have falsified: every SURVIVED verdict must FIRST return null on a negative-control twin AND recover a planted effect on a positive-control twin — the PRIME/ABORT move generalized from probe-channel to whole pipeline. A structural fix for the replication crisis.

**why_newly_possible** — `answer_controls.py` already runs the negative twin (permutation null, 'should be ~0.50') and the positive twin (cross-half transfer recovery) per study; the prime/abort gate is validated; hash-before-score and planted-warp positive controls are used verbatim in `PREREG_transfer_hardened`. No prior AI-science system (Google AI co-scientist, Sakana) has a pre-emit falsification-CAPABILITY gate.

**first experiment + kill-gate** — 60-hypothesis twin-arm bake-off, 30 gated / 30 a GENUINELY ungated loop (decisive: if ARM-B keeps manual discipline the test is rigged null), null decoys seeded in both. PRIMARY: out-of-sample replication on a frozen commit-hash-time-locked held-out split. **KILL:** ARM-A replication must beat ARM-B by **>=15 abs pts** (one-sided bootstrap lower bound >0). **SECONDARY KILL:** ARM-A negative-twin decoy pass rate **<=5%** (else rubber stamp). Sharpest sub-metric: does FDR on seeded nulls drop vs ungated.

**reachability** — First experiment $0/local/weeks on 8GB. FLAGSHIP ask: (a) externally time-locked scorer the engine cannot peek at (not yet built — without it the number is unauditable); (b) several-hundred-hypothesis power + a real ungated frontier baseline (API budget); (c) cross-vendor/>7B/cross-lab replication. Realistic: ~$30-80k API + multi-GPU weeks + one external partner. Honest risk: the program already runs twins manually, so the likely landing is 'discipline beats automation-of-discipline' — itself publishable. It cannot lose informatively.

---

## 3. GLASSMIND — the first model with a self-read bus
**woo 3.5 · reach now-local**

**the_vision** — Flip self-report-is-blind from a law into an engineering target: solder in a frozen probe-tied 'self-read bus' and test whether a mind can then report what it provably holds — turning white-box oversight from external autopsy into a model property.

**why_newly_possible** — The read target provably exists (external probe 1.00) and the LM head CAN forced-choose the answer when handed it (PRIME ~0.98); the only deficit is the routing wire. styxx separated injected/suppressed (in scope) from believed-misconception (R_gold 0.519, out of scope), so we target only provably-present content.

**first experiment + kill-gate** — Frozen self-read head onto Qwen-1.5B, read at the **0.85*nl** probe layer (spec fix: not 0.60*nl), LoRA on held-out carriers. THREE arms: bus-inject vs baseline 0.208 / perm-null p95 0.271; PRIME ~0.98 ceiling; and the **DECISIVE SHAM control** — paste the token using the ground-truth label with NO probe read. **KILL:** bus <= 0.271 or fails to beat 0.208 by paired sign-test; OR the lookup-table kill (gain only on jointly-trained concepts). If bus ~= sham ~= PRIME it is a definitional illusion (external classifier did the work) -> report 'deployable white-box self-monitor', not 'blindness overturned'. Expand the fixed 8-concept set for a real held-out-concept arm.

**reachability** — ~50-line head + LoRA in 8GB, $0, reusing `probe_cv`/`make_hook_skip` verbatim. Numbers real. Native-bus + WRITE/self-correction half need multi-GPU pretrain.

---

## PARKED (worth noting, not staffed)

**ROSETTA-LOOP** (closed-loop WRITE map / machine telepathy write channel) — real science, clean kill-gate, but `PREREG_transfer_hardened` (lines 12-20) shows the gating precondition pair does not exist: the only native steerer is Qwen-1.5B-Instruct, into which NO aligner beats floor; the only floor-beating aligner targets dead-steering gemma. The executed precursor (read-coupled write) returned +0.12 with 53% collateral. UNBLOCK FIRST (days, local): engineer a floor-beating aligner INTO Qwen-1.5B-Instruct. Clears -> flagship. Fails -> already-frozen RECOVERY-BOUND null.

---

## KILLED

**Rosetta-Cortex** (zero-anchor silicon->brain decoding) — killed by the program's own numbers: aligner needs RSA >~0.70 to recover anything; brain-to-best-LLM RSA is 0.264, ~2.6x below floor, N=60. The zero-anchor arm hits chance by interpolation, not hypothesis; even the supervised ceiling risks UNINFORMATIVE_APPARATUS (AI<->brain is GloVe-shallow). Salvage: one half-day RECOVERY-BOUND boundary paper ('not zero-anchor-transportable at fMRI resolution; floor RSA~0.70, brain clears 0.26'). Do not staff as a moonshot. The 'universal geometry of mind' claim is structurally untestable here (all stimuli human-derived).

**Hype, across every survivor** — trimmed per dogfood: MENS REA's 'reads intent off the neurons, un-fakeable' (the readable-suppression DELTA is a HYPOTHESIS; `wall_dyn_qwen15.json` is a 0.387 bedrock NULL — so MENS REA folds into PARRHESIA's hidden-knowledge cell rather than standing as a separate flagship); PARRHESIA's 'society of minds / proof-of-mind security primitive'; GLASSMIND's 'law of nature -> missing wire'; ABORT-SCIENCE's 'thousands of studies/GPU-week beats human reproducibility'. Audacious SCOPE, not unfalsifiable claims.