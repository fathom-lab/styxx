# STYXX v8 — SPEC
### checksums for model behavior
**fathom lab · status: DRAFT v0.1 · 2026-09-07 · becomes binding when committed and its hash is logged as charon entry #0**
---
## 0. What v8 is
Three layers, one rule.
| layer | what it does | depends on keystone? |
|---|---|---|
| **fingerprint** | a reproducible checksum for a model's behavior; `verify` detects drift above a measured noise floor | no |
| **instruments with evidence** | every instrument carries its own prereg → result → promotion certs; certified tier is enforced in code | promotion of *depth* does; the layer itself does not |
| **charon** | append-only transparency log for certs, with inclusion and consistency proofs and a challenge mechanism | no |
The rule: **every claim is a cert, and every cert is reproducible by a stranger from its own recipe.** Anything that can't be reproduced from the cert is not a claim; it's a note.
### Non-goals for v8.0
- No hosted service until a design partner asks for one. Charon v0 is a git repo.
- No conformity assertions. The compliance view organizes evidence; counsel decides.
- No instrument reading in a certified section without a promotion cert that covers it.
- No token in the critical path. Chain anchoring, if used, is a timestamp paid as a fee.
---
## 1. Design principles
1. **Content-addressed.** A cert's id is the sha256 of its canonical bytes. Same evidence, same id, anywhere.
2. **Recipe-complete.** A cert contains everything needed to re-run it: model identity, battery, decoding, template, environment. If it doesn't, `verify` refuses to compare and says so.
3. **Noise floor before drift.** No drift claim without a measured null. "The model changed" is only sayable when the distance exceeds what re-running the *same* model produces.
4. **Instruments carry their own evidence.** Tier is a property of a (instrument, model family, task family) triple, granted only by a promotion cert in the log.
5. **Append-only, publicly auditable, disputable.** The log stores; it does not endorse. Inclusion ≠ validity. Challenges are first-class.
6. **Scope-limited claims.** A promotion cert's claim string is the only text an instrument is allowed to say about itself in compliance output.
---
## 2. Cert envelope (OATH v2)
Keep the OATH envelope; type the payloads. One schema, eight types.
```json
{
  "styxx": "8.0",
  "type": "fingerprint | battery | eval | prereg | result | promotion | action | challenge",
  "id": "sha256:<64 hex>",
  "created": "2026-09-07T18:00:00Z",
  "issuer": { "name": "fathom lab", "key": "ed25519:<base64url pubkey>" },
  "subject": { "...": "model identity, §2.2" },
  "recipe":  { "...": "everything needed to reproduce, §2.3" },
  "body":    { "...": "type-specific payload, §3–§10" },
  "refs":    ["sha256:<id of any cert this depends on or responds to>"],
  "sig":     "ed25519:<base64url signature over id bytes>"
}
```
### 2.1 Canonicalization, id, signature
- Canonical bytes: **RFC 8785 (JSON Canonicalization Scheme)** — sorted keys, fixed number formatting, UTF-8. Use a library; do not hand-roll.
- `id` = `sha256(JCS(cert with "id" and "sig" removed))`.
- `sig` = ed25519 signature over the raw 32-byte digest. Verify = recompute id, check sig against `issuer.key`.
- Any cert failing recompute-or-signature is rejected at log append and at `verify`.
### 2.2 Subject — model identity
Local weights (white-box eligible):
```json
"subject": {
  "kind": "weights",
  "hf_repo": "google/gemma-2-2b", "revision": "<commit sha>",
  "weights_sha256": "<see A.2>", "config_sha256": "...", "tokenizer_sha256": "...",
  "precision": "bf16 | fp16 | q8_0 | q4_k_m | ...",
  "runtime": { "framework": "transformers", "version": "4.x.y", "backend": "torch 2.x cu12x" },
  "hardware": { "gpu": "RTX 4090", "driver": "...", "count": 1 }
}
```
API alias (black-box only):
```json
"subject": {
  "kind": "alias",
  "provider": "...", "alias": "...", "observed_model_id": "<if returned in responses>",
  "observed_at": "2026-09-07T18:00:00Z", "region": "..."
}
```
Alias subjects are labeled `tier: black-box` in every downstream cert. There is no way to make an alias fingerprint as strong as a weights fingerprint; the spec does not pretend otherwise.
### 2.3 Recipe
```json
"recipe": {
  "battery": "sha256:<battery cert id>",
  "decoding": { "temperature": 0, "top_p": 1.0, "max_new_tokens": 64, "stop": ["\n"], "seed": 7 },
  "chat_template_sha256": "...", "system_prompt_sha256": "...",
  "harness": { "name": "styxx", "version": "8.0.0", "commit": "<sha>" },
  "env_lock_sha256": "<sha256 of pip freeze / lockfile>",
  "sae": { "repo": "google/gemma-scope-2b-pt-res", "layer": 12, "width": "16k", "revision": "..." }
}
```
`sae` is present only when a white-box channel or SAE-based instrument is used. Two certs are **comparable** iff `battery`, `decoding`, `chat_template_sha256`, and `system_prompt_sha256` match exactly. Otherwise `verify` exits 3 (recipe mismatch). This is the single most important honesty rule in the product: no number is produced across mismatched recipes.
---
## 3. Fingerprint cert
### 3.1 Body
```json
"body": {
  "runs": 1,
  "items": [
    {
      "item_id": "b7f2…",
      "output_sha256": "…",
      "output_text": "…  (omitted when redacted=true)",
      "token_ids_sha256": "…",
      "seq_logprob": -3.412,
      "topk": [ { "pos": 0, "ids": [..5..], "lps": [..5..] }, "… positions 0..7" ]
    }
  ],
  "channels": {
    "exact":    { "hash": "sha256 of concatenated output_sha256 in battery order" },
    "seqlp":    { "vector_sha256": "…" },
    "topk":     { "vector_sha256": "…" },
    "resid":    { "profile": [[..L..] per item, optional], "mean": [..L..] },
    "lens":     { "converge_layer": [..per item..], "mean": 14.2 }
  },
  "noise_floor": { "runs": ["sha256:…", "…"], "per_channel": { "exact": {...}, "seqlp": {...}, "topk": {...}, "resid": {...}, "lens": {...} } },
  "redacted": false
}
```
### 3.2 Channels
Black-box (any subject):
- **exact** — greedy output token ids per item. The battery-level hash is *the checksum*.
- **seqlp** — sum log-probability of the greedy sequence per item (where the API returns logprobs; else channel absent, never zero-filled).
- **topk** — top-5 log-probs at the first 8 answer positions per item.
White-box (weights subject, one extra forward pass with hooks, no SAE required):
- **resid** — per-layer residual-update norm profile at the final prompt token, normalized to sum 1 across layers.
- **lens** — logit-lens convergence layer: first layer at which the final-position top-1 under the unembedding equals the model's actual top-1. Cheap proxy for computational depth.
Deferred to instrument tier (§7), not a v8.0 channel: SAE attribution depth. It is expensive and its validity is what the keystone decides. It is a *reading*, not a checksum, until promoted.
### 3.3 Distances
Defined in Appendix B. Every channel has exactly one distance function; `verify` never picks.
---
## 4. Battery cert and canary selection (canary-v1)
A fingerprint is only as sensitive as its battery. Two batteries per fingerprint:
- **fixed-v1** — public, model-agnostic, ~512 items, drawn from styxx-bench after the gold fix. Used for cross-model comparison and predecessor diffs. Rotated yearly (fixed-v2 …); old versions stay valid for old certs.
- **canary-v1** — model-specific, selected against the subject to be *sensitive by construction*. Used for drift detection on that subject.
### 4.1 Why canaries
A random battery mostly measures items the model answers with a wide margin; those don't move under quantization, fine-tuning, or a silent swap. Items that sit on decision boundaries do. Glimmer stops being a runbook and becomes the selection oracle.
### 4.2 Perturbation family Δ (run on the reference subject, reference precision)
| id | perturbation | purpose |
|---|---|---|
| δ1 | precision: bf16 → q8_0 → q4_k_m (same weights) | sensitivity to representational change |
| δ2 | nuisance: batch size ∈ {1, 8, 32}, permuted item order, second GPU if available | **noise**, not signal |
| δ3 | margin: temp 0.2, n=3 samples | proximity to decision boundary |
| δ4 | (optional) semantically-null template jitter: trailing whitespace, newline variant | template brittleness |
### 4.3 Scores
For item *i*, at the first answer token under reference precision:
- `margin(i)` = lp(top1) − lp(top2), in nats.
- `flip1(i)` = fraction of δ1 variants whose greedy output ≠ reference.
- `flip2(i)` = fraction of δ2 variants whose greedy output ≠ reference.
- `flip3(i)` = fraction of δ3 samples ≠ reference.
```
s(i) = 0.6 · mean(flip1(i), flip3(i)) + 0.4 · (1 − σ(margin(i) / τ)),   τ = 1.0 nat
```
### 4.4 Selection
1. **Exclude** every item with `flip2(i) > 0`. Those are nuisance-unstable; they go into the noise-floor pool (§5), never the battery. This is the load-bearing step: canaries must be *stable under noise and sensitive to change*.
2. Rank the remainder by `s(i)`.
3. Stratify by task family (recall / short reasoning / instruction-following / format / refusal boundary): no family > 25% of the battery.
4. Take the top **N** (defaults: 64 quick, 256 standard, 1024 docket).
5. Add **K = 64 anchors**: the highest-margin items with zero flips under every δ. Anchors should never change. An anchor flip means a tokenizer or weights swap, not drift — `verify` reports it as a separate verdict (`identity`).
### 4.5 Battery cert body
```json
"body": {
  "kind": "canary-v1 | fixed-v1",
  "selected_against": "sha256:<reference fingerprint cert id>  (canary only)",
  "pool_sha256": "…", "pool_size": 12000,
  "params": { "tau": 1.0, "w_flip": 0.6, "w_margin": 0.4, "n": 256, "k_anchors": 64, "max_family_share": 0.25 },
  "items": [ { "item_id": "…", "prompt_sha256": "…", "prompt_text": "…", "family": "recall", "role": "canary | anchor",
               "score": 0.71, "margin": 0.42, "flip1": 0.67, "flip2": 0.0, "flip3": 0.33 } ]
}
```
Prompts are stored in the clear unless the battery is marked `redacted`. A redacted battery cannot be used for a public claim — only a black-box "internal" fingerprint. The default is clear.
### 4.6 Known limits
- Canaries are specific to the subject they were selected against; a canary battery from model A is just a random battery for model B. Cross-model claims use fixed-v1 only.
- A model fine-tuned *on* its own canaries would defeat them. Re-select after any fine-tune; the selection cost is one δ-sweep.
---
## 5. Noise-floor protocol
Purpose: make "the model changed" a falsifiable statement.
### 5.1 Procedure
For subject S and battery B:
1. Run the recipe **R** times, varying only nuisance factors: batch size, item order, physical GPU / driver where available; for alias subjects, time-of-day and region within the shortest window that yields R runs.
   - R = 5 minimum (10 pairwise distances). R = 8 for dockets (28 pairs).
2. For each channel *c*, compute all pairwise distances between runs → the empirical null `D_c`.
3. `floor_c` = max(D_c) if |D_c| < 30, else the 99th percentile of D_c. Store the full distribution in the cert, not only the threshold.
4. Every run is logged as its own fingerprint cert. The canonical fingerprint (lowest run seed) carries `noise_floor.runs` referencing the others.
### 5.2 Drift decision (per channel)
| condition | verdict |
|---|---|
| no noise floor on record for (S, B, recipe) | `inconclusive` — verify says why, exits 2 |
| d(new, ref) ≤ floor_c | `same` |
| d(new, ref) > floor_c on first run, ≤ floor_c on the mandatory confirmation run | `same (transient)` |
| d(new, ref) > floor_c on first run **and** confirmation run | `drift` |
| any anchor item changed | `identity` (reported separately, always) |
Overall verdict: `drift` if any channel is `drift`; `identity` overrides everything; `inconclusive` if any channel lacks a floor and none drifted; else `same`.
### 5.3 What the floor does not cover
- A provider that changes the model *during* the R-run window contaminates the alias floor. The cert records the window; readers judge.
- Floors are per (S, B, recipe). Change any of them and the floor is void — `verify` exits 2, not 0.
---
## 6. Verify
```
styxx verify --ref <fingerprint cert>            # re-run the recipe now, compare, confirm on drift
styxx verify --diff <cert A> <cert B>            # compare two existing certs; no execution (replaces `migrate`)
```
Emits a `result` cert, `body.kind = "verify"`, with per-channel `{distance, floor, ratio, verdict}` and the overall verdict.
Exit codes (CI contract):
| code | meaning |
|---|---|
| 0 | same |
| 1 | drift (or identity) |
| 2 | inconclusive — no noise floor, or a channel absent on one side |
| 3 | recipe mismatch — certs are not comparable; no distances are printed |
A `--diff` across mismatched recipes prints the mismatched fields and nothing else. Silence beats a misleading number.
---
## 7. Instruments with evidence
### 7.1 Registry
Each instrument is a plugin with `id`, `version`, `code_sha256`, and a **tier** computed from the log — never set by hand:
```
tier(instrument, model_family, task_family) =
    "certified" if a valid promotion cert covering that triple is in charon
    else "lab"
```
`styxx attest` writes certified readings into the certified section and everything else into a `lab` section under a fixed banner: *"lab-tier reading; no validated claim; not for conformity use."* There is no flag to move a reading between sections. If you want it certified, put a promotion cert in the log.
### 7.2 Evidence chain (three cert types)
**prereg** — issued before any confirmatory data exists. Body: hypotheses with one-sided directions, primary and secondary endpoints, exact statistical tests, datasets with loader-verified field names, exclusion rules, kill gates, confound gate, conjunction rule, and `instrument.code_sha256` frozen. The log timestamp is the proof of order; that is charon's first real use.
Sealed preregistration: `body.sealed = true` with `commitment = sha256(salt || JCS(hypotheses))`. The hypotheses are revealed later by a second prereg cert that refs the sealed one and includes the salt. Order is proven by the log; competitors learn nothing until reveal.
**result** — refs its prereg. Body: per-endpoint outcome, effect sizes with CIs, kill-gate status, raw-data hashes, every amendment with its own timestamp, and `deviations` (empty or not; never absent). `kind ∈ {confirmatory, pilot, robustness, verify, response}`.
**promotion** — refs prereg + result. Body:
```json
{ "instrument": "depth", "version": "1.2.0", "code_sha256": "…",
  "scope": { "model_family": "gemma-2", "task_family": "short-answer factual QA" },
  "claim": "Mean SAE-attributed layer adds ΔAUC = 0.041 [0.018, 0.063] over semantic entropy for predicting answer correctness on TriviaQA (ID) and PopQA-rare (OOD), Gemma-2-2B, preregistered.",
  "evidence": ["sha256:<prereg>", "sha256:<result>", "sha256:<robustness result>"] }
```
The `claim` string is scope-limited and is the **only** sentence the compliance view may quote about the instrument. Depth certified on gemma-2 / factual QA says nothing about llama or about long-form reasoning, and the code will not let it.
### 7.3 Goodhart adversary
Every instrument ships `adversary.py`: a bounded search (prompt edits, few-shot scaffolds, formatting tricks; budget fixed in the prereg) that maximizes the instrument reading while a held-out grader says the property is absent. Outputs a `result` cert, `kind = robustness`, with the best attack found, the reading it achieved, and the instrument's discrimination under attack.
- v8.0: a robustness result is **required** for promotion (disclosure).
- v8.1: promotion additionally requires discrimination under attack ≥ a threshold set in the prereg (gate).
### 7.4 Instrument status at v8.0
| instrument | tier | next cert |
|---|---|---|
| depth (mean attributed layer) | lab | keystone v2 prereg cert → result → promotion or a public negative |
| monochord (H1–H3 spectral) | lab | prereg v3 goes into charon as entry #1; result either way |
| glimmer | retired as instrument | becomes δ1 in canary-v1 (§4.2) |
| logprob gate | lab | fix: absent logprobs ⇒ `inconclusive`, never a flag (stacc commitment) |
| styxx-bench | — | becomes fixed-v1 after the monday/tuesday gold fix and strict-gate normalization |
---
## 8. Charon — transparency log
Follows RFC 6962 (Certificate Transparency) with two additions: challenge certs and sealed preregistration.
### 8.1 Structure
- **Entry** = full cert bytes (JCS) + cert id. Redacted certs store hashes of outputs, never the outputs; the recipe stays complete.
- **Leaf hash** = `sha256(0x00 || entry_bytes)`. **Node hash** = `sha256(0x01 || left || right)`. Domain separation exactly as RFC 6962.
- **Signed tree head (STH)** = `{ log_id, tree_size, root_hash, timestamp, sig }`, ed25519 with the log key. Published every 100 entries or every hour, whichever first.
- **Inclusion proof** — audit path leaf → root for a given cert id and STH.
- **Consistency proof** — proves STH(n) is a prefix-extension of STH(m), m < n. This is what makes "append-only" checkable rather than promised.
### 8.2 Storage v0 (weeks 1–4)
A git repository, published on GitHub Pages.
```
charon/
  entries/000000.ndjson   # 10,000 entries per file, one JCS cert per line
  sth/2026-09-07T18.json  # signed tree heads
  keys/log.pub            # log key
  README.md               # the four verification commands, nothing else
```
Git history is a secondary tamper-evidence layer, not the primary one. The Merkle proofs are primary; a mirror needs only the ndjson files and one STH to verify anything.
### 8.3 Submission
- v0: fathom's key only. Append = PR; CI validates schema, recomputes id, checks signature, appends, publishes STH.
- v1: open submission. The log stores; it does not endorse. Rate-limited by issuer key. Inclusion means "this cert existed at this time," nothing more.
### 8.4 Mirrors and gossip
`styxx log mirror` clones, verifies every consistency proof between consecutive STHs, and records the STHs it has seen. A mirror that observes two STHs with the same `tree_size` and different `root_hash` publishes both — that is cryptographic proof the log misbehaved. Gossip in v0 is just mirrors comparing STH files; automate later.
### 8.5 Anchoring (optional)
Publish `root_hash` of each STH to a public chain as a timestamp (opentimestamps-style). Cost is a transaction fee. No token is involved and none is needed.
### 8.6 Threat model
Defends against: silent rewriting of published numbers; post-hoc hypothesis changes; unverifiable claims; "we tested it" with no recipe.
Mitigates (does not prevent): a dishonest issuer fabricating results — challenges (§9) and independent reproduction are the remedy; log-operator misbehavior — consistency proofs, mirrors, anchoring.
Does not address: providers who make reproduction impossible. Alias fingerprints are labeled black-box and stay that way.
---
## 9. Challenge certs
Anyone can dispute a cert by reproducing it.
```json
"type": "challenge",
"refs": ["sha256:<target cert>", "sha256:<challenger's own fingerprint cert>"],
"body": {
  "per_channel": { "exact": { "distance": 0.12, "target_floor": 0.02 }, "…": "…" },
  "recipe_match": true,
  "environment": { "hardware": "…", "runtime": "…" },
  "note": "free text, optional"
}
```
Rules:
- A challenge whose fingerprint cert fails the recipe-match check is `invalid` (computed by clients, not stored). No recipe match, no challenge.
- The original issuer may reply with a `result` cert, `kind = response` (a new noise floor, an identified cause, a concession). Nothing is deleted; the log shows the thread.
- **Disputed** status (client-computed): a cert with ≥ 2 unresolved challenges above the target's own floor from independent issuers (distinct keys, distinct hardware).
- A challenge that lands *below* the target's floor is a reproduction, not a dispute, and is displayed as such. Successful reproductions are how a cert earns weight.
---
## 10. Action certs (agents)
The successor to `cogn_audit_on_send`.
```json
"type": "action",
"refs": ["sha256:<model fingerprint>", "sha256:<parent action>"],
"body": {
  "context_sha256": "…",
  "readings": { "certified": {}, "lab": { "depth": 13.7, "lens": 12 } },
  "action_sha256": "…", "action_kind": "tool_call | message | commit",
  "ts": "…"
}
```
Volume rule: one cert per action would spam the log. Per-action certs are stored locally in a **sub-log** (same Merkle structure); every hour the sub-log's STH is submitted to charon as a single entry. Any individual action is then provable with two inclusion proofs: action → sub-log root, sub-log root → charon.
Glass-box darkflobi: darkflobi runs exactly this, publishes its sub-log alongside its hourly roots, and ships a viewer. That is the reference deployment and the demo. The agent serves the science.
---
## 11. CLI surface (v8)
```
styxx fingerprint  --subject <spec> --battery fixed-v1|canary|<cert id> [--runs 5] [--white-box] [--redact]
styxx verify       --ref <cert> | --diff <a> <b>                              # exit 0/1/2/3
styxx battery      select --pool <file> --subject <spec> --n 256 --k 64      # canary-v1
styxx instrument   list | prereg <file> | run <id> | adversary <id> | promote <prereg> <result> <robustness>
styxx attest       --subject <spec> --view research|compliance
styxx log          append <cert> | prove <id> | verify-inclusion <id> <sth> | verify-consistency <sth1> <sth2> | sth | mirror | challenge <target> <own>
styxx agent        emit | roll   # action certs; `roll` submits the hourly sub-log root
```
Removed, with one-minor-version shims that print the replacement:
| 7.x | 8.0 |
|---|---|
| `migrate` | `verify --diff` |
| `ci-baseline` / `ci-test` | `fingerprint` / `verify` + exit codes |
| `cogn_audit_on_send` | `agent emit` |
| ten candidate instruments in one namespace | `instrument` registry with computed tiers |
---
## 12. Compliance view
`styxx attest --view compliance` is a **read-only projection** of the log, not a product. It groups certs under the GPAI obligation categories a reader will look for (technical documentation, evaluation and adversarial-testing record, change tracking, incident record) and maps cert types onto them:
| category | cert types shown |
|---|---|
| documentation of the model as deployed | fingerprint (canonical + noise floor), battery |
| evaluation and adversarial testing | result (confirmatory, robustness), promotion |
| change tracking / version control | result (verify), fingerprint history for the subject |
| incident and dispute record | challenge, result (response) |
Constraints enforced in code:
- Every sentence in the view either quotes a promotion `claim` string verbatim or is a neutral description of a cert ("fingerprint cert X, 256 canaries + 64 anchors, noise floor from 8 runs").
- The view carries a fixed header: *"This document organizes evidence. It does not assert conformity with any regulation."*
- Lab-tier readings appear only under the lab banner, never in the documentation category.
The value is that the evidence is checkable by the reader's own engineers. That is the whole pitch; anything stronger is counsel's sentence to write, not styxx's.
---
## 13. Day-zero docket
Trigger: an open-weight release. Deadline: 48 hours.
Contents, all as certs in charon:
1. **fingerprint** on fixed-v1 (bf16, R = 8 noise floor).
2. **canary-v1 selection** against the new model, then a canary fingerprint.
3. **quant delta** — `verify --diff` bf16 vs q4_k_m on both batteries. This is the Glimmer runbook, now a routine step.
4. **predecessor diff** — `verify --diff` against the prior model in the family on fixed-v1 only (canaries don't transfer).
5. **instrument readings** — every registered instrument, in the tier the log assigns.
6. **summary** — a one-page human summary generated from the certs. Constraint: no adjectives, and every sentence ends with the cert id it came from. If a sentence can't cite a cert, it doesn't ship.
The generalization runs in the 90-day plan (gemma-2-9b, llama-3.1-8b, one more family) *are* dockets 1–3. Science and marketing on the same GPU hours.
---
## 14. Sequencing
Maps onto the 90-day plan. GPU runs keystone v2; everything below is CPU work until noted.
**Weeks 1–4**
- Envelope: JCS canonicalization, ids, ed25519 signing, schema validation for all eight types.
- fixed-v1 from styxx-bench after the gold fix; battery cert issued.
- `fingerprint` black-box channels; `verify --ref` and `--diff` with exit codes.
- Charon v0: git-backed log, Merkle tree, STH, inclusion + consistency proofs, `mirror`.
- First entries, in order: #0 this spec's hash; #1 monochord prereg v3; #2 keystone v2 prereg. Timestamped preregistration is the first real use of the log.
- Shims for `migrate`, `ci-baseline`, `ci-test`.
**Weeks 4–8**
- canary-v1 selection (GPU: the δ1 precision sweep — schedule between keystone batches).
- White-box channels (`resid`, `lens`).
- Noise-floor protocol end to end, R = 8.
- Dockets 1–3 on the generalization models.
- Instrument registry with computed tiers; `attest` with the lab banner.
**Weeks 8–12**
- Challenge certs, disputed-status computation, gossip between mirrors.
- Action certs, sub-log, `agent roll`; darkflobi as the reference emitter.
- Depth: promotion cert if the keystone result supports it; otherwise a public negative result cert and the README changes that follow from it. Both outcomes ship.
- Monochord result cert either way.
- 8.0 tagged when the four `log verify-*` commands pass on a fresh mirror clone by someone who isn't flobi.
Nothing here requires a hosted service. If a design partner wants one, the partner's need designs it.
---
## 15. Open decisions (flobi)
1. **Log key custody.** Recommend: generate offline, keep the private key on a hardware token, sign STHs from one machine. The Alienware should hold a delegated signing key that can be rotated.
2. **Redaction default.** Recommend clear text for outputs and prompts. Redacted certs can't support public claims; the default should be the strong one.
3. **Is fixed-v1 public?** Recommend yes. A private battery is an unverifiable number, which contradicts principle 1. Contamination is handled by yearly rotation, and canaries — selected per model, after release — can't be trained against in advance.
4. **Battery sizes.** Defaults above (64 / 256 / 1024, K = 64) are guesses; the noise-floor runs will tell you whether 256 is enough to separate q4 from bf16 on gemma-2-2b. Adjust once, then freeze.
5. **Anchoring.** Skip until a mirror exists. A timestamp on a chain no one is checking against is theater.
---
## Appendix A — hashing and encoding
**A.1 Encodings.** Hashes as `sha256:<64 lowercase hex>`. Keys and signatures as `ed25519:<base64url, no padding>`. Timestamps RFC 3339 UTC with `Z`.
**A.2 Weights hash.** For a local model directory: hash every `*.safetensors` shard's bytes; sort the `(filename, hex digest)` pairs by filename; `weights_sha256 = sha256(join("\n", f"{filename} {digest}"))`. Hash `config.json` and `tokenizer.json` (or `tokenizer.model`) separately. A GGUF file is hashed whole and its quant type recorded in `precision`.
**A.3 Battery order.** Items are ordered by `item_id` ascending. Channel-level hashes concatenate per-item hashes in that order, so the checksum is order-independent of how the battery was run.
**A.4 Float storage.** Log-probs are stored as IEEE-754 doubles rendered by JCS. They are compared numerically (Appendix B), never by hash.
---
## Appendix B — distance functions
One per channel. `verify` uses exactly these.
| channel | distance |
|---|---|
| exact | 1 − (matching items / items), over canaries; anchors reported separately as `identity` |
| seqlp | mean over items of \|Δ seq_logprob\| |
| topk | mean over items and positions of the L1 distance between top-k vectors after union-of-vocab alignment; a token absent from one side is assigned lp = −20 |
| resid | mean over items of the Jensen–Shannon divergence between normalized per-layer residual-norm profiles |
| lens | mean over items of \|Δ converge_layer\| / L |
Noise floors are computed with the same functions on same-subject runs, so floor and distance are always in the same units.
---
## Appendix C — canary score, explicit
```
margin(i)  = lp_top1(i) − lp_top2(i)                 # first answer token, reference precision, nats
flip1(i)   = |{ v ∈ δ1 : out_v(i) ≠ out_ref(i) }| / |δ1|
flip2(i)   = |{ v ∈ δ2 : out_v(i) ≠ out_ref(i) }| / |δ2|
flip3(i)   = |{ v ∈ δ3 : out_v(i) ≠ out_ref(i) }| / |δ3|
eligible(i)  = flip2(i) == 0
s(i)         = 0.6 · (flip1(i) + flip3(i)) / 2  +  0.4 · (1 − σ(margin(i) / τ)),   τ = 1.0
anchor(i)    = flip1 = flip2 = flip3 = 0  and  margin(i) in the top-64 of the pool
battery      = top-N eligible by s(i), stratified (≤ 25% per family)  ∪  anchors
```
Weights 0.6 / 0.4 and τ are parameters of canary-v1 and are recorded in the battery cert. Changing them makes canary-v2; it does not silently alter v1.
---
## Appendix D — what a stranger does with a cert
1. Recompute the id from the canonical bytes; check the signature.
2. Fetch the STH; verify the inclusion proof.
3. Read the recipe; obtain the subject (weights + revision, or note that it's an alias).
4. Run `styxx verify --ref <cert>`.
5. If the result exceeds the cert's noise floor on a confirmation run, log a challenge.
If any step is impossible from the cert alone, the cert is defective and the spec has failed. That is the acceptance test for 8.0.
