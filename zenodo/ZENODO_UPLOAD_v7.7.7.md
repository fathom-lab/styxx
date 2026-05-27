# Zenodo deposit checklist — styxx v7.7.7

**Operator action required.** This document is the step-by-step for the manual Zenodo upload (no API token in `secrets/`, so the upload itself can't be automated; the bundle + all metadata is prepared below for copy-paste).

**Bundle:** `zenodo/styxx-v7.7.7-zenodo-bundle.zip` (14.8 MB, 21 files)
**Concept DOI:** `10.5281/zenodo.19326174` (this deposit gets a release-specific DOI under it)
**Previous deposit in this concept:** v7.2.0 / record 20130041 (May 12, 2026)

---

## Step 1 — open Zenodo in a browser

1. Go to https://zenodo.org/uploads
2. Sign in with the Fathom Lab Zenodo account (the one that owns concept DOI 19326174).
3. Find the existing deposit for v7.2.0 (record 20130041) and click **"New version"** at the top right. This auto-links the new deposit as a version of the existing concept DOI, so the academic chain stays connected.

(If "New version" isn't visible, you can also create a fresh upload and manually set `related_identifiers` to point at concept DOI 19326174 with relation `isNewVersionOf` — Step 4 below covers this.)

---

## Step 2 — upload the bundle

Drag-and-drop or browse to: `zenodo/styxx-v7.7.7-zenodo-bundle.zip`

Optional: upload the wheel + sdist + paper individually as separate files (in addition to the bundle zip), which makes them directly clickable on the Zenodo page. From the bundle's `v7.7.7/` directory:
- `styxx-7.7.7-py3-none-any.whl`
- `styxx-7.7.7.tar.gz`
- `PAPER_decorrelation_ceiling_2026_05_27.md`
- `darkcore_benchmark_2026_05_27.json`
- `LEADERBOARD.md`

---

## Step 3 — copy-paste this metadata

### Title

```
Fathom v24 / styxx v7.7.7: The Decorrelation Ceiling — seven-method empirical floor on reference-free detection of cross-vendor consensus hallucination
```

### Authors

```
Rodabaugh, Alexander · Fathom Lab
```

### Resource type

`Publication` → `Working paper`

### Publication date

`2026-05-27`

### Version

`v24` (continuing the v23 → v24 sequence used by the concept DOI)

### Language

`English`

### License

`Creative Commons Attribution 4.0 International (CC-BY-4.0)` for paper/data/findings.
Note in description: styxx code is MIT (see `LICENSE` in the bundle).

### Keywords (paste as comma-separated)

```
decorrelation ceiling, consensus hallucination, reference-free detection, cross-vendor council, AI integrity, cognometric instruments, dark core, justification divergence, perturbation fragility, agreement fracture, injected competitor test, pre-registration, falsifiability, dogfood, self-audit, Pareto frontier, cultural priors, folklore, AI alignment, LLM safety, styxx, Fathom
```

### Description (HTML — paste as the description block; Zenodo accepts HTML)

<!-- COPY EVERYTHING BETWEEN THE START/END MARKERS -->

```html
<!-- DESCRIPTION START -->
<p><strong>Headline: seven independent pre-registered methods on the consensus-hallucination dark core, all closed-negative within their corpus's scope. The synthesis stating the bimodal prediction was committed two days before half of them ran. The data resolved on the load-bearing-floor branch the synthesis named.</strong></p>

<p><strong>Methods (all bars locked + pushed to public origin before each probe fired; verifiable from git history):</strong></p>
<ul>
<li><strong>Detection #1</strong> — perturbation-fragility (Dark Matter swing #1): partial, fragile shell only.</li>
<li><strong>Detection #2</strong> — agreement-fracture (CVPD): clean negative, lift −0.32.</li>
<li><strong>Detection #3</strong> — justification-divergence (JD): clean negative, <strong>INVERTED</strong> — stubborn dark-core items have the <em>most</em> convergent across-vendor justifications (mean JD 0.022 vs truth 0.067). Three vendors share the wrong fact AND the supporting story.</li>
<li><strong>Constructive #1</strong> — neutral injection (ICT, n_folk=4): immovability floor, 0/4 folklore yield.</li>
<li><strong>Constructive #2</strong> — neutral injection on hand-curated 30-folklore corpus: SHORTFALL n_folk=2, but a genuine finding fell out — 28/30 well-known cultural myths are already corrected in 2026 frontier-model council baseline; the practical dark core is narrower than the curation assumed.</li>
<li><strong>Constructive #3</strong> — authoritative injection on same corpus: SHORTFALL + descriptive: same 2 folk items lifted in both framings (no differential framing effect), +0.05 truth-yield to authoritative falsehood (auth-sycophancy direction signal, n=1).</li>
<li><strong>Classification #1</strong> — sentence-transformer + balanced LR: FAIL K2 + K3. The dark core is also dark to text-only classification with this stack at n≈80 training items.</li>
</ul>

<p><strong>Methodological contributions:</strong></p>
<ul>
<li><strong>The closed-loop self-audit demonstration.</strong> The producer's own product (<code>styxx audit</code>, <code>styxx critique</code>) caught the producer drifting from the producer's own derived discipline within the same session; the producer revised in the corrected register; the gate cleared. Composite 0.358 → 0.174 with the Pareto trade-off (sycophancy↔overconfidence on text-only register) observed live. We are unaware of prior published demonstrations of this recursion in the AI-integrity-instrument literature.</li>
<li><strong>Five in-session falsifications</strong> of the producer's own claims, all recorded in place with strikethrough rather than rewritten. Proposed as a methodological pattern for AI-research integrity.</li>
<li><strong>The paper itself self-audits.</strong> The abstract scores sycophancy 0.66 and §5 scores 0.62 — the documented closed-negative restrained-FP firing on the paper's own status-reporting register. The paper acknowledges this in real time rather than gaming the gate.</li>
</ul>

<p><strong>Deployable artifacts:</strong></p>
<ul>
<li><strong>The dark-core benchmark dataset</strong> (108 labeled records across 4 classes; folklore 34 / pseudoscience 6 / factual-error 13 / truth 55) with the seven-method empirical floor baked in as the bar future routing approaches need to beat.</li>
<li><strong>The public-challenge leaderboard</strong> (<code>LEADERBOARD.md</code>) with the seven-method floor as Baseline-001 + three concrete reference baselines (002 classifier, 003 length heuristic, 004 random class). External submissions go through CI auto-verification.</li>
<li><strong>The styxx CLI family</strong>: <code>styxx audit</code> (per-turn auditing), <code>styxx critique</code> (audit + register-fix suggestions with mandatory scope-bound), <code>styxx gauntlet</code> (run a candidate method against the floor), <code>styxx leaderboard</code> (terminal view of the board), <code>styxx data-dir</code> (discover the active chart.jsonl path).</li>
</ul>

<p><strong>What this deposit does NOT claim:</strong></p>
<ul>
<li>A deployable positive routing primitive. Three deployable-positive paths tested closed-negative.</li>
<li>External replication or adoption. Out-of-session by construction.</li>
<li>The "extraordinary" framing the prompt sought. What this work bought is the <em>foundation</em> for credibility — receipts, discipline pattern, falsifications in place, deployable artifacts, public challenge — not the audience reception that compounds over time.</li>
</ul>

<p><strong>Reproduce:</strong></p>
<pre>
pip install styxx==7.7.7
styxx leaderboard --rows-only      # see the floor
styxx gauntlet --method styxx.gauntlet:_majority_baseline_predict --task classification
</pre>

<p><strong>Source:</strong> <a href="https://github.com/fathom-lab/styxx">github.com/fathom-lab/styxx</a> · <strong>Tag:</strong> <a href="https://github.com/fathom-lab/styxx/releases/tag/v7.7.7">v7.7.7</a> · <strong>PyPI:</strong> <a href="https://pypi.org/project/styxx/7.7.7/">pypi.org/project/styxx/7.7.7</a> · <strong>Paper:</strong> <a href="https://github.com/fathom-lab/styxx/blob/main/papers/PAPER_decorrelation_ceiling_2026_05_27.md">PAPER_decorrelation_ceiling_2026_05_27.md</a></p>

<p><strong>Predecessor:</strong> v23 (<a href="https://doi.org/10.5281/zenodo.20130041">10.5281/zenodo.20130041</a>) — Fathom v23 / styxx v7.2.0, F10 Self-Healing Reflex + Cognometric Inversion. v24 advances by shipping the seven-method empirical floor, the closed-loop self-audit demonstration, the public-challenge leaderboard infrastructure, and the styxx CLI family for audit/critique/gauntlet/leaderboard.</p>

<p><strong>License:</strong> CC-BY-4.0 (this deposit, paper, data, findings, leaderboard). MIT (styxx Python code, see <code>LICENSE</code> file in the bundle).</p>

<p><strong>Notes:</strong> Drafted with assistance from Claude Opus 4.7 (1M context) during a 2026-05-27 session. Claude-authored commits in the styxx repo carry the Claude noreply author signature. The closed-loop self-audit demonstration in §5 of the paper means this paper is itself eligible for audit under the same <code>styxx audit</code> CLI it describes — a methodological recursion we expect the field to take less than a year to either replicate or refute.</p>
<!-- DESCRIPTION END -->
```

### Related identifiers (add each as a separate row)

| identifier | relation | scheme | resource type |
|---|---|---|---|
| `10.5281/zenodo.19326174` | Is part of (concept DOI) | DOI | (auto) |
| `10.5281/zenodo.20130041` | Is new version of (v23 predecessor) | DOI | Publication / Working paper |
| `https://github.com/fathom-lab/styxx` | Is supplement to | URL | Software |
| `https://github.com/fathom-lab/styxx/releases/tag/v7.7.7` | Is supplement to | URL | Software |
| `https://pypi.org/project/styxx/7.7.7/` | Is supplement to | URL | Software |
| `https://styxx.org` | Is documented by | URL | Publication / Other |

---

## Step 4 — sanity-check before publish

Before clicking **"Publish"** in Zenodo's UI:

- [ ] Bundle attached (`styxx-v7.7.7-zenodo-bundle.zip`, ~14.8 MB).
- [ ] Title matches what's above.
- [ ] Author is `Rodabaugh, Alexander · Fathom Lab`.
- [ ] Resource type = Working paper.
- [ ] Version = `v24`.
- [ ] License = CC-BY-4.0.
- [ ] Description HTML pasted in full.
- [ ] Related identifiers all added (especially the concept DOI relation).
- [ ] "New version" linked to record 20130041 (so concept DOI 19326174 stays the latest-pointer).

---

## Step 5 — after publishing

Once Zenodo assigns the release-specific DOI:

1. Copy the new DOI (format: `10.5281/zenodo.XXXXXXXX`).
2. Update `papers/PAPER_decorrelation_ceiling_2026_05_27.md` §11 (Citation) with the new DOI.
3. Update `LEADERBOARD.md`'s citation block with the new DOI.
4. Optionally edit the GH release notes at https://github.com/fathom-lab/styxx/releases/tag/v7.7.7 to include `DOI: 10.5281/zenodo.XXXXXXXX` near the top.
5. Add a small commit: `docs: add Zenodo DOI for v7.7.7 release`.

---

## What this deposit accomplishes

Converts the work from "on origin under main" + "pip-installable" to "academic-citable via permanent DOI." Researchers writing follow-up papers will cite `10.5281/zenodo.XXXXXXXX`; future deposits to this concept DOI inherit the citation chain. The Zenodo deposit is the durable academic anchor.

Operator-territory only because: (a) no Zenodo API token in `secrets/`, (b) the v23 deposit was manually curated and the operator owns the Fathom Lab Zenodo account, (c) license + metadata decisions are project-strategy calls. Everything else (bundle, README, full metadata draft, citation update locations) is prepared above for copy-paste.
