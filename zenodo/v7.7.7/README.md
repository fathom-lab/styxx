# styxx 7.7.7 — Zenodo deposit bundle

**Title:** *The Decorrelation Ceiling: a seven-method empirical floor on reference-free detection of cross-vendor consensus hallucination + the styxx integrity-instrument family v7.7.7*

**Release date:** 2026-05-27
**Project:** styxx (Fathom Lab) · https://github.com/fathom-lab/styxx
**PyPI:** https://pypi.org/project/styxx/7.7.7/
**Git tag:** [v7.7.7](https://github.com/fathom-lab/styxx/releases/tag/v7.7.7)
**Concept DOI:** [10.5281/zenodo.19326174](https://doi.org/10.5281/zenodo.19326174) (this deposit gets a release-specific DOI under the same concept)
**License:** CC-BY-4.0 (paper, dataset, findings) · MIT (styxx code)

---

## What this deposit contains

### Paper-grade artifacts

| file | what it is |
|---|---|
| `PAPER_decorrelation_ceiling_2026_05_27.md` | The formal paper draft: 12 sections, ~6000 words, self-auditing §5.4 demonstration, four in-session falsifications recorded in place |
| `REPORT_decorrelation_ceiling_v2_2026_05_27.md` | Field-facing capstone with the seven-method arc summary table + 2026-05-27 update block recording the bimodal prediction's resolution |
| `SYNTHESIS_decorrelation_ceiling_2026_05_25.md` | The original synthesis (2026-05-25) + 2026-05-27 update block. The synthesis was committed **before** half its predicted swings ran; the update records the receipts |

### Pre-registrations (committed BEFORE their respective data runs — verifiable from git history)

| file | bars locked at commit | run + result |
|---|---|---|
| `preregistration_jd_2026_05_25.md` | `959ee64` | clean negative, inverted (FINDING below) |
| `preregistration_ict_2026_05_25.md` | `637b320` | immovability floor (FINDING below) |
| `preregistration_ict_folklore_2026_05_27.md` | `2cffcec` | SHORTFALL n_folk=2 (FINDING below) |
| `preregistration_ict_authoritative_2026_05_27.md` | `b27b42c` | SHORTFALL + descriptive auth-sycophancy (FINDING below) |
| `preregistration_darkcore_classifier_2026_05_27.md` | `646dcb0` | FAIL K2 + K3 (closed-negative baseline) |

### Findings (the run results, each scored against the pre-registered bars)

| file | result against bars |
|---|---|
| `FINDING_jd_2026_05_27.md` | J1 AUC 0.46 / J2 AUC 0.433 — FAIL, inverted direction (stubborn dark core has *most* convergent justifications) |
| `FINDING_ict_2026_05_27.md` | I1 folklore yield 0.00 — immovability floor |
| `FINDING_ict_folklore_2026_05_27.md` | n_folk = 2 SHORTFALL — 28/30 curated folklore already corrected in baseline |
| `FINDING_ict_authoritative_2026_05_27.md` | SHORTFALL + descriptive: same folk lifted in both framings, +0.05 auth-sycophancy direction on truth |
| `FINDING_pareto_frontier_2026_05_27.md` | the closed-loop self-audit + sycoph↔overconfidence Pareto observation |
| `FINDING_product_exploration_2026_05_27.md` | five-primitive cross-check + the falsified-in-place observation #1 |

### Deployable artifacts

| file | what it is |
|---|---|
| `darkcore_benchmark_2026_05_27.json` | 108 labeled records, 4 classes (folklore 34 / pseudoscience 6 / factual-error 13 / truth 55). The empirical floor is baked in as the bar future routing approaches need to beat |
| `LEADERBOARD.md` | The public gauntlet leaderboard with the seven-method floor as Baseline-001 + concrete reference baselines (002, 003, 004) + the submission protocol + the bars + the honor system |
| `CHANGELOG.md` | The 7.4.x → 7.7.7 release history with the discipline pattern visible per release |
| `LICENSE` | MIT (code) + CC-BY-4.0 (paper/data, by separate declaration in this README) |

### Python distributions

| file | what it is |
|---|---|
| `styxx-7.7.7-py3-none-any.whl` | Installable wheel: `pip install styxx==7.7.7` |
| `styxx-7.7.7.tar.gz` | Source distribution |

---

## Headline contributions

1. **Seven-method empirical floor** on the consensus-hallucination dark core: four detection methods (Dark Matter perturbation-fragility, CVPD agreement-fracture, JD justification-divergence, ICT neutral injection), one classification method (sentence-transformer + balanced LR), two constructive variants (ICT-folklore, ICT-authoritative). All seven closed-negative within scope. Bars locked + pushed to public origin before each probe fired.
2. **The justification-divergence inversion** — the stubborn dark-core class has the *most* convergent across-vendor justifications, not the least (mean JD = 0.022 vs truth = 0.067). Three vendors share the wrong fact AND the supporting story.
3. **The closed-loop self-audit demonstration** — the producer's own product (`styxx audit`, `styxx critique`) caught the producer drifting from the producer's own derived discipline within the same session, the producer revised, the gate cleared. Composite 0.358 → 0.174 with the Pareto trade-off observed live.
4. **Four in-session falsifications** of the producer's own claims, all recorded in place with strikethrough rather than rewritten. Proposed as a methodological pattern for AI-research integrity.
5. **A labeled benchmark dataset + public-challenge leaderboard** — the empirical floor is now a runnable, terminal-accessible, CI-verified public challenge. `pip install styxx==7.7.7 && styxx leaderboard --rows-only` shows the current floor immediately.
6. **A `styxx audit` / `styxx critique` / `styxx gauntlet` / `styxx leaderboard` CLI primitive family** — atomic per-turn auditing, register-fix suggestions with mandatory scope-bound, public-challenge runner, terminal leaderboard.

---

## How to cite

```
Rodabaugh, Alexander (Fathom Lab). 2026.
"The Decorrelation Ceiling: A Seven-Method Empirical Floor on Reference-Free
Detection of Cross-Vendor Consensus Hallucination."
styxx 7.7.7, fathom-lab/styxx git main, Zenodo.
DOI: <release-specific DOI assigned on deposit, under concept 10.5281/zenodo.19326174>
URL: https://github.com/fathom-lab/styxx/releases/tag/v7.7.7
Paper: papers/PAPER_decorrelation_ceiling_2026_05_27.md
```

---

## Provenance

- The git history at `github.com/fathom-lab/styxx` (under MIT for code, CC-BY-4.0 for paper/data) is the authoritative source. Every artifact in this deposit is reproducible from commits on `main` between `bcd4208` and the v7.7.7 tag.
- The seven pre-registration commits all preceded their respective probe results — `git log --oneline --reverse papers/consensus-hallucination/` shows the order.
- Five in-session falsifications were recorded in place across the session: the C1 register-law bar, the set_session propagation observation, two probe auto-verdict labels, and the 7.7.5 wheel bundling miss. None were rewritten.

This deposit captures the project state as of 2026-05-27. Subsequent releases continue at the concept DOI 10.5281/zenodo.19326174.
