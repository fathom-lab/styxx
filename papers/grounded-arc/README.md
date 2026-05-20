# grounded-arc — styxx 8.0 research arc

This directory holds the pre-registration, lockable operator decisions,
and executable scaffold for **Bet 0** of the styxx 8.0 grounded arc.

The arc design lives at:
`.styxx/RESEARCH_BRIEF_GROUNDED_ARC_2026_05_19.md` (the bet structure)
and
`.styxx/STYXX_END_TO_END_2026_05_19.md` (the 12-month end-to-end map).

## Files in this directory

- **`preregistration_2026_05_19.md`** — the pre-registration document.
  H1 hypothesis, abandon condition (Spearman ρ ≥ 0.40 on refusal-holdout
  with permutation p < 0.01), holdout corpora identification scaffolding,
  statistical methodology. Committed BEFORE any data was touched; the
  commit hash is the binding public proof of independence-of-data.

- **`operator_decisions.example.json`** — template for the three locked
  decisions (embedding model, H1 abandon ρ, ship target). The operator
  copies this to `operator_decisions.json`, fills in the values, and
  commits the amendment to lock the pre-registration.

- **`holdout_corpora.example.json`** — template for the locked holdout
  corpora identification. Once each holdout is constructed, its
  SHA-256 hash is added to `holdout_corpora.json` and committed; the
  prompts are then immutable for the duration of the arc.

## Workflow

```
1. Read .styxx/RESEARCH_BRIEF_GROUNDED_ARC_2026_05_19.md (the bet)
2. Read preregistration_2026_05_19.md (the locked methodology)
3. Operator decides §4 fields → fills operator_decisions.json
4. git commit (amendment commit hash recorded in JSON)
5. Construct 5 holdout corpora per §5 → fills holdout_corpora.json
6. git commit (corpora hashes locked)
7. Run: python scripts/dogfood/run_bet0_phase1.py
   - exit 2: preconditions not met (decisions / corpora missing)
   - exit 1: H1 FAILED — arc abandoned, write closed-negative paper
   - exit 0: H1 cleared — continue to H2/H3/H4
8. Read the output; H1 PASS → ship as 8.0; H1 FAIL → ship as
   closed-negative paper of independent value
```

The bar is set in the pre-registration. The bar will not move.

## Discipline precedents (same standard applies here)

- **deception-v1**: preregistration-killed on TruthfulQA AUC 0.59
- **text-only overconfidence**: commit `7c36ed9`, H_null on
  preregistered ≥ 0.70 floor
- **cross-vendor universality**: commit `b2675c4`, preregistration-killed
  with vendor-robust refusal labeler showing min transported AUC 0.617

If H1 fails, this arc's closed-negative paper joins that chain. It is
not a failure of the research program — it is the research program
working as intended. The chain is the credibility deposit.

## Scaffold script

`scripts/dogfood/run_bet0_phase1.py` is the executable harness. Runs
the pre-registration's H1 once decisions and corpora are committed.
Currently a structural scaffold — the per-instrument scoring and
embedding compute fills in the next session after operator decisions
land. The decision rule (Spearman ρ ≥ 0.40 → continue; below →
abandon) is already mechanical in the script.

---

*This directory was created during the ten-commit styxx 7.4.2
infrastructure session (2026-05-19). The pre-registration discipline
that produced today's `preflight`, `recover_posture`,
`streaming_preflight`, and recover_posture's PARTIAL falsifiability
result (commit `c557012`) is the same discipline that produces this
arc. Different scope, same posture.*
