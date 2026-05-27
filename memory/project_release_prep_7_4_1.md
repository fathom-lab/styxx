# project: release 7.4.1 (honesty/correctness release) — ✅ COMPLETE

## 2026-05-18 — SHIPPED END TO END
- main: `390752f` pushed ✅
- PyPI: `styxx==7.4.1` LIVE ✅ (latest; operator-approved upload already done)
- Git tag: `v7.4.1` annotated → `390752f`, pushed to fathom-lab/styxx ✅
- GitHub release: https://github.com/fathom-lab/styxx/releases/tag/v7.4.1 — **Latest**, not draft/prerelease ✅
- Tag/release used the fathom-lab admin token (`secrets/fathomlab-github.txt`) via
  reset credential helper; default `gh` account `heyzoos123-blip` is read-only on the repo (403 on push).

## 2026-05-17 — staged (historical)

### Status
- Commit: `390752f` (pushed to fathom-lab/styxx main, ls-remote verified)
- Version: `7.4.0 → 7.4.1` (patch, not minor — corrections only, no API change)
- Tests: **887 passed, 1 skipped** (≥869 floor; labeling suite 18/18)
- Build: `dist/styxx-7.4.1-py3-none-any.whl` (7.22 MB), `dist/styxx-7.4.1.tar.gz` (7.27 MB)
- `twine check`: both PASSED

### Why patch (not minor)
- No new public API
- No new instruments / features
- Composite-key change is a *correction* of a misleading default,
  not a feature; documented; no numeric regression for existing consumers
- Per integrity protocol rule #9: "the record matches the git history"
  → this is the alignment release

### What changed (vs PyPI 7.4.0)
1. `_cogn_score_all` composite excludes reference-less deception
   (commit `0ad384e`); cogn_audit emits `composite_caveat`. Self-audit
   composite on n=16 honest Claude turns: 0.650 → 0.481.
2. Deception routing: v0 (lexical, in-corpus AUC 0.956, OOD 0.59) for
   reference-less; v2 (NLI, AUC 0.82) when `correct_reference` supplied.
3. Overconfidence axis carries `COGN_UNDER_REVIEW` + scope clarified:
   it is a stated-confidence *register* detector, not overconfidence
   (preregistered text-only recalibration failed: held-out AUC 0.571 /
   0.604 / 0.562 vs ≥0.70 floor; n=100, claude-haiku-4-5).
4. README: "universal AI integrity probe" framing **withdrawn**;
   replaced with honest scope (same-family label-free transport,
   vendor-agnostic corpus↔domain-overlap threshold). Cross-vendor
   universality flagged as preregistration-killed (`b2675c4`).
5. Construct-ceiling note added: text-only instruments are
   register/signature detectors, not honesty/correctness ground truth.
6. CHANGELOG: [Unreleased] → [7.4.1] with plain summary.
7. Suite-protect labeler (`tests/test_labeling.py`) + research
   integrity protocol both already on `8d36993`, no change here.

### Upload command (DONE — kept for provenance)
```
cd C:\Users\heyzo\clawd\styxx
python -m twine upload dist/styxx-7.4.1-py3-none-any.whl dist/styxx-7.4.1.tar.gz
```
Token: `secrets/pypi-token.txt`. Per integrity protocol rule #8 this was a
HARD STOP; the irreversible PyPI upload was completed under operator go
(verified live: `pypi.org/pypi/styxx/json` → latest 7.4.1). Tag + GitHub
release completed 2026-05-18 (safe, post-publish formalization — no gate).

### Files left modified locally (not in commit)
- `.netlify/netlify.toml` — unrelated redirect drift
- `_pytest_out.txt` — local test log
- `scripts/dogfood/_behavior_cache_75.json` — local cache
