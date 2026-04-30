# styxx 7.1.0 ship checklist

Run-through for the actual phase 3 publish. All steps are commands flobi runs (or authorizes Claude to run with explicit go).

Pre-flight already complete:
- [x] `styxx/reward.py` shipped (261 LOC, cognometric reward signal)
- [x] `styxx/synth.py` shipped (BOUNDARY BREAK — synthetic preference-pair generator via inverse cognometry; 20/20 craft success, +0.839 mean delta, 20/20 round-trip ranking correct)
- [x] `styxx/_demo_baselines.py` shipped (strawman approval-style baseline)
- [x] `tests/test_reward.py` — 14/14 pass
- [x] `tests/test_synth.py` — 7/7 pass
- [x] full styxx test suite — **821/822 pass** (1 skipped, 0 regressions)
- [x] `data/cognometric_rlhf_demo_v0.jsonl` — 20 curated triples
- [x] `examples/cognometric_reward_basic.py` — basic usage
- [x] `examples/cogn_rlhf_divergence.py` — divergence demo (cogn 17/20 vs approval 6/20)
- [x] `examples/cogn_rlhf_divergence_colab.ipynb` — Colab notebook
- [x] `examples/synth_preference_pairs.py` — synth pair-generator demo
- [x] `examples/trl_ppo_integration.py` — TRL PPOTrainer skeleton
- [x] `release/cogn_rlhf_divergence_v0.json` — saved divergence result
- [x] `release/synth_preference_pairs_v0.jsonl` — 20 synth-generated preference pairs
- [x] `release/v710_release_notes.md` — `gh release --notes-file` source
- [x] `styxx/__init__.py` — top-level `fathom_reward`, `FathomRewardModel`, `craft_preference_pair`, `generate_preference_pairs` exports
- [x] `CHANGELOG.md` — 7.1.0 entry with reward + synth sections
- [x] `pyproject.toml` — version bumped 7.0.0 → 7.1.0
- [x] `dist/styxx-7.1.0-py3-none-any.whl` — built and twine-checked (5.89 MB, includes synth)
- [x] fresh-venv smoke install verified (synth round-trip: chosen=0.112, rejected=1.000, delta=+0.888)

---

## Phase 3 — publish (irreversible — explicit go required for each)

### Step 1. Verify the wheel installs cleanly in a fresh venv

```powershell
cd C:\Users\heyzo\clawd\styxx
python -m venv .venv-71-smoke
.\.venv-71-smoke\Scripts\Activate.ps1
pip install --upgrade pip
pip install dist/styxx-7.1.0-py3-none-any.whl
python -c "from styxx import fathom_reward, FathomRewardModel; print('install ok'); r = fathom_reward(prompt='You agree?', completion='Absolutely!'); print(f'reward={r:.3f}')"
deactivate
```

Expected: `install ok` + a reward print. If it fails, fix and rebuild before continuing.

### Step 2. Git commit

16 changed/added files. Specify exactly to avoid `git add .` polluting:

```powershell
cd C:\Users\heyzo\clawd\styxx
git add styxx/reward.py styxx/synth.py styxx/_demo_baselines.py styxx/__init__.py tests/test_reward.py tests/test_synth.py data/cognometric_rlhf_demo_v0.jsonl examples/cognometric_reward_basic.py examples/cogn_rlhf_divergence.py examples/cogn_rlhf_divergence_colab.ipynb examples/synth_preference_pairs.py examples/synth_multi_instrument.py examples/trl_ppo_integration.py release/cogn_rlhf_divergence_v0.json release/synth_preference_pairs_v0.jsonl release/synth_multi_instrument_v0.json release/v710_release_notes.md CHANGELOG.md pyproject.toml release/SHIP_v710_CHECKLIST.md
git status
```

Verify only those 16+ files are staged (also include any docs/ updates if you add them). Then commit:

```powershell
git commit -m "7.1.0: styxx.reward + styxx.synth - cognometric reward + inverse-cogn synth

styxx.reward: first reward signal calibrated against cognitive failure
modes instead of human approval. Drop-in for trl PPO/GRPO/DPO trainers.

styxx.synth: synthetic preference-pair generator composing v7.0.0
inverse cognometry with the new reward signal. Recursive: fathom's
attack module generates training data for fathom's reward signal.

Results on curated 20-pair sycophancy benchmark:
  cognometric reward       17/20  (85%)
  approval baseline         6/20  (30%, below chance)
  inversions               13/20  (65%)

Synth pair-generation results (target_score=0.85):
  crafted with positive delta:  20/20
  reached saturation:           20/20
  mean delta:                   +0.839
  cogn_reward round-trip:       20/20 ranks chosen above rejected

Universal-perturbation moat: v7.0.0 perturbation lifts cross-fire by
+0.468 in attack mode but produces +0.000 lift on cogn-RLHF reward
(dominant instrument already saturated).

Top-level API:
  from styxx import fathom_reward, FathomRewardModel
  from styxx import craft_preference_pair, generate_preference_pairs

21 new tests, 821/822 full suite pass."
```

### Step 3. Tag v7.1.0

```powershell
git tag -a v7.1.0 -m "7.1.0: cognometric reward signal for RLHF"
git tag -l | tail -5   # verify tag created
```

### Step 4. Push commit + tag

```powershell
git push origin main
git push origin v7.1.0
```

### Step 5. Upload to PyPI (irreversible — version locked forever)

```powershell
cd C:\Users\heyzo\clawd\styxx
python -m pip install -U twine
python -m twine check dist/styxx-7.1.0-py3-none-any.whl
python -m twine upload dist/styxx-7.1.0-py3-none-any.whl
# enter PyPI token when prompted (or use ~/.pypirc)
```

Verify: https://pypi.org/project/styxx/7.1.0/ should show the new version within ~30 seconds.

### Step 6. GitHub release

```powershell
gh release create v7.1.0 dist/styxx-7.1.0-py3-none-any.whl --title "7.1.0: styxx.reward — cognometric reward signal for RLHF" --notes-file release/v710_release_notes.md
```

(Generate `release/v710_release_notes.md` from the CHANGELOG 7.1.0 section before this step.)

### Step 7. Tweet

Post the thread or single tweet from `research/eeg_pilot/v710_tweet_drafts.md` (single tweet preferred for first-fire; thread version if amplification needed).

### Step 8. Update fathom.darkflobi.com

The site mirrors styxx version. Bump the version reference + add a CHANGELOG link.

```powershell
# In Desktop/clawd-clean/darkflobi-site/, update version refs
# Then deploy via: bash clawd/scripts/deploy-fathom-site.sh
# (per memory: NEVER netlify deploy --dir=. from .styxx cwd)
```

---

## Post-ship monitoring (first 24 hours)

- PyPI download count
- GitHub stars delta
- Twitter engagement
- Any installation issues / bug reports

If a critical bug surfaces in the first 24 hours, yank with `python -m twine yank styxx==7.1.0 --reason "..."` and ship 7.1.1 with the fix.

---

## Parallel non-publish work (already drafted, no go required)

- `research/eeg_pilot/arxiv_paper_outline.md` — paper #1 outline, target submission 2026-05-28
- `research/eeg_pilot/outreach_drafts.md` — 4 outreach emails ready to send (OpenBCI / Bitbrain / Pearl IRB / Advarra)
- `research/eeg_pilot/v710_tweet_drafts.md` — single tweet + thread version
