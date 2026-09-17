<!-- Thanks for contributing! Most PRs only need a few lines below. -->

## What does this change?

<!-- One or two sentences. Bug fix, new feature, doc tweak, refactor, etc. -->

## Why?

<!-- Link an issue or describe the failure mode. If this is a new
     calibrated instrument or an update to an existing one, name the
     candidate K=1 critical feature in `calibrated_weights_*` so a
     reviewer can sanity-check the phase-transition replication. -->

## What the diff gate will read

<!-- CI reads this section against the diff (styxx.diffgate, .github/workflows/diffgate.yml).
     It understands plain sentences about: a path this PR creates, modifies or deletes; how many
     files changed in total; how many tests were added; the name of a new function; a directory
     prefix outside of which nothing was touched. Write them with the real path, count or name.
     A contradicted sentence fails the job. Anything else is never judged and is counted as
     never-read, so a PASS covers these sentences and not the prose around them. Test results are
     UNCHECKABLE here, and a path the diff does not show is UNCHECKABLE, never accused.
     Replace this comment with your sentences. -->

## How was this tested?

<!-- - `pytest tests/` shows N passes (was M before)
     - `python scripts/dogfood_v650.py --skip-live` shows X/X green
     - For new instruments: 5-fold CV mean AUC, std, K=1 critical feature -->

## Checklist

- [ ] `pytest tests/` passes locally
- [ ] If new instrument: published `CALIBRATION_FINGERPRINT` + `CALIBRATION_NOTES` (including documented failure modes)
- [ ] If new instrument: added to atlas `benchmarks/cognometry_fingerprint_atlas_v0.json`
- [ ] If user-visible change: README + CHANGELOG updated
