<!-- Thanks for contributing! Most PRs only need a few lines below. -->

## What does this change?

<!-- One or two sentences. Bug fix, new feature, doc tweak, refactor, etc. -->

## Why?

<!-- Link an issue or describe the failure mode. If this is a new
     calibrated instrument or an update to an existing one, name the
     candidate K=1 critical feature in `calibrated_weights_*` so a
     reviewer can sanity-check the phase-transition replication. -->

## How was this tested?

<!-- - `pytest tests/` shows N passes (was M before)
     - `python scripts/dogfood_v650.py --skip-live` shows X/X green
     - For new instruments: 5-fold CV mean AUC, std, K=1 critical feature -->

## Checklist

- [ ] `pytest tests/` passes locally
- [ ] If new instrument: published `CALIBRATION_FINGERPRINT` + `CALIBRATION_NOTES` (including documented failure modes)
- [ ] If new instrument: added to atlas `benchmarks/cognometry_fingerprint_atlas_v0.json`
- [ ] If user-visible change: README + CHANGELOG updated
