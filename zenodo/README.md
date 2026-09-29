# zenodo/ — STALE, historical record only

this directory is the v7.7.x-era zenodo deposit flow (bundles, upload scripts,
`ZENODO_UPLOAD_v7.7.7.md` checklist). it is kept as record of those deposits —
do not follow it for new ones.

paper deposits are prepared as operator packages in `papers/DEPOSIT_*.md`
(e.g. `papers/DEPOSIT_frame_locality_2026_07_28.md`). start there.

the styxx 7.48.0 software version (10.5281/zenodo.23042251, in concept 10.5281/zenodo.19758618)
went through `scripts/zenodo_deposit_software_v7_48_0.py` (draft only; it cannot publish) and
`scripts/zenodo_publish_v7_48_0.py` (re-reads the draft from zenodo's side, then publishes), with
receipts in `release/`. the rule followed: nothing is published without the operator's explicit
authorization for that deposit. on 2026-09-29 the operator authorized it and the agent ran the
publish script; `release/NOTE_zenodo_software_v7_48_0_provenance.md` says how.
