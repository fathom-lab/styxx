# OSF update plan — v6.0.0 deposit

**Current state:** the parent OSF project node `osf.io/wtkzg` was last
synced at styxx v3.x (per references across
`release/v3.5.0-bundle/CHANGELOG.md`, `release/v3.8.0-bundle/`,
`release/v3.9.x-bundle/`). The child node `osf.io/g2epj` is also
likely at v3.x. Nothing on OSF reflects v4 (`@trust`, 8-benchmark
cross-validation), v5 (refusal instrument), or v6 (tool-call drift).

**Goal:** before the marketing push, OSF has to reflect the current
three-instrument product so that anyone landing from a pre-registered
replication link sees matching numbers.

---

## (1) Build the v6.0.0 bundle

```bash
cd C:/Users/heyzo/clawd/styxx
bash release/build-zenodo-osf-bundle.sh v6.0.0    # if a prior build
                                                   # script exists; else
                                                   # model after v3.9.1
```

Bundle should contain, at minimum:

- `styxx-6.0.0-py3-none-any.whl` (from PyPI or `dist/`)
- `styxx-6.0.0.tar.gz`
- `README.md` (current, post-cleanup)
- `CHANGELOG.md` (current, post-5.0/5.1/6.0 backfill)
- `LICENSE`
- `papers/cognometry-v0.5.pdf` (full paper)
- `papers/arxiv/cognometry-v0.pdf` (arXiv submittable)
- `benchmarks/drift_calibrated_v0.json`
- `benchmarks/refusal_xstest_heldout_v2.json`
- `benchmarks/*halueval*.json` (8 AUCs)
- `scripts/drift_calibrated_v0.py`
- `scripts/compete_hhem_halueval.py`
- `tests/test_drift_v1.py`
- `tests/test_refusal_v1.py`

---

## (2) Deposit via `scripts/osf_deposit.py`

The script already exists (`scripts/osf_deposit.py`, uses
`https://api.osf.io/v2/nodes/` + `files.us.osf.io` upload). Requires
an OSF personal access token.

```bash
export OSF_TOKEN=...       # https://osf.io/settings/tokens
python scripts/osf_deposit.py \
  --parent wtkzg \
  --title "styxx v6.0.0 reproducer bundle" \
  --bundle release/v6.0.0-bundle/
```

This creates a new child node under `wtkzg`. Capture the new node ID
(format: 5 alphanumeric chars).

---

## (3) Update the parent node metadata

Log in to https://osf.io/wtkzg/ in a browser, then:

1. **Description** — replace v3.x preamble with:
   > Pre-registered replications and reproducer bundles for the styxx
   > cognometric instrument family. Current release: **v6.0.0 —
   > three calibrated instruments** (hallucination 0.998 AUC
   > HaluEval-QA; refusal 0.976 AUC XSTest GPT-4; tool-call drift
   > 0.916 AUC BFCL v3).
2. **Wiki** — if present, bump version references.
3. **Related works** — add link to Zenodo 19703527 and the new child
   node from step (2).
4. **Contributors** — verify Alexander Rodabaugh is listed with the
   correct ORCID. Add Flobi / Fathom Lab as contributor if missing.

---

## (4) Update repo references

After the new child node is live, grep the repo for `osf.io/wtkzg`
and add the new child node alongside (don't replace — wtkzg is the
parent). Specifically:

- `scripts/create_org_readme.py:86` — add new child node to the OSF
  line.
- `README.md` project section — consider adding an OSF entry if not
  already there.

---

## What does NOT change

The parent node URL `osf.io/wtkzg` is the canonical reference
everywhere. All historical pre-registrations at child nodes stay
intact as the versioned history.
