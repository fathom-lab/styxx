# Zenodo Deposit — Cognometry v0 (DONE)

**Status: deposited 2026-04-23.**

**DOI:** [10.5281/zenodo.19703527](https://doi.org/10.5281/zenodo.19703527)
**Record:** https://zenodo.org/record/19703527
**License:** CC-BY-4.0
**Deposition ID:** 19703527
**Uploaded:** `cognometry-v0.pdf`, `cognometry-v0.md`,
`cognometry-research-agenda-2026.md`

Receipt JSON: `release/zenodo-deposit-receipt.json`

---

## If you need a new deposition

Use `scripts/zenodo_deposit.py`. Set `ZENODO_TOKEN` env var (token in
`clawd/secrets/arxiv-creds.txt` under `[ZENODO]`), then:

```bash
ZENODO_TOKEN=... python scripts/zenodo_deposit.py
```

The script handles create → upload → metadata → publish in one shot,
writes a receipt to `release/zenodo-deposit-receipt.json`, and returns
the DOI on stdout.

## Related identifiers already linked on the deposited record

- software: https://github.com/fathom-lab/styxx
- software: https://pypi.org/project/styxx/4.0.1/
- dataset: https://huggingface.co/datasets/PatronusAI/HaluBench
- dataset: https://huggingface.co/datasets/pminervini/HaluEval
- dataset: https://huggingface.co/datasets/truthfulqa/truthful_qa
- model:   https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

## Where the DOI is cited

- `fathom.darkflobi.com/cognometry` — footer "paper (DOI)" link
- `release/LAUNCH-DAY.md` — HN self-comment + LinkedIn + X quote-tweet
- `release/cognometry-launch-copy.md` — top-level links
- `examples/cognometry_colab.ipynb` — notebook preamble

Re-derive the DOI from the receipt JSON if you need it programmatically:

```bash
python -c "import json; print(json.load(open('release/zenodo-deposit-receipt.json'))['doi'])"
```
