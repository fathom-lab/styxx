# Contributing to styxx

Thanks for thinking about contributing. styxx is an active research-backed project,
and we take contributions from anyone — you don't need a PhD, just a runnable
reproducer and a clean diff.

## Where to start

The open asks, each with what it needs committed and a stated bar:

- **[#43 Re-adjudicate EXTERNAL-1](https://github.com/fathom-lab/styxx/issues/43)** — reading and
  judgement only: 130 sealed items, our panel's answers already committed, and the precision it
  checks (0.23 against a preregistered 0.95 floor) decided a shipped feature.
- **[#44 Replicate the collateral census](https://github.com/fathom-lab/styxx/issues/44)** — 69,654
  public pull requests, CPU-only and deterministic; or attack the six-class ontology where it is crude.
- **[Discussion #103, a VERIFIED that is false](https://github.com/fathom-lab/styxx/discussions/103)** —
  a (description, diff) pair where `styxx.diffgate` says VERIFIED on a claim that is false.
- **[#38 Replicate the mutual-legibility matrix](https://github.com/fathom-lab/styxx/issues/38)** —
  CPU-only, committed data. The matrix itself (`run_b37.py`) takes about an hour on a laptop (see #38,
  which also names the `*_smoke.npz` banks for a fast smoke run); `run_b45.py` reads the same clique
  and island from frame geometry in about four seconds.
- **[#39 Close the OATH trigger-recall gap](https://github.com/fathom-lab/styxx/issues/39)** — read
  [the lab's comment of 2026-08-22](https://github.com/fathom-lab/styxx/issues/39#issuecomment-5382629253)
  before the issue text: its headline share was measured against the wrong denominator.
- **[#40 A direction-sensitive cross-model discovery method](https://github.com/fathom-lab/styxx/issues/40)** —
  an open research problem; the data is committed and CPU-only.

Replicating a published result is a first-class contribution here — see
[REPLICATIONS.md](REPLICATIONS.md), where the first independent re-run of each target earns a
named credit line.

**One rule specific to this repo:** any change to `styxx/certify.py` (the claim verifier) requires
its own preregistration with frozen bars, committed *before* the mutant battery runs. That
discipline is why the verifier can be trusted; see `papers/closed-model-frontier/PREREG_oath_v06*`
for the shape.

## Ground rules

- **Be kind.** This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
- **Ship small.** One logical change per PR. A refactor + a bug fix + a docs tweak
  is three PRs.
- **Write tests.** We run on `pytest`. If you touch a measurement path, add or
  extend a test fixture; if you touch classification logic, add a case to the
  relevant classifier test.
- **No generated prose in the code.** Leave the source file as source code.

## Development setup

```bash
git clone https://github.com/fathom-lab/styxx.git
cd styxx
python -m venv .venv
source .venv/bin/activate    # or: .venv\Scripts\activate on windows
pip install -e ".[test]"     # what CI installs
pytest
```

Tests must be green before you open a PR. The full suite is slow and needs no GPU: CI's
test step took from 16 min 27 s (Python 3.10) to 25 min 4 s (Python 3.9) on
[run 40400834265](https://github.com/fathom-lab/styxx/actions/runs/40400834265). Run the
tests for what you changed while you work, and the whole suite before you push.

## Running the full test suite

```bash
pytest                                   # all tests
pytest tests/test_diffgate.py            # just the diff gate
pytest -k reflex                          # just reflex tests
pytest --lf                              # re-run the last failures
```

## Code style

- **`ruff`** for linting. Run `ruff check .` before committing.
- **Python 3.9+** is the minimum supported version. Don't use `match`/`case` or
  `3.10+`-only typing syntax.
- **Type hints** on public APIs (anything imported at the top of `styxx/__init__.py`).
  Internal helpers don't need them.
- **Docstrings** on public APIs. A one-liner is fine; an example is better.

## Commit messages

Prefer [Conventional Commits](https://www.conventionalcommits.org/):

```
fix(classifier): stop misclassifying imperative phrasing as refusal
feat(reflex): add max_rewinds ceiling to stream_anthropic
docs: add provider compatibility matrix
```

The version tags (`v2.0.1`, `v1.4.0`, …) live in the changelog, not in commit
subjects, so please don't prefix your commits with a version bump unless your PR
**is** the release commit.

## Changes that need discussion first

Before you start coding on any of these, please open an issue and wait for a
maintainer response:

- New classifier category (we have 6 — adding a 7th needs calibration data)
- Changes to the reflex callback API
- Changes to the audit-log schema (`chart.jsonl`)
- New adapters (framework integrations) — we want these, but we want to discuss
  the shape first
- New claim templates for `styxx/diffgate.py`: each one is a new kind of accusation, and no
  kind the gate still accuses with has been measured clearing the 0.95 precision floor
  (README, VERIFY)
- New external runtime dependencies beyond `numpy` and `scikit-learn`

Everything else — bug fixes, tests, docs, refactors that don't touch the public
API, performance wins — is fair game to just ship.

## Where to find things

| you want to | look in |
|---|---|
| fix the diff gate | `styxx/diffgate.py` |
| fix the GitHub Action | `action.yml`, `diffgate_action.py` |
| fix the claim verifier (OATH) | `styxx/certify.py` (needs its own preregistration, see above), `styxx/oathready.py`, `styxx/corpus_audit.py` |
| fix the island instrument | `styxx/islands.py` |
| fix the sense harness | `styxx/sense.py` |
| fix a classifier bug | `styxx/conversation.py`, `styxx/core.py` |
| fix the reflex arc | `styxx/reflex.py` |
| improve the OpenAI wrapper | `styxx/adapters/openai.py` |
| add an adapter | `styxx/adapters/` |
| fix a CLI command | `styxx/cli.py` |
| fix the dashboard | `styxx/dashboard.py` |
| fix calibration | `styxx/calibrate.py` |
| touch the atlas centroids | `styxx/centroids/` (needs a fresh hash — see `styxx doctor`) |

## Reporting bugs

Open an issue with:
1. The version of styxx (`python -c "import styxx; print(styxx.__version__)"`).
2. The model / provider you're using.
3. A minimal runnable reproducer (≤ 20 lines).
4. What you expected vs what you got.

Please don't paste a screenshot of a traceback — paste the text. We search
issues by error string.

## Reporting security issues

See [SECURITY.md](SECURITY.md) — don't open a public issue.

## Research contributions

The cognitive classification pipeline is grounded in real experiments, many of
them pre-registered. If your contribution affects accuracy numbers (classifier
weights, centroids, threshold tuning), it needs to be backed by a dataset run,
not vibes. The preregistrations, findings and negatives are in [`papers/`](papers/)
(indexed by [`papers/INDEX.md`](papers/INDEX.md)), and the replication route is
[REPLICATIONS.md](REPLICATIONS.md). [`fathom-lab/fathom`](https://github.com/fathom-lab/fathom)
holds the atlas.

---

Thanks for caring about proprioception for artificial minds. fair winds.
