# Contributing to styxx

Thanks for thinking about contributing. styxx is an active research-backed project,
and we take contributions from anyone — you don't need a PhD, just a runnable
reproducer and a clean diff.

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
pip install -e ".[dev,openai]"
pytest
```

Tests must be green before you open a PR. The full suite runs in under 60 s on a
laptop — no GPU required for tier-0 work.

## Running the full test suite

```bash
pytest                                   # all tests
pytest tests/test_classifier.py          # just the classifier
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
- New external runtime dependencies beyond `numpy`

Everything else — bug fixes, tests, docs, refactors that don't touch the public
API, performance wins — is fair game to just ship.

## Where to find things

| you want to | look in |
|---|---|
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
not vibes. See [`fathom-lab/fathom`](https://github.com/fathom-lab/fathom) for
the atlas, the pre-registrations, and the replication pipeline.

---

Thanks for caring about proprioception for artificial minds. fair winds.
