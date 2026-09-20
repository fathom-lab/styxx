# telescope — current status

**The daily run has not happened since 2026-05-09. The public scoreboard at
[fathom.darkflobi.com/scoreboard](https://fathom.darkflobi.com/scoreboard) is serving that day's
snapshot.** `README.md` calls this a *daily* measurement layer. Until the two blockers below are
cleared, it is not one, and this file exists so that nothing in the repository says otherwise.

Last real measurement: **`telescope/data/latest.json`, `ts_iso: 2026-05-10T01:47:21Z`** — 21 prompts,
two models (`gpt-5`, `gpt-5-mini`, both OpenAI). `telescope/data/runs/` holds six run ledgers in
total, five from 2026-05-03 and one from 2026-05-09. That is the entire history of a daily runner.

---

## Blocker 1 — no vendor keys on the repository

`.github/workflows/telescope.yml` checks for `TELESCOPE_OPENAI_KEY`, `TELESCOPE_ANTHROPIC_KEY` and
`TELESCOPE_OPENROUTER_KEY`. None is set, so the `check vendor keys` step sets `present=false`, the
three steps that install, run and commit are all skipped by `if: steps.keys.outputs.present ==
'true'`, and **the job succeeds**. It prints a notice while doing it:

> `telescope skipped — no TELESCOPE_* vendor key secrets configured (expected state, not a failure)`

That reasoning is right for a fork. On this repository it is not, because this repository is what
the public scoreboard reads. A green tick here means "did not measure", and the scoreboard says
nothing about the date of what it is showing.

Until 2026-09-18 the job was red anyway, for an unrelated reason: `actions/setup-python@v5` was
configured with `cache: 'pip'` and its post-step failed, 138 consecutive times since 2026-05-04.
#135 removed the cache. That repair is correct and it changes nothing about measurement — it
converts a daily red X into a daily green tick over the same absence. The red was not telling
anyone the telescope was down either, but it was at least not claiming the opposite.

## Blocker 2 — the corpus was not in the repository

`run.py` opens `telescope/prompts.json`, the held-out 21-prompt corpus every telescope number is
derived from:

```python
def load_prompts() -> list:
    p = HERE / "prompts.json"
    if not p.exists():
        sys.exit(f"missing {p} — populate it first.")
```

`.gitignore` ignored it. The rule was a bare `telescope/*.json`, filed in the block headed
`── LaTeX build artifacts ──` between `*.toc` and `packages/styxx-scope-*.zip`; it was there to
sweep up stray run output at `telescope/`'s top level and it caught the corpus as well. So a fresh
checkout had no corpus, and a scheduled run **with** keys configured would have exited 1 at
`load_prompts()` before reaching a single model.

Blocker 1 hid blocker 2 for the whole life of the workflow: the run step never executed, so the
missing corpus was never reached. The `.gitignore` rule is now narrowed (`!telescope/prompts.json`)
and moved into a block of its own that says what it is for.

**`telescope/prompts.json` is still not in the tree.** Un-ignoring it does not create it. It has to
be committed from wherever it currently lives.

---

## What `telescope/data/timeseries.jsonl` actually is

`README.md` describes it as the *long-running per-model trajectory*. It is not one, and nothing
maintains it: `run.py` writes `data/runs/telescope__<ts>.json` and `data/latest.json` and never
mentions `timeseries.jsonl` at all.

What the file holds is three rows dated **2026-04-24**, before any run ledger in `data/runs/`, in
the `_archive/v0_2026-04-24.json` schema (`K`, `C`, `D`, `trust`, `faults`) rather than the schema
the current runner produces (`composite_dishonesty`, `sycophancy`, `deception`, `overconfidence`,
`refusal`). The three rows name `gpt-4o-mini`, `claude-haiku-4-5` and `llama-3.2-3b-instruct` —
none of which appears in any run ledger — and carry **byte-identical metrics across all three**:
`K 0.4355 · C 0.0 · D 0.2044 · trust 0.8193`, with identical fault vectors.

It is v0 seed data wearing a measurement's name. The instrument itself does discriminate: the last
real run separates `gpt-5` from `gpt-5-mini` at `composite_dishonesty` 0.3736 vs 0.2805 and
`deception` 0.3014 vs 0.1499. Nothing is wrong with the telescope's scoring; the trajectory file
simply was never wired up.

---

## To restart it

1. Add at least one of `TELESCOPE_OPENAI_KEY`, `TELESCOPE_ANTHROPIC_KEY`,
   `TELESCOPE_OPENROUTER_KEY` under *Settings → Secrets and variables → Actions*.
   Budget reference is in `RUNBOOK.md`: fast tier only is roughly $2/month.
2. Commit `telescope/prompts.json`. Since the `.gitignore` narrowing, a plain `git add` works.
3. Dispatch the workflow once and read the log rather than the tick — the run should print
   `[telescope] prompts: 21` and then per-model progress. A green job that printed the skip notice
   has measured nothing.
4. Delete the stale sections of this file as they stop being true. `tests/test_telescope_status_is_honest.py`
   fails if this file claims a pause while `latest.json` is fresh, and fails if `latest.json` is
   stale while this file claims the telescope is running.

Until step 1 and step 2 are both done, the scoreboard is showing 2026-05-10.
