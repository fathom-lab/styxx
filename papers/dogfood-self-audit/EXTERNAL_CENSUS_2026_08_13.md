> **WITHDRAWN IN PLACE, same day, by execution.** The cross-repository comparison below
> — styxx 19.4% against a pooled third-party 14.6% — **must not be cited as evidence
> about defect prevalence in either direction.** `FINDING_probe_e_styxx_2026_08_13.md`
> measured what this screen actually detects by running the gates instead of reading
> them: it over-calls by ~16% among functions it can adjudicate and **misses 68.8% of
> real dead terms**, because most dead gates carry no suspicious vocabulary at all.
>
> This file's framing — "the static rate is an **upper bound** on the defect rate" — is
> therefore wrong. It is not a bound. It is a differently biased sample, and the two
> errors point opposite ways rather than cancelling. Every count below is still
> accurate as a count of the *shapes this screen matches*; nothing below licenses an
> inference about how many gates are dead, here or in anyone else's code.
>
> Kept rather than deleted, with the reason, per `PRECOMMIT_ledger_rules_2026_08_13.md`.

# External census — the shape is not ours, and neither is the rate

**Run:** 2026-08-13. **Subject:** nine widely-used Python libraries, installed from
PyPI, written by strangers. **Tool:** `falsifiability_census.py --strict`.

The prior-art assessment closed with a demand: *"Run PROBE E against instruments in a
repository we did not write. n=1 codebase is an existence proof; the prevalence claim
needs strangers' code."* This is the first half of that — the static screen, not PROBE
E, so every number below is a count of **shapes**, never defects. The wording rule from
`PRECOMMIT_ledger_rules_2026_08_13.md` binds here.

---

## The like-for-like result

Gate-vocabulary functions only (`--strict`), which is the mode that may be compared
across repositories:

| repository | decision expressions | at-risk shape | rate |
|---|---:|---:|---:|
| peft | 10 | 0 | 0.0% |
| scipy | 72 | 5 | 6.9% |
| accelerate | 39 | 3 | 7.7% |
| huggingface_hub | 24 | 2 | 8.3% |
| datasets | 22 | 2 | 9.1% |
| sklearn | 73 | 8 | 11.0% |
| torch | 353 | 41 | 11.6% |
| transformers | 571 | 108 | 18.9% |
| numpy | 14 | 3 | 21.4% |
| **pooled third-party** | **1,178** | **172** | **14.6%** |
| *styxx (this repo)* | *283* | *55* | *19.4%* |

**About one in seven gate-shaped decision expressions in mainstream ML infrastructure
carries a term that could be constant.** That is 1,178 expressions of code we did not
write, and it is the first evidence that the pattern is a property of the ecosystem
rather than of one lab's habits.

## A correction to our own headline, published an hour earlier

The first external run used the **broad** mode and reported styxx at 20.5% against
6–15% for the libraries — "we are the worst on the list." That comparison was invalid
and the direction of the error flattered our rigor rather than our results, which makes
it no less wrong.

Broad mode admits any function carrying a boolean decision that reaches its return
value. That is most code. **styxx is a program made of gates**, so it scores higher for
a reason that has nothing to do with quality: a larger fraction of its functions are
verdict-producing by design. The denominators were measuring different things.

Under the comparable mode the gap collapses: styxx 19.4% against a pooled 14.6%, with
`transformers` at 18.9% — statistically unremarkable next to us. The honest sentence is
*"styxx sits modestly above a ~15% ecosystem baseline,"* not *"styxx is the worst
offender."* Both readings are uncomfortable; only one is supported.

Broad mode is still the right tool for auditing a single repository, because it catches
`memory_integrity` — a real dead gate named like nothing at all, which strict mode
misses entirely. **Broad for finding, strict for comparing**, and quoting the wrong one
next to another repo's number is a category error.

## What this does and does not license

**Does:** the shape is common in production ML tooling, at a measurable rate, in a
sample of 1,178 decision expressions across nine libraries. Any claim that
unfalsifiable gates are an exotic failure is refuted.

**Does not:** none of these are defects. A `PRESENCE_TEST` or text `LENGTH_TEST` in a
decision expression is a **candidate for PROBE E**, and the discrimination control
proved this screen cannot tell a dead gate from a live one sharing its syntax. Calling
any line in another project's code broken on this evidence would be exactly the
overstatement this program exists to catch — committed against people who never agreed
to be audited.

**Next, and it is the real work:** PROBE E against a population for a subset of these,
which requires their test corpora, not just their source. The static rate is an upper
bound on the defect rate and the two should never be quoted as if they were one number.

## Reproduction

```
python falsifiability_census.py --pkg <site-packages>/<lib> --strict --json
```

Nine libraries, one command each, no execution of the audited code — the screen is
static and reads ASTs only.
