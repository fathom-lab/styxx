# RESULT — the runtime contract for SP-6: **DEAD by its own criterion, 3 of 5**

**Verdict: the kill criterion was `>= 4 of 5`. The measured score is `3 of 5`.
The idea is dead as stated, and this is the number.**

Criterion frozen and published *before* `styxx/contract.py` was written, in the
module docstring and in the community post. Replay harness:
`scripts/contract_sp6_replay_real.py`. Corpus: `benchmarks/silent_pass`.

---

## the claim that died

**SP-6 (`UNMEASURED_AS_MEASURED`)** is the SILENT-PASS subtype no static screen
reaches, because the defect is *a guard that was never written* and no pass over
source can flag code that does not exist. Measured on the corpus, `styxx.absence`
catches 0 of 5 and `absence`+`loops` together catch 1 of 5.

The proposed fix was to stop looking for the missing guard and start looking for
its consequence, at call time:

> *was there anything to measure?* (inspect the arguments)
> *did it claim something anyway?* (inspect the return)
> the conjunction is the finding.

**That is a boundary test, and it does not reach the class it was built for.**

## the numbers

Each target is the **real shipped function**, extracted with `git archive` at
`<fix_commit>~1` — the last commit where the defect was live — imported in
isolation and wrapped, unmodified. Fidelity check: SP-2026-0008 replays to
`confidence=0.6951217…`, matching the `0.695` recorded in the corpus.

| case | module | input at the boundary | caught |
|---|---|---|:--:|
| SP-2026-0008 | `forecast.py` | `{entropy: [], logprob: [], top2_margin: []}` | **yes** |
| SP-2026-0012 | `temperature.py` | `[], [], []` | **yes** |
| SP-2026-0016 | `divergence.py` | `[], []` | **yes** |
| SP-2026-0011 | `cognometrics.py` | a well-formed **20-token** response | no |
| SP-2026-0020 | `divergence.py` | four valid Japanese strings | no |

**3 / 5.**

## why it failed, exactly

The two misses are not near-misses. They are a second mechanism.

**SP-6 contains two structurally different defects, and the corpus label did not
distinguish them:**

- **boundary-degenerate** — nothing arrives, something confident leaves.
  `is_degenerate` sees it. **3 of 3 caught.**
- **interior-degenerate** — a *perfectly normal* argument arrives, and the
  emptiness is manufactured **inside** the function. **0 of 2 caught, and no
  boundary test of any tuning can catch them.**

SP-2026-0011: a 20-token response is not a degenerate input by any definition.
Scoring simply never reached phase4, so `gate` stayed `'pending'`, and the test
was `gate != "fail"`. The return *is* flagged as confident (`valid=True`); the
input side is what has nothing to see.

SP-2026-0020: four distinct Japanese strings are not a degenerate input either.
The tokenizer `[a-z0-9]+` produced the emptiness internally. This case fails on
**both** sides — and the second failure is worth recording separately: the
function returns a **bare unnamed float**, and polarity is unknowable without a
name. `looks_confident(-0.0)` → `None`; `looks_confident({"entropy": -0.0})` →
flagged. An honest measurement cannot tell whether `0.0` is the flattering end
or the alarming one unless the value carries its own identity.

## what is NOT being claimed

**3 of 3 on the reachable subset is not a pass.** The criterion was 5 cases, not
3, and redefining the denominator after seeing which cases failed is the exact
move this program forbids. The subset structure above is an **observation
generated post-hoc by a failed run**. It licenses a new preregistration on
**cases this analysis has never seen**; it licenses nothing today.

It is also not evidence that interior degeneracy is undetectable — only that
*this* instrument cannot see it. A different mechanism (contracts on the
*intermediate* values, not the boundary) is the obvious successor, and it is
deliberately not being built in the same session that watched this one die.

## what happens to the code

`styxx/contract.py` ships, labelled with the number that killed it. It is not
"the SP-6 fix"; it is a guard that caught **three real shipped defects** in this
repo and is **blind to two others by construction**, and its docstring says so
in those words. A tool that documents its own blind spot is the only kind this
program is willing to ship — shipping it as a solution, after measuring 3/5,
would reproduce the defect class it was written to remove.

The alternative — deleting it — was considered and rejected: 3 real defects is a
real yield, and deleting the evidence of a failed criterion is worse than
publishing it.

## the harness bug that would have inflated this to 4/5

The first replay used **hand-written reproductions** of the pre-fix functions and
scored **4 of 5** — a pass. It was wrong. Two reproductions passed an empty
sequence at the call boundary where the shipped code received a well-formed
argument and generated the emptiness internally. I wrote the bug into the test by
writing the test from memory of the defect rather than from the defect.

**The reconstruction was easier to catch than the code.** That is the failure mode
this benchmark exists to prevent, committed inside the benchmark, and caught only
by re-running against `git archive`. It is the seventh time in this program that a
tool committed the defect class it hunts, and the first where doing so would have
converted a death into a published success.

## cost

Five git extractions, five isolated imports, no API spend, no GPU.
